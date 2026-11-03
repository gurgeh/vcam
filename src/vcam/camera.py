"""Find the webcam and lock its auto controls.

Clean-plate matting (BackgroundMattingV2) compares every frame with a picture of
the empty room, so exposure, gain, white balance and focus must not drift
between the plate and the live frames. The BRIO only adjusts its auto controls
while it streams, so `lock` streams for a few seconds and then reproduces the
picture it settled on in manual mode (see `lock` for why that takes calibration).
"""

import json
import subprocess
import time
from pathlib import Path

AUTO = {"auto_exposure": 3, "white_balance_automatic": 1, "focus_automatic_continuous": 1}


# Preferred cameras, best first. The laptop's own camera also exposes an IR node
# ("USB2.0 I"), which must never be picked.
CAMERAS = ["BRIO", "FHD UVC WebCam: USB2.0 F"]


def find_device(names: list[str] | None = None) -> str:
    """First capture node (sysfs index 0) whose name contains one of `names`, in preference order."""
    nodes = sorted(Path("/sys/class/video4linux").iterdir(), key=lambda p: int(p.name[5:]))
    for name in names or CAMERAS:
        for p in nodes:
            if name.lower() in (p / "name").read_text().lower() and (p / "index").read_text().strip() == "0":
                return f"/dev/{p.name}"
    raise SystemExit(f"no camera whose name contains any of {names or CAMERAS}")


def _ctl(dev: str, *args: str) -> str:
    return subprocess.run(["v4l2-ctl", "-d", dev, *args], check=True, capture_output=True, text=True).stdout


def get(dev: str, names: list[str]) -> dict[str, int]:
    out = _ctl(dev, "-C", ",".join(names))
    return {k.strip(): int(v) for k, v in (line.split(":") for line in out.splitlines() if ":" in line)}


def set_(dev: str, **values: int) -> None:
    for k, v in values.items():  # one at a time: preserves the mode-before-value order
        _ctl(dev, "-c", f"{k}={v}")


def unlock(dev: str) -> None:
    set_(dev, **AUTO)


def open_capture(dev: str, width: int, height: int, raw: bool = False):
    """MJPEG capture. raw=True returns the undecoded JPEG bytes of each frame."""
    import cv2
    cap = cv2.VideoCapture(int(dev.removeprefix("/dev/video")), cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if raw:
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
    ok, f = cap.read()
    if ok and raw:
        f = cv2.imdecode(f.reshape(-1), cv2.IMREAD_COLOR)
        ok = f is not None
    if not ok:
        users = subprocess.run(["fuser", dev], capture_output=True, text=True).stdout.split()
        busy = f" (in use by pid {', '.join(users)})" if users else ""
        raise SystemExit(f"cannot read frames from {dev}{busy}")
    if f.shape[:2] != (height, width):
        raise SystemExit(f"{dev} gave {f.shape[1]}x{f.shape[0]}, wanted {width}x{height}")
    return cap


def _luma(cap, skip: int = 6, n: int = 5) -> float:
    for _ in range(skip):  # controls take a few frames to reach the sensor
        cap.read()
    return sum(cap.read()[1].mean() for _ in range(n)) / n


def _bisect(dev: str, cap, name: str, lo: int, hi: int, target: float) -> int:
    """Smallest-error integer value of a control whose effect on luma is monotonic increasing."""
    vals = {}
    def at(v):
        if v not in vals:
            set_(dev, **{name: v})
            vals[v] = _luma(cap)
        return vals[v]
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if at(mid) < target else (lo, mid)
    best = min((lo, hi), key=lambda v: abs(at(v) - target))
    set_(dev, **{name: best})
    return best


EXPOSURE = 300  # 100 µs units: 30 ms, a whole number of 50 Hz flicker half-periods, and fits 30 fps


def lock(dev: str, width: int, height: int, settle: float = 4.0) -> dict[str, int]:
    """Let the auto controls settle on the scene, then reproduce that picture in manual mode.

    The BRIO reports white balance and focus truthfully, but not what auto exposure
    does: it raises an internal gain and leaves exposure_time_absolute and gain at
    stale values. So brightness is calibrated against the image instead: measure the
    auto picture's mean luma, fix the exposure, and bisect gain until luma matches
    (or, in bright light, bisect exposure at gain 0).
    """
    unlock(dev)
    cap = open_capture(dev, width, height)
    try:
        t_end = time.monotonic() + settle
        while time.monotonic() < t_end:
            cap.read()
        target = _luma(cap, skip=0, n=10)
        wb_focus = get(dev, ["white_balance_temperature", "focus_absolute"])
        set_(dev, white_balance_automatic=0, white_balance_temperature=wb_focus["white_balance_temperature"],
             focus_automatic_continuous=0, focus_absolute=wb_focus["focus_absolute"],
             auto_exposure=1, exposure_time_absolute=EXPOSURE, gain=0)
        if _luma(cap) >= target:
            exposure, gain = _bisect(dev, cap, "exposure_time_absolute", 3, EXPOSURE, target), 0
        else:
            exposure, gain = EXPOSURE, _bisect(dev, cap, "gain", 0, 255, target)
        reached = _luma(cap)
    finally:
        cap.release()
    print(f"  brightness: auto {target:.0f}, locked {reached:.0f}")
    return {"exposure_time_absolute": exposure, "gain": gain, **wb_focus}


def apply(dev: str, settings: dict[str, int]) -> None:
    set_(dev, auto_exposure=1, exposure_time_absolute=settings["exposure_time_absolute"],
         gain=settings["gain"], white_balance_automatic=0,
         white_balance_temperature=settings["white_balance_temperature"],
         focus_automatic_continuous=0, focus_absolute=settings["focus_absolute"])


def save(path: Path, dev: str, width: int, height: int, settings: dict[str, int]) -> None:
    path.write_text(json.dumps({"device": dev, "width": width, "height": height, "controls": settings}, indent=2))


def load(path: Path) -> dict:
    return json.loads(path.read_text())
