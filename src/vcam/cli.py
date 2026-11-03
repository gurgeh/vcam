import argparse
import time
from pathlib import Path

from .record import OUT


def session_path(s: str | None) -> Path:
    p = Path(s) if s else OUT / "latest"
    if not p.is_absolute() and not p.exists():
        p = OUT / s
    return p.resolve()


def main() -> None:
    ap = argparse.ArgumentParser(prog="vcam", description="Virtual camera with background replacement (RVM / BackgroundMattingV2)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("record", help="lock the camera, record a test clip and the empty room")
    r.add_argument("--device")
    r.add_argument("--seconds", type=int, default=30)
    r.add_argument("--size", default="1920x1080")

    f = sub.add_parser("reference", aliases=["facit"], help="build the reference alpha (slow, offline)")
    f.add_argument("--session", help="session directory or name (default: latest)")
    f.add_argument("--band", type=int, default=24, help="half-width of the unknown band around the edge, px")
    f.add_argument("--limit", type=int, help="only the first N frames")

    e = sub.add_parser("eval", help="score real-time configurations against the reference")
    e.add_argument("--session")
    e.add_argument("--config", action="append", help="model spec, repeat for several (default: the whole grid)")
    e.add_argument("--limit", type=int)
    e.add_argument("--fps", type=float, default=30, help="drop frames evenly, as live does at a lower fps")
    e.add_argument("--plate", type=Path, help="a different plate than the session's (test a stale plate)")

    lv = sub.add_parser("live", help="run the virtual camera")
    lv.add_argument("--mode", default="max", help="performance mode: max or light (--config/--size/--fps/--cpu override parts)")
    lv.add_argument("--config")
    lv.add_argument("--fps", type=float)
    lv.add_argument("--cpu", action="store_true", default=None, help="run the model on the CPU so the dGPU can sleep")
    lv.add_argument("--session", help="plate + locked camera settings (for bgm/hyb models; 'latest' = most recent)")
    lv.add_argument("--background", type=Path, help="image to put behind you")
    lv.add_argument("--size")
    lv.add_argument("--device")
    lv.add_argument("--out", default="/dev/video10")

    b = sub.add_parser("bench", help="measure power, load and fans per performance mode")
    b.add_argument("--modes", default="max,light")
    b.add_argument("--seconds", type=float, default=30, help="per mode; power responds at once (fans: 90)")
    b.add_argument("--rest", type=float, default=10, help="idle rest before each mode (fans: 45)")
    b.add_argument("--session", help="required if a mode uses bgm/hyb")

    pl = sub.add_parser("plate", help="take a fresh plate before a meeting (locks the camera on you first)")
    pl.add_argument("--device")
    pl.add_argument("--size", default="1920x1080")

    sub.add_parser("unlock", help="turn auto exposure, white balance and focus back on").add_argument("--device")

    a = ap.parse_args()
    if a.cmd == "record":
        from . import record
        w, h = map(int, a.size.split("x"))
        record.run(a.device, w, h, a.seconds)
    elif a.cmd in ("reference", "facit"):
        from . import facit
        facit.run(session_path(a.session), a.band, a.limit)
    elif a.cmd == "eval":
        from . import evaluate
        evaluate.run(session_path(a.session), a.config or evaluate.GRID, a.limit, a.plate, a.fps)
    elif a.cmd == "live":
        from . import live
        m = dict(live.MODES[a.mode])
        m.update({k: v for k in ("config", "size", "fps", "cpu") if (v := getattr(a, k)) is not None})
        w, h = map(int, m["size"].split("x"))
        live.run(m["config"], session_path(a.session) if a.session else None, a.background, w, h, a.device, a.out,
                 m["fps"], m["cpu"])
    elif a.cmd == "bench":
        from . import bench
        s = session_path(a.session) if a.session else None
        bench.run(a.modes.split(","), a.seconds, a.rest, s, OUT.parent / "bench" / time.strftime("%Y%m%d-%H%M%S"))
    elif a.cmd == "plate":
        from . import record
        w, h = map(int, a.size.split("x"))
        record.run(a.device, w, h, 0)
    elif a.cmd == "unlock":
        from . import camera
        camera.unlock(a.device or camera.find_device())
