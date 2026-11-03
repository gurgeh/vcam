"""Webcam → matting → new background → v4l2loopback ("Hugin Cam", /dev/video10)."""

import time
from pathlib import Path

import cv2
import pyvirtualcam
import torch

from . import camera
from . import matting as M

# Performance modes: what `vcam live --mode` and `vcam bench` mean by a name.
# Chosen 2026-11-03 from the evaluation (quality vs facit), judging by eye and `vcam bench`:
# - RVM-ResNet50 everywhere: judged best by eye (least halo, least flicker); vs
#   MobileNet it costs ~3 W more on the dGPU, which is awake anyway (desktop on HDMI).
# - No plate in the defaults: BGM keeps more hair but flickered in corners and broke
#   when the camera was nudged. Plate users: --config hyb:... --session ...
# - stab=0.3 halves edge flicker (0.73 -> 0.41) and moves no other metric.
# - 720p, not 1080p: Meet scales down anyway; 1080p cost +15 W on the dGPU.
# - No CPU mode: 15 fps at 540p, CPU at 85 °C, and saves nothing while the dGPU is awake.
# Power over idle (APU + dGPU): max +19.4 W; light +12.7 W. 540p alone or 20 fps alone
# saved under 2 W each, so only the combination is a mode. Quality at 20 fps held
# (flicker 0.43 vs 0.41). Fans did not move in any mode.
MODES = {
    "max": dict(config="rvm:resnet50:dr=0.25:stab=0.3", size="1280x720", fps=30, cpu=False),
    "light": dict(config="rvm:resnet50:dr=0.25:stab=0.3", size="960x540", fps=20, cpu=False),
}


class Pipeline:
    def __init__(self, spec: str, session: Path | None, background: Path | None, width: int, height: int,
                 device: str | None, cpu: bool = False, cpu_io: bool = False):
        # Set every time: a previous Pipeline in the same process (bench) may have left "cpu".
        M.DEVICE = "cpu" if cpu else "cuda"
        if cpu:
            torch.set_num_threads(4)
        self.dev = device or camera.find_device()
        self.model = M.parse(spec)
        self.plate = None
        if self.model.needs_plate:
            if session is None or not (session / "plate.png").exists():
                raise SystemExit("this model needs a plate: pass --session with a plate.png")
            # The plate is only valid with the exposure, white balance and focus it was taken with.
            camera.apply(self.dev, camera.load(session / "camera.json")["controls"])
            img = cv2.imread(str(session / "plate.png"))
            self.plate = M.to_tensor(cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA))
        else:
            camera.unlock(self.dev)
        # GPU modes decode on the GPU (nvJPEG) and convert to I420 there; the CPU mode
        # leaves the dGPU asleep, so it decodes and converts on the CPU.
        self.gpu_io = not cpu and not cpu_io
        self.cap = camera.open_capture(self.dev, width, height, raw=self.gpu_io)
        self.bg = M.load_image(background, width, height) if background else M.default_background(width, height)

    @torch.no_grad()
    def frame(self):
        ok, f = self.cap.read()
        if not ok:
            raise SystemExit("the camera stopped delivering frames")
        return f

    @torch.no_grad()
    def process(self, f):
        src = M.decode_jpeg(f) if self.gpu_io else M.to_tensor(f)
        fgr, pha = self.model(src, self.plate)
        comp = M.composite(fgr, pha, self.bg)
        return M.to_i420(comp) if self.gpu_io else M.to_bgr(comp)

    def close(self):
        self.cap.release()


def run(spec: str, session: Path | None, background: Path | None, width: int, height: int,
        device: str | None, out_device: str, fps: float = 30, cpu: bool = False,
        seconds: float | None = None, on_second=None, cpu_io: bool = False) -> float:
    """Run until Ctrl+C, or for `seconds`. Returns the achieved output fps.

    The camera delivers 30 fps; below that, frames are read and dropped so the
    output never lags. on_second(elapsed) is called about once a second (bench).
    """
    p = Pipeline(spec, session, background, width, height, device, cpu, cpu_io)
    interval = 1 / fps
    fmt = pyvirtualcam.PixelFormat.I420 if p.gpu_io else pyvirtualcam.PixelFormat.BGR
    with pyvirtualcam.Camera(width, height, round(fps), device=out_device, fmt=fmt) as cam:
        if seconds is None:
            print(f"{p.dev} → {cam.device} ({width}x{height}@{fps:g}, {spec}{', CPU' if cpu else ''}). Ctrl+C to stop.",
                  flush=True)
        start = tick = due = time.perf_counter()
        sent = window = 0
        half = 0.5 / 30  # half a camera frame of slack
        try:
            while seconds is None or time.perf_counter() - start < seconds:
                f = p.frame()
                now = time.perf_counter()
                # Book each output frame `interval` after the previous *scheduled* one, so
                # 20 fps from a 30 fps camera keeps 2 of 3 frames instead of every other.
                if now < due - half:
                    continue
                due = max(due + interval, now - interval)
                cam.send(p.process(f))
                sent += 1
                window += 1
                if now - tick >= 1:
                    if on_second:
                        on_second(now - start)
                    elif window and int(now - start) % 5 == 0:
                        print(f"  {window / (now - tick):.1f} fps", flush=True)
                    tick, window = now, 0
        except KeyboardInterrupt:
            pass
        finally:
            p.close()
    return sent / (time.perf_counter() - start)
