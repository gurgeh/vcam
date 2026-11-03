"""Record a test session: locked camera, a 30 s clip with you in it, then the empty room.

Layout of a session directory:
    camera.json   locked control values (live mode reapplies them)
    clip.mkv      MJPEG stream copied straight from the camera, no re-encode
    plate.mkv     3 s of the empty room
    plate.png     per-pixel median of plate.mkv (the clean background plate)
"""

import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from . import camera

OUT = Path.home() / "out" / "vcam" / "sessions"

SCRIPT = [  # (second, instruction) during the clip
    (0, "Sit or stand as in a meeting and talk."),
    (5, "Shake your head, slowly, then fast."),
    (10, "Wave. Run a hand through your hair."),
    (15, "Turn sideways, show your profile both ways."),
    (20, "Lean back, then towards the camera."),
    (25, "Duck out of frame and come back up."),
]


def say(msg: str) -> None:
    print(f"\a\n>>> {msg}", flush=True)


def countdown(n: int, msg: str) -> None:
    say(msg)
    for i in range(n, 0, -1):
        print(f"  {i}…", flush=True)
        time.sleep(1)


def capture(dev: str, width: int, height: int, seconds: float, out: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "v4l2", "-input_format", "mjpeg",
         "-video_size", f"{width}x{height}", "-framerate", "30", "-i", dev,
         "-t", str(seconds), "-c", "copy", str(out)],
        check=True, stdin=subprocess.DEVNULL,
    )


def median_plate(video: Path, out: Path) -> None:
    cap = cv2.VideoCapture(str(video))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    if not frames:
        raise SystemExit(f"no frames in {video}")
    cv2.imwrite(str(out), np.median(np.stack(frames), axis=0).astype(np.uint8))


def run(device: str | None, width: int, height: int, seconds: int) -> Path:
    dev = device or camera.find_device()
    session = OUT / datetime.now().strftime("%Y%m%d-%H%M%S")
    session.mkdir(parents=True)
    latest = OUT / "latest"
    latest.unlink(missing_ok=True)
    latest.symlink_to(session.name)

    countdown(3, "Get in front of the camera as usual. Exposure, white balance and focus lock on you.")
    settings = camera.lock(dev, width, height)
    camera.save(session / "camera.json", dev, width, height, settings)
    print(f"  locked: {settings}")

    if seconds == 0:  # `vcam plate`: lock on you, then only the empty room
        countdown(8, "Step out of frame completely.")
        say("Recording the empty room (3 s)…")
        capture(dev, width, height, 3, session / "plate.mkv")
        median_plate(session / "plate.mkv", session / "plate.png")
        say(f"Done. Session: {session}  (live: vcam live --config hyb:… --session {session.name})")
        return session

    countdown(3, f"The clip starts ({seconds} s).")
    t0 = time.monotonic()

    def prompts():
        for sec, msg in SCRIPT:
            if sec >= seconds:
                return
            time.sleep(max(0, t0 + sec - time.monotonic()))
            say(msg)

    threading.Thread(target=prompts, daemon=True).start()
    capture(dev, width, height, seconds, session / "clip.mkv")

    countdown(8, "Done. Step out of frame completely, and move nothing in the room.")
    say("Recording the empty room (3 s)…")
    capture(dev, width, height, 3, session / "plate.mkv")
    median_plate(session / "plate.mkv", session / "plate.png")
    say(f"Done. Session: {session}")
    return session

