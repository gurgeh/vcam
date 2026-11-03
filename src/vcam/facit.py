"""Build the reference alpha ("facit") for a recorded session, offline and slowly.

Facit is a model too, and its mistakes become the truth every candidate is
scored against. To keep it from simply crowning one candidate family, the
trimap draws on two different techniques: RVM (no plate) and BackgroundMattingV2
(clean plate). Only where both are confident and agree is a pixel fixed as
foreground or background. Everything else (a band around the consensus edge,
anywhere either model is fractional, anywhere they disagree) is left unknown
and solved by ViTMatte, a trimap-based matting transformer that puts all its
capacity into that band.

Writes into <session>/facit/:
    alpha/NNNNN.png    16-bit alpha
    trimap/NNNNN.png   0 / 128 / 255; the 128 band is where the evaluation looks
    preview/NNNNN.png  every 30th frame: image | facit composite | alpha | trimap
"""

import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from . import matting as M

VITMATTE = "hustvl/vitmatte-base-composition-1k"


def dilate(x: torch.Tensor, r: int) -> torch.Tensor:
    return F.max_pool2d(x, 2 * r + 1, stride=1, padding=r)


def erode(x: torch.Tensor, r: int) -> torch.Tensor:
    return -dilate(-x, r)


def make_trimap(alphas: list[torch.Tensor], band: int) -> torch.Tensor:
    a = torch.cat(alphas, 1)
    hard = (a.mean(1, keepdim=True) > 0.5).float()
    edge = (dilate(hard, band) - erode(hard, band)) > 0
    fractional = ((a > 0.03) & (a < 0.97)).any(1, keepdim=True)
    disagree = (a.amax(1, keepdim=True) - a.amin(1, keepdim=True)) > 0.1
    unknown = edge | (dilate((fractional | disagree).float(), 3) > 0)
    return torch.where(unknown, torch.full_like(hard, 0.5), hard)


class ViTMatte:
    def __init__(self):
        from transformers import VitMatteForImageMatting
        self.m = VitMatteForImageMatting.from_pretrained(VITMATTE, dtype=torch.float16).eval().to(M.DEVICE)

    @torch.no_grad()
    def __call__(self, src: torch.Tensor, tri: torch.Tensor) -> torch.Tensor:
        h, w = src.shape[-2:]
        x = torch.cat([(src - 0.5) / 0.5, tri], 1)
        x = F.pad(x, (0, -w % 32, 0, -h % 32))
        alpha = self.m(pixel_values=x.half()).alphas[..., :h, :w].float()
        return torch.where(tri == 0.5, alpha, tri)  # known regions stay known


def frames(video: Path):
    cap = cv2.VideoCapture(str(video))
    while True:
        ok, f = cap.read()
        if not ok:
            return
        yield f


def save16(path: Path, a: torch.Tensor) -> None:
    cv2.imwrite(str(path), (a[0, 0].clamp(0, 1) * 65535).round().to(torch.int32).cpu().numpy().astype(np.uint16))


def load16(path: Path) -> torch.Tensor:
    a = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    return torch.from_numpy(a.astype(np.float32) / 65535).to(M.DEVICE)[None, None]


def gray3(a: torch.Tensor) -> torch.Tensor:
    return a.expand(-1, 3, -1, -1)


def run(session: Path, band: int, limit: int | None) -> None:
    out = session / "facit"
    for d in ("alpha", "trimap", "preview"):
        (out / d).mkdir(parents=True, exist_ok=True)

    plate = M.to_tensor(cv2.imread(str(session / "plate.png")))
    h, w = plate.shape[-2:]
    bg = M.default_background(w, h)
    rvm = M.RVM("resnet50", dr=0.4)
    bgm = M.BGM("resnet101", mode="full")
    vit = ViTMatte()

    t0, n, bad = time.perf_counter(), 0, []
    for i, f in enumerate(frames(session / "clip.mkv")):
        if limit is not None and i >= limit:
            break
        src = M.to_tensor(f)
        _, a_rvm = rvm(src)
        _, a_bgm = bgm(src, plate)
        # A plate with you in it, or a moved chair or changed light, makes BGMv2 contradict
        # RVM over whole regions. The trimap then has no confident foreground and
        # ViTMatte returns background: a silently wrong facit. Stop instead.
        fg = (a_rvm > 0.5).sum().item()
        clash = ((a_rvm - a_bgm).abs() > 0.5).sum().item() / max(1, fg)
        if clash > 0.25:
            bad.append(i)
            if len(bad) == 10 and i < 20:
                raise SystemExit(
                    f"RVM and BGMv2 disagree on {clash:.0%} of the foreground from the start. "
                    "plate.png probably contains you, or the room or light changed since it was taken.")
        tri = make_trimap([a_rvm, a_bgm], band)
        alpha = vit(src, tri)
        save16(out / "alpha" / f"{i:05d}.png", alpha)
        cv2.imwrite(str(out / "trimap" / f"{i:05d}.png"), (tri[0, 0] * 255).round().byte().cpu().numpy())
        if i % 30 == 0:
            comp = M.composite(M.blur_fusion(src, alpha), alpha, bg)
            row = torch.cat([src, comp, gray3(alpha), gray3(tri)], 3)
            cv2.imwrite(str(out / "preview" / f"{i:05d}.png"), M.to_bgr(F.interpolate(row, scale_factor=0.5, mode="area")))
        n += 1
        if n % 50 == 0:
            print(f"  {n} frames, {(time.perf_counter() - t0) / n:.2f} s each", flush=True)
    print(f"reference: {n} frames in {time.perf_counter() - t0:.0f} s → {out}")
    if bad:
        print(f"WARNING: {len(bad)} frames where RVM and BGMv2 disagreed on >25 % of the foreground "
              f"(first: {bad[:10]}). Look at preview/ before trusting them.")
