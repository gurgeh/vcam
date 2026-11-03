"""Score real-time configurations against the facit of a session.

Metrics (lower is better), averaged over frames:
    comp_band  mean |composite - facit composite| in the trimap band, in 0..255 units.
               What a viewer sees: alpha error, halo and colour spill together.
    sad_band   sum |alpha - facit alpha| in the band, /1000. The matting literature's SAD,
               split by direction into:
    lost_band  alpha facit has and the candidate lacks: curls and strands cut away.
    extra_band alpha the candidate adds: halo, background left around you.
    mse        mean squared alpha error over the whole frame, x1000. Catches gross
               mistakes outside the band (a lost arm, a chair kept as foreground).
    dtssd      RMS of (alpha change - facit alpha change) between frames, x100.
               Temporal error relative to facit; inherits facit's own flicker.
    flicker    mean |alpha change| on pixels whose image did not change, x1000.
               Reference-free: pure model noise. Also reported for facit itself.
    ms         GPU time for the model alone, per frame.

Configs marked * fed the facit trimap, so they have a home advantage.
"""

import csv
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from . import matting as M
from .facit import frames, load16

GRID = [
    "rvm:mobilenetv3:dr=0.25",
    "rvm:mobilenetv3:dr=0.4",
    "rvm:mobilenetv3:dr=0.25:ema=0.6",
    "rvm:resnet50:dr=0.25",
    "rvm:resnet50:dr=0.4",
    "bgm:mobilenetv2:bs=0.25:mode=sampling:px=80000",
    "bgm:mobilenetv2:bs=0.25:mode=full",
    "bgm:mobilenetv2:bs=0.25:mode=full:ema=0.6",
    "bgm:resnet50:bs=0.25:mode=sampling:px=80000",
    "bgm:resnet50:bs=0.25:mode=full",
    "bgm:resnet101:bs=0.25:mode=full",
]
FACIT_SOURCES = {"rvm:resnet50:dr=0.4", "bgm:resnet101:bs=0.25:mode=full"}
STATIC = 6 / 255  # image change below this counts as "nothing moved"
SHEET_FRAMES = 6
TILE = 360


def crop_box(tri: torch.Tensor) -> tuple[int, int, int, int]:
    """Square box around the upper part of the band (head and hair), where matting is hard."""
    ys, xs = torch.nonzero(tri[0, 0] == 0.5, as_tuple=True)
    h, w = tri.shape[-2:]
    if len(ys) == 0:
        return 0, 0, min(h, w), min(h, w)
    top = int(ys.min())
    cx = int(xs[ys < top + h // 3].float().mean()) if (ys < top + h // 3).any() else w // 2
    side = min(h, w, max(TILE, int(0.45 * h)))
    y0 = max(0, min(h - side, top - side // 8))
    x0 = max(0, min(w - side, cx - side // 2))
    return y0, x0, y0 + side, x0 + side


def tile(img: torch.Tensor, box) -> np.ndarray:
    y0, x0, y1, x1 = box
    t = F.interpolate(img[..., y0:y1, x0:x1], size=(TILE, TILE), mode="area")
    return M.to_bgr(t)


def label(img: np.ndarray, text: str) -> np.ndarray:
    img = img.copy()
    for color, thick in (((0, 0, 0), 4), ((255, 255, 255), 1)):
        cv2.putText(img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, thick, cv2.LINE_AA)
    return img


def kept(i: int, fps: float, source_fps: float = 30) -> bool:
    """Frame i of a source_fps clip survives resampling to fps (drops evenly, like live does)."""
    return int(i * fps / source_fps) != int((i - 1) * fps / source_fps) if i else True


def run(session: Path, specs: list[str], limit: int | None, plate_path: Path | None = None,
        fps: float = 30) -> None:
    fdir = session / "facit"
    n = len(list((fdir / "alpha").glob("*.png")))
    if limit is not None:
        n = min(n, limit)
    if n == 0:
        raise SystemExit("no reference alpha: run `vcam reference` first")
    import time
    out = session / "eval" / (time.strftime("%Y%m%d-%H%M%S") + ("" if plate_path is None else f"-plate-{plate_path.parent.name}")
                              + ("" if fps == 30 else f"-{fps:g}fps"))
    out.mkdir(parents=True)

    plate = M.to_tensor(cv2.imread(str(plate_path or session / "plate.png")))
    h, w = plate.shape[-2:]
    bg = M.default_background(w, h)
    sheet_idx = [round(i * (n - 1) / (SHEET_FRAMES - 1)) for i in range(SHEET_FRAMES)]
    sheet_idx = [i if kept(i, fps) else i - 1 for i in sheet_idx]
    boxes, tiles = {}, {}

    # Facit pass: band masks, composites for the sheet, and facit's own flicker.
    print("reference pass…", flush=True)
    facit_flicker, prev = [], None
    for i, f in enumerate(frames(session / "clip.mkv")):
        if i >= n:
            break
        if not kept(i, fps):
            continue
        src = M.to_tensor(f)
        a = load16(fdir / "alpha" / f"{i:05d}.png")
        if prev is not None:
            static = (src - prev[0]).abs().amax(1, keepdim=True) < STATIC
            facit_flicker.append(((a - prev[1]).abs() * static).sum().item() / max(1, static.sum().item()) * 1000)
        prev = (src, a)
        if i in sheet_idx:
            tri = cv2.imread(str(fdir / "trimap" / f"{i:05d}.png"), cv2.IMREAD_GRAYSCALE)
            tri = torch.from_numpy(tri).to(M.DEVICE)[None, None].float() / 255
            boxes[i] = crop_box(tri.round(decimals=1))
            tiles.setdefault("image", []).append(tile(src, boxes[i]))
            tiles.setdefault("reference", []).append(tile(M.composite(M.blur_fusion(src, a), a, bg), boxes[i]))

    rows = []
    for spec in specs:
        print(spec, flush=True)
        model = M.parse(spec)
        acc = {k: [] for k in ("comp_band", "sad_band", "lost_band", "extra_band", "mse", "dtssd", "flicker", "ms")}
        prev = None
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for i, f in enumerate(frames(session / "clip.mkv")):
            if i >= n:
                break
            if not kept(i, fps):
                continue
            src = M.to_tensor(f)
            start.record()
            fgr, pha = model(src, plate)
            end.record()
            torch.cuda.synchronize()
            acc["ms"].append(start.elapsed_time(end))

            fa = load16(fdir / "alpha" / f"{i:05d}.png")
            tri = cv2.imread(str(fdir / "trimap" / f"{i:05d}.png"), cv2.IMREAD_GRAYSCALE)
            band = torch.from_numpy((tri > 64) & (tri < 192)).to(M.DEVICE)[None, None]
            nb = max(1, band.sum().item())
            comp = M.composite(fgr, pha, bg)
            fcomp = M.composite(M.blur_fusion(src, fa), fa, bg)
            acc["comp_band"].append(((comp - fcomp).abs().mean(1, keepdim=True) * band).sum().item() / nb * 255)
            acc["sad_band"].append(((pha - fa).abs() * band).sum().item() / 1000)
            acc["lost_band"].append(((fa - pha).clamp(min=0) * band).sum().item() / 1000)
            acc["extra_band"].append(((pha - fa).clamp(min=0) * band).sum().item() / 1000)
            acc["mse"].append(((pha - fa) ** 2).mean().item() * 1000)
            if prev is not None:
                psrc, ppha, pfa = prev
                acc["dtssd"].append(((pha - ppha) - (fa - pfa)).pow(2).mean().sqrt().item() * 100)
                static = (src - psrc).abs().amax(1, keepdim=True) < STATIC
                acc["flicker"].append(((pha - ppha).abs() * static).sum().item() / max(1, static.sum().item()) * 1000)
            prev = (src, pha, fa)
            if i in sheet_idx:
                tiles.setdefault(spec, []).append(tile(comp, boxes[i]))
        acc["ms"] = acc["ms"][5:]  # drop warm-up
        rows.append({"config": spec + (" *" if spec in FACIT_SOURCES else ""),
                     **{k: float(np.mean(v)) for k, v in acc.items()}})
        del model
        torch.cuda.empty_cache()

    rows.sort(key=lambda r: r["comp_band"])
    keys = ["comp_band", "sad_band", "lost_band", "extra_band", "mse", "dtssd", "flicker", "ms"]
    with open(out / "results.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, ["config", *keys])
        wr.writeheader()
        wr.writerows(rows)
    lines = ["| config | " + " | ".join(keys) + " |", "|---|" + "---:|" * len(keys)]
    lines += [f"| {r['config']} | " + " | ".join(f"{r[k]:.2f}" for k in keys) + " |" for r in rows]
    lines.append("| reference (own flicker) |" + " |" * 6 + f" {np.mean(facit_flicker):.2f} | |")
    table = "\n".join(lines)
    (out / "results.md").write_text(table + "\n")
    print(table)

    order = ["image", "reference"] + [r["config"].rstrip(" *") for r in rows]
    sheet = np.vstack([np.hstack([label(t, name if j == 0 else f"#{sheet_idx[j]}") for j, t in enumerate(tiles[name])])
                       for name in order])
    cv2.imwrite(str(out / "sheet.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"→ {out}")
