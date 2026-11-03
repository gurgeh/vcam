"""Matting models behind one interface, plus the image helpers they share.

Tensors are float32 on the GPU, shape (1, 3|1, H, W), RGB, range 0..1.

A model spec is a colon-separated string:
    rvm:mobilenetv3:dr=0.25
    bgm:resnet50:bs=0.25:mode=sampling:px=80000
    ...:ema=0.6                     (optional temporal smoothing of alpha)
"""

import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

MODELS = Path(__file__).resolve().parents[2] / "models"
DEVICE = "cuda"


def to_tensor(bgr: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(bgr).to(DEVICE, non_blocking=True)
    return t.permute(2, 0, 1)[[2, 1, 0]].unsqueeze(0).float().div_(255)


def to_bgr(t: torch.Tensor) -> np.ndarray:
    return t[0, [2, 1, 0]].clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).contiguous().cpu().numpy()


def load_image(path: Path, width: int, height: int) -> torch.Tensor:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"cannot read {path}")
    return to_tensor(cover(img, width, height))


def cover(img: np.ndarray, width: int, height: int) -> np.ndarray:
    """Scale to cover width x height and centre-crop, like CSS background-size: cover."""
    h, w = img.shape[:2]
    s = max(width / w, height / h)
    img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    y, x = (img.shape[0] - height) // 2, (img.shape[1] - width) // 2
    return img[y:y + height, x:x + width]


def default_background(width: int, height: int) -> torch.Tensor:
    """A smooth teal-to-sand gradient: neither black nor white, so halos of either show."""
    y = torch.linspace(0, 1, height, device=DEVICE).view(1, 1, height, 1)
    x = torch.linspace(0, 1, width, device=DEVICE).view(1, 1, 1, width)
    a = torch.tensor([0.16, 0.42, 0.45], device=DEVICE).view(1, 3, 1, 1)
    b = torch.tensor([0.86, 0.76, 0.58], device=DEVICE).view(1, 3, 1, 1)
    t = (0.6 * x + 0.4 * y).clamp(0, 1)
    return a + (b - a) * t


def composite(fgr: torch.Tensor, pha: torch.Tensor, bg: torch.Tensor) -> torch.Tensor:
    return fgr * pha + bg * (1 - pha)


def _box(x: torch.Tensor, r: int) -> torch.Tensor:
    """Separable box blur of radius r, edge-normalised."""
    k = 2 * r + 1
    x = F.avg_pool2d(x, (1, k), stride=1, padding=(0, r), count_include_pad=False)
    return F.avg_pool2d(x, (k, 1), stride=1, padding=(r, 0), count_include_pad=False)


def _dilate(x: torch.Tensor, r: int) -> torch.Tensor:
    """Square dilation of radius r; max is separable, so two 1-D passes instead of one (2r+1)^2."""
    k = 2 * r + 1
    x = F.max_pool2d(x, (1, k), stride=1, padding=(0, r))
    return F.max_pool2d(x, (k, 1), stride=1, padding=(r, 0))


def decode_jpeg(raw: np.ndarray) -> torch.Tensor:
    """Camera MJPEG frame → (1,3,H,W) float RGB on the GPU via nvJPEG.

    Only the ~300 kB JPEG crosses the bus instead of a 6 MB decoded frame; Huffman
    decoding still runs on the CPU (consumer GPUs lack the NVJPG engine), so this
    roughly halves the CPU cost of decoding (10 → 4.4 ms at 1080p), not more.
    """
    from torchvision.io import decode_jpeg as dj
    return dj(torch.from_numpy(raw.reshape(-1)), device=DEVICE).unsqueeze(0).float().div_(255)


def to_i420(rgb: torch.Tensor) -> np.ndarray:
    """(1,3,H,W) RGB 0..1 → I420 bytes, BT.601 limited range like libyuv's ARGBToI420,
    which is what pyvirtualcam would otherwise run on the CPU for a BGR frame."""
    return fetch(i420_gpu(rgb))


_host: dict = {}


def fetch(t: torch.Tensor) -> np.ndarray:
    """GPU tensor → numpy without spinning a CPU core.

    A plain .cpu() blocks in CUDA's default spin-wait, so a core runs flat out while
    the GPU works, eating what GPU decoding saved. Copy asynchronously into pinned
    memory and poll the event with short sleeps instead.
    """
    if t.device.type == "cpu":
        return t.numpy()
    key = (t.shape, t.dtype)
    if key not in _host:
        _host[key] = (torch.empty(t.shape, dtype=t.dtype, pin_memory=True), torch.cuda.Event())
    host, ev = _host[key]
    host.copy_(t, non_blocking=True)
    ev.record()
    while not ev.query():
        time.sleep(0.0005)
    return host.numpy()


def i420_gpu(rgb: torch.Tensor) -> torch.Tensor:
    x = rgb[0].clamp(0, 1) * 255
    r, g, b = x[0], x[1], x[2]
    y = (66 * r + 129 * g + 25 * b + 128) / 256 + 16
    sub = F.avg_pool2d(x.unsqueeze(0), 2)[0]
    r, g, b = sub[0], sub[1], sub[2]
    u = (112 * b - 74 * g - 38 * r + 128) / 256 + 128
    v = (112 * r - 94 * g - 18 * b + 128) / 256 + 128
    return torch.cat([y.reshape(-1), u.reshape(-1), v.reshape(-1)]).round().clamp(0, 255).byte()


def blur_fusion(img: torch.Tensor, pha: torch.Tensor) -> torch.Tensor:
    """Foreground colour from image + alpha (Forte & Pitié 2021, 'blur fusion', two passes)."""
    def step(fg, bg, r):
        ba = _box(pha, r)
        bf = _box(fg * pha, r) / (ba + 1e-5)
        bb = _box(bg * (1 - pha), r) / (_box(1 - pha, r) + 1e-5)
        fg = bf + pha * (img - pha * bf - (1 - pha) * bb)
        return fg.clamp(0, 1), bb
    fg, bg = step(img, img, 90)
    fg, _ = step(fg, bg, 6)
    return fg


class Model:
    """Base: wraps a raw model with optional temporal smoothing of alpha.

    ema=k    plain exponential smoothing: alpha = k*new + (1-k)*previous. Cuts flicker
             but ghosts when you move.
    stab=k   motion-adaptive: where the camera image did not change since the last
             frame there is no reason for alpha to change, so it moves only k of the
             way to the new value; where the image changed (motion > STAB_MOTION), the
             new alpha applies at once. Targets edge flicker on a still head without
             lagging behind motion.
    """

    needs_plate = False
    STAB_MOTION = 12 / 255  # image change that counts as real motion (sensor noise is ~2-4)

    def __init__(self, ema: float = 1.0, stab: float = 1.0):
        self.ema, self.stab = ema, stab
        self.prev = None
        self.prev_src = None

    def reset(self) -> None:
        self.prev = None
        self.prev_src = None

    def __call__(self, src, plate=None):
        fgr, pha = self.raw(src, plate)
        if self.ema < 1.0 and self.prev is not None:
            pha = self.ema * pha + (1 - self.ema) * self.prev
        if self.stab < 1.0 and self.prev is not None:
            motion = _box((src - self.prev_src).abs().amax(1, keepdim=True), 2)
            w = (motion / self.STAB_MOTION).clamp(self.stab, 1.0)
            pha = self.prev + w * (pha - self.prev)
        self.prev, self.prev_src = pha, src
        return fgr, pha


class RVM(Model):
    def __init__(self, variant="mobilenetv3", dr=0.25, **kw):
        super().__init__(**kw)
        m = torch.jit.load(str(MODELS / f"rvm_{variant}_fp32.torchscript")).eval().to(DEVICE)
        self.m, self.dr = torch.jit.freeze(m), dr
        self.rec = [None] * 4

    def reset(self):
        super().reset()
        self.rec = [None] * 4

    @torch.no_grad()
    def raw(self, src, plate=None):
        fgr, pha, *self.rec = self.m(src, *self.rec, self.dr)
        return fgr, pha


class BGM(Model):
    needs_plate = True

    def __init__(self, variant="mobilenetv2", bs=0.25, mode="sampling", px=80000, **kw):
        super().__init__(**kw)
        self.m = torch.jit.load(str(MODELS / f"torchscript_{variant}_fp32.pth")).eval().to(DEVICE)
        self.m.backbone_scale = bs
        self.m.refine_mode = mode
        if mode == "sampling":
            self.m.refine_sample_pixels = int(px)

    @torch.no_grad()
    def raw(self, src, plate):
        pha, fgr = self.m(src, plate)[:2]
        return fgr, pha


class Hybrid(Model):
    """BGM's alpha near you, RVM's judgement of where you are.

    BGM treats anything that differs from the plate as foreground: your shadow on
    the wall, a lamp that changed, a nudged camera. RVM has no plate and knows a
    ceiling is not a person, but cuts curls. So BGM decides inside a zone around
    RVM's coarse mask, dilated by `gate` px (at 1080p; scaled with the frame) so
    curls fit, and alpha is forced to 0 beyond it.

    When the plate stops matching (moved camera, changed light), the two disagree
    over a large part of the foreground; then this falls back to RVM alone and says
    so once, rather than showing ghosts. `stale` frames in a row trigger it, and it
    recovers by itself if agreement returns.
    """

    needs_plate = True

    def __init__(self, variant="mobilenetv2", gate=48, stale=15, **kw):
        bgm_kw = {k: kw.pop(k) for k in ("bs", "mode", "px") if k in kw}
        super().__init__(**kw)
        self.bgm = BGM(variant, **bgm_kw)
        self.rvm = RVM("mobilenetv3", dr=0.25)
        self.gate, self.stale = gate, int(stale)
        self.bad = 0
        self.warned = False

    def reset(self):
        super().reset()
        self.rvm.reset()
        self.bad = 0

    @torch.no_grad()
    def raw(self, src, plate):
        rfg, ra = self.rvm(src)
        bfg, ba = self.bgm(src, plate)
        fg = (ra > 0.5).sum().item()
        clash = ((ra - ba).abs() > 0.5).sum().item() / max(1, fg)
        self.bad = self.bad + 1 if clash > 0.25 else 0
        if self.bad >= self.stale:
            if not self.warned:
                print("  the plate no longer matches (camera moved or light changed): "
                      "running RVM alone; take a fresh one with `vcam plate`", flush=True)
                self.warned = True
            return rfg, ra
        self.warned = False
        r = max(1, round(self.gate * src.shape[-2] / 1080))
        zone = _dilate((ra > 0.05).float(), r)
        zone = _box(zone, max(1, r // 4))  # soft edge, so the cut-off itself cannot flicker
        return bfg, ba * zone


def parse(spec: str) -> Model:
    kind, variant, *opts = spec.split(":")
    kw = {}
    for o in opts:
        k, v = o.split("=")
        kw[k] = v if k == "mode" else float(v)
    return {"rvm": RVM, "bgm": BGM, "hyb": Hybrid}[kind](variant, **kw)
