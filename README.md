# vcam

A virtual webcam with background replacement for Linux, for meeting tools that
won't let you upload your own background (Teams on the web as a guest) or that cut
curly hair down to a helmet (all of the built-in ones).

Camera → matting model on the GPU (or CPU) → your background → `v4l2loopback`
device "Hugin Cam" (`/dev/video10`), which any browser or meeting app can select.

## Setup

```
sudo apt install v4l2loopback-dkms v4l-utils
echo 'options v4l2loopback devices=1 video_nr=10 card_label="Hugin Cam" exclusive_caps=1' | sudo tee /etc/modprobe.d/v4l2loopback.conf
echo v4l2loopback | sudo tee /etc/modules-load.d/v4l2loopback.conf
sudo modprobe v4l2loopback
uv sync --extra eval
```

`exclusive_caps=1` is what makes Chrome list the device; it then appears only while
`vcam live` is producing frames. Model weights go in `models/` (gitignored), from
the v1.0.0 GitHub releases of PeterL1n/RobustVideoMatting
(`rvm_*_fp32.torchscript`) and PeterL1n/BackgroundMattingV2
(`torchscript_*_fp32.pth`).

## Use

```
vcam live --mode max --background ~/Pictures/office.jpg
```

| mode | model | size | power over idle |
|---|---|---|---:|
| `max` (default) | RVM ResNet50 | 1280×720 @ 30 | +19 W |
| `light` | RVM ResNet50 | 960×540 @ 20 | +13 W |

Both use `stab=0.3`, a motion-adaptive temporal filter on alpha that halves edge
flicker. Override any part with `--config`, `--size`, `--fps`, `--cpu`. On the GPU
path the camera's MJPEG is decoded with nvJPEG and the I420 conversion for the
loopback runs on the GPU; that saved ~12 W of CPU-side power over doing both on
the CPU. Lowering only the resolution or only the frame rate saved under 2 W each.

### Clean-plate matting

BackgroundMattingV2 compares each frame with a photo of the empty room and keeps
more hair, but breaks when the camera moves or the light changes. `hyb:` limits it
to a zone around RVM's mask and falls back to RVM when the two disagree:

```
vcam plate          # lock the camera on you, then step out for 3 s
vcam live --config hyb:resnet50:bs=0.25:mode=sampling:px=80000:gate=48 --session latest
```

## Measuring quality: reference first

Eyeballing edges does not compare configurations. Instead:

```
vcam record         # lock camera, 30 s guided clip, 3 s empty room
vcam reference      # offline reference alpha (~0.5 s/frame)
vcam eval           # score the grid of real-time configs against it
```

The reference comes from ViTMatte, a trimap-based matting transformer,
run at full resolution. Its trimap is the band where RVM and BackgroundMattingV2
(two different techniques) are unsure or disagree, so neither candidate family
grades its own homework. `reference` refuses to run when the two disagree over most of
the foreground: that means the plate contains you or the room changed.

Metrics, lower is better: `comp_band` (error in the final composite within the
band: what a viewer sees), `sad_band` split into `lost_band` (hair cut away) and
`extra_band` (halo), `mse`, `dtssd`, `flicker` (alpha change where the image did
not change; reference-free) and `ms`. Each run writes `results.md` and a contact
sheet to `<session>/eval/<timestamp>/`.

## Measuring cost

```
vcam bench --modes max,light
```

Runs each mode live and reports achieved fps, the process's CPU share, GPU
utilisation, AMD socket power (PPT), NVIDIA board power (NVML, read only while the
card is awake, since an open handle keeps it from suspending), temperatures and fan
speeds, averaged over the second half of each run. Rank by power: the firmware fan
curve is a temperature staircase and did not move in any mode.

## Camera locking

`vcam record`/`plate` lock exposure, white balance and focus. The Logitech BRIO
does not report what its auto exposure is doing (it raises an internal gain and
leaves the controls at stale values), so `camera.lock` calibrates against the image:
fixed 30 ms exposure (flicker-free at 50 Hz) and bisection on gain until mean luma
matches what auto mode produced.

## Licence

MIT for the code in this repository (see `LICENSE`). The models are downloaded
separately and keep their own licences: Robust Video Matting is GPL-3.0,
BackgroundMattingV2 and ViTMatte are MIT.
