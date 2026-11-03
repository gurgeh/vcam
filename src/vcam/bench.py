"""What each performance mode costs in fan noise and watts, measured on the live pipeline.

Sensors (all readable without root):
    cpu_fan, gpu_fan  rpm, asus-nb-wmi hwmon: what you actually hear. The firmware curve
                      is a staircase (55/60/64/74 °C ...), so small loads often leave the fans
                      on the same step: the temperatures show how close the next step is.
    cpu_c, gpu_c      k10temp Tctl, NVIDIA core (NVML; 0 when the dGPU is suspended)
    apu_w             AMD socket power (PPT: CPU cores + iGPU), amdgpu hwmon
    dgpu_w            NVIDIA board power via NVML; 0 when the dGPU is runtime-suspended.
                      NVML is only touched while the card is already awake, since an open
                      handle would itself keep it from suspending.

Fans react over tens of seconds, so each mode runs `seconds` and only the second half
counts; an idle rest before each mode keeps one mode's heat from billing the next.
"""

import csv
import time
from pathlib import Path

from . import live

# Extra points used to choose the modes (2026-11-03); not user-facing.
BENCH_MODES = {
    "cpu-mbv3-540": dict(config="rvm:mobilenetv3:dr=0.25:stab=0.3", size="960x540", fps=20, cpu=True),
    "mbv3-720": dict(config="rvm:mobilenetv3:dr=0.25:stab=0.3", size="1280x720", fps=30, cpu=False),
    "r50-540": dict(config="rvm:resnet50:dr=0.25:stab=0.3", size="960x540", fps=30, cpu=False),
    "r50-540-20": dict(config="rvm:resnet50:dr=0.25:stab=0.3", size="960x540", fps=20, cpu=False),
    "r50-720-20": dict(config="rvm:resnet50:dr=0.25:stab=0.3", size="1280x720", fps=20, cpu=False),
    "r50-1080": dict(config="rvm:resnet50:dr=0.25:stab=0.3", size="1920x1080", fps=30, cpu=False),
    # max with decoding and YUV conversion on the CPU, to measure what the GPU path saves.
    "max-cpuio": dict(config="rvm:resnet50:dr=0.25:stab=0.3", size="1280x720", fps=30, cpu=False, cpu_io=True),
    "mbv3-1080": dict(config="rvm:mobilenetv3:dr=0.25:stab=0.3", size="1920x1080", fps=30, cpu=False),
}

HWMON = Path("/sys/class/hwmon")


def _hwmon(name: str) -> Path:
    for p in HWMON.iterdir():
        if (p / "name").read_text().strip() == name:
            return p
    raise SystemExit(f"no hwmon named {name!r}")


class Sensors:
    def __init__(self):
        self.asus = _hwmon("asus")
        self.amd = _hwmon("amdgpu")
        self.k10 = _hwmon("k10temp")
        nv = sorted(Path("/sys/bus/pci/drivers/nvidia").glob("0000:*"))
        self.nv_status = nv[0] / "power" / "runtime_status" if nv else None

    def dgpu(self) -> tuple[float, float, float]:
        """(watts, °C, utilisation %), or zeros while the dGPU is runtime-suspended."""
        if self.nv_status is None or self.nv_status.read_text().strip() != "active":
            return 0.0, 0.0, 0.0
        import pynvml
        pynvml.nvmlInit()
        try:
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            return (pynvml.nvmlDeviceGetPowerUsage(h) / 1000,
                    float(pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)),
                    float(pynvml.nvmlDeviceGetUtilizationRates(h).gpu))
        finally:
            pynvml.nvmlShutdown()

    def read(self) -> dict[str, float]:
        return {
            "cpu_fan": int((self.asus / "fan1_input").read_text()),
            "gpu_fan": int((self.asus / "fan2_input").read_text()),
            "apu_w": int((self.amd / "power1_input").read_text()) / 1e6,
            "dgpu_w": (d := self.dgpu())[0],
            "cpu_c": int((self.k10 / "temp1_input").read_text()) / 1000,
            "gpu_c": d[1],
            "gpu_util": d[2],
            "dgpu_state": self.nv_status.read_text().strip() if self.nv_status else "none",
        }


def _summary(name: str, samples: list[dict], fps: float | None) -> dict:
    tail = samples[len(samples) // 2:]
    mean = lambda k: sum(s[k] for s in tail) / len(tail)
    return {"mode": name, "fps": fps, **{k: mean(k) for k in ("cpu_fan", "gpu_fan", "apu_w", "dgpu_w", "cpu_c", "gpu_c", "gpu_util")},
            "dgpu_awake": sum(s["dgpu_state"] == "active" for s in tail) / len(tail)}


def idle(sensors: Sensors, seconds: float) -> list[dict]:
    samples = []
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        samples.append(sensors.read())
        time.sleep(1)
    return samples


def run(modes: list[str], seconds: float, rest: float, session: Path | None, out: Path) -> None:
    sensors = Sensors()
    rows, log = [], []
    print(f"idle {rest:.0f} s (baseline)…", flush=True)
    s = idle(sensors, rest)
    rows.append({**_summary("idle", s, None), "proc_cpu": None})
    log += [{"mode": "idle", **x} for x in s]
    for name in modes:
        m = {**live.MODES, **BENCH_MODES}[name]
        w, h = map(int, m["size"].split("x"))
        print(f"{name}: {m} for {seconds:.0f} s…", flush=True)
        samples = []
        cpu0 = time.process_time()
        fps = live.run(m["config"], session, None, w, h, None, "/dev/video10", m["fps"], m["cpu"],
                       seconds=seconds, on_second=lambda t: samples.append(sensors.read()),
                       cpu_io=m.get("cpu_io", False))
        cpu_pct = (time.process_time() - cpu0) / seconds * 100
        rows.append({**_summary(name, samples, fps), "proc_cpu": cpu_pct})
        log += [{"mode": name, **x} for x in samples]
        print(f"  idle {rest:.0f} s…", flush=True)
        idle(sensors, rest)

    out.mkdir(parents=True, exist_ok=True)
    with open(out / "bench_samples.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, list(log[0]))
        wr.writeheader()
        wr.writerows(log)
    keys = ["fps", "proc_cpu", "gpu_util", "apu_w", "dgpu_w", "cpu_c", "gpu_c", "cpu_fan", "gpu_fan", "dgpu_awake"]
    lines = ["| mode | " + " | ".join(keys) + " |", "|---|" + "---:|" * len(keys)]
    for r in rows:
        lines.append(f"| {r['mode']} | " + " | ".join("" if r[k] is None else f"{r[k]:.0f}" if k.endswith("fan") else f"{r[k]:.1f}"
                                                         for k in keys) + " |")
    table = "\n".join(lines)
    (out / "bench.md").write_text(table + "\n")
    print(table)
