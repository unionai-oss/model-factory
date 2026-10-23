"""Measure a workload's GPU for real: device memory and utilization over time.

The steps are the other benches' (prepare inputs in a child, run the
workload in a fresh interpreter, observe). The observation is NVML -- the
library dcgm-exporter itself reads -- sampled every 0.5 s:

- ``memory.used`` of the device: what ``DCGM_FI_DEV_FB_USED`` reports,
  CUDA context and the caching allocator's reserve included. Its maximum
  is the label: a request is a device, and the device has to hold all of it.
- ``utilization.gpu``: the share of the last sample period a kernel ran
  (``DCGM_FI_DEV_GPU_UTIL``). Its mean over the run is the second label;
  the 30 s windows DCGM would have reported are kept too.

A workload that dies with a CUDA OOM is *censored*: it needs a bigger
device than the one it was measured on, and the station re-measures it on
one. The child also reports torch's own ``max_memory_reserved`` as a
cross-check of NVML. Without a GPU (a laptop) the NVML fields are absent
and the workload runs on CPU.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from ..bench.measure import _base_env, prepare_inputs
from ..bench_cpu.measure import cgroup_cpu_seconds
from . import grid

_CHILD = r"""
import importlib, json, sys, time
spec = json.loads(sys.argv[1])
sys.path.insert(0, spec["dir"])
t0 = time.monotonic()
try:
    result = importlib.import_module("workload").main(*spec["paths"])
    status = {"ok": True, "result": result}
except Exception as e:
    name = type(e).__name__
    oom = name == "OutOfMemoryError" or "CUDA out of memory" in str(e)
    status = {"ok": False, "oom": oom, "error": name + ": " + str(e)[:300]}
status["wall_s"] = time.monotonic() - t0
try:
    import torch
    if torch.cuda.is_available():
        status["torch_max_reserved_mib"] = torch.cuda.max_memory_reserved() / 2**20
        status["torch_max_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
        status["device_name"] = torch.cuda.get_device_name(0)
except Exception:
    pass
with open(spec["status"], "w") as f:
    json.dump(status, f, default=str)
"""


def _nvml():
    try:
        import pynvml

        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        total = pynvml.nvmlDeviceGetMemoryInfo(h).total / 2**20

        def read() -> tuple[float, float]:
            mem = pynvml.nvmlDeviceGetMemoryInfo(h).used / 2**20
            util = float(pynvml.nvmlDeviceGetUtilizationRates(h).gpu)
            return mem, util

        return read, total
    except Exception:  # noqa: BLE001 -- no driver, no GPU, no pynvml
        return None, 0.0


def dcgm_windows(samples: list[tuple[float, float, float]], interval_s: int = grid.DCGM_INTERVAL_S) -> list[float]:
    """Mean utilization in each `interval_s` window: the points DCGM would chart."""
    out: list[float] = []
    if not samples:
        return out
    end = samples[-1][0]
    t = 0.0
    while t < end:
        w = [u for (s, _m, u) in samples if t <= s < t + interval_s]
        if w:
            out.append(sum(w) / len(w))
        t += interval_s
    return out


def run_workload(harness_code: str, paths: list[str], workdir: Path, timeout_s: float = 900,
                 tick_s: float = 0.5) -> dict:
    (workdir / "workload.py").write_text(harness_code)
    status_path = workdir / "status.json"
    spec = json.dumps({"dir": str(workdir), "paths": paths, "status": str(status_path)})
    read, total_mib = _nvml()
    base_mem = read()[0] if read else 0.0
    cpu0 = cgroup_cpu_seconds()
    t0 = time.monotonic()
    proc = subprocess.Popen([sys.executable, "-c", _CHILD, spec], cwd=workdir, env=_base_env(1))
    samples: list[tuple[float, float, float]] = []
    killed = {"why": ""}
    done = threading.Event()

    def sample() -> None:
        while not done.wait(tick_s):
            now = time.monotonic() - t0
            if read:
                try:
                    m, u = read()
                    samples.append((now, m, u))
                except Exception:  # noqa: BLE001
                    pass
            if now > timeout_s:
                killed["why"] = f"timeout {timeout_s:.0f}s"
                proc.kill()
                return

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    _, _, ru = os.wait4(proc.pid, 0)
    done.set()
    proc.returncode = 0
    sampler.join(timeout=2)
    wall = time.monotonic() - t0
    cpu1 = cgroup_cpu_seconds()
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    mems = [m for _t, m, _u in samples]
    utils = [u for _t, _m, u in samples]
    peak = max(mems, default=0.0)
    t_peak = next((t for t, m, _u in samples if m >= 0.9 * peak), wall) if peak else wall
    windows = dcgm_windows(samples)
    oom = bool(status.get("oom"))
    return {
        "ok": bool(status.get("ok")) and not killed["why"],
        "censored": oom,
        "error": killed["why"] or status.get("error", "" if status else "no status (killed?)"),
        "wall_s": float(wall),
        "device_total_mib": float(total_mib),
        "device_name": status.get("device_name", ""),
        "base_mem_mib": float(base_mem),
        "peak_gpu_mib": float(peak),
        "t_peak_frac": float(min(1.0, t_peak / wall)) if wall > 0 else 1.0,
        "mean_util": float(sum(utils) / len(utils)) if utils else 0.0,
        "peak_util_window": float(max(windows, default=0.0)),
        "util_windows": [round(w, 1) for w in windows],
        "torch_max_reserved_mib": float(status.get("torch_max_reserved_mib") or 0.0),
        "torch_max_allocated_mib": float(status.get("torch_max_allocated_mib") or 0.0),
        "mean_cpu_cores": float((cpu1 - cpu0) / wall) if cpu0 is not None and cpu1 is not None and wall > 0
        else float((ru.ru_utime + ru.ru_stime) / wall) if wall > 0 else 0.0,
        "n_samples": len(samples),
        "result": status.get("result"),
    }


def measure_run(harness_code: str, inputs: list[dict], seed: int, timeout_s: float = 900) -> dict:
    with tempfile.TemporaryDirectory(prefix="rsg-") as d:
        workdir = Path(d)
        t0 = time.monotonic()
        paths, sizes = prepare_inputs(inputs, workdir, seed)
        prep_s = time.monotonic() - t0
        out = run_workload(harness_code, paths, workdir, timeout_s)
    out["input_bytes"] = sizes
    out["prep_s"] = prep_s
    return out
