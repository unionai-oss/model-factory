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
machine than the one it was measured on, and the station re-measures it on
one. Every GPU a run can see is sampled; a sharded run's per-GPU peaks are
folded into one single-GPU-equivalent peak (``summarize``). The child also reports torch's own ``max_memory_reserved`` as a
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


def _nvml(indices: list[int] | None = None):
    """A reader of (memory.used MiB, utilization %) for each of `indices` (all GPUs if None)."""
    try:
        import pynvml

        pynvml.nvmlInit()
        idx = list(range(pynvml.nvmlDeviceGetCount())) if indices is None else list(indices)
        hs = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in idx]
        if not hs:
            return None, 0.0
        total = pynvml.nvmlDeviceGetMemoryInfo(hs[0]).total / 2**20

        def read() -> list[tuple[float, float]]:
            return [(pynvml.nvmlDeviceGetMemoryInfo(h).used / 2**20, float(pynvml.nvmlDeviceGetUtilizationRates(h).gpu))
                    for h in hs]

        return read, total
    except Exception:  # noqa: BLE001 -- no driver, no GPU, no pynvml
        return None, 0.0


def gpu_count() -> int:
    try:
        import pynvml

        pynvml.nvmlInit()
        return int(pynvml.nvmlDeviceGetCount())
    except Exception:  # noqa: BLE001
        return 0


#: A GPU counts as used by a run once its memory rises this far above where it started.
USED_MIB = 256.0


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


def summarize(samples: list[tuple[float, list[float], list[float]]], base: list[float], wall: float,
              ctx_mib: float = grid.CONTEXT_GIB * 1024) -> dict:
    """Per-GPU NVML samples -> the run's labels.

    The peak is the *single-GPU equivalent*: the per-GPU peaks summed, less
    the extra CUDA contexts (k GPUs carry k of them, one GPU one), so a
    sharded run measured on 8 T4s and on one L40S label the same need.
    ``gpu_imbalance`` is how much more than an even share its busiest GPU
    held; ``grid.fits`` reads it to place the run on other shapes."""
    if not samples:
        return {"peak_gpu_mib": 0.0, "gpu_n_used": 0, "gpu_max_dev_mib": 0.0, "gpu_imbalance": 1.0,
                "t_peak_frac": 1.0, "mean_util": 0.0, "util_windows": [], "peak_util_window": 0.0}
    n = len(base)
    peaks = [max(m[i] for _t, m, _u in samples) for i in range(n)]
    used = [i for i in range(n) if peaks[i] - base[i] >= USED_MIB] or [int(max(range(n), key=lambda i: peaks[i]))]
    k = len(used)
    core = max(sum(peaks[i] for i in used) - k * ctx_mib, 0.0)
    peak = core + ctx_mib if k > 1 else peaks[used[0]]
    max_dev = max(peaks[i] for i in used)
    imbalance = max((max_dev - ctx_mib) * k / core, 1.0) if k > 1 and core > 0 else 1.0
    totals = [(t, sum(m[i] for i in used)) for t, m, _u in samples]
    top = max(v for _t, v in totals)
    t_peak = next((t for t, v in totals if v >= 0.9 * top), wall)
    flat = [(t, 0.0, sum(u[i] for i in used) / k) for t, _m, u in samples]
    windows = dcgm_windows(flat)
    utils = [u for _t, _m, u in flat]
    return {"peak_gpu_mib": float(peak), "gpu_n_used": k, "gpu_max_dev_mib": float(max_dev),
            "gpu_imbalance": float(imbalance), "t_peak_frac": float(min(1.0, t_peak / wall)) if wall > 0 else 1.0,
            "mean_util": float(sum(utils) / len(utils)), "util_windows": [round(w, 1) for w in windows],
            "peak_util_window": float(max(windows, default=0.0))}


def run_workload(harness_code: str, paths: list[str], workdir: Path, timeout_s: float = 900,
                 tick_s: float = 0.5, gpus: list[int] | None = None) -> dict:
    """Run one workload in a fresh interpreter and sample its GPUs.

    `gpus`: the NVML indices the run may use (CUDA_VISIBLE_DEVICES, in PCI
    order so NVML and CUDA agree); None = every GPU, nothing pinned. With
    several runs in one pod (one per GPU) the pod's cgroup is shared, so CPU
    use is read from the child's own rusage."""
    (workdir / "workload.py").write_text(harness_code)
    status_path = workdir / "status.json"
    spec = json.dumps({"dir": str(workdir), "paths": paths, "status": str(status_path)})
    read, total_mib = _nvml(gpus)
    base = [m for m, _u in read()] if read else []
    env = _base_env(1)
    if gpus is not None:
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
    pinned = gpus is not None and gpu_count() > len(gpus)
    cpu0 = None if pinned else cgroup_cpu_seconds()
    t0 = time.monotonic()
    proc = subprocess.Popen([sys.executable, "-c", _CHILD, spec], cwd=workdir, env=env)
    samples: list[tuple[float, list[float], list[float]]] = []
    killed = {"why": ""}
    done = threading.Event()

    def sample() -> None:
        while not done.wait(tick_s):
            now = time.monotonic() - t0
            if read:
                try:
                    r = read()
                    samples.append((now, [m for m, _u in r], [u for _m, u in r]))
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
    cpu1 = None if pinned else cgroup_cpu_seconds()
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    labels = summarize(samples, base, wall)
    oom = bool(status.get("oom"))
    return {
        "ok": bool(status.get("ok")) and not killed["why"],
        "censored": oom,
        "error": killed["why"] or status.get("error", "" if status else "no status (killed?)"),
        "wall_s": float(wall),
        "device_total_mib": float(total_mib),
        "device_name": status.get("device_name", ""),
        "n_gpus_visible": len(base),
        "base_mem_mib": float(sum(base)),
        **labels,
        "torch_max_reserved_mib": float(status.get("torch_max_reserved_mib") or 0.0),
        "torch_max_allocated_mib": float(status.get("torch_max_allocated_mib") or 0.0),
        "mean_cpu_cores": float((cpu1 - cpu0) / wall) if cpu0 is not None and cpu1 is not None and wall > 0
        else float((ru.ru_utime + ru.ru_stime) / wall) if wall > 0 else 0.0,
        "n_samples": len(samples),
        "result": status.get("result"),
    }


def measure_run(harness_code: str, inputs: list[dict], seed: int, timeout_s: float = 900,
                gpus: list[int] | None = None) -> dict:
    with tempfile.TemporaryDirectory(prefix="rsg-") as d:
        workdir = Path(d)
        t0 = time.monotonic()
        paths, sizes = prepare_inputs(inputs, workdir, seed)
        prep_s = time.monotonic() - t0
        out = run_workload(harness_code, paths, workdir, timeout_s, gpus=gpus)
    out["input_bytes"] = sizes
    out["prep_s"] = prep_s
    return out


def measure_many(runs: list[tuple[str, list[dict], int]], measure_one, mode: str = "per_gpu") -> list[dict]:
    """Measure `runs` in one pod. ``per_gpu``: one run per GPU at a time, each
    pinned to its own GPU (a 4-GPU node measures 4 single-GPU runs at once).
    ``whole``: one run at a time with every GPU visible (sharded code spreads
    over all of them). `measure_one(run, gpus)` measures one run."""
    n = gpu_count()
    if mode == "whole" or n <= 1:
        return [measure_one(r, None) for r in runs]
    import queue

    todo: queue.Queue = queue.Queue()
    for i, r in enumerate(runs):
        todo.put((i, r))
    out: list[dict | None] = [None] * len(runs)

    def worker(g: int) -> None:
        while True:
            try:
                i, r = todo.get_nowait()
            except queue.Empty:
                return
            out[i] = measure_one(r, [g])

    threads = [threading.Thread(target=worker, args=(g,)) for g in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return [o for o in out if o is not None]
