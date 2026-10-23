"""Measure a workload's CPU for real: cores used, second by second.

Same three steps as RightSizeMem's ``measure`` (prepare inputs in a child,
run the workload in a fresh interpreter, observe), with a different
observation: cumulative CPU seconds sampled once a second, from

- the container's **cgroup** (``cpu.stat`` usage), where it exists: the
  same counter cAdvisor exports as ``container_cpu_usage_seconds_total``,
  so the label is what the platform's monitoring sees, process pools and
  all; or
- the workload's **process tree** (psutil), elsewhere (a laptop).

The per-second series is kept with the run: the busiest 15-second window
is the label (``grid.peak_window``), and the scoring simulates a CPU limit
against the series itself (``scoring.throttled_seconds``).

The workload is imported as a module, not exec'd, so a process pool can
pickle its worker function by name.
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

from ..bench.measure import _base_env, _maxrss_mib, prepare_inputs
from . import grid

_CHILD = r"""
import importlib, json, sys, time
spec = json.loads(sys.argv[1])
sys.path.insert(0, spec["dir"])
t0 = time.monotonic()
try:
    result = importlib.import_module("workload").main(*spec["paths"])
    status = {"ok": True, "result": result}
except MemoryError as e:
    status = {"ok": False, "error": "MemoryError: " + str(e)[:300]}
except Exception as e:
    status = {"ok": False, "error": type(e).__name__ + ": " + str(e)[:300]}
status["wall_s"] = time.monotonic() - t0
with open(spec["status"], "w") as f:
    json.dump(status, f, default=str)
"""

_CGROUP_V2 = Path("/sys/fs/cgroup/cpu.stat")
_CGROUP_V1 = (Path("/sys/fs/cgroup/cpuacct/cpuacct.usage"), Path("/sys/fs/cgroup/cpu,cpuacct/cpuacct.usage"))


def cgroup_cpu_seconds() -> float | None:
    """The container's cumulative CPU seconds, or None outside a cgroup."""
    try:
        for line in _CGROUP_V2.read_text().splitlines():
            if line.startswith("usage_usec"):
                return int(line.split()[1]) / 1e6
    except OSError:
        pass
    for p in _CGROUP_V1:
        try:
            return int(p.read_text()) / 1e9
        except OSError:
            continue
    return None


def _tree_cpu_seconds(pid: int) -> float:
    import psutil

    try:
        root = psutil.Process(pid)
        procs = [root, *root.children(recursive=True)]
    except psutil.Error:
        return 0.0
    total = 0.0
    for p in procs:
        try:
            t = p.cpu_times()
            total += t.user + t.system
        except psutil.Error:
            continue
    return total


def _source() -> str:
    forced = os.environ.get("RSB_CPU_SOURCE")
    if forced:
        return forced
    return "cgroup" if cgroup_cpu_seconds() is not None else "psutil"


def per_second(stamps: list[tuple[float, float]]) -> list[float]:
    """(elapsed s, cumulative CPU s) samples -> cores used in each whole second.

    Interpolated onto a 1 s grid, so a short final interval (the child
    exits 0.05 s after the last tick) cannot turn into a 20-core "spike";
    a trailing partial second is folded into the last whole one."""
    import numpy as np

    t = np.array([s[0] for s in stamps], dtype=float)
    c = np.maximum.accumulate(np.array([s[1] for s in stamps], dtype=float))
    end = t[-1]
    if end <= 0:
        return []
    edges = np.arange(0.0, max(1.0, np.floor(end)) + 1.0)
    edges[-1] = end  # last bin absorbs the partial second
    cum = np.interp(edges, t, c)
    widths = np.diff(edges)
    return [float(x) for x in np.maximum(0.0, np.diff(cum) / np.maximum(widths, 1e-6))]


def run_workload(harness_code: str, paths: list[str], workdir: Path, timeout_s: float = 400,
                 tick_s: float = 1.0) -> dict:
    """Run one harness module in a fresh interpreter; sample its CPU every `tick_s`."""
    (workdir / "workload.py").write_text(harness_code)
    status_path = workdir / "status.json"
    spec = json.dumps({"dir": str(workdir), "paths": paths, "status": str(status_path)})
    source = _source()
    t0 = time.monotonic()
    proc = subprocess.Popen([sys.executable, "-c", _CHILD, spec], cwd=workdir, env=_base_env(1))
    read = cgroup_cpu_seconds if source == "cgroup" else (lambda: _tree_cpu_seconds(proc.pid))
    base = read() or 0.0
    stamps: list[tuple[float, float]] = [(0.0, 0.0)]
    killed = {"why": ""}
    done = threading.Event()

    def sample() -> None:
        last_tree = 0.0
        while not done.wait(tick_s):
            now = time.monotonic() - t0
            used = (read() or 0.0) - base
            if source != "cgroup":  # a process tree loses its exited children
                used = max(used, last_tree)
                last_tree = used
            stamps.append((now, used))
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
    child_cpu = ru.ru_utime + ru.ru_stime
    if source == "cgroup":
        stamps.append((wall, (read() or 0.0) - base))
    else:
        stamps.append((wall, max(stamps[-1][1], child_cpu)))
    series = per_second(stamps)
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    cpu_s = stamps[-1][1]
    return {
        "ok": bool(status.get("ok")) and not killed["why"],
        "censored": False,
        "error": killed["why"] or status.get("error", "" if status else "no status (killed?)"),
        "cpu_source": source,
        "wall_s": float(wall),
        "cpu_s": float(cpu_s),
        "mean_cores": float(cpu_s / wall) if wall > 0 else 0.0,
        "peak_cores": float(grid.peak_window(series)),
        "peak_cores_1s": float(max(series, default=0.0)),
        "cpu_series": [round(x, 3) for x in series],
        "peak_rss_mib": float(_maxrss_mib(ru)),
        "result": status.get("result"),
    }


def measure_run(harness_code: str, inputs: list[dict], seed: int, timeout_s: float = 400) -> dict:
    with tempfile.TemporaryDirectory(prefix="rsc-") as d:
        workdir = Path(d)
        t0 = time.monotonic()
        paths, sizes = prepare_inputs(inputs, workdir, seed)
        prep_s = time.monotonic() - t0
        out = run_workload(harness_code, paths, workdir, timeout_s)
    out["input_bytes"] = sizes
    out["prep_s"] = prep_s
    return out
