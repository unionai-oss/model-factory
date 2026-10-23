"""Measure a workload for real: write its inputs, run it, read the kernel's numbers.

Labels in this bench are never formulas. ``measure_run`` does three things in
separate processes so their memory never mixes:

1. **prepare** -- write the run's input files to a scratch dir, in chunks,
   so preparing a 3 GiB parquet file costs a few hundred MiB, not 3 GiB
   (this matters doubly in live validation, where prep and workload share
   one pod limit);
2. **run** -- exec the harness body in a fresh interpreter;
3. **observe** -- ``os.wait4`` returns that child's ``ru_maxrss`` (the
   kernel's high-water mark, which is what an OOM decision is made against)
   and its CPU seconds. A sampler thread polls RSS meanwhile, for the time
   to peak and to kill a run that would take the whole pod down with it.

Everything here is plain Python + numpy/pyarrow: the station task calls it,
the tests call it, and so can a laptop.
"""

from __future__ import annotations

import json
import os
import resource
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

# Values per write chunk: ~32 MiB of float64. Live validation lowers it so
# writing inputs stays far below even the smallest request on the grid.
_CHUNK_CELLS = int(os.environ.get("RSB_PREP_CHUNK_CELLS", "4000000"))


# ── input writers ───────────────────────────────────────────────────────


def _column(rng: np.random.Generator, spec: dict, n: int):
    import pyarrow as pa

    kind = spec["kind"]
    if kind == "f64":
        return pa.array(rng.standard_normal(n))
    if kind == "f32":
        return pa.array(rng.standard_normal(n).astype(np.float32))
    if kind in ("i64", "i32"):
        hi = int(spec.get("distinct") or 1_000_000)
        dtype = np.int64 if kind == "i64" else np.int32
        return pa.array(rng.integers(0, hi, n, dtype=dtype))
    if kind == "cat":
        k = int(spec.get("distinct") or 16)
        labels = pa.array([f"{spec['name']}_{i:04d}" for i in range(k)])
        return pa.DictionaryArray.from_arrays(pa.array(rng.integers(0, k, n, dtype=np.int32)), labels)
    if kind == "str":
        length = int(spec.get("len", 12))
        raw = rng.integers(97, 123, (n, length), dtype=np.uint8)
        return pa.array(raw.view(f"S{length}").ravel(), type=pa.binary()).cast(pa.string())
    if kind == "text":
        words, vocab = int(spec.get("words", 40)), int(spec.get("vocab", 30_000))
        # Zipf-ish word ids so the vocabulary has a realistic head and tail.
        ids = np.minimum(rng.zipf(1.3, (n, words)), vocab) - 1
        return pa.array([" ".join(f"w{j}" for j in row) for row in ids.tolist()])
    raise ValueError(f"unknown column kind {kind!r}")


def _table_chunks(spec: dict, seed: int):
    import pyarrow as pa

    rng = np.random.default_rng(seed)
    ncols = max(1, len(spec["columns"]))
    text_cols = sum(c["kind"] == "text" for c in spec["columns"])
    step = max(1_000, _CHUNK_CELLS // ncols // (20 if text_cols else 1))
    done = 0
    while done < spec["rows"]:
        n = min(step, spec["rows"] - done)
        yield pa.table({c["name"]: _column(rng, c, n) for c in spec["columns"]})
        done += n


def write_input(spec: dict, directory: Path, seed: int) -> Path:
    """Write one input file; returns its path. Memory stays O(chunk)."""
    fmt = spec["fmt"]
    path = directory / f"{spec['name']}.{fmt}"
    if fmt == "npy":
        rng = np.random.default_rng(seed)
        shape = tuple(spec["shape"])
        dtype = np.dtype(spec["dtype"])
        out = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
        per_row = int(np.prod(shape[1:])) or 1
        step = max(1, _CHUNK_CELLS // per_row)
        for i in range(0, shape[0], step):
            n = min(step, shape[0] - i)
            if dtype == np.uint8:
                out[i : i + n] = rng.integers(0, 256, (n, *shape[1:]), dtype=np.uint8)
            else:
                out[i : i + n] = rng.standard_normal((n, *shape[1:])).astype(dtype)
        out.flush()
        del out
        return path
    if fmt == "parquet":
        import pyarrow.parquet as pq

        writer = None
        for chunk in _table_chunks(spec, seed):
            if writer is None:
                writer = pq.ParquetWriter(path, chunk.schema)
            writer.write_table(chunk)
        writer.close()
        return path
    if fmt == "csv":
        import pyarrow as pa
        import pyarrow.csv as pacsv

        writer = None
        for chunk in _table_chunks(spec, seed):
            chunk = pa.table({n: (c.cast(pa.string()) if pa.types.is_dictionary(c.type) else c)
                              for n, c in zip(chunk.column_names, chunk.columns)})
            if writer is None:
                writer = pacsv.CSVWriter(str(path), chunk.schema)
            writer.write_table(chunk)
        writer.close()
        return path
    if fmt == "jsonl":
        with open(path, "w") as f:
            for chunk in _table_chunks(spec, seed):
                cols = {n: c.to_pylist() for n, c in zip(chunk.column_names, chunk.columns)}
                names = list(cols)
                for row in zip(*cols.values()):
                    f.write(json.dumps(dict(zip(names, row))))
                    f.write("\n")
        return path
    raise ValueError(f"unknown input format {fmt!r}")


# ── child-process driver ────────────────────────────────────────────────
# Runs inside the measured interpreter. Kept as source text so the child
# starts from a bare `python -c`, with nothing of the parent's imports.

_CHILD = r"""
import json, sys, time
spec = json.loads(sys.argv[1])
ns = {}
code = open(spec["harness"]).read()
t0 = time.monotonic()
try:
    exec(compile(code, "workload", "exec"), ns)
    result = ns["main"](*spec["paths"])
    status = {"ok": True, "result": result}
except MemoryError as e:
    status = {"ok": False, "error": "MemoryError: " + str(e)[:300]}
except Exception as e:
    status = {"ok": False, "error": type(e).__name__ + ": " + str(e)[:300]}
status["wall_s"] = time.monotonic() - t0
with open(spec["status"], "w") as f:
    json.dump(status, f, default=str)
"""

_PREP = r"""
import json, sys
from pathlib import Path
sys.path[:0] = json.loads(sys.argv[2])
from resource_tuner.bench.measure import write_input
spec = json.loads(sys.argv[1])
paths = [str(write_input(s, Path(spec["dir"]), spec["seed"] + i)) for i, s in enumerate(spec["inputs"])]
print(json.dumps(paths))
"""


def _rss_mib(pid: int) -> float:
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    try:  # macOS / no procfs
        import psutil

        return psutil.Process(pid).memory_info().rss / 2**20
    except Exception:  # noqa: BLE001
        return 0.0


def _kill(pid: int) -> None:
    import signal

    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _maxrss_mib(ru) -> float:
    # Linux reports KiB, macOS bytes.
    return ru.ru_maxrss / (2**20 if sys.platform == "darwin" else 1024)


def _base_env(threads: int) -> dict:
    env = dict(os.environ)
    # BLAS/OpenMP default to every core of the NODE, not the pod. Workloads
    # that want more threads say so in their own code (it is part of what
    # an arm reads); everything else gets this floor.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[var] = str(threads)
    return env


def prepare_inputs(inputs: list[dict], directory: Path, seed: int) -> tuple[list[str], dict[str, int]]:
    """Write inputs in a child process (so its memory is not attributed to
    anything) and return (paths, {name: bytes on disk})."""
    spec = json.dumps({"inputs": inputs, "dir": str(directory), "seed": seed})
    out = subprocess.run(
        [sys.executable, "-c", _PREP, spec, json.dumps(sys.path)],
        capture_output=True, text=True, check=True, env=_base_env(1),
    )
    paths = json.loads(out.stdout.strip().splitlines()[-1])
    return paths, {s["name"]: os.path.getsize(p) for s, p in zip(inputs, paths)}


def run_workload(
    harness_code: str,
    paths: list[str],
    workdir: Path,
    kill_above_mib: float | None = None,
    timeout_s: float = 600,
    threads: int = 1,
) -> dict:
    """Run one harness body in a fresh interpreter and measure it."""
    harness = workdir / "workload.py"
    harness.write_text(harness_code)
    status_path = workdir / "status.json"
    spec = json.dumps({"harness": str(harness), "paths": paths, "status": str(status_path)})
    t0 = time.monotonic()
    proc = subprocess.Popen([sys.executable, "-c", _CHILD, spec], cwd=workdir, env=_base_env(threads))
    samples: list[tuple[float, float]] = []
    killed = {"why": ""}
    # Never proc.poll() here: it can reap the child before os.wait4 does,
    # and the rusage -- the whole point -- goes with it.
    done = threading.Event()

    def sample() -> None:
        while not done.is_set():
            rss = _rss_mib(proc.pid)
            samples.append((time.monotonic() - t0, rss))
            if kill_above_mib and rss > kill_above_mib:
                killed["why"] = f"rss {rss:.0f}MiB > guard {kill_above_mib:.0f}MiB"
                _kill(proc.pid)
                return
            if time.monotonic() - t0 > timeout_s:
                killed["why"] = f"timeout {timeout_s:.0f}s"
                _kill(proc.pid)
                return
            time.sleep(0.02)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    _, _, ru = os.wait4(proc.pid, 0)
    done.set()
    proc.returncode = 0  # reaped by wait4; keep Popen from waiting again
    sampler.join(timeout=2)
    wall = time.monotonic() - t0
    peak = _maxrss_mib(ru)
    cpu_s = ru.ru_utime + ru.ru_stime
    status: dict = {}
    if status_path.exists():
        status = json.loads(status_path.read_text())
    # Time to (90% of) peak, as a fraction of the run: how much of a run an
    # under-provisioned attempt burns before the OOM kill lands.
    t_peak = next((t for t, r in samples if r >= 0.9 * peak), wall)
    return {
        "ok": bool(status.get("ok")) and not killed["why"],
        "censored": bool(killed["why"]),
        "error": killed["why"] or status.get("error", "" if status else "no status (killed?)"),
        "peak_rss_mib": float(peak),
        "cpu_s": float(cpu_s),
        "wall_s": float(wall),
        "avg_cores": float(cpu_s / wall) if wall > 0 else 0.0,
        "t_peak_frac": float(min(1.0, t_peak / wall)) if wall > 0 else 1.0,
        "result": status.get("result"),
    }


def measure_run(
    harness_code: str,
    inputs: list[dict],
    seed: int,
    kill_above_mib: float | None = None,
    timeout_s: float = 600,
) -> dict:
    """Prepare inputs, run, measure; the scratch dir is gone afterwards."""
    with tempfile.TemporaryDirectory(prefix="rsb-") as d:
        workdir = Path(d)
        t0 = time.monotonic()
        paths, sizes = prepare_inputs(inputs, workdir, seed)
        prep_s = time.monotonic() - t0
        out = run_workload(harness_code, paths, workdir, kill_above_mib, timeout_s)
    out["input_bytes"] = sizes
    out["prep_s"] = prep_s
    return out


def interpreter_baseline_mib() -> float:
    """Peak RSS of an interpreter that does nothing: subtracted nowhere, but
    recorded so the per-workload numbers can be read against it."""
    out = subprocess.run([sys.executable, "-c", "pass"], capture_output=True)
    del out
    return _maxrss_mib(resource.getrusage(resource.RUSAGE_CHILDREN))
