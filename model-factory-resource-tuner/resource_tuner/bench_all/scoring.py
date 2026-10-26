"""One cost model for the joint request: dollars to get a run to SUCCEEDED.

Each axis fails the way it fails on its own bench, all in one simulation:

- **memory** below the need: OOM-killed after ``t_peak_frac`` of the run,
  retried at double the memory (RightSizeMem's ``simulate_run``);
- **CPU** below the need: the run is throttled and finishes late -- each
  second at u cores takes max(1, u / limit) seconds, replayed against the
  measured per-second series where there is one (RightSizeCPU), or against
  the mean where there is not (RightSizeMem rows);
- **GPU** too small: a CUDA OOM after ``t_peak_frac``, retried on the next
  device up (RightSizeGPU's ``simulate``); **no GPU for a GPU task**: it
  fails at start (no CUDA device) and the platform retries it on a T4;
  **a GPU for a CPU task**: it runs, on a GPU node, at the GPU node's price.

A pod is priced by its dominant share of an m6i.xlarge (RightSizeMem's
"packed" model); a pod with a GPU at the GPU instance's on-demand price
(host memory and cores come with it). Every retry adds ``OOM_PENALTY`` x
the oracle's cost (the delay), and every hour late ``DELAY_PENALTY`` x the
oracle's hourly price, as on the benches.

The **oracle** asks for each measured axis's label cell (and no GPU for a
CPU task). ``regret`` is per run: cost / oracle cost - 1, averaged over
runs -- every run counts once, whether it costs $0.01 or $2 -- and the RL
arm's reward is exactly its negative.

Pure logic: no flyte, no torch.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np

from ..bench.grid import dollars_per_hr
from ..bench_cpu.grid import within_margin as cpu_within_margin
from ..bench_cpu.scoring import throttled_seconds
from ..bench_gpu import scoring as gpu_scoring
from . import grid

ATTEMPT_OVERHEAD = 0.02  # task-hours per attempt: pull, start, read inputs
OOM_PENALTY = 1.0
DELAY_PENALTY = 1.0
MAX_ATTEMPTS = 8
#: A GPU task scheduled without a GPU dies at CUDA init.
NO_GPU_FAIL_HOURS = 0.05
#: Regret ratios are capped here for the RL reward (a GPU asked for a 0.25-core
#: task is ~50x the oracle; the cap keeps one such run from swamping a batch).
REWARD_CAP = 20.0


@dataclass
class Outcome:
    cost: float  # dollars, all attempts, penalties included
    attempts: int
    first_ok: bool  # the first attempt succeeded (no OOM, no CUDA OOM, no missing GPU)
    failed: bool  # ran out of attempts
    slowdown: float  # throttled hours / nominal hours, >= 1


def _series(row: dict) -> np.ndarray:
    s = row.get("_series")
    if s is None:
        s = np.array(json.loads(row.get("cpu_series_json") or "[]"), dtype=float)
        row["_series"] = s
    return s


def stretch(row: dict, cores: float) -> float:
    """Hours the run takes per nominal hour under a CPU limit of `cores`."""
    s = _series(row)
    if len(s):
        return throttled_seconds(s, cores) / len(s)
    return max(1.0, max(float(row["mean_cores"]), 0.05) / cores)


def _cpu_task(row: dict, m: int, c: int, g: int) -> tuple[float, int, bool, float]:
    cores = grid.CPU_GRID[c]
    st = stretch(row, cores)
    gpu_price = grid.DEVICES[grid.device_of(g)][2] if g != grid.NO_GPU else None
    need, tpf = float(row["need_mib"]), float(row["t_peak_frac"])
    cost, attempts, mi = 0.0, 0, m
    while attempts < MAX_ATTEMPTS:
        attempts += 1
        mem = grid.MEMORY_GRID_MIB[mi]
        rate = gpu_price if gpu_price is not None else dollars_per_hr(mem, cores)
        if mem >= need:
            return cost + rate * (st + ATTEMPT_OVERHEAD), attempts, False, st
        cost += rate * (st * max(tpf, 0.05) + ATTEMPT_OVERHEAD)
        if mi == grid.N_MEM - 1:
            break
        mi = grid.mem_cell(mem * 2)
    return cost, attempts, True, st


def _gpu_task(row: dict, m: int, c: int, g: int) -> tuple[float, int, bool, float]:
    peak, tpf = float(row["peak_gpu_gib"]), float(row["t_peak_frac"])
    cost, attempts = 0.0, 0
    if g == grid.NO_GPU:
        cost += dollars_per_hr(grid.MEMORY_GRID_MIB[m], grid.CPU_GRID[c]) * NO_GPU_FAIL_HOURS
        attempts += 1
        g = 1  # the platform's retry: the smallest cell, i.e. a T4
    c2, att, failed = gpu_scoring.simulate(g - 1, peak, tpf)
    return cost + c2, attempts + att, failed, 1.0


def raw_cost(row: dict, m: int, c: int, g: int) -> Outcome:
    fn = _gpu_task if row["source"] == "gpu" else _cpu_task
    cost, attempts, failed, st = fn(row, m, c, g)
    return Outcome(cost, attempts, attempts == 1 and not failed, failed, st)


def oracle_action(row: dict) -> tuple[int, int, int]:
    if row["source"] == "gpu":
        return 0, 0, int(row["label_gpu"])
    return int(row["label_mem"]), int(row["label_cpu"]), grid.NO_GPU


def oracle_cost(row: dict) -> float:
    oc = row.get("_oracle")
    if oc is None:
        oc = raw_cost(row, *oracle_action(row)).cost
        row["_oracle"] = oc
    return oc


def cost(row: dict, m: int, c: int, g: int) -> Outcome:
    """The run's cost with the delay penalties: OOM_PENALTY x oracle per
    retry, DELAY_PENALTY x the oracle's hourly price per hour late."""
    o = raw_cost(row, m, c, g)
    base = oracle_cost(row)
    o.cost += OOM_PENALTY * base * (o.attempts - 1) + DELAY_PENALTY * base * (o.slowdown - 1.0)
    return o


def ratio(row: dict, m: int, c: int, g: int) -> float:
    return cost(row, m, c, g).cost / max(oracle_cost(row), 1e-9)


def reward(row: dict, m: int, c: int, g: int) -> float:
    """The RL arm's reward: minus the run's regret ratio (capped)."""
    return -min(ratio(row, m, c, g), REWARD_CAP)


# ── per-axis accuracy, as each bench defines it ─────────────────────────


def mem_right(row: dict, m: int) -> bool:
    """Fits, and at most one ladder step above the cheapest cell that fits."""
    return 0 <= m - int(row["label_mem"]) <= 1


def cpu_right(row: dict, c: int) -> bool:
    return cpu_within_margin(grid.CPU_GRID[c], float(row["need_cores"]))


def gpu_right(row: dict, g: int) -> bool:
    """CPU task: no GPU. GPU task: the device the oracle lands on."""
    if row["source"] != "gpu":
        return g == grid.NO_GPU
    return g != grid.NO_GPU and grid.device_of(g) == grid.device_of(int(row["label_gpu"]))


def score(rows: list[dict], preds: dict[str, dict]) -> dict:
    """`preds[run_id]` = {"mem_idx", "cpu_idx", "gpu_idx"}."""
    ratios, costs, ocosts, first, fails = [], [], [], [], []
    m_ok, c_ok, g_ok, all_ok, need_gpu_ok, gpu_waste = [], [], [], [], [], []
    missing = 0
    for r in rows:
        p = preds.get(r["run_id"])
        if p is None:
            missing += 1
            continue
        m, c, g = int(p["mem_idx"]), int(p["cpu_idx"]), int(p["gpu_idx"])
        o = cost(r, m, c, g)
        oc = oracle_cost(r)
        ratios.append(o.cost / max(oc, 1e-9)), costs.append(o.cost), ocosts.append(oc)
        first.append(o.first_ok), fails.append(o.failed)
        gr = gpu_right(r, g)
        g_ok.append(gr)
        need_gpu_ok.append((g != grid.NO_GPU) == (r["source"] == "gpu"))
        if r["source"] == "gpu":
            all_ok.append(gr)
        else:
            mr, cr = mem_right(r, m), cpu_right(r, c)
            m_ok.append(mr), c_ok.append(cr)
            all_ok.append(mr and cr and gr)
            gpu_waste.append(g != grid.NO_GPU)
    n = len(ratios)
    if not n:
        return {"n": 0, "missing": missing}
    lr = np.log(np.maximum(np.array(ratios), 1e-9))

    def mean(v):
        return float(np.mean(v)) if v else float("nan")

    return {
        "n": n, "missing": missing,
        "regret": float(np.mean(ratios)) - 1.0,
        "geo_regret": float(math.exp(lr.mean())) - 1.0,
        "dollar_regret": float(np.sum(costs) / np.sum(ocosts)) - 1.0,
        "cost_per_1k_runs": 1000 * float(np.mean(costs)),
        "oracle_cost_per_1k_runs": 1000 * float(np.mean(ocosts)),
        "first_attempt_ok": mean(first),
        "fail_rate": mean(fails),
        "all_right": mean(all_ok),
        "mem_right": mean(m_ok),
        "cpu_right": mean(c_ok),
        "gpu_right": mean(g_ok),
        "needs_gpu_right": mean(need_gpu_ok),
        "gpu_for_cpu_task": mean(gpu_waste),
    }


def score_slices(rows: list[dict], preds: dict[str, dict]) -> dict[str, dict]:
    out = {"all": score(rows, preds)}
    for split in ("val", "test", "ood"):
        sub = [r for r in rows if r["split"] == split]
        out[f"split={split}"] = score(sub, preds)
        for src in ("mem", "cpu", "gpu"):
            out[f"{split}/source={src}"] = score([r for r in sub if r["source"] == src], preds)
    test = [r for r in rows if r["split"] == "test"]
    for cold in (True, False):
        out[f"test/cold={cold}"] = score([r for r in test if (r["n_history"] == 0) == cold], preds)
    return out


# ── decoding a distribution per axis ────────────────────────────────────


def decode(row_source_probs: tuple, gpu_threshold: float, cpu_decode, mem_decode, gpu_decode) -> tuple[int, int, int]:
    """(P(mem), P(cpu), P(gpu incl. none)) -> one joint request.

    No GPU when P(none) >= `gpu_threshold` (tuned on val: a missing GPU costs
    a failed start; a needless one costs the GPU's price for the whole run).
    Otherwise the GPU cell decoded from P(gpu | some GPU), and memory / CPU
    by their bench's decoders (they are unused on a GPU node, but cost
    nothing to answer)."""
    pm, pc, pg = (np.asarray(x, dtype=float) for x in row_source_probs)
    m, c = mem_decode(pm), cpu_decode(pc)
    if pg[grid.NO_GPU] >= gpu_threshold:
        return m, c, grid.NO_GPU
    rest = pg[1:] / max(pg[1:].sum(), 1e-12)
    return m, c, 1 + gpu_decode(rest)
