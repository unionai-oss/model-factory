"""Score GPU decisions: the right machine, the right memory, the right utilization.

**Cost** (``gpu_regret``). A predicted peak memory is a request for the
cheapest machine shape whose GPU memory holds it (``grid.device_for``). If
the run fits that shape (``grid.fits``: one GPU holds the whole peak, or a
sharded run's share fits each of several) it pays one nominal task-hour at
the shape's price; if not, it dies with a CUDA OOM after the part of the
run before its peak (``t_peak_frac``, measured -- GPU peaks come early: the
first forward/backward), and is retried on the next shape up. Each OOM
also costs OOM_PENALTY oracle runs of delay. ``gpu_regret`` = dollars /
the oracle's dollars - 1; the oracle runs on the cheapest shape it fits.

Headline and decoding target: memory **accuracy**, not cost (round 18:
under expected-cost decoding every distributional arm collapsed onto the
one cell below the T4's capacity). Distributions are decoded by
``decode_index`` (the cell most likely within margin of the peak, ties
up); ``bayes_index`` (expected cost) is the ``-cost`` read-out and the
risk frontier.

**Accuracy.** Memory: exact cell, within one cell, within 20% of the peak
(``grid.mem_within_margin``; skipping runs whose peak filled the device), first-attempt OOM rate, and whether the
machine matches the oracle's. Utilization: mean absolute error in points and
the share within 10 points (``grid.UTIL_MARGIN``).

Pure numpy, no flyte.
"""

from __future__ import annotations

import math

import numpy as np

from ..bench.scoring import expected_calibration_error
from . import grid

N = len(grid.GPU_MEM_GRID_GIB)
CELLS = np.array(grid.GPU_MEM_GRID_GIB, dtype=float)
ATTEMPT_OVERHEAD = 0.05  # image pull + CUDA init + data load, per attempt (task-hours)
OOM_PENALTY = 1.0
MAX_ATTEMPTS = 4


def simulate(cell_idx: int, peak_gib: float, t_peak_frac: float, sharded: bool = False,
             imbalance: float = grid.DEFAULT_IMBALANCE) -> tuple[float, int, bool]:
    """(dollars, attempts, failed) to get a run to SUCCEEDED from this request."""
    dev = grid.device_for(CELLS[cell_idx])
    cost, attempts = 0.0, 0
    while dev is not None and attempts < MAX_ATTEMPTS:
        attempts += 1
        price = grid.DEVICES[dev].price
        if grid.fits(dev, peak_gib, sharded, imbalance):
            return cost + price * (1 + ATTEMPT_OVERHEAD), attempts, False
        cost += price * (max(t_peak_frac, 0.02) + ATTEMPT_OVERHEAD)
        dev = grid.next_device_up(dev)
    return cost, attempts, True


def oracle_cost(peak_gib: float, sharded: bool = False, imbalance: float = grid.DEFAULT_IMBALANCE) -> float:
    dev = grid.oracle_device(peak_gib, sharded, imbalance)
    return grid.DEVICES[grid.device_for(peak_gib) if dev is None else dev].price * (1 + ATTEMPT_OVERHEAD)


def run_fit(row: dict) -> tuple[bool, float]:
    """(sharded, imbalance) of a measured run, as `simulate` and `grid.fits` take them."""
    return bool(row.get("sharded")), float(row.get("gpu_imbalance") or grid.DEFAULT_IMBALANCE)


def cost_matrix(penalty: float = OOM_PENALTY, t_peak_frac: float = 0.1) -> np.ndarray:
    C = np.zeros((N, N))
    for j in range(N):
        need = (CELLS[j - 1] + CELLS[j]) / 2 if j else CELLS[0] * 0.7
        base = oracle_cost(need)
        for m in range(N):
            c, att, _ = simulate(m, need, t_peak_frac)
            C[m, j] = (c + penalty * base * (att - 1)) / base
    return C


_C: np.ndarray | None = None


def hit_matrix() -> np.ndarray:
    H = np.zeros((N, N))
    for j in range(N):
        need = (CELLS[j - 1] + CELLS[j]) / 2 if j else CELLS[0] * 0.7
        for m in range(N):
            H[m, j] = float(grid.mem_within_margin(CELLS[m], need))
    return H


_H: np.ndarray | None = None


def decode_index(probs) -> int:
    """The cell most likely within margin of the peak; ties go up."""
    global _H
    if _H is None:
        _H = hit_matrix()
    p = np.asarray(probs, dtype=float)
    return int(np.argmax(_H @ (p / p.sum()) + 1e-6 * np.arange(N)))


def bayes_index(probs, C: np.ndarray | None = None) -> int:
    global _C
    if C is None:
        if _C is None:
            _C = cost_matrix()
        C = _C
    p = np.asarray(probs, dtype=float)
    return int(np.argmin(C @ (p / p.sum())))


def score(rows: list[dict], preds: dict[str, dict]) -> dict:
    """`preds[run_id]` = {"gmem_idx", "util_idx", optional "gmem_probs"}."""
    acc = {k: [] for k in ("cost", "ocost", "oom", "dev_exact", "dev_over", "exact", "adj",
                           "u_err", "u_hit", "u_exact", "fail")}
    mem_hit, mem_err = [], []
    briers, nlls, p_ooms, real_ooms = [], [], [], []
    missing = 0
    for r in rows:
        p = preds.get(r["run_id"])
        if p is None:
            missing += 1
            continue
        peak = float(r["peak_gpu_gib"])
        sharded, imb = run_fit(r)
        oi, i = int(r["label_gmem_idx"]), int(p["gmem_idx"])
        c, att, failed = simulate(i, peak, float(r["t_peak_frac"]), sharded, imb)
        oc = oracle_cost(peak, sharded, imb)
        acc["cost"].append(c + OOM_PENALTY * oc * (att - 1)), acc["ocost"].append(oc)
        acc["oom"].append(att > 1), acc["fail"].append(failed)
        dev = grid.device_for(CELLS[i])
        odev = grid.oracle_device(peak, sharded, imb)
        odev = grid.device_for(peak) if odev is None else odev
        acc["dev_exact"].append(dev == odev), acc["dev_over"].append(grid.DEVICES[dev].price > grid.DEVICES[odev].price)
        acc["exact"].append(i == oi), acc["adj"].append(abs(i - oi) <= 1)
        if not r.get("fills_device"):  # a device-bound peak is not the task's need
            mem_hit.append(grid.mem_within_margin(CELLS[i], peak)), mem_err.append(abs(CELLS[i] - peak))
        ui = int(p["util_idx"])
        acc["u_err"].append(abs(grid.util_mid(ui) - r["mean_util"]))
        acc["u_hit"].append(abs(grid.util_mid(ui) - r["mean_util"]) <= grid.UTIL_MARGIN)
        acc["u_exact"].append(ui == int(r["label_util_idx"]))
        probs = p.get("gmem_probs")
        if probs is not None:
            pr = np.asarray(probs, dtype=float)
            pr = pr / pr.sum()
            onehot = np.zeros(N)
            onehot[oi] = 1
            briers.append(float(((pr - onehot) ** 2).sum()))
            nlls.append(float(-math.log(max(pr[oi], 1e-9))))
            usable = grid.DEVICES[dev].capacity_gib
            p_ooms.append(float(pr[CELLS > usable + 1e-9].sum()))
            real_ooms.append(att > 1)
    n = len(acc["cost"])
    if not n:
        return {"n": 0, "missing": missing}
    m = {k: float(np.mean(v)) for k, v in acc.items()}
    out = {"n": n, "missing": missing,
           "gpu_regret": float(np.sum(acc["cost"]) / np.sum(acc["ocost"]) - 1.0),
           "oom_rate": m["oom"], "fail_rate": m["fail"], "device_exact": m["dev_exact"], "device_over": m["dev_over"],
           "mem_exact": m["exact"], "mem_within_one_cell": m["adj"],
           "mem_within_margin": float(np.mean(mem_hit)) if mem_hit else float("nan"),
           "mem_mae_gib": float(np.mean(mem_err)) if mem_err else float("nan"), "n_mem_scored": len(mem_hit),
           "util_mae": m["u_err"], "util_within_10": m["u_hit"], "util_exact": m["u_exact"]}
    if briers:
        out.update({"brier": float(np.mean(briers)), "nll": float(np.mean(nlls)),
                    "ece_oom": expected_calibration_error(p_ooms, real_ooms)})
    return out


FRONTIER_PENALTIES = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
FRONTIER_SHIFTS = (-1, 0, 1, 2, 3)


def frontier(rows: list[dict], preds: dict[str, dict], point_estimate: bool = False) -> list[dict]:
    distributional = not point_estimate and all(p.get("gmem_probs") is not None for p in preds.values())
    points = []
    for k in FRONTIER_PENALTIES if distributional else FRONTIER_SHIFTS:
        if distributional:
            C = cost_matrix(k)
            dec = {rid: {**p, "gmem_idx": bayes_index(p["gmem_probs"], C)} for rid, p in preds.items()}
            name = f"penalty={k:g}"
        else:
            dec = {rid: {**p, "gmem_idx": int(min(N - 1, max(0, p["gmem_idx"] + k)))} for rid, p in preds.items()}
            name = f"shift={k:+d}"
        s = score(rows, dec)
        points.append({"knob": name, "oom_rate": s["oom_rate"], "gpu_regret": s["gpu_regret"]})
    return points


def score_slices(rows: list[dict], preds: dict[str, dict]) -> dict[str, dict]:
    slices: dict[str, list[dict]] = {}
    for r in rows:
        hist = "cold" if r["n_history"] == 0 else "warm"
        keys = [f"split={r['split']}", f"driver={r['driver']}", f"history={hist}", f"family={r['family']}"]
        if r["split"] in ("test", "ood"):
            keys += [f"{r['split']}/history={hist}"]
        for k in keys:
            slices.setdefault(k, []).append(r)
    return {k: score(v, preds) for k, v in sorted(slices.items())}
