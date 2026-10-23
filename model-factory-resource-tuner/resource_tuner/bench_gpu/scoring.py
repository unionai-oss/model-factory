"""Score GPU decisions: the right device, the right memory, the right utilization.

**Cost** (``gpu_regret``, the headline). A predicted peak memory is a
request for the cheapest device that holds it (``grid.device_for``). If the
run's real peak fits that device it pays one nominal task-hour at the
device's price; if not, it dies with a CUDA OOM after the part of the run
before its peak (``t_peak_frac``, measured -- GPU peaks come early: the
first forward/backward), and is retried on the next device up. Each OOM
also costs OOM_PENALTY oracle runs of delay. ``gpu_regret`` = dollars /
the oracle's dollars - 1; the oracle runs on the cheapest device that fits.

**Accuracy.** Memory: exact cell, within one cell, within 20% of the peak
(``grid.mem_within_margin``), first-attempt OOM rate, and whether the
device matches the oracle's. Utilization: mean absolute error in points and
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


def simulate(cell_idx: int, peak_gib: float, t_peak_frac: float) -> tuple[float, int, bool]:
    """(dollars, attempts, failed) to get a run to SUCCEEDED from this request."""
    dev = grid.device_for(CELLS[cell_idx])
    cost, attempts = 0.0, 0
    while dev is not None and attempts < MAX_ATTEMPTS:
        attempts += 1
        _name, usable, price, _shape = grid.DEVICES[dev]
        if peak_gib <= usable:
            return cost + price * (1 + ATTEMPT_OVERHEAD), attempts, False
        cost += price * (max(t_peak_frac, 0.02) + ATTEMPT_OVERHEAD)
        dev = grid.next_device_up(dev)
    return cost, attempts, True


def oracle_cost(peak_gib: float) -> float:
    return grid.DEVICES[grid.device_for(peak_gib)][2] * (1 + ATTEMPT_OVERHEAD)


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
    acc = {k: [] for k in ("cost", "ocost", "oom", "dev_exact", "dev_over", "exact", "adj", "hit", "err",
                           "u_err", "u_hit", "u_exact", "fail")}
    briers, nlls, p_ooms, real_ooms = [], [], [], []
    missing = 0
    for r in rows:
        p = preds.get(r["run_id"])
        if p is None:
            missing += 1
            continue
        peak = float(r["peak_gpu_gib"])
        oi, i = int(r["label_gmem_idx"]), int(p["gmem_idx"])
        c, att, failed = simulate(i, peak, float(r["t_peak_frac"]))
        oc = oracle_cost(peak)
        acc["cost"].append(c + OOM_PENALTY * oc * (att - 1)), acc["ocost"].append(oc)
        acc["oom"].append(att > 1), acc["fail"].append(failed)
        dev, odev = grid.device_for(CELLS[i]), grid.device_for(peak)
        acc["dev_exact"].append(dev == odev), acc["dev_over"].append(grid.DEVICES[dev][2] > grid.DEVICES[odev][2])
        acc["exact"].append(i == oi), acc["adj"].append(abs(i - oi) <= 1)
        acc["hit"].append(grid.mem_within_margin(CELLS[i], peak)), acc["err"].append(abs(CELLS[i] - peak))
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
            usable = grid.DEVICES[dev][1]
            p_ooms.append(float(pr[CELLS > usable + 1e-9].sum()))
            real_ooms.append(att > 1)
    n = len(acc["cost"])
    if not n:
        return {"n": 0, "missing": missing}
    m = {k: float(np.mean(v)) for k, v in acc.items()}
    out = {"n": n, "missing": missing,
           "gpu_regret": float(np.sum(acc["cost"]) / np.sum(acc["ocost"]) - 1.0),
           "oom_rate": m["oom"], "fail_rate": m["fail"], "device_exact": m["dev_exact"], "device_over": m["dev_over"],
           "mem_exact": m["exact"], "mem_within_one_cell": m["adj"], "mem_within_margin": m["hit"],
           "mem_mae_gib": m["err"], "util_mae": m["u_err"], "util_within_10": m["u_hit"], "util_exact": m["u_exact"]}
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
