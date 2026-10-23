"""Score CPU decisions: is the peak right, and what does a wrong one cost?

Two questions, both answered per run and aggregated:

1. **Accuracy** (the headline: the task is *predict peak CPU*): a predicted
   peak is *within margin* when it is within 25% of the measured peak 15 s
   window (or 0.25 core for small peaks; ``grid.within_margin``). Also:
   the share of runs predicted *under* the peak (they will be throttled),
   and mean absolute error in cores.
2. **Cost** (``cpu_regret``): the predicted peak is used as the CPU limit,
   and the run is replayed against its own measured per-second series. A
   second that wanted u cores under a limit L < u takes u/L seconds
   (CFS throttling stretches the work, it never kills it). The run pays
   L x (its stretched duration) in core-seconds, plus DELAY_PENALTY oracle
   core-seconds per second of delay -- a late result costs something even
   though the cores were cheap. ``cpu_regret`` = total / the oracle's
   total - 1, where the oracle asks for the cell covering the peak window.

Accuracy, not cost, is the headline and the decoding target. Under a
throttling-only cost, asking for *less* than the peak is often cheaper in
core-seconds (a throttled second costs less than an idle core), so a
cost-optimal policy predicts low on purpose -- the opposite of the task.
Distributions are decoded by ``decode_index``: the cell most likely to land
within the margin of the peak, ties broken upward (an under-prediction
throttles; an over-prediction only idles). The cost-optimal decoder
(``bayes_index``) is kept for the risk frontier.

Pure numpy, no flyte.
"""

from __future__ import annotations

import json
import math

import numpy as np

from ..bench.scoring import expected_calibration_error
from . import grid

N = len(grid.CPU_GRID)
CELLS = np.array(grid.CPU_GRID, dtype=float)
#: Oracle core-seconds charged per second a throttled run finishes late.
DELAY_PENALTY = 1.0
#: For decoding a distribution without the series: a run is assumed to sit
#: at its peak half the time and at half its peak the other half.
PEAK_SHARE = 0.5


def series_of(row: dict) -> np.ndarray:
    s = row.get("_series")
    if s is None:
        s = np.array(json.loads(row.get("cpu_series_json") or "[]"), dtype=float)
        row["_series"] = s
    return s


def throttled_seconds(series: np.ndarray, limit: float) -> float:
    """How long the run takes under a CPU limit: each second's work at u
    cores needs max(1, u / limit) seconds."""
    if not len(series):
        return 0.0
    return float(np.maximum(1.0, series / max(limit, 1e-6)).sum())


def run_cost(series: np.ndarray, limit: float, oracle_limit: float) -> tuple[float, float]:
    """(core-seconds charged incl. delay penalty, delay seconds)."""
    t = throttled_seconds(series, limit)
    delay = t - len(series)
    return limit * t + DELAY_PENALTY * oracle_limit * delay, delay


# ── decision theory ─────────────────────────────────────────────────────


def cost_matrix(penalty: float = DELAY_PENALTY) -> np.ndarray:
    """C[m, j]: normalized cost of asking for cell m when the peak is in cell j."""
    C = np.zeros((N, N))
    for j in range(N):
        need = (CELLS[j - 1] + CELLS[j]) / 2 if j else CELLS[0] * 0.8  # cell j's interior
        oracle = CELLS[j]
        for m in range(N):
            L = CELLS[m]
            t = PEAK_SHARE * max(1.0, need / L) + (1 - PEAK_SHARE) * max(1.0, need / 2 / L)
            C[m, j] = (L * t + penalty * oracle * (t - 1)) / oracle
    return C


_C: np.ndarray | None = None


def hit_matrix() -> np.ndarray:
    """H[m, j] = 1 if asking for cell m is within margin of a peak inside cell j."""
    H = np.zeros((N, N))
    for j in range(N):
        need = (CELLS[j - 1] + CELLS[j]) / 2 if j else CELLS[0] * 0.8
        for m in range(N):
            H[m, j] = float(grid.within_margin(CELLS[m], need))
    return H


_H: np.ndarray | None = None


def decode_index(probs) -> int:
    """The cell with the highest probability of being within margin of the
    peak; ties go to the larger cell."""
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


# ── scoring ─────────────────────────────────────────────────────────────


def score(rows: list[dict], preds: dict[str, dict]) -> dict:
    """`preds[run_id]` = {"peak_idx", "mean_idx", optional "peak_probs"}."""
    costs, ocosts, delays, hit, under, exact, adj, err, logerr = [], [], [], [], [], [], [], [], []
    mhit, merr, briers, nlls, p_under, real_under = [], [], [], [], [], []
    missing = 0
    for r in rows:
        p = preds.get(r["run_id"])
        if p is None:
            missing += 1
            continue
        s = series_of(r)
        peak, mean = float(r["peak_cores"]), float(r["mean_cores"])
        oi = int(r["label_peak_idx"])
        i = int(p["peak_idx"])
        pred = CELLS[i]
        c, d = run_cost(s, pred, CELLS[oi])
        oc, _ = run_cost(s, CELLS[oi], CELLS[oi])
        costs.append(c), ocosts.append(oc), delays.append(d / max(len(s), 1))
        hit.append(grid.within_margin(pred, peak))
        under.append(pred < peak * 0.95)
        exact.append(i == oi), adj.append(abs(i - oi) <= 1)
        err.append(abs(pred - peak)), logerr.append(abs(math.log2(pred / max(peak, 0.05))))
        mpred = CELLS[int(p["mean_idx"])]
        mhit.append(grid.within_margin(mpred, mean)), merr.append(abs(mpred - mean))
        probs = p.get("peak_probs")
        if probs is not None:
            pr = np.asarray(probs, dtype=float)
            pr = pr / pr.sum()
            onehot = np.zeros(N)
            onehot[oi] = 1
            briers.append(float(((pr - onehot) ** 2).sum()))
            nlls.append(float(-math.log(max(pr[oi], 1e-9))))
            p_under.append(float(pr[i + 1 :].sum()))
            real_under.append(oi > i)
    n = len(costs)
    if not n:
        return {"n": 0, "missing": missing}
    out = {
        "n": n, "missing": missing,
        "within_margin": float(np.mean(hit)),
        "under_rate": float(np.mean(under)),
        "exact": float(np.mean(exact)), "within_one_cell": float(np.mean(adj)),
        "mae_cores": float(np.mean(err)), "mean_abs_log2_err": float(np.mean(logerr)),
        "cpu_regret": float(np.sum(costs) / np.sum(ocosts) - 1.0),
        "mean_slowdown": float(np.mean(delays)),
        "mean_within_margin": float(np.mean(mhit)), "mean_mae_cores": float(np.mean(merr)),
    }
    if briers:
        out.update({"brier": float(np.mean(briers)), "nll": float(np.mean(nlls)),
                    "ece_under": expected_calibration_error(p_under, real_under)})
    return out


FRONTIER_PENALTIES = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
FRONTIER_SHIFTS = (-2, -1, 0, 1, 2, 3)


def frontier(rows: list[dict], preds: dict[str, dict], point_estimate: bool = False) -> list[dict]:
    """Under-prediction rate vs cost as an arm's risk knob moves."""
    distributional = not point_estimate and all(p.get("peak_probs") is not None for p in preds.values())
    points = []
    knobs = FRONTIER_PENALTIES if distributional else FRONTIER_SHIFTS
    for k in knobs:
        if distributional:
            C = cost_matrix(k)
            dec = {rid: {**p, "peak_idx": bayes_index(p["peak_probs"], C)} for rid, p in preds.items()}
            name = f"penalty={k:g}"
        else:
            dec = {rid: {**p, "peak_idx": int(min(N - 1, max(0, p["peak_idx"] + k)))} for rid, p in preds.items()}
            name = f"shift={k:+d}"
        s = score(rows, dec)
        points.append({"knob": name, "under_rate": s["under_rate"], "cpu_regret": s["cpu_regret"],
                       "within_margin": s["within_margin"]})
    return points


def score_slices(rows: list[dict], preds: dict[str, dict]) -> dict[str, dict]:
    slices: dict[str, list[dict]] = {}
    for r in rows:
        hist = "cold" if r["n_history"] == 0 else "warm"
        keys = [f"split={r['split']}", f"driver={r['driver']}", f"history={hist}", f"family={r['family']}"]
        if r["split"] in ("test", "ood"):
            keys += [f"{r['split']}/history={hist}", f"{r['split']}/driver={r['driver']}"]
        for k in keys:
            slices.setdefault(k, []).append(r)
    return {k: score(v, preds) for k, v in sorted(slices.items())}
