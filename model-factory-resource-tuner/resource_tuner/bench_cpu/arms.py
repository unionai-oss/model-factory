"""Non-LLM arms for RightSizeCPU: the bars an LLM has to clear.

Each predicts the peak cell (and the mean cell) and has one knob, tuned on
validation for the headline -- the share of peaks predicted within margin
(ties broken by ``cpu_regret``):

- ``static``     one request for every run: a quantile of training peaks.
- ``vpa``        the max peak this task has been seen to use, times a
                 margin (VPA-style); cold start falls back to static.
- ``code-rule``  the simple algorithm that reads code: the largest pool
                 size the source declares (``n_jobs=``, ``max_workers=``,
                 ``SET threads``, ``set_num_threads``, ...), else one core,
                 times a tuned factor. It cannot know the GIL, a pool
                 capped by its work items, or a serial phase.
- ``gbt``        quantile gradient-boosted trees over every
                 scheduler-visible feature (including ``max_pool``).
- ``gbt-dist``   the same trees at 19 quantiles -> a distribution over
                 cells -> decoded to the cell most likely within margin.
"""

from __future__ import annotations

import json
import math

import numpy as np

from ..bench.arms import DIST_QUANTILES, _gbt, quantiles_to_cell_probs
from . import grid, scoring
from .features import declared_pools, feature_names, row_features

MARGINS = (0.6, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0)
QUANTILES = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
_EDGES = np.log2(np.array(grid.CPU_GRID, dtype=float))
_FLOOR = 0.05  # log2 of a peak near zero


def _cell(cores: float) -> int:
    return grid.index_covering(max(cores, 0.0))


def _pred(peak: float, mean: float) -> dict:
    return {"peak_idx": _cell(peak), "mean_idx": _cell(mean)}


def _tune(candidates, make_preds, val_rows):
    best = None
    for c in candidates:
        s = scoring.score(val_rows, make_preds(c))
        cost = (1.0 - s["within_margin"], s["cpu_regret"])
        if best is None or cost < best[1]:
            best = (c, cost)
    return best[0]


def _hist(row: dict) -> list[dict]:
    return json.loads(row.get("history_json") or "[]")


# ── static ──────────────────────────────────────────────────────────────


def fit_static(train: list[dict], val: list[dict]) -> dict:
    peaks = np.array([r["peak_cores"] for r in train])
    means = np.array([r["mean_cores"] for r in train])
    mean = float(np.median(means))
    q = _tune((0.5, 0.75, 0.9, 0.95),
              lambda q: {r["run_id"]: _pred(float(np.quantile(peaks, q)), mean) for r in val}, val)
    return {"q": q, "peak": float(np.quantile(peaks, q)), "mean": mean}


def predict_static(state: dict, rows: list[dict]) -> dict:
    return {r["run_id"]: _pred(state["peak"], state["mean"]) for r in rows}


# ── vpa ─────────────────────────────────────────────────────────────────


def _vpa_one(row: dict, margin: float, static: dict) -> dict:
    h = _hist(row)
    if not h:
        return _pred(static["peak"], static["mean"])
    return _pred(max(x["peak_cores"] for x in h) * margin, max(x["mean_cores"] for x in h))


def fit_vpa(train: list[dict], val: list[dict]) -> dict:
    static = fit_static(train, val)
    warm = [r for r in val if r["n_history"]] or val
    margin = _tune(MARGINS, lambda m: {r["run_id"]: _vpa_one(r, m, static) for r in warm}, warm)
    return {"margin": margin, "static": static}


def predict_vpa(state: dict, rows: list[dict]) -> dict:
    return {r["run_id"]: _vpa_one(r, state["margin"], state["static"]) for r in rows}


# ── code-rule ───────────────────────────────────────────────────────────


def code_rule_cores(code: str) -> float:
    pools = declared_pools(code)
    return float(min(max(pools), 8)) if pools else 1.0


def _rule_one(row: dict, factor: float) -> dict:
    c = code_rule_cores(row["task_code"]) * factor
    return _pred(c, c * 0.8)


def fit_code_rule(train: list[dict], val: list[dict]) -> dict:
    return {"factor": _tune(MARGINS, lambda f: {r["run_id"]: _rule_one(r, f) for r in val}, val)}


def predict_code_rule(state: dict, rows: list[dict]) -> dict:
    return {r["run_id"]: _rule_one(r, state["factor"]) for r in rows}


# ── gbt / gbt-dist ──────────────────────────────────────────────────────


def _matrix(rows: list[dict]) -> np.ndarray:
    names = feature_names()
    return np.array([[row_features(r).get(n, 0.0) for n in names] for r in rows])


def _log_peak(rows: list[dict]) -> np.ndarray:
    return np.array([math.log2(max(r["peak_cores"], _FLOOR)) for r in rows])


def _mean_model(X: np.ndarray, rows: list[dict]):
    return _gbt(0.5).fit(X, np.array([r["mean_cores"] for r in rows]))


def fit_gbt(train: list[dict], val: list[dict]) -> dict:
    X, Xv = _matrix(train), _matrix(val)
    y = _log_peak(train)
    mean_m = _mean_model(X, train)
    mv = mean_m.predict(Xv)
    models = {q: _gbt(q).fit(X, y) for q in QUANTILES}
    pv = {q: m.predict(Xv) for q, m in models.items()}
    q = _tune(QUANTILES, lambda q: {r["run_id"]: _pred(2 ** pv[q][i], mv[i]) for i, r in enumerate(val)}, val)
    return {"q": q, "peak": models[q], "mean": mean_m}


def predict_gbt(state: dict, rows: list[dict]) -> dict:
    X = _matrix(rows)
    p, m = state["peak"].predict(X), state["mean"].predict(X)
    return {r["run_id"]: _pred(2 ** p[i], m[i]) for i, r in enumerate(rows)}


def fit_gbt_dist(train: list[dict], val: list[dict]) -> dict:
    X = _matrix(train)
    y = _log_peak(train)
    return {"levels": list(DIST_QUANTILES), "peak": [_gbt(float(q)).fit(X, y) for q in DIST_QUANTILES],
            "mean": _mean_model(X, train)}


def predict_gbt_dist(state: dict, rows: list[dict]) -> dict:
    X = _matrix(rows)
    Q = np.stack([m.predict(X) for m in state["peak"]], axis=1)
    mean = state["mean"].predict(X)
    levels = np.array(state["levels"])
    out = {}
    for i, r in enumerate(rows):
        probs = quantiles_to_cell_probs(Q[i], levels, _EDGES)
        out[r["run_id"]] = {"peak_idx": scoring.decode_index(probs), "mean_idx": _cell(mean[i]),
                            "peak_probs": probs.tolist()}
    return out


ARMS = {
    "static": (fit_static, predict_static),
    "vpa": (fit_vpa, predict_vpa),
    "code-rule": (fit_code_rule, predict_code_rule),
    "gbt": (fit_gbt, predict_gbt),
    "gbt-dist": (fit_gbt_dist, predict_gbt_dist),
}
