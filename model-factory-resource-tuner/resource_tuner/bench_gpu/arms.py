"""Non-LLM arms for RightSizeGPU. One knob each, tuned on val against ``gpu_regret``.

- ``static``     one GPU memory request for every run (a training-peak quantile).
- ``vpa``        the largest peak this task has used, times a margin; cold
                 start falls back to static.
- ``size-rule``  peak GPU memory as a power law of input bytes (fit as a
                 quantile); with history, the nearest past run rescaled by
                 the input-size ratio to that power.
- ``gbt``        quantile gradient-boosted trees over the scheduler-visible
                 features (PyTorch API tokens, Linear sizes, batch slice, history).
- ``gbt-dist``   the same trees at 19 quantiles -> expected-cost decoding.

Utilization: training median (static), the task's last observed value
(vpa / size-rule, when there is history), a median-regression GBT (gbt arms).
"""

from __future__ import annotations

import json
import math

import numpy as np

from ..bench.arms import DIST_QUANTILES, _gbt, quantiles_to_cell_probs
from . import grid, scoring
from .features import feature_names, row_features

MARGINS = (1.0, 1.1, 1.25, 1.5, 2.0, 3.0)
QUANTILES = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
_EDGES = np.log2(np.array(grid.GPU_MEM_GRID_GIB, dtype=float))


def _pred(gib: float, util: float) -> dict:
    return {"gmem_idx": grid.mem_index_covering(gib), "util_idx": grid.util_index(util)}


def _tune(candidates, make_preds, val_rows):
    best = None
    for c in candidates:
        cost = scoring.score(val_rows, make_preds(c))["gpu_regret"]
        if best is None or cost < best[1]:
            best = (c, cost)
    return best[0]


def _hist(row: dict) -> list[dict]:
    return json.loads(row.get("history_json") or "[]")


def _bytes(row: dict) -> float:
    return float(sum(json.loads(row["input_bytes_json"]).values()) or 1)


def fit_static(train, val):
    peaks = np.array([r["peak_gpu_gib"] for r in train])
    util = float(np.median([r["mean_util"] for r in train]))
    q = _tune((0.5, 0.75, 0.9, 0.95), lambda q: {r["run_id"]: _pred(float(np.quantile(peaks, q)), util) for r in val}, val)
    return {"q": q, "gib": float(np.quantile(peaks, q)), "util": util}


def predict_static(state, rows):
    return {r["run_id"]: _pred(state["gib"], state["util"]) for r in rows}


def _vpa_one(row, margin, static):
    h = _hist(row)
    if not h:
        return _pred(static["gib"], static["util"])
    return _pred(max(x["peak_gpu_mib"] for x in h) / 1024 * margin, h[-1]["mean_util"])


def fit_vpa(train, val):
    static = fit_static(train, val)
    warm = [r for r in val if r["n_history"]] or val
    return {"margin": _tune(MARGINS, lambda m: {r["run_id"]: _vpa_one(r, m, static) for r in warm}, warm),
            "static": static}


def predict_vpa(state, rows):
    return {r["run_id"]: _vpa_one(r, state["margin"], state["static"]) for r in rows}


def _fit_law(train, q):
    from sklearn.linear_model import QuantileRegressor

    X = np.array([[math.log2(_bytes(r))] for r in train])
    y = np.array([math.log2(max(r["peak_gpu_gib"], 0.1)) for r in train])
    m = QuantileRegressor(quantile=q, alpha=0.0, solver="highs").fit(X, y)
    return float(m.intercept_), float(m.coef_[0])


def _size_one(row, law, margin, util):
    a, b = law
    h = _hist(row)
    if h:
        x = min(h, key=lambda x: abs(math.log((sum(x["input_bytes"].values()) or 1) / _bytes(row))))
        ratio = _bytes(row) / (sum(x["input_bytes"].values()) or 1)
        return _pred(x["peak_gpu_mib"] / 1024 * ratio ** max(b, 0.0) * margin, x["mean_util"])
    return _pred(2 ** (a + b * math.log2(_bytes(row))), util)


def fit_size_rule(train, val):
    util = float(np.median([r["mean_util"] for r in train]))
    cold = [r for r in val if not r["n_history"]] or val
    q = _tune((0.5, 0.75, 0.9), lambda q: {r["run_id"]: _size_one(r, _fit_law(train, q), 1.0, util) for r in cold}, cold)
    law = _fit_law(train, q)
    warm = [r for r in val if r["n_history"]] or val
    margin = _tune(MARGINS, lambda m: {r["run_id"]: _size_one(r, law, m, util) for r in warm}, warm)
    return {"law": law, "q": q, "margin": margin, "util": util}


def predict_size_rule(state, rows):
    return {r["run_id"]: _size_one(r, tuple(state["law"]), state["margin"], state["util"]) for r in rows}


def _matrix(rows):
    names = feature_names()
    return np.array([[row_features(r).get(n, 0.0) for n in names] for r in rows])


def _y(rows):
    return np.array([math.log2(max(r["peak_gpu_gib"], 0.1)) for r in rows])


def _util_model(X, rows):
    return _gbt(0.5).fit(X, np.array([r["mean_util"] for r in rows]))


def fit_gbt(train, val):
    X, Xv = _matrix(train), _matrix(val)
    um = _util_model(X, train)
    uv = um.predict(Xv)
    models = {q: _gbt(q).fit(X, _y(train)) for q in QUANTILES}
    pv = {q: m.predict(Xv) for q, m in models.items()}
    q = _tune(QUANTILES, lambda q: {r["run_id"]: _pred(2 ** pv[q][i], uv[i]) for i, r in enumerate(val)}, val)
    return {"q": q, "mem": models[q], "util": um}


def predict_gbt(state, rows):
    X = _matrix(rows)
    m, u = state["mem"].predict(X), state["util"].predict(X)
    return {r["run_id"]: _pred(2 ** m[i], u[i]) for i, r in enumerate(rows)}


def fit_gbt_dist(train, val):
    X = _matrix(train)
    return {"levels": list(DIST_QUANTILES), "mem": [_gbt(float(q)).fit(X, _y(train)) for q in DIST_QUANTILES],
            "util": _util_model(X, train)}


def predict_gbt_dist(state, rows):
    X = _matrix(rows)
    Q = np.stack([m.predict(X) for m in state["mem"]], axis=1)
    u = state["util"].predict(X)
    levels = np.array(state["levels"])
    out = {}
    for i, r in enumerate(rows):
        probs = quantiles_to_cell_probs(Q[i], levels, _EDGES)
        out[r["run_id"]] = {"gmem_idx": scoring.bayes_index(probs), "util_idx": grid.util_index(u[i]),
                            "gmem_probs": probs.tolist()}
    return out


ARMS = {
    "static": (fit_static, predict_static),
    "vpa": (fit_vpa, predict_vpa),
    "size-rule": (fit_size_rule, predict_size_rule),
    "gbt": (fit_gbt, predict_gbt),
    "gbt-dist": (fit_gbt_dist, predict_gbt_dist),
}
