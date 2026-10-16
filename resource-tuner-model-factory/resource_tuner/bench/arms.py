"""Non-LLM arms: the bars an LLM has to clear.

Every arm has exactly one risk knob and it is tuned the same way: on the
validation split, against `scoring.score`'s cost. Nobody gets a hand-picked
margin. That is what makes the leaderboard a comparison of *information
and model*, not of who guessed the best fudge factor.

- ``static``     one request for every run: the cheapest cell that would
                 have covered ``q`` of training runs. The status quo.
- ``vpa``        Kubernetes-VPA-style: the max peak this task has been
                 seen to use, times a margin. Cold start falls back to static.
- ``size-rule``  the simple algorithm a platform could ship tomorrow:
                 resident memory is a power law in input bytes per format
                 (fit as a high quantile); with history, rescale the
                 nearest past run by the input-size ratio.
- ``gbt``        quantile gradient-boosted trees over every
                 scheduler-visible feature (input metadata, code tokens,
                 history). The classical-ML ceiling.
- ``gbt-dist``   the same trees at 19 quantiles, read as a distribution
                 and decoded by expected cost -- the control that separates
                 "a better model" from "a better decision rule".
"""

from __future__ import annotations

import json
import math

import numpy as np

from . import grid, scoring
from .features import feature_names, row_features

MARGINS = (1.0, 1.1, 1.2, 1.35, 1.5, 1.75, 2.0, 2.5, 3.0)
QUANTILES = (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98)
DIST_QUANTILES = tuple(np.round(np.linspace(0.05, 0.95, 19), 2))


def _cell(need_mib: float) -> int:
    return grid.mem_index_covering(need_mib)


def _cpu_from_history(row: dict) -> int:
    hist = json.loads(row.get("history_json") or "[]")
    cores = max((h["avg_cores"] for h in hist), default=1.0)
    return grid.cpu_index_covering(cores)


def _tune(candidates, make_preds, val_rows):
    """The candidate whose predictions cost least on validation."""
    best = None
    for c in candidates:
        cost = scoring.score(val_rows, make_preds(c))["cost_per_1k_task_hr"]
        if best is None or cost < best[1]:
            best = (c, cost)
    return best[0]


# ── static ──────────────────────────────────────────────────────────────


def fit_static(train: list[dict], val: list[dict]) -> dict:
    req = np.array([r["required_mib"] for r in train])
    cores = np.array([r["avg_cores"] for r in train])

    def preds(q):
        cell = _cell(float(np.quantile(req, q)))
        cpu = grid.cpu_index_covering(float(np.quantile(cores, 0.9)))
        return {r["run_id"]: {"mem_idx": cell, "cpu_idx": cpu} for r in val}

    # The status quo is a *safe* static request, not a cost-optimal one:
    # tune only among coverage levels a platform team would sign off on.
    q = _tune((0.9, 0.95, 0.99), preds, val)
    return {"mem_idx": _cell(float(np.quantile(req, q))),
            "cpu_idx": grid.cpu_index_covering(float(np.quantile(cores, 0.9))), "q": q}


def predict_static(state: dict, rows: list[dict]) -> dict:
    return {r["run_id"]: {"mem_idx": state["mem_idx"], "cpu_idx": state["cpu_idx"]} for r in rows}


# ── vpa ─────────────────────────────────────────────────────────────────


def _vpa_one(row: dict, margin: float, static: dict) -> dict:
    hist = json.loads(row.get("history_json") or "[]")
    if not hist:
        return {"mem_idx": static["mem_idx"], "cpu_idx": static["cpu_idx"]}
    peak = max(h["peak_rss_mib"] for h in hist)
    return {"mem_idx": _cell(grid.required_memory_mib(peak) * margin), "cpu_idx": _cpu_from_history(row)}


def fit_vpa(train: list[dict], val: list[dict]) -> dict:
    static = fit_static(train, val)
    margin = _tune(MARGINS, lambda m: {r["run_id"]: _vpa_one(r, m, static) for r in val}, val)
    return {"margin": margin, "static": static}


def predict_vpa(state: dict, rows: list[dict]) -> dict:
    return {r["run_id"]: _vpa_one(r, state["margin"], state["static"]) for r in rows}


# ── size-rule ───────────────────────────────────────────────────────────


def _fmt_of(row: dict) -> str:
    inputs = json.loads(row["inputs_json"])
    sizes = json.loads(row["input_bytes_json"])
    return max(inputs, key=lambda s: sizes.get(s["name"], 0))["fmt"]


def _bytes(row: dict) -> float:
    return float(sum(json.loads(row["input_bytes_json"]).values()) or 1)


def _fit_power_law(train: list[dict], q: float) -> dict:
    """Per format: log2(required) ~ a + b*log2(bytes), at quantile q."""
    from sklearn.linear_model import QuantileRegressor

    out = {}
    for fmt in ("parquet", "csv", "jsonl", "npy"):
        rs = [r for r in train if _fmt_of(r) == fmt]
        if len(rs) < 10:
            continue
        X = np.array([[math.log2(_bytes(r))] for r in rs])
        y = np.array([math.log2(r["required_mib"]) for r in rs])
        m = QuantileRegressor(quantile=q, alpha=0.0, solver="highs").fit(X, y)
        out[fmt] = (float(m.intercept_), float(m.coef_[0]))
    return out


def _size_one(row: dict, laws: dict, margin: float) -> dict:
    hist = json.loads(row.get("history_json") or "[]")
    total = _bytes(row)
    if hist:
        h = min(hist, key=lambda h: abs(math.log((sum(h["input_bytes"].values()) or 1) / total)))
        need = grid.required_memory_mib(h["peak_rss_mib"]) * total / (sum(h["input_bytes"].values()) or 1)
        return {"mem_idx": _cell(need * margin), "cpu_idx": _cpu_from_history(row)}
    a, b = laws.get(_fmt_of(row), (math.log2(4096), 0.0))
    return {"mem_idx": _cell(2 ** (a + b * math.log2(total))), "cpu_idx": grid.cpu_index_covering(1.0)}


def fit_size_rule(train: list[dict], val: list[dict]) -> dict:
    cold_val = [r for r in val if not r["n_history"]]
    warm_val = [r for r in val if r["n_history"]]
    q = _tune((0.5, 0.75, 0.9, 0.95),
              lambda q: {r["run_id"]: _size_one(r, _fit_power_law(train, q), 1.0) for r in cold_val},
              cold_val)
    laws = _fit_power_law(train, q)
    margin = _tune(MARGINS, lambda m: {r["run_id"]: _size_one(r, laws, m) for r in warm_val}, warm_val)
    return {"laws": laws, "q": q, "margin": margin}


def predict_size_rule(state: dict, rows: list[dict]) -> dict:
    laws = {k: tuple(v) for k, v in state["laws"].items()}
    return {r["run_id"]: _size_one(r, laws, state["margin"]) for r in rows}


# ── gbt ─────────────────────────────────────────────────────────────────


def _matrix(rows: list[dict]) -> np.ndarray:
    names = feature_names()
    return np.array([[row_features(r).get(n, 0.0) for n in names] for r in rows])


def _gbt(q: float):
    from sklearn.ensemble import HistGradientBoostingRegressor

    return HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=300,
                                         learning_rate=0.06, max_leaf_nodes=31, random_state=0)


def fit_gbt(train: list[dict], val: list[dict]) -> dict:
    X, Xv = _matrix(train), _matrix(val)
    y = np.array([math.log2(r["required_mib"]) for r in train])
    yc = np.array([r["avg_cores"] for r in train])
    cpu_model = _gbt(0.7).fit(X, yc)
    cpu_v = cpu_model.predict(Xv)
    models = {q: _gbt(q).fit(X, y) for q in QUANTILES}
    pv = {q: m.predict(Xv) for q, m in models.items()}

    def preds(q):
        return {r["run_id"]: {"mem_idx": _cell(2 ** pv[q][i]), "cpu_idx": grid.cpu_index_covering(cpu_v[i])}
                for i, r in enumerate(val)}

    q = _tune(QUANTILES, preds, val)
    return {"q": q, "mem": models[q], "cpu": cpu_model}


def predict_gbt(state: dict, rows: list[dict]) -> dict:
    X = _matrix(rows)
    m, c = state["mem"].predict(X), state["cpu"].predict(X)
    return {r["run_id"]: {"mem_idx": _cell(2 ** m[i]), "cpu_idx": grid.cpu_index_covering(c[i])}
            for i, r in enumerate(rows)}


def quantiles_to_cell_probs(qs: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Predicted quantiles of log2(required MiB) -> P(needed cell = j).

    Piecewise-linear CDF through (quantile value, level), with the tails
    closed 1.5 doublings beyond the outermost predicted quantiles.
    """
    qs = np.maximum.accumulate(np.asarray(qs, dtype=float))
    edges = np.log2(np.array(grid.MEMORY_GRID_MIB, dtype=float))
    x = np.concatenate([[qs[0] - 1.5], qs + np.arange(len(qs)) * 1e-6, [qs[-1] + 1.5]])
    y = np.concatenate([[0.0], levels, [1.0]])
    cdf_at_edges = np.interp(edges, x, y)
    cdf_at_edges[-1] = 1.0
    p = np.diff(np.concatenate([[0.0], cdf_at_edges]))
    p = np.clip(p, 1e-6, None)
    return p / p.sum()


def fit_gbt_dist(train: list[dict], val: list[dict]) -> dict:
    X = _matrix(train)
    y = np.array([math.log2(r["required_mib"]) for r in train])
    yc = np.array([r["avg_cores"] for r in train])
    return {"levels": list(DIST_QUANTILES),
            "mem": [_gbt(float(q)).fit(X, y) for q in DIST_QUANTILES],
            "cpu": _gbt(0.7).fit(X, yc)}


def predict_gbt_dist(state: dict, rows: list[dict]) -> dict:
    X = _matrix(rows)
    Q = np.stack([m.predict(X) for m in state["mem"]], axis=1)
    c = state["cpu"].predict(X)
    levels = np.array(state["levels"])
    out = {}
    for i, r in enumerate(rows):
        probs = quantiles_to_cell_probs(Q[i], levels)
        out[r["run_id"]] = {"mem_idx": scoring.bayes_mem_index(probs), "cpu_idx": grid.cpu_index_covering(c[i]),
                            "mem_probs": probs.tolist()}
    return out


ARMS = {
    "static": (fit_static, predict_static),
    "vpa": (fit_vpa, predict_vpa),
    "size-rule": (fit_size_rule, predict_size_rule),
    "gbt": (fit_gbt, predict_gbt),
    "gbt-dist": (fit_gbt_dist, predict_gbt_dist),
}
