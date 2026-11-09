"""Non-LLM arms for the combined bench. Each makes the whole request (memory,
CPU, GPU) and tunes its knobs on val for the headline (mean regret).

- ``static``     one request for every run: a training quantile per axis, no GPU.
- ``rules``      what a platform team would write: a GPU if the code mentions
                 CUDA (sized by the task's history, else a training quantile);
                 memory from history x margin (history as the platform
                 reports it: Flyte's peaks plus the runtime, a K8s
                 container's as is), else a quantile; CPU from
                 history, else the largest pool the code declares, else 1.
- ``rules-sized`` the rules a platform team would write knowing this bench
                 (round 22, after review): memory scales the nearest-size
                 earlier run's peak by the input-size ratio; on a first run it
                 is a base plus k x the input bytes; and it reads the Job's
                 staging -- + the staged bytes for ``medium: Memory``, + boto3's
                 48 MiB for ``import boto3`` -- off the manifest and code.
- ``gbt-dist``   quantile trees per axis (19 levels, read as a distribution)
                 plus a needs-a-GPU classifier; decoded per axis by its
                 bench's decoder, with the no-GPU threshold tuned on val.
"""

from __future__ import annotations

import itertools
import json

import numpy as np

from ..bench.arms import DIST_QUANTILES, _gbt, quantiles_to_cell_probs
from ..bench.scoring import bayes_mem_index
from ..bench_cpu.features import declared_pools
from ..bench_cpu.scoring import decode_index as cpu_decode
from ..bench_gpu.scoring import decode_index as gpu_decode
from . import grid, scoring
from .dataset import platform_of
from .features import CUDA_TOKENS, feature_names, row_features
from . import k8s
from .k8s import need_from_peak

QUANTILES = (0.5, 0.75, 0.9, 0.95)
MARGINS = (1.0, 1.1, 1.25, 1.5, 2.0)
GPU_THRESHOLDS = (0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98)

_MEM_EDGES = np.log2(np.array(grid.MEMORY_GRID_MIB, dtype=float))
_CPU_EDGES = np.log2(np.array(grid.CPU_GRID, dtype=float))
_GPU_EDGES = np.log2(np.array(grid.GPU_MEM_GRID_GIB, dtype=float))


def _p(m: int, c: int, g: int, **extra) -> dict:
    return {"mem_idx": int(m), "cpu_idx": int(c), "gpu_idx": int(g), **extra}


def _tune(candidates, make_preds, val):
    return min(candidates, key=lambda k: scoring.score(val, make_preds(k))["regret"])


def _hist(row):
    return json.loads(row.get("history_json") or "[]")


def _cpu_rows(rows):
    return [r for r in rows if r["source"] != "gpu"]


def _gpu_rows(rows):
    return [r for r in rows if r["source"] == "gpu"]


def mentions_cuda(code: str) -> bool:
    return any(t in code for t in CUDA_TOKENS)


# ── static ──────────────────────────────────────────────────────────────


def fit_static(train, val):
    need = np.array([r["need_mib"] for r in _cpu_rows(train)])
    cores = np.array([r["need_cores"] for r in _cpu_rows(train)])

    def preds(qs):
        m, c = grid.mem_cell(float(np.quantile(need, qs[0]))), grid.cpu_cell(float(np.quantile(cores, qs[1])))
        return {r["run_id"]: _p(m, c, grid.NO_GPU) for r in val}

    qm, qc = _tune(list(itertools.product(QUANTILES, QUANTILES)), preds, val)
    return {"q_mem": qm, "q_cpu": qc, "mem": grid.mem_cell(float(np.quantile(need, qm))),
            "cpu": grid.cpu_cell(float(np.quantile(cores, qc)))}


def predict_static(state, rows):
    return {r["run_id"]: _p(state["mem"], state["cpu"], grid.NO_GPU) for r in rows}


# ── rules ───────────────────────────────────────────────────────────────


def _rules_one(row, k, static):
    margin_m, margin_c, margin_g = k
    h = _hist(row)
    if mentions_cuda(row["task_code"]):
        gpus = [x["peak_gpu_mib"] for x in h if "peak_gpu_mib" in x]
        g = grid.gpu_cell(max(gpus) / 1024 * margin_g) if gpus else static["gpu"]
        return _p(static["mem"], static["cpu"], g)
    mems = [x["peak_mem_mib"] for x in h if "peak_mem_mib" in x]
    m = grid.mem_cell(need_from_peak(max(mems), platform_of(row)) * margin_m) if mems else static["mem"]
    cores = [x.get("peak_cores", x.get("mean_cores", 0.0)) for x in h]
    pools = declared_pools(row["task_code"])
    if cores:
        c = grid.cpu_cell(max(cores) * margin_c)
    else:
        c = grid.cpu_cell(float(max(pools))) if pools else grid.cpu_cell(1.0)
    return _p(m, c, grid.NO_GPU)


def fit_rules(train, val):
    st = fit_static(train, val)
    peaks = np.array([r["peak_gpu_gib"] for r in _gpu_rows(train)] or [8.0])
    cold_gpu = [r for r in _gpu_rows(val) if not r["n_history"]] or val
    qg = _tune(QUANTILES, lambda q: {r["run_id"]: _p(st["mem"], st["cpu"], grid.gpu_cell(float(np.quantile(peaks, q))))
                                     for r in cold_gpu}, cold_gpu)
    static = {"mem": st["mem"], "cpu": st["cpu"], "gpu": grid.gpu_cell(float(np.quantile(peaks, qg)))}
    warm = [r for r in val if r["n_history"]] or val
    k = _tune(list(itertools.product(MARGINS, (1.0, 1.25), MARGINS)),
              lambda k: {r["run_id"]: _rules_one(r, k, static) for r in warm}, warm)
    return {"static": static, "margins": list(k), "q_gpu": qg}


def predict_rules(state, rows):
    return {r["run_id"]: _rules_one(r, tuple(state["margins"]), state["static"]) for r in rows}


# ── rules-sized ─────────────────────────────────────────────────────────

#: A first run's workload peak: base + k x input MiB (tuned on val).
SIZED_BASE_MIB = (64.0, 128.0, 256.0, 512.0)
SIZED_K = (0.5, 1.0, 2.0, 4.0, 8.0)


def _staging_mib(row) -> float:
    """What the Job's staging adds on top of the workload, read off its manifest and code."""
    if (row.get("platform") or "flyte") != "k8s":
        return 0.0
    sizes = json.loads(row.get("input_bytes_json") or "{}")
    tmpfs = "medium: Memory" in (row.get("manifest") or "")
    boto3 = "import boto3" in row["task_code"]
    return (k8s.BOTO3_MIB if boto3 else 0.0) + (k8s.staged_bytes(sizes) / 2**20 if tmpfs else 0.0)


def _sized_one(row, k, static):
    margin, base, slope = k
    h = _hist(row)
    if mentions_cuda(row["task_code"]):
        return _rules_one(row, (1.0, 1.0, static.get("margin_g", 1.0)), static)
    sizes = json.loads(row.get("input_bytes_json") or "{}")
    total = sum(sizes.values()) or 1
    mems = [(x["peak_mem_mib"], sum((x.get("input_bytes") or {}).values()) or 1) for x in h if "peak_mem_mib" in x]
    plat = platform_of(row)
    if mems:  # the earlier run nearest in size, scaled to this run's size; history already holds the staging
        peak, hb = min(mems, key=lambda x: abs(np.log(x[1] / total)))
        m = grid.mem_cell(need_from_peak(peak * total / hb, plat) * margin)
    else:
        workload = base + slope * total / 2**20
        m = grid.mem_cell(need_from_peak(workload + _staging_mib(row), plat))
    c = _rules_one(row, (1.0, static["margin_c"], 1.0), static)["cpu_idx"]
    return _p(m, c, grid.NO_GPU)


def fit_rules_sized(train, val):
    st = fit_rules(train, val)
    static = {**st["static"], "margin_c": st["margins"][1], "margin_g": st["margins"][2]}
    cold = [r for r in _cpu_rows(val) if not r["n_history"]] or val
    bs = _tune(list(itertools.product((1.0,), SIZED_BASE_MIB, SIZED_K)),
               lambda k: {r["run_id"]: _sized_one(r, k, static) for r in cold}, cold)
    warm = [r for r in _cpu_rows(val) if r["n_history"]] or val
    mg = _tune([(m, bs[1], bs[2]) for m in MARGINS],
               lambda k: {r["run_id"]: _sized_one(r, k, static) for r in warm}, warm)
    return {"static": static, "k": list(mg)}


def predict_rules_sized(state, rows):
    return {r["run_id"]: _sized_one(r, tuple(state["k"]), state["static"]) for r in rows}


# ── gbt-dist ────────────────────────────────────────────────────────────


def _matrix(rows):
    names = feature_names()
    return np.array([[row_features(r).get(n, 0.0) for n in names] for r in rows], dtype=np.float32)


def _dist(X, y):
    return [_gbt(float(q)).fit(X, y) for q in DIST_QUANTILES]


def fit_gbt_dist(train, val):
    from sklearn.ensemble import HistGradientBoostingClassifier

    X = _matrix(train)
    is_gpu = np.array([r["source"] == "gpu" for r in train])
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, random_state=0).fit(X, is_gpu)
    cpu_rows, gpu_rows = ~is_gpu, is_gpu
    state = {
        "levels": list(DIST_QUANTILES), "clf": clf,
        "mem": _dist(X[cpu_rows], np.log2([r["need_mib"] for r in train if r["source"] != "gpu"])),
        "cpu": _dist(X[cpu_rows], np.log2([max(r["need_cores"], 0.05) for r in train if r["source"] != "gpu"])),
        "gpu": _dist(X[gpu_rows], np.log2([max(r["peak_gpu_gib"], 0.1) for r in train if r["source"] == "gpu"])),
        "threshold": 0.5,
    }
    probs = distributions(state, val)
    state["threshold"] = _tune(GPU_THRESHOLDS, lambda t: decode_all(probs, t), val)
    return state


def distributions(state, rows) -> dict[str, tuple]:
    """run_id -> (P(mem cell), P(cpu cell), P(gpu choice, "none" first))."""
    X = _matrix(rows)
    levels = np.array(state["levels"])
    p_gpu = state["clf"].predict_proba(X)[:, 1]
    Q = {k: np.stack([m.predict(X) for m in state[k]], axis=1) for k in ("mem", "cpu", "gpu")}
    out = {}
    for i, r in enumerate(rows):
        pm = quantiles_to_cell_probs(Q["mem"][i], levels, _MEM_EDGES)
        pc = quantiles_to_cell_probs(Q["cpu"][i], levels, _CPU_EDGES)
        pg_cells = quantiles_to_cell_probs(Q["gpu"][i], levels, _GPU_EDGES)
        pg = np.concatenate([[1.0 - p_gpu[i]], p_gpu[i] * pg_cells])
        out[r["run_id"]] = (pm, pc, pg)
    return out


def decode_all(probs: dict[str, tuple], threshold: float) -> dict[str, dict]:
    out = {}
    for rid, (pm, pc, pg) in probs.items():
        m, c, g = scoring.decode((pm, pc, pg), threshold, cpu_decode, bayes_mem_index, gpu_decode)
        out[rid] = _p(m, c, g, probs=[pm.tolist(), pc.tolist(), pg.tolist()])
    return out


def predict_gbt_dist(state, rows):
    return decode_all(distributions(state, rows), state["threshold"])


ARMS = {
    "static": (fit_static, predict_static),
    "rules": (fit_rules, predict_rules),
    "rules-sized": (fit_rules_sized, predict_rules_sized),
    "gbt-dist": (fit_gbt_dist, predict_gbt_dist),
}

__all__ = ["ARMS", "mentions_cuda"]
