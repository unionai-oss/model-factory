"""Scheduler-visible features for RightSizeCPU (same boundary as RightSizeMem).

Input metadata reuses RightSizeMem's ``input_features``. Code features are
the APIs that decide parallelism, plus the pool sizes the code declares
(``declared_pools``): the literal a careful reader would look for first.
Never the knobs, the family, or anything measured on the run.
"""

from __future__ import annotations

import json
import math
import re

from ..bench.features import input_features

CODE_TOKENS = (
    "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "POLARS_MAX_THREADS", "SET threads", "n_jobs=",
    "threadpool_limits", "set_num_threads", "set_cpu_count", "use_threads=True", "use_threads=False",
    "ThreadPoolExecutor", "ProcessPoolExecutor", "max_workers=", "zlib.compress", "hashlib", "np.sort",
    "time.sleep", "json.loads", ".tolist()", "for b in chunk", "pd.read_parquet", "groupby", ".apply(",
    ".str.", "merge(", "scan_parquet", "collect()", "duckdb", "OVER (", "JOIN", "RandomForest", "ExtraTrees",
    "HistGradientBoosting", "n_estimators=", "torch.nn", "X.T @ X", "linalg.svd", "linalg.solve", "np.tanh",
    "pacsv.read_csv", "splitlines", "lines[i::", "phase 1", "phase 2",
)

#: `n_jobs=4`, `max_workers=8`, `SET threads = 3`, `set_num_threads(2)`,
#: `limits=6`, `set_cpu_count(4)`, `NUM_THREADS"] = "4"`, `MAX_THREADS"] = "2"`
_POOL = re.compile(
    r"(?:n_jobs=|max_workers=|SET threads = |set_num_threads\(|limits=|set_cpu_count\(|"
    r"NUM_THREADS\"\] = \"|MAX_THREADS\"\] = \")(\d+)"
)
_COUNT = re.compile(r"n_estimators=(\d+)|range\((\d+)\)\]")
_NUM = re.compile(r"(?<![\w.])(\d[\d_]*)(?![\w.])")


def declared_pools(code: str) -> list[int]:
    return [int(x) for x in _POOL.findall(code)]


def code_features(code: str) -> dict[str, float]:
    f = {f"tok_{t}": float(t in code) for t in CODE_TOKENS}
    pools = declared_pools(code)
    f["max_pool"] = float(max(pools)) if pools else 0.0
    f["n_pool_decls"] = float(len(pools))
    counts = [int(a or b) for a, b in _COUNT.findall(code)]
    f["min_work_items"] = float(min(counts)) if counts else 0.0
    nums = [int(n.replace("_", "")) for n in _NUM.findall(code)]
    nums = [n for n in nums if n > 1]
    f["log_max_literal"] = math.log2(max(nums)) if nums else 0.0
    f["code_lines"] = float(code.count("\n"))
    return f


def history_features(history: list[dict]) -> dict[str, float]:
    if not history:
        return {"n_hist": 0.0, "hist_peak_max": 0.0, "hist_peak_last": 0.0, "hist_mean_max": 0.0}
    return {"n_hist": float(len(history)),
            "hist_peak_max": max(h["peak_cores"] for h in history),
            "hist_peak_last": history[-1]["peak_cores"],
            "hist_mean_max": max(h["mean_cores"] for h in history)}


def row_features(row: dict) -> dict[str, float]:
    inputs = json.loads(row["inputs_json"])
    sizes = json.loads(row["input_bytes_json"])
    history = json.loads(row.get("history_json") or "[]")
    return {**input_features(inputs, sizes), **code_features(row["task_code"]), **history_features(history)}


def feature_names() -> list[str]:
    demo = [{"name": "x", "fmt": "npy", "shape": [2, 2], "dtype": "float64"}]
    return sorted(row_features({"inputs_json": json.dumps(demo), "input_bytes_json": json.dumps({"x": 10}),
                                "task_code": "", "history_json": "[]"}))
