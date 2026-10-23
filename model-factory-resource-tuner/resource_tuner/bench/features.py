"""Scheduler-visible features: the information boundary every arm shares.

An arm may use what a platform knows when a run is submitted, and nothing
else:

- the task's **source** (as text);
- each input's **format metadata** -- bytes on disk always; rows and schema
  for parquet (the footer); shape and dtype for npy (the header); only the
  header line for CSV and only the first record's keys for JSONL (no row
  count: you would have to read the file);
- the task's **run history**: earlier runs' input sizes and observed peaks.

Never: the generator's knobs, the family label, or anything measured on
the run being decided. ``params_json`` in the old corpus leaked the
generator's own parameters into the GBT baseline's features; that is the
mistake this module exists not to repeat. ``test_features`` pins it.
"""

from __future__ import annotations

import json
import math
import re

#: API surface whose presence changes how bytes become resident memory.
CODE_TOKENS = (
    "read_parquet", "read_csv", "read_json", "scan_parquet", "read_parquet(", "columns=",
    "dtype_backend", "astype", "float32", "chunksize", "iter_batches", "streaming",
    "collect(", "mmap_mode", "np.load", "toarray", "TruncatedSVD", "TfidfVectorizer",
    "merge(", "drop_duplicates", "groupby", "transform(", "rank(", "describe(", "sort_values",
    "to_parquet", "json.loads", "for line in f", "records = [", "defaultdict(list)", "Counter(",
    ".append(", "concat(", "duckdb", "memory_limit", "threads", "ORDER BY", "OVER (",
    "GROUP BY", "SELECT *", ".df()", "svd(", " @ ", ".T @", "cov(", "np.asarray(X[:",
    "RandomForest", "KNeighbors", "KMeans", "HistGradientBoosting", "LogisticRegression",
    "max_depth=None", "n_jobs=", "torch.tensor", "Embedding(", "AdamW", "Linear(",
    "batch_size", "usecols", "pl.read_parquet", "uint8", "/ 255",
)
_NUM = re.compile(r"(?<![\w.])(\d[\d_]*)(?![\w.])")


def _visible_rows(spec: dict) -> float:
    return float(spec["rows"]) if spec["fmt"] == "parquet" else 0.0


def input_features(inputs: list[dict], sizes: dict[str, int]) -> dict[str, float]:
    f: dict[str, float] = {}
    total = sum(sizes.values()) or 1
    f["log_total_bytes"] = math.log2(total)
    f["n_inputs"] = len(inputs)
    for fmt in ("parquet", "csv", "jsonl", "npy"):
        b = sum(sizes.get(s["name"], 0) for s in inputs if s["fmt"] == fmt)
        f[f"log_bytes_{fmt}"] = math.log2(b) if b else 0.0
    rows = sum(_visible_rows(s) for s in inputs)
    f["log_rows_visible"] = math.log2(rows) if rows else 0.0
    kinds = {"f64": 0, "f32": 0, "i64": 0, "i32": 0, "cat": 0, "str": 0, "text": 0}
    ncols = 0
    for s in inputs:
        if s["fmt"] == "parquet":  # schema visible
            for c in s["columns"]:
                kinds[c["kind"]] += 1
            ncols += len(s["columns"])
        elif s["fmt"] in ("csv", "jsonl"):  # names only
            ncols += len(s["columns"])
    for k, v in kinds.items():
        f[f"ncol_{k}"] = float(v)
    f["ncols"] = float(ncols)
    arrays = [s for s in inputs if s["fmt"] == "npy"]
    if arrays:
        a = arrays[0]
        f["npy_log_n"] = math.log2(max(a["shape"][0], 1))
        f["npy_log_row"] = math.log2(max(math.prod(a["shape"][1:]), 1))
        f["npy_itemsize"] = {"uint8": 1, "float32": 4, "float64": 8}.get(a["dtype"], 8)
    else:
        f["npy_log_n"] = f["npy_log_row"] = f["npy_itemsize"] = 0.0
    return f


def code_features(code: str) -> dict[str, float]:
    f = {f"tok_{t}": float(t in code) for t in CODE_TOKENS}
    nums = [int(n.replace("_", "")) for n in _NUM.findall(code)]
    nums = [n for n in nums if n > 1]
    f["log_max_literal"] = math.log2(max(nums)) if nums else 0.0
    f["code_lines"] = float(code.count("\n"))
    return f


def history_features(history: list[dict], sizes: dict[str, int]) -> dict[str, float]:
    total = sum(sizes.values()) or 1
    if not history:
        return {"n_hist": 0.0, "log_hist_peak_max": 0.0, "log_hist_scaled": 0.0,
                "log_hist_size_ratio": 0.0, "hist_peak_per_byte": 0.0}
    peaks = [h["peak_rss_mib"] for h in history]
    hbytes = [sum(h["input_bytes"].values()) or 1 for h in history]
    # Nearest-size history run, scaled linearly to this run's input size.
    i = min(range(len(history)), key=lambda k: abs(math.log(hbytes[k] / total)))
    scaled = peaks[i] * total / hbytes[i]
    return {
        "n_hist": float(len(history)),
        "log_hist_peak_max": math.log2(max(max(peaks), 1)),
        "log_hist_scaled": math.log2(max(scaled, 1)),
        "log_hist_size_ratio": math.log2(total / hbytes[i]),
        "hist_peak_per_byte": peaks[i] * 2**20 / hbytes[i],
    }


def row_features(row: dict) -> dict[str, float]:
    inputs = json.loads(row["inputs_json"])
    sizes = json.loads(row["input_bytes_json"])
    history = json.loads(row.get("history_json") or "[]")
    return {**input_features(inputs, sizes), **code_features(row["task_code"]),
            **history_features(history, sizes)}


def feature_names() -> list[str]:
    demo_inputs = [{"name": "x", "fmt": "npy", "shape": [2, 2], "dtype": "float64"}]
    return sorted(row_features({"inputs_json": json.dumps(demo_inputs),
                                "input_bytes_json": json.dumps({"x": 10}),
                                "task_code": "", "history_json": "[]"}))
