"""Scheduler-visible features for the combined bench: the union of the three
benches' code features, the shared input features, and history features over
the unified history schema. Same boundary as every bench (no knobs, no family,
no source, nothing measured on the run itself); `test_all_features_cannot_see_the_answer`
pins it.
"""

from __future__ import annotations

import json
import math

from ..bench import features as mem_features
from ..bench.features import input_features
from ..bench_cpu import features as cpu_features
from ..bench_gpu import features as gpu_features

#: The one signal that a task needs a GPU, as a reader would spot it.
CUDA_TOKENS = ('"cuda"', "torch.cuda", ".to(device)", "device =")


def code_features(code: str) -> dict[str, float]:
    f: dict[str, float] = {}
    for prefix, mod in (("m_", mem_features), ("c_", cpu_features), ("g_", gpu_features)):
        f.update({prefix + k: v for k, v in mod.code_features(code).items()})
    f.update({f"cuda_{t}": float(t in code) for t in CUDA_TOKENS})
    return f


def history_features(history: list[dict], sizes: dict[str, int]) -> dict[str, float]:
    total = sum(sizes.values()) or 1
    out = {"n_hist": float(len(history)), "log_hist_mem": 0.0, "hist_peak_cores": 0.0, "hist_mean_cores": 0.0,
           "log_hist_gpu": 0.0, "hist_util": 0.0, "hist_used_gpu": 0.0, "log_hist_size_ratio": 0.0,
           "log_hist_mem_scaled": 0.0}
    if not history:
        return out
    hb = [sum(h["input_bytes"].values()) or 1 for h in history]
    i = min(range(len(history)), key=lambda k: abs(math.log(hb[k] / total)))
    h = history[i]
    out["log_hist_size_ratio"] = math.log2(total / hb[i])
    mems = [x["peak_mem_mib"] for x in history if "peak_mem_mib" in x]
    if mems:
        out["log_hist_mem"] = math.log2(max(max(mems), 1))
        if "peak_mem_mib" in h:
            out["log_hist_mem_scaled"] = math.log2(max(h["peak_mem_mib"] * total / hb[i], 1))
    cores = [x.get("peak_cores", x.get("mean_cores", 0.0)) for x in history]
    out["hist_peak_cores"] = max(cores)
    out["hist_mean_cores"] = max(x.get("mean_cores", 0.0) for x in history)
    gpus = [x["peak_gpu_mib"] for x in history if "peak_gpu_mib" in x]
    if gpus:
        out["hist_used_gpu"] = 1.0
        out["log_hist_gpu"] = math.log2(max(max(gpus), 1))
        out["hist_util"] = float(h.get("mean_util", 0.0))
    return out


def row_features(row: dict) -> dict[str, float]:
    inputs = json.loads(row["inputs_json"])
    sizes = json.loads(row["input_bytes_json"])
    history = json.loads(row.get("history_json") or "[]")
    return {**input_features(inputs, sizes), **code_features(row["task_code"]), **history_features(history, sizes)}


_NAMES: list[str] | None = None


def feature_names() -> list[str]:
    global _NAMES
    if _NAMES is None:
        demo = [{"name": "x", "fmt": "npy", "shape": [2, 2], "dtype": "float64"}]
        _NAMES = sorted(row_features({"inputs_json": json.dumps(demo), "input_bytes_json": json.dumps({"x": 10}),
                                      "task_code": "", "history_json": "[]"}))
    return _NAMES
