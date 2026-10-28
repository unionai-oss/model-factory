"""Scheduler-visible features for RightSizeGPU (the same information boundary).

Code features are the PyTorch APIs that move GPU memory and utilization,
plus a few numbers a reader pulls out of code: the largest ``Linear``
weight, the batch the loop slices, the declared vocabulary and width.
"""

from __future__ import annotations

import json
import math
import re

from ..bench.features import input_features

CODE_TOKENS = (
    "torch.optim.Adam(", "torch.optim.AdamW(", "torch.optim.SGD(", "SparseAdam", "sparse=True",
    "torch.autocast", "GradScaler", "enabled=AMP", "AMP = True", "model.half()", "X.half()", "float16",
    "checkpoint.checkpoint", "inference_mode", "nullcontext", "outputs.append(emb)", ".cpu()",
    "channels_last", "to(device)  # whole", "images.to(device)", "Embedding(", "MultiheadAttention",
    "TransformerEncoder", "Conv2d(", "BatchNorm2d", "cdist", "topk", "@ X.T", "DataLoader", "num_workers=0",
    "num_workers=1", "num_workers=2", "pin_memory", "mmap_mode", "per-row", "for p in img", "np.random.normal",
    "cross_entropy", "VOCAB", "logits",
    # big-2: code that spreads over every GPU, frozen layers, factored optimizers, KV caches
    "device_count()", "RowSharded", "place[i]", "requires_grad_(False)", "blocks[:-4]", "Adafactor",
    "scaled_dot_product_attention", "caches", "DTYPE", "momentum=0.9",
)
_LINEAR = re.compile(r"Linear\((\d+), (\d+)")
_SLICE = re.compile(r"\[i : i \+ (\d+)\]")
_ASSIGN3 = re.compile(r"= (\d+), (\d+), (\d+)\n")
_ASSIGN4 = re.compile(r"= (\d+), (\d+), (\d+), (\d+)\n")
_NUM = re.compile(r"(?<![\w.])(\d[\d_]*)(?![\w.])")


def code_features(code: str) -> dict[str, float]:
    f = {f"tok_{t}": float(t in code) for t in CODE_TOKENS}
    lin = [int(a) * int(b) for a, b in _LINEAR.findall(code)]
    f["log_max_linear"] = math.log2(max(lin)) if lin else 0.0
    f["log_sum_linear"] = math.log2(sum(lin)) if lin else 0.0
    sl = [int(x) for x in _SLICE.findall(code)]
    f["log_batch"] = math.log2(max(sl)) if sl else 0.0
    trip = _ASSIGN3.findall(code)
    f["log_assign_a"] = math.log2(int(trip[0][0])) if trip else 0.0
    f["log_assign_b"] = math.log2(int(trip[0][1])) if trip else 0.0
    f["assign_c"] = float(trip[0][2]) if trip else 0.0
    quad = _ASSIGN4.findall(code)
    if quad:  # VOCAB, D, LAYERS, NEW
        f["log_assign_a"], f["log_assign_b"] = math.log2(int(quad[0][0])), math.log2(int(quad[0][1]))
        f["assign_c"] = float(quad[0][2])
    f["log_assign_d"] = math.log2(int(quad[0][3])) if quad else 0.0
    # params a reader would count: d^2 x layers for a transformer of width d
    f["log_d2_layers"] = (math.log2(int(trip[0][1]) ** 2 * int(trip[0][2])) if trip else
                          math.log2(int(quad[0][1]) ** 2 * int(quad[0][2])) if quad else 0.0)
    nums = [int(n.replace("_", "")) for n in _NUM.findall(code)]
    nums = [n for n in nums if n > 1]
    f["log_max_literal"] = math.log2(max(nums)) if nums else 0.0
    f["code_lines"] = float(code.count("\n"))
    return f


def history_features(history: list[dict], sizes: dict[str, int]) -> dict[str, float]:
    if not history:
        return {"n_hist": 0.0, "log_hist_peak": 0.0, "hist_util": 0.0, "log_hist_size_ratio": 0.0}
    total = sum(sizes.values()) or 1
    h = min(history, key=lambda h: abs(math.log((sum(h["input_bytes"].values()) or 1) / total)))
    return {"n_hist": float(len(history)),
            "log_hist_peak": math.log2(max(max(x["peak_gpu_mib"] for x in history), 1)),
            "hist_util": float(h["mean_util"]),
            "log_hist_size_ratio": math.log2(total / (sum(h["input_bytes"].values()) or 1))}


def row_features(row: dict) -> dict[str, float]:
    inputs = json.loads(row["inputs_json"])
    sizes = json.loads(row["input_bytes_json"])
    history = json.loads(row.get("history_json") or "[]")
    return {**input_features(inputs, sizes), **code_features(row["task_code"]), **history_features(history, sizes)}


def feature_names() -> list[str]:
    demo = [{"name": "x", "fmt": "npy", "shape": [2, 2], "dtype": "float64"}]
    return sorted(row_features({"inputs_json": json.dumps(demo), "input_bytes_json": json.dumps({"x": 10}),
                                "task_code": "", "history_json": "[]"}))
