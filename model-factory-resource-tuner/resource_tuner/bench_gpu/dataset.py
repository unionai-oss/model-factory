"""RightSizeGPU rows: tasks -> runs -> measured, labelled, split, with history.

Conventions are RightSizeMem's. Labels: the GPU memory cell covering the
run's peak device memory, and its mean-utilization bucket. ``measured_on``
records the device (pass 1: T4; runs that did not fit: a bigger machine,
``stations._climb``). ``sharded`` is a property of the task's code (its
family spreads the model over every visible GPU), known before it runs.
"""

from __future__ import annotations

import json
import zlib
from random import Random

from ..bench.dataset import SPLIT_FRACTIONS, SPLITS, usable
from ..bench.workloads import describe_inputs, render_harness, render_task
from . import grid
from .workloads import FAMILIES, HELDOUT_FAMILIES, SHARDED_FAMILIES, make_task

__all__ = ["SPLITS", "usable", "generate_runs", "attach_measurement", "attach_history"]


def _split_for(task_id: str, family: str) -> str:
    if family in HELDOUT_FAMILIES:
        return "ood"
    u = (zlib.crc32(task_id.encode()) % 10_000) / 10_000
    if u < SPLIT_FRACTIONS["train"]:
        return "train"
    if u < SPLIT_FRACTIONS["train"] + SPLIT_FRACTIONS["val"]:
        return "val"
    return "test"


def generate_runs(tasks_per_family: int, seed: int = 0, runs_per_task: int = 2,
                  families: list[str] | None = None, profile: str = "default") -> list[dict]:
    rng = Random(seed)
    rows: list[dict] = []
    for family in families or sorted(FAMILIES):
        for _ in range(tasks_per_family):
            task = make_task(family, rng.randint(0, 2**31), n_runs=runs_per_task, profile=profile)
            code, harness = render_task(task), render_harness(task)
            for k in range(runs_per_task):
                rows.append({
                    "bench": "gpu", "profile": profile, "run_id": f"{task.task_id}-r{k}", "task_id": task.task_id,
                    "family": family, "driver": task.memory_driver, "run_idx": k,
                    "sharded": FAMILIES[family].name in SHARDED_FAMILIES,
                    "split": _split_for(task.task_id, family),
                    "knobs_json": json.dumps(task.knobs, sort_keys=True),
                    "inputs_json": json.dumps(task.run_inputs(k)), "size_factor": task.size_factors[k],
                    "task_code": code, "harness_code": harness,
                    "seed": zlib.crc32(f"{task.task_id}-{k}".encode()) % 2**31,
                })
    return rows


def attach_measurement(row: dict, m: dict) -> dict:
    sizes = m.get("input_bytes") or {}
    out = dict(row)
    out.update({
        "ok": bool(m.get("ok")),
        "censored": bool(m.get("censored")),
        "error": str(m.get("error") or "")[:500],
        "input_bytes_json": json.dumps(sizes),
        "total_input_bytes": int(sum(sizes.values())),
        "inputs_desc": describe_inputs(json.loads(row["inputs_json"]), sizes),
        "peak_gpu_mib": float(m.get("peak_gpu_mib") or 0.0),
        "base_mem_mib": float(m.get("base_mem_mib") or 0.0),
        "mean_util": float(m.get("mean_util") or 0.0),
        "peak_util_window": float(m.get("peak_util_window") or 0.0),
        "util_windows_json": json.dumps(m.get("util_windows") or []),
        "t_peak_frac": float(m.get("t_peak_frac") or 1.0),
        "wall_s": float(m.get("wall_s") or 0.0),
        "prep_s": float(m.get("prep_s") or 0.0),
        "torch_max_reserved_mib": float(m.get("torch_max_reserved_mib") or 0.0),
        "gpu_n_used": int(m.get("gpu_n_used") or (1 if m.get("peak_gpu_mib") else 0)),
        "gpu_max_dev_mib": float(m.get("gpu_max_dev_mib") or m.get("peak_gpu_mib") or 0.0),
        "gpu_imbalance": float(m.get("gpu_imbalance") or 1.0),
        "n_gpus_visible": int(m.get("n_gpus_visible") or 1),
        "mean_cpu_cores": float(m.get("mean_cpu_cores") or 0.0),
        "measured_on": str(m.get("device_name") or ""),
        "device_total_mib": float(m.get("device_total_mib") or 0.0),
        "n_samples": int(m.get("n_samples") or 0),
    })
    out["peak_gpu_gib"] = out["peak_gpu_mib"] / 1024
    # cuDNN's workspace search and the caching allocator grow into whatever the
    # device has (rsg-pilot-1: a CNN logged failed 3-8 GB workspace allocations,
    # fell back, and peaked at 15.1 of 15.1 GiB -- and succeeded). Such a peak is
    # the device's size, not the task's need: the device label stands, the
    # memory-accuracy columns skip it.
    total = out["device_total_mib"]
    out["fills_device"] = bool(out["ok"] and total and out["gpu_max_dev_mib"] >= 0.95 * total)
    # measured on one GPU, a sharded run's split across several is not known
    if out["gpu_n_used"] <= 1:
        out["gpu_imbalance"] = grid.DEFAULT_IMBALANCE if row.get("sharded") else 1.0
    out["label_gmem_idx"] = grid.label_mem_index(out["peak_gpu_gib"])
    out["label_util_idx"] = grid.util_index(out["mean_util"])
    return out


def attach_history(rows: list[dict]) -> list[dict]:
    by_task: dict[str, list[dict]] = {}
    for r in rows:
        by_task.setdefault(r["task_id"], []).append(r)
    out = []
    for runs in by_task.values():
        runs.sort(key=lambda r: r["run_idx"])
        seen: list[dict] = []
        for r in runs:
            r = dict(r)
            r["history_json"] = json.dumps(seen)
            r["n_history"] = len(seen)
            out.append(r)
            if r.get("ok"):
                seen = seen + [{"input_bytes": json.loads(r["input_bytes_json"]), "inputs_desc": r["inputs_desc"],
                                "peak_gpu_mib": round(r["peak_gpu_mib"]), "mean_util": round(r["mean_util"], 1),
                                "measured_on": r["measured_on"], "wall_s": round(r["wall_s"], 1)}]
    return out
