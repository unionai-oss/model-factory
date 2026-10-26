"""RightSizeCPU rows: tasks -> runs -> measured, labelled, split, with history.

The conventions are RightSizeMem's (``resource_tuner.bench.dataset``):
splits by task (70/10/20, held-out families -> ``ood``), run k carries runs
0..k-1 as history. The labels are the CPU grid cells covering the run's
peak 15-second window and its mean.
"""

from __future__ import annotations

import json
import zlib
from random import Random

from ..bench.dataset import SPLIT_FRACTIONS, SPLITS, usable
from ..bench.workloads import describe_inputs, render_harness, render_task
from . import grid
from .workloads import FAMILIES, HELDOUT_FAMILIES, make_task

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
                    "bench": "cpu",
                    "profile": profile,
                    "run_id": f"{task.task_id}-r{k}",
                    "task_id": task.task_id,
                    "family": family,
                    "driver": task.memory_driver,
                    "run_idx": k,
                    "split": _split_for(task.task_id, family),
                    "knobs_json": json.dumps(task.knobs, sort_keys=True),
                    "inputs_json": json.dumps(task.run_inputs(k)),
                    "size_factor": task.size_factors[k],
                    "task_code": code,
                    "harness_code": harness,
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
        "peak_cores": float(m.get("peak_cores") or 0.0),
        "peak_cores_1s": float(m.get("peak_cores_1s") or 0.0),
        "mean_cores": float(m.get("mean_cores") or 0.0),
        "wall_s": float(m.get("wall_s") or 0.0),
        "prep_s": float(m.get("prep_s") or 0.0),
        "peak_rss_mib": float(m.get("peak_rss_mib") or 0.0),
        "cpu_source": str(m.get("cpu_source") or ""),
        "cpu_s": float(m.get("cpu_s") or 0.0),
        "ru_cpu_s": float(m.get("ru_cpu_s") or 0.0),
        "cpu_series_json": json.dumps(m.get("cpu_series") or []),
    })
    out["label_peak_idx"] = grid.label_peak_index(out["peak_cores"])
    out["label_mean_idx"] = grid.label_mean_index(out["mean_cores"])
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
                seen = seen + [{
                    "input_bytes": json.loads(r["input_bytes_json"]),
                    "inputs_desc": r["inputs_desc"],
                    "peak_cores": round(r["peak_cores"], 2),
                    "mean_cores": round(r["mean_cores"], 2),
                    "wall_s": round(r["wall_s"], 1),
                }]
    return out
