"""Bench rows: tasks -> runs -> measured, labelled, split, with history.

One row per *run* (a task at one input size). Rows are plain dicts so the
same code builds the table in a station, in a test, and on a laptop.

Splits are by task, never by run: every run of a task lands in one split,
so no arm ever sees a test task's code at training time. Two families are
held out entirely (``HELDOUT_FAMILIES``) -- the out-of-distribution test.

History is the bench's "the platform has seen this task before" signal:
run k of a task carries runs 0..k-1 as history (their input sizes and the
peak they were *observed* to use), exactly what a run ledger would hold.
Run 0 is always a cold start.
"""

from __future__ import annotations

import json
import zlib
from random import Random

from . import grid
from .workloads import (
    FAMILIES,
    HELDOUT_FAMILIES,
    describe_inputs,
    make_task,
    render_harness,
    render_task,
)

SPLITS = ("train", "val", "test", "ood")
#: Fractions of in-distribution tasks; the rest of the family set is "ood".
SPLIT_FRACTIONS = {"train": 0.7, "val": 0.1, "test": 0.2}


def _split_for(task_id: str, family: str) -> str:
    if family in HELDOUT_FAMILIES:
        return "ood"
    u = (zlib.crc32(task_id.encode()) % 10_000) / 10_000
    if u < SPLIT_FRACTIONS["train"]:
        return "train"
    if u < SPLIT_FRACTIONS["train"] + SPLIT_FRACTIONS["val"]:
        return "val"
    return "test"


def generate_runs(tasks_per_family: int, seed: int = 0, runs_per_task: int = 3,
                  families: list[str] | None = None) -> list[dict]:
    """Unmeasured run rows, deterministic in ``seed``."""
    rng = Random(seed)
    rows: list[dict] = []
    for family in families or sorted(FAMILIES):
        for _ in range(tasks_per_family):
            task = make_task(family, rng.randint(0, 2**31), n_runs=runs_per_task)
            code, harness = render_task(task), render_harness(task)
            for k in range(runs_per_task):
                rows.append({
                    "run_id": f"{task.task_id}-r{k}",
                    "task_id": task.task_id,
                    "family": family,
                    "memory_driver": task.memory_driver,
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
    """Fold one measurement into a run row and derive its labels."""
    sizes = m.get("input_bytes") or {}
    out = dict(row)
    out.update({
        "ok": bool(m.get("ok")),
        "censored": bool(m.get("censored")),
        "error": str(m.get("error") or "")[:500],
        "input_bytes_json": json.dumps(sizes),
        "total_input_bytes": int(sum(sizes.values())),
        "inputs_desc": describe_inputs(json.loads(row["inputs_json"]), sizes),
        "peak_rss_mib": float(m.get("peak_rss_mib") or 0.0),
        "avg_cores": float(m.get("avg_cores") or 0.0),
        "wall_s": float(m.get("wall_s") or 0.0),
        "prep_s": float(m.get("prep_s") or 0.0),
        "t_peak_frac": float(m.get("t_peak_frac") or 1.0),
    })
    out["required_mib"] = grid.required_memory_mib(out["peak_rss_mib"])
    out["label_mem_idx"] = grid.label_memory_index(out["peak_rss_mib"])
    out["label_cpu_idx"] = grid.label_cpu_index(out["avg_cores"])
    return out


def attach_history(rows: list[dict]) -> list[dict]:
    """Give run k the observed runs 0..k-1 of the same task, as a ledger would."""
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
                    "peak_rss_mib": round(r["peak_rss_mib"], 1),
                    "avg_cores": round(r["avg_cores"], 2),
                    "wall_s": round(r["wall_s"], 2),
                }]
    return out


def usable(row: dict) -> bool:
    """Rows an arm is scored on: the workload ran to completion."""
    return bool(row.get("ok")) and not row.get("censored")
