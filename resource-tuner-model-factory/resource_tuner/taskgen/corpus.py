"""Corpus builder: sample GeneratedTasks into the tuning-task-corpus table.

Pure logic (records as dicts); the Flyte task that publishes the corpus
as an artifact lives in training/stations.py.
"""

from __future__ import annotations

import json
from random import Random

from ..contracts import CORPUS_COLUMNS
from .templates import FAMILIES, GeneratedTask, generate_task


def task_to_record(t: GeneratedTask, split: str) -> dict:
    return {
        "task_id": t.task_id,
        "family": t.family,
        "source_code": t.source_code,
        "harness_code": t.harness_code,
        "input_profile": t.input_profile,
        "params_json": json.dumps(t.params, sort_keys=True),
        "true_peak_memory_mib": float(t.true_peak_memory_mib),
        "true_cpu_cores": float(t.true_cpu_cores),
        "duration_s": int(t.duration_s),
        "split": split,
    }


def build_corpus(n_train: int, n_heldout: int, seed: int = 0) -> list[dict]:
    """Round-robin families, deterministic in `seed`, heldout drawn from a
    disjoint seed range so re-sampling train can never leak into eval."""
    families = sorted(FAMILIES)
    rng = Random(seed)
    records: list[dict] = []
    for i in range(n_train):
        fam = families[i % len(families)]
        records.append(task_to_record(generate_task(fam, seed=rng.randint(0, 2**30)), "train"))
    heldout_rng = Random(seed + 1_000_003)
    for i in range(n_heldout):
        fam = families[i % len(families)]
        records.append(
            task_to_record(
                generate_task(fam, seed=2**30 + heldout_rng.randint(0, 2**30)), "heldout"
            )
        )
    assert all(set(r) == set(CORPUS_COLUMNS) for r in records)
    return records
