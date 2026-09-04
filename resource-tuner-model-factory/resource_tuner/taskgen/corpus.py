"""Corpus builder: sample GeneratedTasks into the tuning-task-corpus table.

Pure logic (records as dicts); the Flyte task that publishes the corpus
as an artifact lives in training/stations.py.

Beyond code + input profile, rows carry the PRD §8 context fields the
serving path now feeds the policy:

- ``prior_json`` — a synthetic author-declared prior. Deliberately
  imperfect (padded defaults, over-bucketed guesses, occasional
  under-asks) so the policy learns the prior is a HINT, not the answer.
- ``history_json`` — 0–3 simulated past runs (requested resources,
  observed peak, fit/OOM), the same shape the tune service extracts from
  its value ledger. History entries derive from the task's true footprint
  on purpose: reading them is the skill being trained, exactly as reading
  real ledger history is the skill at serve time.

Both are empty on a fraction of rows — cold-start must keep working.
"""

from __future__ import annotations

import json
import zlib
from random import Random

from ..contracts import CORPUS_COLUMNS
from ..policy.actions import (
    CPU_GRID,
    MEMORY_GRID_MIB,
    bucket_cpu,
    bucket_memory_mib,
    format_memory,
)
from .templates import FAMILIES, GeneratedTask, generate_task

PRIOR_RATE = 0.5  # fraction of rows carrying an author prior
HISTORY_RATE = 0.4  # fraction carrying past runs


def _synthetic_prior(t: GeneratedTask, rng: Random) -> dict:
    """One plausible author guess. Styles mirror reality: the padded
    one-size default, the over-bucketed 'measured once, doubled twice',
    and the stale under-ask."""
    style = rng.random()
    if style < 0.4:  # the classic hard-coded default
        prior = {"cpu": 4, "memory": "8Gi"}
    elif style < 0.8:  # over-bucketed: 1-3 grid steps above truth
        mem_idx = MEMORY_GRID_MIB.index(bucket_memory_mib(t.true_peak_memory_mib))
        mem = MEMORY_GRID_MIB[min(mem_idx + rng.randint(1, 3), len(MEMORY_GRID_MIB) - 1)]
        cpu_idx = CPU_GRID.index(bucket_cpu(t.true_cpu_cores))
        cpu = CPU_GRID[min(cpu_idx + rng.randint(0, 2), len(CPU_GRID) - 1)]
        prior = {"cpu": cpu, "memory": format_memory(mem)}
    else:  # stale under-ask (the task grew since the author measured)
        mem_idx = MEMORY_GRID_MIB.index(bucket_memory_mib(t.true_peak_memory_mib))
        prior = {"cpu": 1, "memory": format_memory(MEMORY_GRID_MIB[max(mem_idx - 1, 0)])}
    if t.true_gpu_mem_mib > 0 and rng.random() < 0.6:
        prior["gpu"] = "T4:1"  # authors guess the cheapest card, right or not
    return prior


def _synthetic_history(t: GeneratedTask, rng: Random) -> list[dict]:
    """1–3 simulated past runs against the task's true footprint — the
    ledger-shaped signal (requested / peak / fit)."""
    entries = []
    for _ in range(rng.randint(1, 3)):
        mem_idx = MEMORY_GRID_MIB.index(bucket_memory_mib(t.true_peak_memory_mib))
        req_mem = MEMORY_GRID_MIB[
            min(max(mem_idx + rng.randint(-1, 2), 0), len(MEMORY_GRID_MIB) - 1)
        ]
        peak = t.true_peak_memory_mib * rng.uniform(0.92, 1.05)
        ok = req_mem >= peak * 1.05
        entries.append(
            {
                "resources": {"cpu": bucket_cpu(t.true_cpu_cores), "memory": format_memory(req_mem)},
                "peak": f"{peak:.0f}MiB",
                "ok": ok,
            }
        )
    return entries


def task_to_record(t: GeneratedTask, split: str, rng: Random | None = None) -> dict:
    prior_json = history_json = ""
    if rng is not None:
        if rng.random() < PRIOR_RATE:
            prior_json = json.dumps(_synthetic_prior(t, rng), sort_keys=True)
        if rng.random() < HISTORY_RATE:
            history_json = json.dumps(_synthetic_history(t, rng))
    return {
        "task_id": t.task_id,
        "family": t.family,
        "source_code": t.source_code,
        "harness_code": t.harness_code,
        "input_profile": t.input_profile,
        "params_json": json.dumps(t.params, sort_keys=True),
        "prior_json": prior_json,
        "history_json": history_json,
        "true_peak_memory_mib": float(t.true_peak_memory_mib),
        "true_cpu_cores": float(t.true_cpu_cores),
        "true_gpu_mem_mib": float(t.true_gpu_mem_mib),
        "duration_s": int(t.duration_s),
        "split": split,
    }


def _context_rng(t: GeneratedTask) -> Random:
    """Deterministic per-task randomness for prior/history — crc of the
    task_id, not hash() (salted per process)."""
    return Random(zlib.crc32(t.task_id.encode()))


def build_corpus(
    n_train: int,
    n_heldout: int,
    seed: int = 0,
    gpu_max_vram_mib: float | None = None,
) -> list[dict]:
    """Round-robin families, deterministic in `seed`, heldout drawn from a
    disjoint seed range so re-sampling train can never leak into eval.
    `gpu_max_vram_mib` constrains GPU families (e.g. 14000 → every GPU
    task fits, and should be proposed, a single T4)."""
    families = sorted(FAMILIES)
    rng = Random(seed)
    records: list[dict] = []
    for i in range(n_train):
        fam = families[i % len(families)]
        t = generate_task(fam, seed=rng.randint(0, 2**30), gpu_max_vram_mib=gpu_max_vram_mib)
        records.append(task_to_record(t, "train", rng=_context_rng(t)))
    heldout_rng = Random(seed + 1_000_003)
    for i in range(n_heldout):
        fam = families[i % len(families)]
        t = generate_task(
            fam,
            seed=2**30 + heldout_rng.randint(0, 2**30),
            gpu_max_vram_mib=gpu_max_vram_mib,
        )
        records.append(task_to_record(t, "heldout", rng=_context_rng(t)))
    assert all(set(r) == set(CORPUS_COLUMNS) for r in records)
    return records
