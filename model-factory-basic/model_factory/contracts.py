"""The inter-team contract: artifact names, payload schemas, partitioning.

Four teams own the factory (see docs/SPEC.md §6). They are wired by the
*factory graph* (`model_factory/factory.py`), not by calls between their
modules — a team's task takes `File`/`Dir` arguments and returns `File`/`Dir`,
and the factory decides which artifact each return value becomes:

| team             | builds                                                  |
|------------------|---------------------------------------------------------|
| data engineering | `heldout-tasks`, `seed-tasks`, `synthetic-tasks`, `rl-tasks-dataset` |
| model training   | `policy-checkpoint`                                     |
| model eval       | `baseline-report`, `eval-report`, `promoted-model` (+ crowns `champion`) |
| inference        | the `mf-inference` endpoint                             |

Teams may import THIS module and `model_factory.shared.*` (platform
libraries), but never each other's task modules.

Nothing in the factory publishes by hand. Artifact creation is the factory's
job: its driver calls each build's task with `produces_artifacts=True` inside a
`flyte.artifacts.produces(...)` block. The one exception is `mf-champion`,
which is published OUTSIDE the factory on purpose — see below.
"""

from __future__ import annotations

import hashlib
import re

# ── artifact registry ───────────────────────────────────────────────────
#
# The `mf-` prefix is not decoration. An artifact's partition schema is fixed
# in the registry by its first version and cannot change afterwards, and the
# pre-factory trigger chain had already published `synthetic-tasks`,
# `rl-tasks-dataset`, `policy-checkpoint`, `eval-report` and `promoted-model`
# with NO partitions. Declaring them `{date: Daily}` is rejected at
# `factory deploy`, so the partitioned versions live under new names.

#: Built ONCE, not per day (no partitions): the fixed held-out split every day
#: is scored on, and the base model's score on it. Every daily materialization
#: reuses the same version, which is what makes day-over-day numbers comparable.
ARTIFACT_HELDOUT = "mf-heldout-tasks"
ARTIFACT_BASELINE = "mf-baseline-report"

#: Built per release day, partitioned by `date`.
ARTIFACT_SEED_TASKS = "mf-seed-tasks"
ARTIFACT_SYNTHETIC = "mf-synthetic-tasks"
ARTIFACT_RL_DATASET = "mf-rl-tasks-dataset"
ARTIFACT_CHECKPOINT = "mf-policy-checkpoint"
ARTIFACT_EVAL_REPORT = "mf-eval-report"
ARTIFACT_PROMOTED = "mf-promoted-model"

#: The loop's state: `mf-champion[D]` is the model day D starts from and must
#: beat. It is a `factory.source`, published by `crown_champion` (an
#: OnArtifact trigger on `mf-promoted-model`) as `champion[D+1] =
#: promoted[D]`.
#:
#: Why not let day D read `promoted-model[D-1]` directly? A factory graph is a
#: DAG over artifact *names*: `promoted-model -> ... -> policy-checkpoint ->
#: promoted-model` is a cycle and `factory deploy` rejects it, and there is no
#: "previous partition" mapping (`.window()` always includes the consumer's own
#: day). So the edge that crosses days is a separate artifact the factory
#: only reads. Keeping it partitioned makes day D a pure function of
#: `champion[D]`: re-materializing a day is a cache hit, and a missing champion
#: stops the day before any GPU work instead of training from the wrong model.
ARTIFACT_CHAMPION = "mf-champion"

#: The factory's `serve` node: a running app, not an artifact.
ENDPOINT_APP = "mf-inference"

#: The partition dimension every per-day artifact carries.
PARTITION_DATE = "date"

# ── payload schemas ─────────────────────────────────────────────────────
# heldout-tasks / seed-tasks / synthetic-tasks / rl-tasks-dataset: parquet
# File with exactly these columns.
DATASET_COLUMNS = [
    "task_id",
    "question",
    "function_declaration",
    "tests",
    "reference_solution",
    "difficulty",
    "n_tests",
    "source",
    "split",  # "train" | "heldout"
]

# policy-checkpoint / promoted-model / champion: Dir containing manifest.json
# with at least these keys, plus (when `adapter` is true) a PEFT LoRA adapter
# and tokenizer files. A champion with `adapter: false` is the base model
# itself — that is how the loop starts (`seed_champion`).
CHECKPOINT_MANIFEST_KEYS = ["base_model", "adapter", "generation"]
#: Optional files a checkpoint carries forward to the next day.
HARD_TASKS_FILE = "hard_tasks.parquet"  # train tasks the policy kept failing
EVAL_FILE = "eval_report.json"  # the report that got it promoted

# eval-report / baseline-report: JSON File with at least these keys.
EVAL_REPORT_KEYS = [
    "base_model",
    "n_heldout",
    "heldout_fingerprint",
    "candidate_pass_at_1",
    "base_pass_at_1",
    "champion_pass_at_1",
    "delta_vs_base",
    "delta_vs_champion",
    "auto_gate_passed",
]


# ── held-out isolation ──────────────────────────────────────────────────
_WS = re.compile(r"\s+")


def question_key(question: str) -> str:
    """Identity of a task for leak checks: its whitespace/case-normalized text."""
    return hashlib.sha1(_WS.sub(" ", question).strip().lower().encode()).hexdigest()


def heldout_fingerprint(task_ids) -> str:
    """One short hash naming a held-out split, so reports can prove it is fixed."""
    return hashlib.sha1("\n".join(sorted(map(str, task_ids))).encode()).hexdigest()[:16]


def find_leaks(questions, heldout_questions) -> list[int]:
    """Positions in ``questions`` whose task is also in the held-out split."""
    held = {question_key(q) for q in heldout_questions}
    return [i for i, q in enumerate(questions) if question_key(q) in held]
