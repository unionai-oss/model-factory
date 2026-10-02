"""The inter-team contract: artifact names, payload schemas, partitioning.

Four teams own the factory (see docs/SPEC.md §6). They are wired by the
*factory graph* (`model_factory/factory.py`), not by calls between their
modules — a team's task takes `File`/`Dir` arguments and returns `File`/`Dir`,
and the factory decides which artifact each return value becomes:

| team             | builds                                  | from                     |
|------------------|-----------------------------------------|--------------------------|
| data engineering | `seed-tasks`, `synthetic-tasks`, `rl-tasks-dataset` | the seed dataset |
| model training   | `policy-checkpoint`                     | `rl-tasks-dataset`       |
| model eval       | `eval-report`, `promoted-model`         | `policy-checkpoint`      |
| inference        | the `mf-inference` endpoint             | `promoted-model`         |

Teams may import THIS module and `model_factory.shared.*` (platform
libraries), but never each other's task modules.

Nothing here publishes anything. Artifact creation is the factory's job: its
driver calls each build's task with `produces_artifacts=True` inside a
`flyte.artifacts.produces(...)` block, declaring the name, kind, partition
values and parent versions itself. A station task that wrapped its own output
in `flyte.artifacts.new()` would be fighting that, so none of them do.
"""

from __future__ import annotations

# ── artifact registry ───────────────────────────────────────────────────
# One release day of the factory produces one version of each of these,
# partitioned by `date` (a `factory.Daily` dimension).
#
# The `mf-` prefix is not decoration. An artifact's partition schema is fixed
# in the registry by its first version and cannot change afterwards, and the
# pre-factory trigger chain had already published `synthetic-tasks`,
# `rl-tasks-dataset`, `policy-checkpoint`, `eval-report` and `promoted-model`
# with NO partitions. Declaring them `{date: Daily}` is rejected at
# `factory deploy` with "the registry already fixes this artifact's partitions
# ... Match the registry or publish under a new artifact name".
#
# So: partitioned names under a new prefix, which keeps the daily dimension
# (and with it per-day reuse and backfill) and leaves the old unpartitioned
# versions in place as history. Reusing the original names would mean deleting
# those artifacts — recoverable via `flyte undelete`, but it throws away the
# lineage earlier research rounds refer to, so it is not done here.
ARTIFACT_SEED_TASKS = "mf-seed-tasks"
ARTIFACT_SYNTHETIC = "mf-synthetic-tasks"
ARTIFACT_RL_DATASET = "mf-rl-tasks-dataset"
ARTIFACT_CHECKPOINT = "mf-policy-checkpoint"
ARTIFACT_EVAL_REPORT = "mf-eval-report"
ARTIFACT_PROMOTED = "mf-promoted-model"

#: The factory's `serve` node: a running app, not an artifact. It is a node in
#: the graph like the others, but it is never published to or read from the
#: artifact registry — materializing it deploys the app with the
#: `promoted-model` version the same materialization resolved.
ENDPOINT_APP = "mf-inference"

#: The partition dimension every artifact above carries: one release per day.
PARTITION_DATE = "date"

# ── payload schemas ─────────────────────────────────────────────────────
# seed-tasks / synthetic-tasks / rl-tasks-dataset: parquet File with exactly
# these columns.
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

# policy-checkpoint / promoted-model: Dir containing a PEFT LoRA adapter,
# tokenizer files, and manifest.json with at least these keys.
CHECKPOINT_MANIFEST_KEYS = ["base_model", "profile", "max_steps", "final_metrics"]

# eval-report: JSON File with at least these keys.
EVAL_REPORT_KEYS = [
    "base_model",
    "n_heldout",
    "candidate_pass_at_1",
    "base_pass_at_1",
    "delta",
    "auto_gate_passed",
]
