"""Artifact contracts for the resource-tuner factory.

Same rule as basic-model-factory: artifacts are the interface between
stations; names and schemas here are load-bearing.
"""

from __future__ import annotations

# ── artifact registry ───────────────────────────────────────────────────
ARTIFACT_TASK_CORPUS = "tuning-task-corpus"
ARTIFACT_SYNTHETIC = "synthetic-task-corpus"  # oracle-verified teacher tasks
ARTIFACT_TUNER_CHECKPOINT = "tuner-checkpoint"
ARTIFACT_EVAL_REPORT = "tuner-eval-report"
ARTIFACT_PROMOTED = "promoted-tuner"
# A/B attribution: tuned proposals vs a hard-coded prior on real pods —
# the PRD's auditable savings record, prototype-sized. JSON File.
ARTIFACT_AB_REPORT = "tuning-ab-report"
AB_REPORT_KEYS = [
    "n_tasks",
    "prior",
    "prior_oom_rate",
    "tuned_oom_rate",
    "prior_fit_rate",
    "tuned_fit_rate",
    "prior_median_overprovision_pct",
    "tuned_median_overprovision_pct",
    "episodes",
]

# tuning-task-corpus: parquet File with exactly these columns.
CORPUS_COLUMNS = [
    "task_id",
    "family",  # data_engineering | data_science | ml_training | batch_inference | etl
    "source_code",  # the rendered flyte task the policy sees
    "harness_code",  # the same workload as a plain function, for episode pods
    "input_profile",  # human-readable input description shown to the policy
    "params_json",  # sampled template params (ground truth generator state)
    "prior_json",  # author-declared prior (JSON kwargs; "" = cold start)
    "history_json",  # past runs [{resources, peak, ok}]; "" = none
    "true_peak_memory_mib",  # analytic footprint estimate
    "true_cpu_cores",  # sustained parallel CPU demand
    "true_gpu_mem_mib",  # VRAM the task needs; 0 = CPU task
    "duration_s",  # how long the workload holds its footprint
    "split",  # "train" | "heldout"
]

# tuner-checkpoint-intermediate: same Dir shape as tuner-checkpoint, but
# published mid-training by publish_intermediate_checkpoint. A separate
# artifact name on purpose: the eval-on-new-checkpoint trigger binds to
# tuner-checkpoint, so intermediates never fire dark evals.
ARTIFACT_TUNER_CHECKPOINT_INTERMEDIATE = "tuner-checkpoint-intermediate"

# tuner-checkpoint: Dir with a PEFT adapter + tokenizer + manifest.json.
CHECKPOINT_MANIFEST_KEYS = ["base_model", "profile", "reward_stage", "max_steps", "final_metrics"]

# tuner-eval-report: JSON File with at least these keys.
EVAL_REPORT_KEYS = [
    "base_model",
    "reward_stage",  # which reward trained the checkpoint (the arm key)
    "n_contexts",
    "schema_validity",  # % proposals that parse + bounds-check
    "success_rate",  # % episodes where the task fit in the proposal
    "median_overprovision_pct",  # waste on successful episodes
    "baseline_success_rate",  # rule-based baseline on the same contexts
    "baseline_median_overprovision_pct",
    "policy_cost_per_task_hr",  # the business metric ($, pricing.py)
    "baseline_cost_per_task_hr",
    "dollars_saved_per_1k_task_hrs",
    "cluster_episodes",  # how many episodes ran on the real cluster
    "auto_gate_passed",
]


def publish(obj, name: str, description: str = "", kind: str = "data"):
    """Publish an offloaded asset as a versioned artifact (team hand-off)."""
    import flyte.artifacts as artifacts

    return artifacts.new(obj, artifacts.Metadata(name=name, description=description, kind=kind))
