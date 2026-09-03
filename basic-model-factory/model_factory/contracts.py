"""The inter-team contract: artifacts are the ONLY interface between teams.

Four teams own the factory (see docs/SPEC.md §6):

| team             | consumes (OnArtifact)     | publishes                          |
|------------------|---------------------------|------------------------------------|
| data engineering | `synthetic-tasks` (own)   | `rl-tasks-dataset`, `synthetic-tasks` |
| model training   | `rl-tasks-dataset`        | `policy-checkpoint`                |
| model eval       | `policy-checkpoint`       | `eval-report`, `promoted-model`    |
| inference        | `policy-checkpoint`       | `inference-endpoint`               |

Teams may import THIS module and `model_factory.shared.*` (platform
libraries), but never each other's task modules. Downstream work starts via
OnArtifact triggers on these names; on backends without artifact events the
integration driver (integration.py) plays the event bus.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

# ── artifact registry ───────────────────────────────────────────────────
ARTIFACT_RL_DATASET = "rl-tasks-dataset"
ARTIFACT_SYNTHETIC = "synthetic-tasks"
ARTIFACT_CHECKPOINT = "policy-checkpoint"
ARTIFACT_EVAL_REPORT = "eval-report"
ARTIFACT_PROMOTED = "promoted-model"
ARTIFACT_INFERENCE_ENDPOINT = "inference-endpoint"

# ── payload schemas ─────────────────────────────────────────────────────
# rl-tasks-dataset / synthetic-tasks: parquet File with exactly these columns.
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


@dataclass(frozen=True)
class InferenceEndpoint:
    """Payload of the `inference-endpoint` artifact (JSON File)."""

    url: str  # public base URL of the serving app
    base_model: str
    checkpoint_path: str  # object-store path of the loaded policy-checkpoint
    checkpoint_run: str  # run that produced the checkpoint ("" if unknown)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, raw: str) -> "InferenceEndpoint":
        return cls(**json.loads(raw))


def publish(obj, name: str, description: str = "", kind: str = "data"):
    """Publish an offloaded asset (File/Dir/DataFrame) as a versioned artifact.

    This is the hand-off point between teams: returning the wrapped value from
    a task creates a new artifact version, which fires downstream OnArtifact
    triggers (where the backend supports artifact events).
    """
    import flyte.artifacts as artifacts

    return artifacts.new(obj, artifacts.Metadata(name=name, description=description, kind=kind))
