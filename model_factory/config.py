"""Factory configuration: profiles and secret wiring.

Profiles size the whole loop (model, dataset slice, GRPO steps) so the same
pipeline runs as a minutes-long smoke test or a multi-hour training run.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

# Secret names on the Union cluster (see TODO.md for creation commands).
HF_TOKEN_SECRET = "NIELS_HUGGINGFACE_TOKEN"
WANDB_SECRET = "NIELS_WANDB_API_KEY"

# Secrets are only attached to task environments when this env var is set at
# deploy/run time (from the machine that bundles the code). Flyte fails to
# schedule a task whose declared secret doesn't exist on the cluster, and the
# secrets have not been created yet — everything the POC needs (KodCode,
# Qwen models) is public, so the loop runs without them.
USE_SECRETS = os.environ.get("MF_USE_SECRETS", "0") == "1"

WANDB_PROJECT = "model-factory"

SEED_DATASET = "KodCode/KodCode-Light-RL-10K"

# Artifact names — the factory's asset registry.
ARTIFACT_RL_DATASET = "rl-tasks-dataset"
ARTIFACT_SYNTHETIC = "synthetic-tasks"
ARTIFACT_CHECKPOINT = "policy-checkpoint"
ARTIFACT_EVAL_REPORT = "eval-report"
ARTIFACT_PROMOTED = "promoted-model"


@dataclass(frozen=True)
class FactoryProfile:
    """Sizing knobs for one factory iteration."""

    name: str
    base_model: str
    # data
    train_tasks: int  # curated tasks fed to GRPO
    eval_tasks: int  # held-out tasks for the eval gate
    min_test_functions: int  # curation floor (anti reward-hacking)
    # synthetic generation
    synthetic_seeds: int  # seed problems to mutate
    synthetic_max_new_tokens: int
    # GRPO
    max_steps: int
    num_generations: int
    per_device_batch: int  # prompts per device step (before generations)
    max_completion_length: int
    learning_rate: float
    lora_r: int
    use_vllm: bool  # colocate rollouts; off for smoke (robustness first)
    # eval gate: candidate pass@1 must beat base by this margin to auto-pass
    promotion_margin: float
    gpu: str = "A10G:1"

    @property
    def wandb_enabled(self) -> bool:
        return USE_SECRETS


SMOKE = FactoryProfile(
    name="smoke",
    base_model="Qwen/Qwen2.5-Coder-0.5B-Instruct",
    train_tasks=64,
    eval_tasks=32,
    min_test_functions=3,
    synthetic_seeds=8,
    synthetic_max_new_tokens=512,
    max_steps=10,
    num_generations=4,
    per_device_batch=4,
    max_completion_length=512,
    learning_rate=2e-5,
    lora_r=16,
    use_vllm=False,
    promotion_margin=-1.0,  # smoke never blocks on quality
    gpu="A10G:1",
)

# Longer smoke: same tiny model, enough steps for the reward curve to move.
SMOKE_PLUS = FactoryProfile(
    name="smoke-plus",
    base_model="Qwen/Qwen2.5-Coder-0.5B-Instruct",
    train_tasks=64,
    eval_tasks=32,
    min_test_functions=3,
    synthetic_seeds=8,
    synthetic_max_new_tokens=512,
    max_steps=60,
    num_generations=8,
    per_device_batch=8,
    max_completion_length=512,
    learning_rate=2e-5,
    lora_r=16,
    use_vllm=False,
    promotion_margin=-1.0,
    gpu="A10G:1",
)

DEV = FactoryProfile(
    name="dev",
    base_model="Qwen/Qwen2.5-Coder-1.5B-Instruct",
    train_tasks=2000,
    eval_tasks=200,
    min_test_functions=5,
    synthetic_seeds=64,
    synthetic_max_new_tokens=768,
    max_steps=100,
    num_generations=8,
    per_device_batch=4,
    max_completion_length=768,
    learning_rate=1e-5,
    lora_r=16,
    use_vllm=True,
    promotion_margin=0.0,
    gpu="A10G:1",
)

FULL = FactoryProfile(
    name="full",
    base_model="Qwen/Qwen2.5-Coder-1.5B-Instruct",
    train_tasks=10000,
    eval_tasks=500,
    min_test_functions=5,
    synthetic_seeds=256,
    synthetic_max_new_tokens=1024,
    max_steps=500,
    num_generations=8,
    per_device_batch=4,
    max_completion_length=1024,
    learning_rate=1e-5,
    lora_r=32,
    use_vllm=True,
    promotion_margin=0.01,
    gpu="A10G:1",
)

PROFILES: dict[str, FactoryProfile] = {p.name: p for p in (SMOKE, SMOKE_PLUS, DEV, FULL)}


def get_profile(name: str) -> FactoryProfile:
    try:
        return PROFILES[name]
    except KeyError:
        raise ValueError(f"unknown profile {name!r}; choose from {sorted(PROFILES)}")
