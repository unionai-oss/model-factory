"""Factory configuration: cluster sizing, model ladder, and RL profiles.

Follows basic-model-factory's pattern: everything cluster-specific lives in
a profile selected at deploy time, resolved values travel into containers
via env vars (child task specs are re-derived at runtime inside the parent's
container — see `cluster_env_vars`).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# ── cluster ─────────────────────────────────────────────────────────────
# demo.hosted pools (verified on the basic factory): A10G (training pool,
# scarce — ~1 spare), T4 (g4dn.xlarge serving pool, 3670m CPU / 14000Mi
# allocatable), plenty of CPU nodes. L4 never schedules.
#
# Training GPU: smallest that works. Qwen3.5-0.8B + LoRA + GRPO fits a T4's
# 16GB VRAM at fp16 with short sequences; T4 (Turing, sm75) has NO bf16, so
# the trainer selects dtype from the device. 2B+ or bf16 needs A10G.
APP_ORG = os.environ.get("RT_ORG", "demo")
APP_PROJECT = os.environ.get("RT_PROJECT", "resource-tuner-model-factory")
APP_DOMAIN = os.environ.get("RT_DOMAIN", "development")

TRAIN_GPU = os.environ.get("RT_TRAIN_GPU", "T4:1")
TRAIN_CPU = int(os.environ.get("RT_TRAIN_CPU", "2"))
TRAIN_MEMORY = os.environ.get("RT_TRAIN_MEMORY", "10Gi")
TRAIN_DISK = os.environ.get("RT_TRAIN_DISK", "50Gi")

CPU_TASK_CPU = int(os.environ.get("RT_CPU", "2"))
# 8Gi: the archetype release materializes a ~10^5-row corpus in pandas.
CPU_TASK_MEMORY = os.environ.get("RT_CPU_MEMORY", "8Gi")

# The episode harness env: the resource request is the WHOLE experiment, so
# these are only the env defaults — every episode overrides them with the
# policy's proposal.
HARNESS_TIMEOUT_S = int(os.environ.get("RT_HARNESS_TIMEOUT", "600"))

# Secrets on the cluster (project-scoped; all three exist in
# resource-tuner-model-factory/development as of 2026-09-03).
HF_TOKEN_SECRET = "HUGGINGFACE_TOKEN"
WANDB_SECRET = "WANDB_API_KEY"
# API key for the llm-service teacher apps' PUBLIC endpoints (the OIDC
# gateway accepts it as a bearer token). Mounted on the driver env
# unconditionally — the synthetic station needs it.
LLM_SERVICE_SECRET = "LLM_SERVICE_API_KEY"
# Default ON: all three secrets exist in this project; RT_USE_SECRETS=0
# remains the escape hatch for tenants without them.
USE_SECRETS = os.environ.get("RT_USE_SECRETS", "1") == "1"

WANDB_PROJECT = "resource-tuner-model-factory"


def cluster_env_vars() -> dict[str, str]:
    """Settings that must travel INTO containers (children re-derive specs)."""
    return {
        "RT_ORG": APP_ORG,
        "RT_PROJECT": APP_PROJECT,
        "RT_DOMAIN": APP_DOMAIN,
        "RT_TRAIN_GPU": TRAIN_GPU,
        "RT_TRAIN_CPU": str(TRAIN_CPU),
        "RT_TRAIN_MEMORY": TRAIN_MEMORY,
        "RT_TRAIN_DISK": TRAIN_DISK,
        "RT_CPU": str(CPU_TASK_CPU),
        "RT_CPU_MEMORY": CPU_TASK_MEMORY,
    }


def train_resources():
    import flyte

    return flyte.Resources(
        cpu=TRAIN_CPU, memory=TRAIN_MEMORY, gpu=TRAIN_GPU, disk=TRAIN_DISK, shm="auto"
    )


def cpu_resources():
    import flyte

    return flyte.Resources(cpu=CPU_TASK_CPU, memory=CPU_TASK_MEMORY)


# ── model ladder ────────────────────────────────────────────────────────
# Parameterized via RT_MODEL. Default is TEXT-ONLY Qwen3-1.7B (fastest
# rung that trains well). The Qwen3.5 multimodal-arch TRL blocker
# (trl#5269 / vllm#39993) no longer reproduces: probe run
# u2248fp4bgnpds44b24q (2026-09-04) trained Qwen3.5-4B QLoRA GRPO on a T4
# with real gradients — the *-qwen35 rungs are live.
DEFAULT_MODEL = os.environ.get("RT_MODEL", "Qwen/Qwen3-1.7B")
MODEL_LADDER: dict[str, str] = {
    # text-only tier: known-good with TRL GRPO today
    "xs": "Qwen/Qwen3-0.6B",
    "s": "Qwen/Qwen3-1.7B",
    "m": "Qwen/Qwen3-4B",
    # Qwen3.5 tier: blocked on trl#5269 / vllm#39993 for RL as of 2026-09
    "xs-qwen35": "Qwen/Qwen3.5-0.8B",
    "s-qwen35": "Qwen/Qwen3.5-2B",
    "m-qwen35": "Qwen/Qwen3.5-4B",
    "l-qwen35": "Qwen/Qwen3.5-9B",
}


# ── RL profiles ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TunerProfile:
    """Sizing knobs for one factory iteration."""

    name: str
    base_model: str
    # corpus
    train_contexts: int  # generated task contexts for training
    eval_contexts: int  # held-out contexts
    # rewards: "success" (stage A) or "composite" (stage B)
    reward_stage: str
    # GRPO
    max_steps: int
    num_generations: int
    per_device_batch: int
    max_completion_length: int
    learning_rate: float
    lora_r: int
    use_qlora: bool  # 4-bit base weights; needed above ~2B on small GPUs
    # episode mix: fraction of reward episodes run on the real cluster
    # (the rest are simulated). Prototype trains sim-only, evals on-cluster.
    cluster_episode_fraction: float
    # on-cluster validation episodes in eval
    eval_cluster_episodes: int
    # ── checkpointing (defaults off; long runs turn these on) ──
    # intra-task cadence: TRL saves full trainer state every N steps and
    # the flyte Checkpoint uploads it, so a retried attempt resumes
    # mid-run instead of restarting. 0 = no intra-task saves.
    save_steps: int = 0
    # every N steps, publish the adapter as a tuner-checkpoint-intermediate
    # artifact via a child task (Union-native lineage; resumable input for
    # a later train_tuner via resume_from_artifact). 0 = off.
    artifact_checkpoint_every: int = 0


SMOKE = TunerProfile(
    name="smoke",
    base_model=DEFAULT_MODEL,
    train_contexts=64,
    eval_contexts=32,
    reward_stage="success",
    max_steps=10,
    num_generations=8,
    # TRL requires generation_batch_size % num_generations == 0; keeping
    # batch == group size means each step trains on whole groups.
    per_device_batch=8,
    max_completion_length=128,
    learning_rate=1e-5,
    lora_r=16,
    use_qlora=False,
    cluster_episode_fraction=0.0,
    eval_cluster_episodes=4,
)

# Stage-A saturates immediately (the base model's generous proposals fit
# most tasks), so groups go all-pass and gradient vanishes. This rung turns
# on the composite reward at smoke scale: waste ranks the all-pass groups,
# restoring advantage variance — the "reward goes up" signal to watch here
# is waste_penalty shrinking while success holds.
SMOKE_COMPOSITE = TunerProfile(
    name="smoke-composite",
    base_model=DEFAULT_MODEL,
    train_contexts=64,
    eval_contexts=32,
    reward_stage="composite",
    max_steps=30,
    num_generations=8,
    per_device_batch=8,
    max_completion_length=128,
    learning_rate=1e-5,
    lora_r=16,
    use_qlora=False,
    cluster_episode_fraction=0.0,
    eval_cluster_episodes=6,
)

DEV = TunerProfile(
    name="dev",
    base_model=DEFAULT_MODEL,
    train_contexts=512,
    eval_contexts=128,
    reward_stage="composite",
    max_steps=150,
    num_generations=16,
    per_device_batch=16,
    max_completion_length=128,
    learning_rate=5e-6,
    lora_r=16,
    use_qlora=False,
    cluster_episode_fraction=0.0,
    eval_cluster_episodes=16,
)

FULL = TunerProfile(
    name="full",
    base_model=MODEL_LADDER["s"],
    train_contexts=4096,
    eval_contexts=512,
    reward_stage="composite",
    max_steps=500,
    num_generations=16,
    per_device_batch=16,
    max_completion_length=192,
    learning_rate=5e-6,
    lora_r=32,
    use_qlora=True,
    cluster_episode_fraction=0.05,
    eval_cluster_episodes=64,
)

# Reward-shaping experiment arms (round 7): identical dev-scale training,
# reward shape varied — the only free variable. The mutually exclusive
# waste_form is the per-experiment judgment; each named shape composes its
# own set of the composable knobs (see rewards/shaping.py SHAPES).
import dataclasses as _dc

_SHAPE_ARMS = ("c-linear", "c-log", "c-bucket", "c-cost")
_DEV_SHAPED = tuple(
    _dc.replace(DEV, name=f"dev-{stage}", reward_stage=stage) for stage in _SHAPE_ARMS
)

# Checkpointing smoke rung: 10 steps with aggressive save/publish cadence
# so the whole intra-task + artifact-checkpoint + resume surface can be
# exercised end-to-end in minutes.
SMOKE_CKPT = _dc.replace(
    SMOKE, name="smoke-ckpt", save_steps=4, artifact_checkpoint_every=5
)

# 5-step trainability probe for the Qwen3.5 tier (trl#5269 gate): cheap
# way to learn whether the multimodal-arch blocker still bites before
# committing a 3-day run to the model.
PROBE_QWEN35 = None  # defined after AMBITIOUS (needs _dc)

# The most ambitious rung: 0.5M-row corpus, ~3-day budget on a SINGLE T4
# (minimal GPU spend), 4B-class model via QLoRA. Qwen3.5-4B first; if the
# known TRL-multimodal blocker (trl#5269) still bites, the text-only
# Qwen3-4B is the 4B-class fallback.
AMBITIOUS = TunerProfile(
    name="ambitious",
    base_model=MODEL_LADDER["m-qwen35"],
    train_contexts=16384,
    eval_contexts=512,
    reward_stage="c-cost",  # round-7's best arm on the business metric
    max_steps=1200,
    num_generations=8,
    per_device_batch=8,
    max_completion_length=128,
    learning_rate=5e-6,
    lora_r=32,
    use_qlora=True,
    cluster_episode_fraction=0.0,
    eval_cluster_episodes=24,
    save_steps=25,
    artifact_checkpoint_every=100,
)

PROBE_QWEN35 = _dc.replace(
    SMOKE,
    name="probe-qwen35",
    base_model=MODEL_LADDER["m-qwen35"],
    use_qlora=True,
    max_steps=5,
    train_contexts=32,
)

# Round-8 arms: the 250k mixed corpus (archetypes + templates incl.
# single-T4 GPU tasks + prior/history context fields), 8x dev's contexts,
# 2x its steps. Same fixed train-subset seed across arms → controlled.
_R8_BASE = _dc.replace(DEV, train_contexts=4096, eval_contexts=256, max_steps=300)
_R8_SHAPED = tuple(
    _dc.replace(_R8_BASE, name=f"r8-{stage}", reward_stage=stage) for stage in _SHAPE_ARMS
)

PROFILES: dict[str, TunerProfile] = {
    p.name: p
    for p in (
        SMOKE, SMOKE_COMPOSITE, SMOKE_CKPT, DEV, FULL, AMBITIOUS, PROBE_QWEN35,
        *_DEV_SHAPED, *_R8_SHAPED,
    )
}


def get_profile(name: str) -> TunerProfile:
    try:
        return PROFILES[name]
    except KeyError:
        raise ValueError(f"unknown profile {name!r}; choose from {sorted(PROFILES)}")
