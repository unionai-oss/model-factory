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
    "m8": "Qwen/Qwen3-8B",
    "l": "Qwen/Qwen3-14B",
    # Biggest dense Qwen that full-fine-tunes on ONE node: 8x L40S
    # (g6e.48xlarge, 384GB VRAM) holds bf16 weights + grads + 8-bit Adam
    # (~6B/param ≈ 192GB) with room for activations + generation KV.
    "xl": "Qwen/Qwen3-32B",
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
    # Prompt budget, in TOKENS. Load-bearing twice over, which is why it is
    # a profile knob and not left to the library default:
    #
    # CORRECTNESS — TRL's GRPOConfig defaults this to 512 and truncates
    # LEFT. Our prompts run ~1,000-1,700 tokens once the round-13 corpora
    # put real library code in them, so the default was silently eating the
    # FRONT of every prompt: the system instructions and the task
    # description, keeping only the tail. A policy cannot learn a rule it
    # never sees.
    #
    # MEMORY — the GRPO backward pass holds a B x T x vocab logits tensor
    # AND its gradient, and Qwen3.5's vocab is 151,936. Prompt length is
    # therefore the dominant term in whether a rung fits its GPU. Pinning
    # it makes the requirement explicit instead of an accident of whatever
    # the corpus happens to contain.
    max_prompt_length: int = 1024
    # ── GPU sizing ──
    # The accelerator this rung NEEDS, as a flyte.Resources gpu string.
    # Recorded per profile because it is a property of the arm, not of the
    # deployment: "ambitious" was written as a T4 rung when it meant a
    # 1.7B model, and quietly became wrong when it moved to 4B-class with a
    # 151,936-token vocab. The trainer env still takes its actual request
    # from RT_TRAIN_GPU at deploy time, but `preflight_vram` checks the GPU
    # it actually landed on against what the knobs require, so a mismatch
    # fails in seconds with the numbers rather than OOMing in the backward
    # pass an hour later.
    train_gpu: str = "T4:1"
    train_memory: str = "10Gi"
    # ── checkpointing (defaults off; long runs turn these on) ──
    # intra-task cadence: TRL saves full trainer state every N steps and
    # the flyte Checkpoint uploads it, so a retried attempt resumes
    # mid-run instead of restarting. 0 = no intra-task saves.
    save_steps: int = 0
    # every N steps, publish the adapter as a tuner-checkpoint-intermediate
    # artifact via a child task (Union-native lineage; resumable input for
    # a later train_tuner via resume_from_artifact). 0 = off.
    artifact_checkpoint_every: int = 0
    # ── round-11 arms ──
    # extend LoRA to the MLP projections (gate/up/down_proj) — the honest
    # capacity test before anything more radical.
    lora_mlp: bool = False
    # False = FULL fine-tune (no adapters; 8-bit Adam; small rungs only —
    # a 0.6B fits a T4 at ~5-6GB with grad checkpointing).
    use_lora: bool = True
    # GBT-hint composition: quantile-GBT estimates ride in the prompt
    # (out-of-fold at train time) AND the baseline_relative reward term
    # references the GBT's cost — the policy is paid for beating the
    # strongest classical estimator, not the family-median strawman.
    gbt_hint: bool = False


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

# The most ambitious rung: ~3-day budget, 4B-class model via QLoRA.
# Qwen3.5-4B first; if the known TRL-multimodal blocker (trl#5269) still
# bites, the text-only Qwen3-4B is the 4B-class fallback.
#
# GPU: L40S, not T4 (changed 2026-09-11 after run u5cqz99dmhgmkllmn9w4
# OOMed in `accelerator.backward()`). This rung was written as "minimal GPU
# spend on a single T4" when the arm was a 1.7B model. Two things broke
# that: the model is now 4B-class with a 151,936-token vocab, and fixing
# the silent 512-token prompt truncation (below) triples the sequence the
# logits tensor is built over. Sized by `vram_estimate_gib`: batch 8 x
# 1,664 tokens x 151,936 vocab in fp32 needs ~34 GiB for logits + gradient
# + loss temporaries, which clears neither the T4 (14.7) nor the L4 (24).
#
# The cheaper alternative was halving num_generations and per_device_batch
# to fit an L4 at $0.512/hr, but num_generations IS the GRPO group size —
# cutting it to 4 weakens every advantage estimate and stops this being
# the same arm as the previous ambitious run. Paying $1.634/hr to keep the
# arm intact is the better trade for a comparison experiment.
AMBITIOUS = TunerProfile(
    name="ambitious",
    base_model=MODEL_LADDER["m-qwen35"],
    train_contexts=16384,
    eval_contexts=512,
    reward_stage="c-cost",  # round-7's best arm on the business metric
    max_steps=1200,
    # Group size 4, not 8 — forced by the VRAM arithmetic once the prompt
    # truncation was fixed, and the preflight on run uzd8ktrh2qqjlqgzgqfx
    # measured the real cost: Qwen3.5's tokenizer is ~248k tokens (NOT the
    # 151,936 of Qwen3, which is what I first assumed), so batch 8 x 1,664
    # tokens needs ~55 GiB and does not fit even one L40S's 44.
    #
    # Given the choice between a shorter prompt and a smaller group, the
    # group loses: a left-truncated prompt is CORRUPT input — it drops the
    # system instructions entirely — whereas 4 completions per group is a
    # perfectly ordinary GRPO configuration, just a noisier advantage
    # estimate. Never trade data integrity for batch size.
    num_generations=4,
    per_device_batch=4,
    max_completion_length=128,
    # Round-13/14 corpora carry real library code, so prompts run past the
    # 512 TRL would have silently left-truncated to. 1536 keeps the whole
    # prompt for the overwhelming majority of rows.
    max_prompt_length=1536,
    train_gpu="L40s:1",
    train_memory="32Gi",
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

# Round-11 arms: r8 scale (4096 ctx / 300 steps, c-cost reward — the
# round-7/8 winner) so results drop straight into the standing comparison
# table. One variable each:
_R11_BASE = _dc.replace(_R8_BASE, reward_stage="c-cost")
R11_R64 = _dc.replace(_R11_BASE, name="r11-r64-mlp", lora_r=64, lora_mlp=True)
R11_GBT = _dc.replace(_R11_BASE, name="r11-gbt-hint", gbt_hint=True)
R11_FULLFT = _dc.replace(
    _R11_BASE,
    name="r11-fullft-06b",
    base_model=MODEL_LADDER["xs"],  # Qwen3-0.6B — full FT fits a T4
    use_lora=False,
    use_qlora=False,
)
# The expanded-budget redo: full FT of the biggest Qwen that fits a
# RELIABLY-provisionable node — Qwen3-14B on g6e.12xlarge (L40s:4,
# 192GB VRAM; ~84GB of bf16 weights+grads+8-bit-Adam states). The 32B/
# 8-GPU variant needs g6e.48xlarge, which rarely provisions. Naive
# model-parallel via device_map — single-node as specified. Intra-task
# saves are ~56GB tarballs, so cadence stays at every 100 steps and
# intermediate ARTIFACTS stay off (the final checkpoint is the artifact).
# Bottom provisioning rung that RELIABLY schedules: Qwen3-4B full FT on
# ONE L40S (g6e.2xlarge — the pool llm-service cold-starts routinely).
# ~24GB training state in 48GB VRAM, and no naive-MP tax: single GPU.
R11_FULLFT_4B = _dc.replace(
    _R11_BASE,
    name="r11-fullft-4b",
    base_model=MODEL_LADDER["m"],
    use_lora=False,
    use_qlora=False,
    learning_rate=1e-6,
    num_generations=8,
    per_device_batch=8,
    save_steps=100,
    artifact_checkpoint_every=0,
)

# Provisioning-ladder fallback: Qwen3-8B on g6.12xlarge (L4:4, 96GB —
# ~48GB of full-FT states). Same recipe one rung down; the 32B/L40s:8
# and 14B/L40s:4 configs stay on the ladder for when big nodes provision.
R11_FULLFT_8B = _dc.replace(
    _R11_BASE,
    name="r11-fullft-8b",
    base_model=MODEL_LADDER["m8"],
    use_lora=False,
    use_qlora=False,
    learning_rate=1e-6,
    num_generations=8,
    per_device_batch=8,
    save_steps=100,
    artifact_checkpoint_every=0,
)

R11_FULLFT_14B = _dc.replace(
    _R11_BASE,
    name="r11-fullft-14b",
    base_model=MODEL_LADDER["l"],
    use_lora=False,
    use_qlora=False,
    learning_rate=1e-6,  # full-FT RL wants a gentler lr than LoRA
    num_generations=8,
    per_device_batch=8,
    save_steps=100,
    artifact_checkpoint_every=0,
)

PROFILES: dict[str, TunerProfile] = {
    p.name: p
    for p in (
        SMOKE, SMOKE_COMPOSITE, SMOKE_CKPT, DEV, FULL, AMBITIOUS, PROBE_QWEN35,
        *_DEV_SHAPED, *_R8_SHAPED, R11_R64, R11_GBT, R11_FULLFT, R11_FULLFT_4B,
        R11_FULLFT_8B, R11_FULLFT_14B,
    )
}


def get_profile(name: str) -> TunerProfile:
    try:
        return PROFILES[name]
    except KeyError:
        raise ValueError(f"unknown profile {name!r}; choose from {sorted(PROFILES)}")
