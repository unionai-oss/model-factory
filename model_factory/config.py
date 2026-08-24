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

# ── cluster profiles ────────────────────────────────────────────────────
# Tenants differ in node pools and org policy, so the parts of a deployment
# that are cluster-specific — accelerator, task sizing, whether apps may be
# anonymous — live here rather than being hardcoded in each env. Pick one at
# deploy (bundle) time with MF_CLUSTER; individual fields can still be
# overridden with the MF_* vars below for one-off experiments.
#
# Note on memory: `flyte.Resources(memory=...)` is HOST (CPU) memory. VRAM is
# not requested directly — it follows from the accelerator named in `gpu`,
# e.g. "V100:4" asks for four V100s and gets their VRAM with them.


@dataclass(frozen=True)
class ClusterProfile:
    """Deployment settings for one Union tenant."""

    name: str
    org: str
    gpu: str  # accelerator for training / synthetic-gen / eval, e.g. "V100:4"
    inference_gpu: str  # accelerator for the serving app
    gpu_task_cpu: int  # host CPUs for GPU envs
    gpu_task_memory: str  # host memory for GPU envs (not VRAM)
    gpu_task_disk: str
    cpu_task_cpu: int  # host CPUs for CPU-bound envs
    cpu_task_memory: str
    requires_app_auth: bool  # False = publicly reachable apps


DEMO_CLUSTER = ClusterProfile(
    name="demo",
    org="demo",
    gpu="A10G:1",
    inference_gpu="L4:1",
    gpu_task_cpu=6,
    gpu_task_memory="24Gi",
    gpu_task_disk="100Gi",
    cpu_task_cpu=2,
    cpu_task_memory="4Gi",
    requires_app_auth=False,
)

# playground.canary has no A10G or L4 — its GPU pools are V100 (p3.8xlarge,
# 4 GPUs / 31600m CPU / 227700Mi; p3.16xlarge, 8 GPUs) and T4 (g4dn.xlarge,
# tainted NoSchedule). "V100:4" takes a whole p3.8xlarge, and 24 CPU /
# 200Gi sits inside that node's allocatable. The org also sets
# `app.disallow_anonymous`, so apps need auth. CPU envs at 12 / 24Gi target
# the c5.4xlarge pool (15640m / 26900Mi) — too big for t3a.xlarge.
PLAYGROUND_CLUSTER = ClusterProfile(
    name="playground",
    org="playground",
    gpu="V100:4",
    inference_gpu="V100:4",
    gpu_task_cpu=24,
    gpu_task_memory="100Gi",
    gpu_task_disk="100Gi",
    cpu_task_cpu=12,
    cpu_task_memory="24Gi",
    requires_app_auth=True,
)

CLUSTERS: dict[str, ClusterProfile] = {c.name: c for c in (DEMO_CLUSTER, PLAYGROUND_CLUSTER)}


def get_cluster(name: str) -> ClusterProfile:
    try:
        return CLUSTERS[name]
    except KeyError:
        raise ValueError(f"unknown cluster {name!r}; choose from {sorted(CLUSTERS)}")


CLUSTER = get_cluster(os.environ.get("MF_CLUSTER", "demo"))

# Per-field escape hatches; default to the selected cluster profile.
REQUIRE_APP_AUTH = os.environ.get("MF_REQUIRE_AUTH", "1" if CLUSTER.requires_app_auth else "0") == "1"
APP_ORG = os.environ.get("MF_ORG", CLUSTER.org)
APP_PROJECT = os.environ.get("MF_PROJECT", "model-factory")
APP_DOMAIN = os.environ.get("MF_DOMAIN", "development")

GPU = os.environ.get("MF_GPU", CLUSTER.gpu)
INFERENCE_GPU = os.environ.get("MF_INFERENCE_GPU", CLUSTER.inference_gpu)
GPU_TASK_CPU = int(os.environ.get("MF_GPU_CPU", str(CLUSTER.gpu_task_cpu)))
GPU_TASK_MEMORY = os.environ.get("MF_GPU_MEMORY", CLUSTER.gpu_task_memory)
GPU_TASK_DISK = os.environ.get("MF_GPU_DISK", CLUSTER.gpu_task_disk)
CPU_TASK_CPU = int(os.environ.get("MF_CPU", str(CLUSTER.cpu_task_cpu)))
CPU_TASK_MEMORY = os.environ.get("MF_CPU_MEMORY", CLUSTER.cpu_task_memory)


def cluster_env_vars() -> dict[str, str]:
    """Settings that must travel INTO the container.

    Child task specs are re-derived at runtime inside the parent task's
    container, so that container has to resolve the same cluster profile the
    deploy did. Without this the children silently fall back to the default
    profile while the locally-launched parent gets the right one — the task
    queues forever on an accelerator the tenant doesn't have.

    Resolved values (not just the profile name) are propagated so per-field
    MF_* overrides used at deploy time reach the container too.
    """
    return {
        "MF_CLUSTER": CLUSTER.name,
        "MF_GPU": GPU,
        "MF_INFERENCE_GPU": INFERENCE_GPU,
        "MF_GPU_CPU": str(GPU_TASK_CPU),
        "MF_GPU_MEMORY": GPU_TASK_MEMORY,
        "MF_GPU_DISK": GPU_TASK_DISK,
        "MF_CPU": str(CPU_TASK_CPU),
        "MF_CPU_MEMORY": CPU_TASK_MEMORY,
    }


def gpu_resources(gpu: str | None = None):
    """`flyte.Resources` for a GPU task env, sized for the target cluster.

    ``memory`` is host memory; VRAM comes with the accelerator named in
    ``gpu``.
    """
    import flyte

    return flyte.Resources(
        cpu=GPU_TASK_CPU,
        memory=GPU_TASK_MEMORY,
        gpu=gpu or GPU,
        disk=GPU_TASK_DISK,
        shm="auto",
    )


def cpu_resources():
    """`flyte.Resources` for a CPU-bound task env, sized for the cluster."""
    import flyte

    return flyte.Resources(cpu=CPU_TASK_CPU, memory=CPU_TASK_MEMORY)


WANDB_PROJECT = "model-factory"

SEED_DATASET = "KodCode/KodCode-Light-RL-10K"

# Artifact names live in model_factory.contracts — the inter-team interface.


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
