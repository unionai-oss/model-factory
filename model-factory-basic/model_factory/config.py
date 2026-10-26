"""Factory configuration: profiles and secret wiring.

Profiles size the whole loop (model, dataset slice, GRPO steps) so the same
pipeline runs as a minutes-long smoke test or a multi-hour training run.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace

# Secret names on the Union cluster, scoped to the model-factory project
# (created in demo/development; see TODO.md for creation commands).
HF_TOKEN_SECRET = "HUGGINGFACE_TOKEN"
WANDB_SECRET = "WANDB_API_KEY"

# Secrets are only attached to task environments when this env var is set at
# deploy/run time (from the machine that bundles the code). Flyte fails to
# schedule a task whose declared secret doesn't exist on the cluster, and not
# every tenant has them — everything the POC needs (KodCode, Qwen models) is
# public, so the loop still runs without them (W&B disabled). CI deploys to
# the demo tenant, where they exist, with MF_USE_SECRETS=1.
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
    # Serving-app host sizing, kept separate from the GPU *task* sizing: the
    # serving pool is deliberately a different (smaller) instance type, and a
    # request sized for a training node simply never schedules on it.
    inference_task_cpu: int
    inference_task_memory: str
    inference_task_disk: str
    cpu_task_cpu: int  # host CPUs for CPU-bound envs
    cpu_task_memory: str
    requires_app_auth: bool  # False = publicly reachable apps


# Serving must sit on a DIFFERENT accelerator pool from training/eval, or the
# two starve each other: this tenant has ~one spare A10G, and an app parked on
# it blocks every GPU stage of the factory (a scaled-up app held the only A10G
# for 20+ minutes while synthetic generation sat in waiting_for_resources).
# The original L4 choice had the right shape but the wrong device — L4 revisions
# never schedule here ("0/30 nodes are available: ... didn't match Pod's node
# affinity/selector"). T4 is the small serving pool that does.
DEMO_CLUSTER = ClusterProfile(
    name="demo",
    org="demo",
    gpu="A10G:1",
    inference_gpu="T4:1",
    gpu_task_cpu=6,
    gpu_task_memory="24Gi",
    gpu_task_disk="100Gi",
    # Sized for g4dn.xlarge (3670m CPU / 14000Mi allocatable), the T4 instance.
    inference_task_cpu=2,
    inference_task_memory="10Gi",
    inference_task_disk="50Gi",
    cpu_task_cpu=2,
    cpu_task_memory="4Gi",
    requires_app_auth=False,
)

# playground.canary has no A10G or L4. Its GPU pools are V100
# (p3.8xlarge / p3.16xlarge) and T4 (g4dn.xlarge). On-demand V100 capacity
# is scarce here — `V100:4` needs a whole free p3 node and sat queued for
# 90+ minutes with the node group at max size — so target the much smaller
# g4dn.xlarge (1x T4, 3670m CPU / 14000Mi allocatable) instead. 2 CPU /
# 10Gi fits inside that node's allocatable with room for DaemonSets.
#
# The org also sets `app.disallow_anonymous`, so apps need auth. CPU envs at
# 12 / 24Gi target the c5.4xlarge pool (15640m / 26900Mi).
PLAYGROUND_CLUSTER = ClusterProfile(
    name="playground",
    org="playground",
    gpu="T4:1",
    inference_gpu="T4:1",
    gpu_task_cpu=2,
    gpu_task_memory="10Gi",
    gpu_task_disk="100Gi",
    inference_task_cpu=2,
    inference_task_memory="10Gi",
    inference_task_disk="50Gi",
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
INFERENCE_CPU = int(os.environ.get("MF_INFERENCE_CPU", str(CLUSTER.inference_task_cpu)))
INFERENCE_MEMORY = os.environ.get("MF_INFERENCE_MEMORY", CLUSTER.inference_task_memory)
INFERENCE_DISK = os.environ.get("MF_INFERENCE_DISK", CLUSTER.inference_task_disk)


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
        "MF_INFERENCE_CPU": str(INFERENCE_CPU),
        "MF_INFERENCE_MEMORY": INFERENCE_MEMORY,
        "MF_INFERENCE_DISK": INFERENCE_DISK,
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


def inference_resources():
    """`flyte.Resources` for the serving app, sized for the serving pool.

    Separate from `gpu_resources` because serving runs on a smaller instance
    type than training; asking a T4 node for a training node's CPU/memory is
    an unschedulable request, not a merely generous one.
    """
    import flyte

    return flyte.Resources(
        cpu=INFERENCE_CPU,
        memory=INFERENCE_MEMORY,
        gpu=INFERENCE_GPU,
        disk=INFERENCE_DISK,
        shm="auto",
    )


def cpu_resources():
    """`flyte.Resources` for a CPU-bound task env, sized for the cluster."""
    import flyte

    # Disk for full-weight model copies (promotion copies a ~1GB checkpoint).
    return flyte.Resources(cpu=CPU_TASK_CPU, memory=CPU_TASK_MEMORY, disk="20Gi")


WANDB_PROJECT = "model-factory"

SEED_DATASET = "KodCode/KodCode-Light-RL-10K"

# Artifact names live in model_factory.contracts — the inter-team interface.


@dataclass(frozen=True)
class FactoryProfile:
    """Sizing knobs for one factory day."""

    name: str
    base_model: str  # the policy being trained (the "student")
    # data
    eval_tasks: int  # size of the FIXED held-out split (built once, reused daily)
    heldout_pool: int  # seed-dataset rows reserved for held-out; training never reads them
    train_tasks: int  # fresh curated tasks per day, from a window that rotates by date
    min_test_functions: int  # curation floor (anti reward-hacking)
    # synthetic generation (failure-driven curriculum)
    synthetic_model: str  # the "teacher" that mutates seed problems
    synthetic_seeds: int  # problems to mutate per day
    synthetic_max_new_tokens: int
    # GRPO
    max_steps: int
    num_generations: int  # completions per prompt (the GRPO group)
    per_device_batch: int  # completions per device micro-batch
    grad_accum: int
    max_completion_length: int
    learning_rate: float
    lora_r: int
    # eval gate: candidate pass@1 must beat the CHAMPION's by this margin
    promotion_margin: float
    eval_batch: int = 32
    #: Serve the synthetic-task teacher from an OpenAI-compatible app instead of
    #: loading ``synthetic_model`` on a GPU: ``"<app>@<project>"``. None = local.
    teacher_app: str | None = None
    teacher_max_tokens: int = 2048
    #: TRL colocated vLLM rollouts (needs the trainer's vLLM image).
    use_vllm: bool = False
    vllm_gpu_memory_utilization: float = 0.3

    @property
    def wandb_enabled(self) -> bool:
        return USE_SECRETS

    @property
    def prompts_per_step(self) -> int:
        return self.per_device_batch * self.grad_accum // self.num_generations


#: Machinery check: tiny everything, minutes end to end. Its margin is -1 so the
#: gate always passes — a smoke run proves the plumbing, never the quality.
SMOKE = FactoryProfile(
    name="smoke",
    base_model="Qwen/Qwen2.5-Coder-0.5B-Instruct",
    eval_tasks=16,
    heldout_pool=60,
    train_tasks=32,
    min_test_functions=3,
    synthetic_model="Qwen/Qwen2.5-Coder-0.5B-Instruct",
    synthetic_seeds=8,
    synthetic_max_new_tokens=512,
    max_steps=4,
    num_generations=4,
    per_device_batch=8,
    grad_accum=1,
    max_completion_length=384,
    learning_rate=2e-5,
    lora_r=16,
    promotion_margin=-1.0,
    use_vllm=True,
)

#: Round 2's day: 200 held-out tasks, HF-generate rollouts, one LoRA continued
#: across days.
#:
#: A profile NAME is part of every build's cache key; its contents are not
#: (the factory passes `profile_name="daily"` as a constant). So a profile is
#: immutable once a day has been built with it: new settings get a new name,
#: or the factory silently reuses artifacts built under the old ones.
DAILY_R2 = FactoryProfile(
    name="daily",
    base_model="Qwen/Qwen2.5-Coder-0.5B-Instruct",
    eval_tasks=200,
    heldout_pool=600,
    train_tasks=256,
    min_test_functions=3,
    synthetic_model="Qwen/Qwen2.5-Coder-1.5B-Instruct",
    synthetic_seeds=48,
    synthetic_max_new_tokens=768,
    max_steps=50,
    num_generations=8,
    per_device_batch=16,
    grad_accum=2,
    max_completion_length=512,
    learning_rate=3e-5,
    lora_r=16,
    promotion_margin=0.01,
)

#: The real day (round 3). 500 held-out tasks bring the paired 95% CI on a
#: day-over-day delta down to roughly +/-3 points; vLLM rollouts buy ~3x the
#: GRPO steps of round 2 in the same hour; a 27B teacher writes the curriculum.
DAILY = FactoryProfile(
    name="daily-r3",
    base_model="Qwen/Qwen2.5-Coder-0.5B-Instruct",
    eval_tasks=500,
    heldout_pool=2000,
    train_tasks=384,
    min_test_functions=3,
    synthetic_model="Qwen/Qwen2.5-Coder-7B-Instruct",
    synthetic_seeds=64,
    synthetic_max_new_tokens=768,
    max_steps=80,
    num_generations=8,
    per_device_batch=32,
    grad_accum=2,
    max_completion_length=512,
    learning_rate=2e-5,
    lora_r=16,
    promotion_margin=0.01,
    use_vllm=True,
)

#: `daily`'s batch shape for a handful of steps: measures step time (and that
#: vLLM rollouts work at all) without spending a day. Run train_grpo directly.
PROBE = replace(DAILY, name="probe-r3", max_steps=6)

PROFILES: dict[str, FactoryProfile] = {p.name: p for p in (SMOKE, DAILY_R2, DAILY, PROBE)}


def get_profile(name: str) -> FactoryProfile:
    try:
        return PROFILES[name]
    except KeyError:
        raise ValueError(f"unknown profile {name!r}; choose from {sorted(PROFILES)}")
