"""Model suite, sizing profiles, and cluster settings.

Two independent axes:

- **The suite** (`MODEL_SUITE`): which open-weights models get fine-tuned and
  compared. Each entry is a short slug, because the slug becomes a *partition
  value* in the factory and a HF repo id (`Qwen/Qwen2.5-0.5B-Instruct`) has a
  `/` in it, which a partition value cannot carry.
- **The profile** (`PROFILES`): how big the run is — episode counts, epochs,
  which slice of the suite to train.

Model choice. The default suite is small, ungated decoder-only instruct
models, verified ungated on the Hub (2026-10-02): a model that needs a
licence acceptance makes the factory fail for anyone without an
`HUGGINGFACE_TOKEN`, and the point here is a loop that runs end to end
unattended. The suite deliberately spans an order of magnitude in size
(135M -> 1.5B) so the scorecard shows a real capability gradient on the same
task rather than six variations of one model.

`gemma-3-270m-it` and `Llama-3.2-1B-Instruct` are both `gated="manual"` on the
Hub, so they are listed in `GATED_MODELS` as opt-in rather than defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# ── cluster settings ────────────────────────────────────────────────────
# Same shape as the sibling factories: everything tenant-specific lives here
# so a different cluster is one env var, not a code edit.

APP_PROJECT = os.environ.get("DF_PROJECT", "model-factory")
APP_DOMAIN = os.environ.get("DF_DOMAIN", "development")

#: Accelerator for fine-tuning and eval. The demo tenant's training pool is
#: A10G; the small serving pool is T4 (see the sibling factory's notes — an
#: app parked on the single spare A10G starves every GPU stage).
GPU = os.environ.get("DF_GPU", "A10G:1")

#: Serving is CPU by default, and that is a design choice rather than a
#: concession. The champion is a sub-1B model emitting one ~30-token JSON
#: object per request, which is a second or two on CPU — while a GPU app holds
#: a whole accelerator idle between requests. On the demo tenant that is not
#: hypothetical: the sibling factory's `mf-inference` app occupies the single
#: T4 serving node, and a second GPU app simply never schedules
#: ("node(s) had untolerated taint(s)", run df-serve-r1).
#:
#: Set DF_SERVING_GPU to an accelerator (e.g. "T4:1") to serve on GPU anyway.
SERVING_GPU = os.environ.get("DF_SERVING_GPU", "")

GPU_CPU = int(os.environ.get("DF_GPU_CPU", "6"))
GPU_MEMORY = os.environ.get("DF_GPU_MEMORY", "24Gi")  # host memory, not VRAM
GPU_DISK = os.environ.get("DF_GPU_DISK", "100Gi")

CPU = int(os.environ.get("DF_CPU", "2"))
CPU_MEMORY = os.environ.get("DF_CPU_MEMORY", "4Gi")

# Sized for g4dn.xlarge (3670m CPU / 14000Mi allocatable), the T4 instance.
SERVING_CPU = int(os.environ.get("DF_SERVING_CPU", "2"))
SERVING_MEMORY = os.environ.get("DF_SERVING_MEMORY", "10Gi")
SERVING_DISK = os.environ.get("DF_SERVING_DISK", "50Gi")

#: Apps are publicly reachable on the demo tenant; orgs that set
#: `app.disallow_anonymous` need this on.
REQUIRE_APP_AUTH = os.environ.get("DF_REQUIRE_AUTH", "0") == "1"


def env_vars() -> dict[str, str]:
    """Settings that must travel INTO the container.

    Child task specs are re-derived at runtime inside the parent's container,
    so that container has to resolve the same settings the deploy did.
    Resolved values (not just names) are propagated so per-field overrides
    used at deploy time reach the container too.
    """
    return {
        "DF_PROJECT": APP_PROJECT,
        "DF_DOMAIN": APP_DOMAIN,
        "DF_GPU": GPU,
        "DF_SERVING_GPU": SERVING_GPU,
        "DF_GPU_CPU": str(GPU_CPU),
        "DF_GPU_MEMORY": GPU_MEMORY,
        "DF_GPU_DISK": GPU_DISK,
        "DF_CPU": str(CPU),
        "DF_CPU_MEMORY": CPU_MEMORY,
        "DF_SERVING_CPU": str(SERVING_CPU),
        "DF_SERVING_MEMORY": SERVING_MEMORY,
        "DF_SERVING_DISK": SERVING_DISK,
    }


def gpu_resources():
    import flyte

    return flyte.Resources(
        cpu=GPU_CPU, memory=GPU_MEMORY, gpu=GPU, disk=GPU_DISK, shm="auto"
    )


def cpu_resources():
    import flyte

    return flyte.Resources(cpu=CPU, memory=CPU_MEMORY)


def serving_resources():
    """Sized for the serving pool.

    No `gpu=` unless DF_SERVING_GPU is set: a GPU request pins the app to an
    accelerator pool, and on a tenant whose serving pool is one node that
    means the second app to ask never schedules. Passing `gpu=""` would be a
    malformed request, so the key is omitted entirely instead.
    """
    import flyte

    kwargs = dict(cpu=SERVING_CPU, memory=SERVING_MEMORY, disk=SERVING_DISK, shm="auto")
    if SERVING_GPU:
        kwargs["gpu"] = SERVING_GPU
    return flyte.Resources(**kwargs)


# ── the model suite ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class Candidate:
    """One model in the suite."""

    #: Short, partition-safe key. Becomes the `model` partition value, so it
    #: must be lowercase alphanumerics, '-', '_', '.' — never '/'.
    slug: str
    hf_id: str
    #: Rough parameter count, for the scorecard's size-vs-accuracy column.
    params_m: int
    #: True when the Hub requires a manual licence acceptance (needs a token).
    gated: bool = False


SMOLLM2_135M = Candidate("smollm2-135m", "HuggingFaceTB/SmolLM2-135M-Instruct", 135)
SMOLLM2_360M = Candidate("smollm2-360m", "HuggingFaceTB/SmolLM2-360M-Instruct", 360)
QWEN25_05B = Candidate("qwen2.5-0.5b", "Qwen/Qwen2.5-0.5B-Instruct", 494)
QWEN3_06B = Candidate("qwen3-0.6b", "Qwen/Qwen3-0.6B", 596)
QWEN25_15B = Candidate("qwen2.5-1.5b", "Qwen/Qwen2.5-1.5B-Instruct", 1540)

#: Every model the factory knows how to train, keyed by partition value.
MODEL_SUITE: dict[str, Candidate] = {
    c.slug: c
    for c in (SMOLLM2_135M, SMOLLM2_360M, QWEN25_05B, QWEN3_06B, QWEN25_15B)
}

#: Opt-in only: these need an accepted licence plus HUGGINGFACE_TOKEN.
GATED_MODELS: dict[str, Candidate] = {
    c.slug: c
    for c in (
        Candidate("gemma3-270m", "google/gemma-3-270m-it", 270, gated=True),
        Candidate("llama3.2-1b", "meta-llama/Llama-3.2-1B-Instruct", 1240, gated=True),
    )
}


def get_candidate(slug: str) -> Candidate:
    """The suite entry for a partition value."""
    if slug in MODEL_SUITE:
        return MODEL_SUITE[slug]
    if slug in GATED_MODELS:
        return GATED_MODELS[slug]
    raise ValueError(
        f"unknown model {slug!r}; the suite is {sorted(MODEL_SUITE)} "
        f"(gated, opt-in: {sorted(GATED_MODELS)})"
    )


# ── sizing profiles ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class Profile:
    """Sizing knobs for one pass through the factory."""

    name: str
    #: Which suite members to train and compare.
    models: tuple[str, ...]
    train_episodes: int
    eval_episodes: int
    epochs: float
    per_device_batch: int
    grad_accum: int
    learning_rate: float
    lora_r: int
    max_seq_length: int
    max_new_tokens: int
    #: Champion selection floor: a model must beat this exact-accuracy to be
    #: promoted at all, so a failed suite deploys nothing rather than the
    #: least-bad thing.
    min_exact_accuracy: float
    #: Also score each base model untuned, to show what fine-tuning bought.
    score_base_model: bool

    @property
    def candidates(self) -> list[Candidate]:
        return [get_candidate(s) for s in self.models]


#: Minutes end to end: two tiny models, a few hundred episodes, one epoch.
SMOKE = Profile(
    name="smoke",
    models=("smollm2-135m", "qwen2.5-0.5b"),
    train_episodes=600,
    eval_episodes=120,
    epochs=1.0,
    per_device_batch=8,
    grad_accum=2,
    learning_rate=2e-4,
    lora_r=16,
    max_seq_length=1024,
    max_new_tokens=64,
    # Smoke proves the loop runs; it must not block on quality. The majority
    # baseline is ~20% on this mix, so 0 means "deploy whatever won".
    min_exact_accuracy=0.0,
    score_base_model=True,
)

DEV = Profile(
    name="dev",
    models=("smollm2-135m", "smollm2-360m", "qwen2.5-0.5b", "qwen3-0.6b"),
    train_episodes=4000,
    eval_episodes=600,
    epochs=2.0,
    per_device_batch=8,
    grad_accum=2,
    learning_rate=1e-4,
    lora_r=16,
    max_seq_length=1024,
    max_new_tokens=64,
    min_exact_accuracy=0.60,
    score_base_model=True,
)

FULL = Profile(
    name="full",
    models=("smollm2-135m", "smollm2-360m", "qwen2.5-0.5b", "qwen3-0.6b", "qwen2.5-1.5b"),
    train_episodes=20000,
    eval_episodes=2000,
    epochs=3.0,
    per_device_batch=4,
    grad_accum=4,
    learning_rate=5e-5,
    lora_r=32,
    max_seq_length=1024,
    max_new_tokens=64,
    min_exact_accuracy=0.85,
    score_base_model=True,
)

PROFILES: dict[str, Profile] = {p.name: p for p in (SMOKE, DEV, FULL)}

#: Which profile the factory graph is declared at. The suite a factory trains
#: is part of its *structure* (one build instance per model partition value),
#: so changing this changes the graph and needs a redeploy — unlike the
#: per-build constants, which `--param` can override per materialization.
DEFAULT_PROFILE = os.environ.get("DF_PROFILE", "smoke")


def get_profile(name: str) -> Profile:
    try:
        return PROFILES[name]
    except KeyError:
        raise ValueError(f"unknown profile {name!r}; choose from {sorted(PROFILES)}")
