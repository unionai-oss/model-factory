"""TaskEnvironments and images — the single source of infra config.

Three tiers:

- ``cpu_env``    — data ingestion/curation, reward verification, eval scoring,
                   promotion. Sandboxed test execution happens here too (it
                   only needs pytest).
- ``gpu_env``    — GRPO training + synthetic generation + eval generation.
- ``factory_env``— the lightweight driver that chains stations and holds the
                   human-in-the-loop condition gates.

Secrets are attached only when MF_USE_SECRETS=1 at bundle time (see
config.py — the cluster refuses to schedule tasks whose secrets don't exist).
"""

from __future__ import annotations

import flyte

from .config import HF_TOKEN_SECRET, USE_SECRETS, WANDB_SECRET

_PYTHON = (3, 12)


def _secrets() -> list[flyte.Secret]:
    if not USE_SECRETS:
        return []
    return [
        flyte.Secret(key=HF_TOKEN_SECRET, as_env_var="HF_TOKEN"),
        flyte.Secret(key=WANDB_SECRET, as_env_var="WANDB_API_KEY"),
    ]


cpu_image = (
    flyte.Image.from_debian_base(name="mf-cpu", python_version=_PYTHON)
    .with_pip_packages(
        "flyte>=2.6.0",
        "pandas>=2.2",
        "pyarrow>=17",
        "datasets>=3.2",
        "huggingface_hub>=0.27",
        "pytest>=8",
    )
)

# Torch/TRL stack. vLLM intentionally absent from the POC image: the smoke
# profile uses transformers generation (use_vllm=False) to keep the first
# E2E runs robust; flip to a vllm-enabled image alongside profile "dev".
gpu_image = (
    flyte.Image.from_debian_base(name="mf-gpu", python_version=_PYTHON)
    .with_apt_packages("git")
    .with_pip_packages(
        "flyte>=2.6.0",
        "torch>=2.6",
        "transformers>=4.57",
        "trl>=1.10",
        "peft>=0.17",
        "accelerate>=1.7",
        "datasets>=3.2",
        "pandas>=2.2",
        "pyarrow>=17",
        "huggingface_hub>=0.27",
        "hf-transfer",
        "wandb>=0.19",
        "pytest>=8",
    )
    .with_env_vars({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

cpu_env = flyte.TaskEnvironment(
    name="mf-cpu",
    resources=flyte.Resources(cpu=2, memory="4Gi"),
    image=cpu_image,
    secrets=_secrets(),
    cache="auto",
)

gpu_env = flyte.TaskEnvironment(
    name="mf-gpu",
    resources=flyte.Resources(cpu=6, memory="24Gi", gpu="A10G:1", disk="100Gi", shm="auto"),
    image=gpu_image,
    secrets=_secrets(),
)

factory_env = flyte.TaskEnvironment(
    name="mf-factory",
    resources=flyte.Resources(cpu=1, memory="2Gi"),
    image=cpu_image,
    secrets=_secrets(),
    depends_on=[cpu_env, gpu_env],
)
