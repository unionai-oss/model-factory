"""Container images + secret wiring shared across team environments.

In a larger org each team would own its images; here they share two base
images (CPU data/ops, GPU torch/TRL stack) published by the platform team.
"""

from __future__ import annotations

import flyte

from ..config import HF_TOKEN_SECRET, USE_SECRETS, WANDB_SECRET

PYTHON = (3, 12)

#: The task images' flyte must track the SDK that deploys the factory: an image
#: a minor behind cannot read a factory task spec, and an unpinned one drifts to
#: whatever is newest on PyPI. flyteplugins-union 0.17 caps flyte below 2.11.
FLYTE_PIN = "flyte>=2.10.6,<2.11"


def secrets() -> list[flyte.Secret]:
    """Secret attachments, gated until the secrets exist on the cluster."""
    if not USE_SECRETS:
        return []
    return [
        flyte.Secret(key=HF_TOKEN_SECRET, as_env_var="HF_TOKEN"),
        flyte.Secret(key=WANDB_SECRET, as_env_var="WANDB_API_KEY"),
    ]


cpu_image = (
    flyte.Image.from_debian_base(name="mf-cpu", python_version=PYTHON)
    .with_pip_packages(
        FLYTE_PIN,
        "pandas>=2.2",
        "pyarrow>=17",
        "datasets>=3.2",
        "huggingface_hub>=0.27",
        "pytest>=8",
    )
)

gpu_image = (
    flyte.Image.from_debian_base(name="mf-gpu", python_version=PYTHON)
    .with_apt_packages("git")
    .with_pip_packages(
        FLYTE_PIN,
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

#: The trainer's image adds vLLM for colocated GRPO rollouts. vLLM 0.31 pins
#: torch 2.13 (what the stack already resolves); TRL 1.10 only *warns* above
#: the vLLM 0.26 it was tested with. Kept separate so eval/synthetic/serving
#: images do not carry it.
#: FlashInfer's top-k/top-p sampler JIT-compiles a CUDA kernel at engine
#: warm-up and needs nvcc, which this image does not carry ("Could not find
#: nvcc"); the PyTorch-native sampler needs nothing.
trainer_image = (
    gpu_image.with_pip_packages("vllm==0.31.0")
    .with_env_vars({"HF_HUB_ENABLE_HF_TRANSFER": "1", "VLLM_USE_FLASHINFER_SAMPLER": "0"})
)
