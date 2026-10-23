"""Container images + secret wiring shared across team environments.

In a larger org each team would own its images; here they share two base
images (CPU data/ops, GPU torch/TRL stack) published by the platform team.
"""

from __future__ import annotations

import flyte

from ..config import HF_TOKEN_SECRET, USE_SECRETS, WANDB_SECRET

PYTHON = (3, 12)


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
        "flyte>=2.6.0",
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
