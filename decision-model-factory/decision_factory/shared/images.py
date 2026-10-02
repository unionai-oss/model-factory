"""Container images for the factory's task environments.

Three images. `cpu_image` is for episode generation and champion selection —
both pure pandas/json work. `gpu_image` carries the training stack and is
shared by fine-tuning and eval. `serving_image` is the CPU-torch build the
decision API runs on: serving a sub-1B model needs no accelerator, and the
CUDA wheel is multiple GB of pull time an app pays on every cold start.

Pins are lower bounds on purpose: these are demo images rebuilt often, and a
hard pin would silently rot against the SDK the tasks are deployed with.
"""

from __future__ import annotations

import os

import flyte

PYTHON = (3, 12)

#: Attached only when the token exists on the cluster: Flyte refuses to
#: schedule a task whose declared secret is missing, and the default suite is
#: entirely ungated, so the loop must run without it. Needed only for the
#: opt-in gated models (see config.GATED_MODELS).
USE_HF_TOKEN = os.environ.get("DF_USE_HF_TOKEN", "0") == "1"

HF_TOKEN_SECRET = "HUGGINGFACE_TOKEN"


def secrets() -> list[flyte.Secret]:
    if not USE_HF_TOKEN:
        return []
    return [flyte.Secret(key=HF_TOKEN_SECRET, as_env_var="HF_TOKEN")]


cpu_image = flyte.Image.from_debian_base(name="df-cpu", python_version=PYTHON).with_pip_packages(
    "flyte>=2.10.0",
    "pandas>=2.2",
    "pyarrow>=17",
)

gpu_image = (
    flyte.Image.from_debian_base(name="df-gpu", python_version=PYTHON)
    .with_pip_packages(
        "flyte>=2.10.0",
        "torch>=2.6",
        "transformers>=4.57",
        "trl>=0.21",
        "peft>=0.13",
        "accelerate>=1.0",
        "datasets>=3.2",
        "pandas>=2.2",
        "pyarrow>=17",
        "hf-transfer",
    )
    # Without this the base-weight download is the slowest part of a smoke run.
    .with_env_vars({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)


# The decision API serves on CPU (see config.serving_resources), so it gets the
# CPU torch wheel. Same code as the GPU image would run, but the CUDA build is
# ~5GB the app would pull on every cold start and never use.
serving_image = (
    flyte.Image.from_debian_base(name="df-serve", python_version=PYTHON)
    .with_pip_packages("flyte>=2.10.0", "fastapi", "uvicorn", "pandas>=2.2", "pyarrow>=17")
    .with_pip_packages("torch>=2.6", index_url="https://download.pytorch.org/whl/cpu")
    .with_pip_packages("transformers>=4.57", "peft>=0.13", "accelerate>=1.0")
)
