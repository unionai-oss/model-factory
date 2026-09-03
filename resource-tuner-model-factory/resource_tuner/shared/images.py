"""Container images for the resource-tuner factory.

Three images because the workloads differ by an order of magnitude:
- harness: CPU-only, must import everything the CORPUS templates import
  (numpy/pandas/sklearn/torch-cpu). Kept lean — episode pods are the
  experiment, and image pull time pads every episode.
- gpu: the trainer stack (TRL/peft/bitsandbytes).
- driver: CPU orchestration + optionally the private metrics plugin.

`with_pip_packages` everywhere; `.with_requirements()` stores a relative
path that breaks under the remote builder.
"""

from __future__ import annotations

import os

import flyte

from ..config import GH_BUILD_TOKEN_SECRET, HF_TOKEN_SECRET, USE_SECRETS, WANDB_SECRET

PYTHON = (3, 12)

# The metrics plugin lives in a PRIVATE repo (unionai/flyteplugins-union,
# branch niels/get-metrics). The remote image builder can only fetch it with
# a token. RT_GH_TOKEN is read at BUNDLE time on the deploy machine; note
# the token is visible in the image's pip metadata, so use a short-lived,
# read-only fine-grained token scoped to that one repo (prototype-grade
# wiring; production would mount a build secret named
# RT_GH_BUILD_TOKEN — see config.GH_BUILD_TOKEN_SECRET).
METRICS_PLUGIN_REF = "github.com/unionai/flyteplugins-union.git@niels/get-metrics"


def _metrics_plugin_spec() -> str | None:
    if os.environ.get("RT_WITH_METRICS", "0") != "1":
        return None
    token = os.environ.get("RT_GH_TOKEN", "")
    auth = f"x-access-token:{token}@" if token else ""
    return f"flyteplugins-union @ git+https://{auth}{METRICS_PLUGIN_REF}"


def secrets() -> list[flyte.Secret]:
    if not USE_SECRETS:
        return []
    return [
        flyte.Secret(key=HF_TOKEN_SECRET, as_env_var="HF_TOKEN"),
        flyte.Secret(key=WANDB_SECRET, as_env_var="WANDB_API_KEY"),
    ]


harness_image = (
    flyte.Image.from_debian_base(name="rt-harness", python_version=PYTHON)
    .with_pip_packages("numpy>=1.26", "pandas>=2.2", "scikit-learn>=1.5")
    # CPU wheel: the harness never sees a GPU, the CUDA wheel is ~5GB dead weight.
    .with_pip_packages("torch>=2.4", index_url="https://download.pytorch.org/whl/cpu")
)

gpu_image = (
    flyte.Image.from_debian_base(name="rt-gpu", python_version=PYTHON)
    .with_apt_packages("git")
    .with_pip_packages(
        "torch>=2.4",
        "transformers>=4.57",
        "trl>=0.21",
        "peft>=0.13",
        "datasets>=3.2",
        "accelerate>=0.34",
        "bitsandbytes>=0.44",
        "pandas>=2.2",
        "pyarrow>=17",
        "wandb>=0.28",
    )
)

driver_image = flyte.Image.from_debian_base(name="rt-driver", python_version=PYTHON).with_pip_packages(
    "pandas>=2.2", "pyarrow>=17"
)

_spec = _metrics_plugin_spec()
if _spec:
    gpu_image = gpu_image.with_pip_packages(_spec)
    driver_image = driver_image.with_pip_packages(_spec)
