"""Container images for the resource-tuner factory.

Three images because the workloads differ by an order of magnitude:
- harness: CPU-only, must import everything the CORPUS templates import
  (numpy/pandas/sklearn/torch-cpu). Kept lean — episode pods are the
  experiment, and image pull time pads every episode.
- gpu: the trainer stack (TRL/peft/bitsandbytes) + the metrics plugin.
- driver: CPU orchestration + the metrics plugin.

`with_pip_packages` everywhere; `.with_requirements()` stores a relative
path that breaks under the remote builder.
"""

from __future__ import annotations

import flyte

from ..config import HF_TOKEN_SECRET, USE_SECRETS, WANDB_SECRET

PYTHON = (3, 12)

# flyteplugins-union is public as of 0.10.0 (Metrics interface). Its PyPI
# metadata still caps flyte<2.7.0, so pip may downgrade flyte when it
# installs; the re-pin layer after it restores flyte 2.7.x deterministically.
# Collapse to one plain layer once a plugins release declares 2.7 support.
_METRICS_LAYER = ("flyteplugins-union>=0.10.0",)
_FLYTE_REPIN_LAYER = ("flyte>=2.7.0,<2.8.0",)


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
    .with_pip_packages(*_METRICS_LAYER)
    .with_pip_packages(*_FLYTE_REPIN_LAYER)
)

driver_image = (
    flyte.Image.from_debian_base(name="rt-driver", python_version=PYTHON)
    .with_pip_packages("pandas>=2.2", "pyarrow>=17")
    .with_pip_packages(*_METRICS_LAYER)
    .with_pip_packages(*_FLYTE_REPIN_LAYER)
)
