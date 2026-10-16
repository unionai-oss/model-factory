"""Container images for the resource-tuner factory.

Four images because the workloads differ by an order of magnitude:
- harness: CPU-only, must import everything the CORPUS templates import
  (numpy/pandas/sklearn/torch-cpu). Kept lean — episode pods are the
  experiment, and image pull time pads every episode.
- gpu: the trainer stack (TRL/peft/bitsandbytes) + the metrics plugin.
- driver: CPU orchestration + the metrics plugin.
- decision: CPU torch, for the multi-head decision-model arm. A separate
  image from `driver` because torch roughly triples the pull, and the
  orchestration env pulls on every single action.

`with_pip_packages` everywhere; `.with_requirements()` stores a relative
path that breaks under the remote builder.
"""

from __future__ import annotations

import flyte

from ..config import HF_TOKEN_SECRET, USE_SECRETS, WANDB_SECRET

PYTHON = (3, 12)

# flyteplugins-union carries both the Metrics interface and (as of 0.12.0)
# factories. Its PyPI metadata has historically capped the flyte version, so
# pip can downgrade flyte while installing it; the re-pin layer after it
# restores the SDK line deterministically. The floor MUST track the version
# the project deploys with — an image pinned a minor behind the deploying SDK
# runs tasks on a runtime that cannot read the factory task spec.
_METRICS_LAYER = ("flyteplugins-union>=0.12.0",)
_FLYTE_REPIN_LAYER = ("flyte>=2.10.0",)


def secrets() -> list[flyte.Secret]:
    if not USE_SECRETS:
        return []
    return [
        flyte.Secret(key=HF_TOKEN_SECRET, as_env_var="HF_TOKEN"),
        flyte.Secret(key=WANDB_SECRET, as_env_var="WANDB_API_KEY"),
    ]


harness_image = (
    flyte.Image.from_debian_base(name="rt-harness", python_version=PYTHON)
    # scipy/pyarrow: the round-12 synthetic allowlist admits sparse and
    # columnar workloads — the oracle must be able to run them.
    .with_pip_packages("numpy>=1.26", "pandas>=2.2", "scikit-learn>=1.5", "scipy>=1.13", "pyarrow>=17")
    # CPU wheel: the harness never sees a GPU, the CUDA wheel is ~5GB dead weight.
    .with_pip_packages("torch>=2.4", index_url="https://download.pytorch.org/whl/cpu")
    # ── round 13: the stack axis ────────────────────────────────────────
    # The oracle can only MEASURE what it can RUN, so every library the
    # teacher is allowed to import must live here. Grouped by layer, all
    # usable offline (no model downloads, no API clients).
    .with_pip_packages(  # dataframe / OLAP / out-of-core
        "polars>=1.0", "duckdb>=1.1", "dask[dataframe]>=2024.8", "ibis-framework[duckdb]>=9.0"
    )
    .with_pip_packages(  # classical ML + stats + graphs
        "xgboost>=2.1", "lightgbm>=4.5", "statsmodels>=0.14", "networkx>=3.3"
    )
    .with_pip_packages(  # DL frameworks beyond bare torch (CPU builds)
        "jax[cpu]>=0.4.30", "lightning>=2.4", "transformers>=4.57", "safetensors>=0.4"
    )
    .with_pip_packages(  # agent / RAG shapes — offline-usable cores only
        "faiss-cpu>=1.8",
        "langchain-core>=0.3",
        "langchain-text-splitters>=0.3",
        "langgraph>=0.2",
        "llama-index-core>=0.11",
    )
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
        # tune service serves the joblib'd quantile-GBT baseline as a
        # fallback/AB estimator next to the LLM.
        "scikit-learn>=1.5",
    )
    .with_pip_packages(*_METRICS_LAYER)
    .with_pip_packages(*_FLYTE_REPIN_LAYER)
)

driver_image = (
    flyte.Image.from_debian_base(name="rt-driver", python_version=PYTHON)
    # scikit-learn: the classical ML baseline (quantile GBTs) trains inside
    # eval_tuner on this env.
    .with_pip_packages("pandas>=2.2", "pyarrow>=17", "scikit-learn>=1.5")
    .with_pip_packages(*_METRICS_LAYER)
    .with_pip_packages(*_FLYTE_REPIN_LAYER)
)


# The decision-model arm (training/decision_model.py) is a small MLP: torch,
# but strictly CPU — a 256x128 trunk over 43 features trains in seconds, and
# putting it on the GPU image would queue it behind the LLM trainer for an
# accelerator it never uses.
decision_image = (
    flyte.Image.from_debian_base(name="rt-decision", python_version=PYTHON)
    .with_pip_packages("pandas>=2.2", "pyarrow>=17")
    # CPU wheel: the CUDA build is ~5GB of dead weight here.
    .with_pip_packages("torch>=2.4", index_url="https://download.pytorch.org/whl/cpu")
    .with_pip_packages(*_METRICS_LAYER)
    .with_pip_packages(*_FLYTE_REPIN_LAYER)
)
