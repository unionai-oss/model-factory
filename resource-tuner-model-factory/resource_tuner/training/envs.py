"""Task environments for the tuner factory stations."""

from __future__ import annotations

import flyte

import flyte as _flyte

from ..config import LLM_SERVICE_SECRET, cluster_env_vars, cpu_resources, train_resources
from ..environment.harness import harness_env
from ..shared.images import driver_image, gpu_image, secrets
from .generator import generator_env

trainer_env = flyte.TaskEnvironment(
    name="rt-trainer",
    image=gpu_image,
    resources=train_resources(),
    secrets=secrets(),
    env_vars={
        **cluster_env_vars(),
        # Fragmentation guard for larger ladder rungs on a single GPU.
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TOKENIZERS_PARALLELISM": "false",
    },
    depends_on=[harness_env],
)

driver_env = flyte.TaskEnvironment(
    name="rt-driver",
    image=driver_image,
    resources=cpu_resources(),
    # The synthetic station reaches the llm-service teachers through their
    # PUBLIC endpoints with this key (svc DNS vanishes when the platform
    # unassigns an idle app). Unconditional: the secret exists in this
    # project.
    secrets=[_flyte.Secret(key=LLM_SERVICE_SECRET, as_env_var="LLM_SERVICE_API_KEY")],
    env_vars=cluster_env_vars(),
    # The pipeline driver (and eval, which lives here — CPU-only since
    # generation moved to the reusable GPU generator) awaits GPU children;
    # a GPU parent awaiting GPU children deadlocks a small pool.
    depends_on=[trainer_env, harness_env, generator_env],
)
