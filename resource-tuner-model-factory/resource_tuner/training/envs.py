"""Task environments for the tuner factory stations."""

from __future__ import annotations

import flyte

from ..config import cluster_env_vars, cpu_resources, train_resources
from ..environment.harness import harness_env
from ..shared.images import driver_image, gpu_image, secrets

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
    # Eval runs cluster episodes: harness tasks are called from trainer-env
    # tasks, so the harness env must ship with this deploy unit.
    depends_on=[harness_env],
)

driver_env = flyte.TaskEnvironment(
    name="rt-driver",
    image=driver_image,
    resources=cpu_resources(),
    env_vars=cluster_env_vars(),
    # The pipeline driver awaits GPU children; it must itself be CPU-only
    # (a GPU parent awaiting GPU children deadlocks a small pool).
    depends_on=[trainer_env, harness_env],
)
