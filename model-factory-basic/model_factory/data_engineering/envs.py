"""Data engineering team environments (deploy unit: team_data.py)."""

from __future__ import annotations

import flyte

from ..config import cluster_env_vars, cpu_resources, gpu_resources
from ..shared.images import cpu_image, secrets, trainer_image

# GPU env for synthetic task generation (batch inference).
de_gpu_env = flyte.TaskEnvironment(
    name="de-gpu",
    resources=gpu_resources(),
    env_vars=cluster_env_vars(),
    image=trainer_image,  # vLLM for the teacher
    secrets=secrets(),
)

de_cpu_env = flyte.TaskEnvironment(
    name="de-cpu",
    resources=cpu_resources(),
    env_vars=cluster_env_vars(),
    image=cpu_image,
    secrets=secrets(),
    cache="auto",
)
