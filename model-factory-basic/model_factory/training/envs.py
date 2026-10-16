"""Training team environment (deploy unit: team_training.py)."""

from __future__ import annotations

import flyte

from ..config import cluster_env_vars, gpu_resources
from ..shared.images import secrets, trainer_image

trainer_env = flyte.TaskEnvironment(
    name="trainer",
    resources=gpu_resources(),
    env_vars=cluster_env_vars(),
    image=trainer_image,
    secrets=secrets(),
)
