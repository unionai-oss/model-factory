"""Training team environment (deploy unit: team_training.py)."""

from __future__ import annotations

import flyte

from ..shared.images import gpu_image, secrets

trainer_env = flyte.TaskEnvironment(
    name="trainer",
    resources=flyte.Resources(cpu=6, memory="24Gi", gpu="A10G:1", disk="100Gi", shm="auto"),
    image=gpu_image,
    secrets=secrets(),
)
