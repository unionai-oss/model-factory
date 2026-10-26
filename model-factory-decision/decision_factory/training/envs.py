"""Training station environment: GPU."""

from __future__ import annotations

import flyte

from ..config import env_vars, gpu_resources
from ..shared.images import gpu_image, secrets

trainer_env = flyte.TaskEnvironment(
    name="df-trainer",
    resources=gpu_resources(),
    env_vars=env_vars(),
    # Off by default in Flyte 2. Without it a factory materialization
    # re-runs every build even when the plan shows it as existing.
    cache="auto",
    image=gpu_image,
    secrets=secrets(),
)
