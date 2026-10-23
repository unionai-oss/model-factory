"""Data station environment: CPU only, no model involved."""

from __future__ import annotations

import flyte

from ..config import cpu_resources, env_vars
from ..shared.images import cpu_image

data_env = flyte.TaskEnvironment(
    name="df-data",
    resources=cpu_resources(),
    env_vars=env_vars(),
    image=cpu_image,
)
