"""Eval station environments.

`scorer_env` needs a GPU (it generates from each fine-tune and from the
untuned base). `selector_env` must NOT have one: champion selection only
reads scorecards and copies a directory, and a GPU coordinator would sit on
the accelerator the fine-tunes are queued for.
"""

from __future__ import annotations

import flyte

from ..config import cpu_resources, env_vars, gpu_resources
from ..shared.images import cpu_image, gpu_image, secrets

scorer_env = flyte.TaskEnvironment(
    name="df-scorer",
    resources=gpu_resources(),
    env_vars=env_vars(),
    # Off by default in Flyte 2. Without it a factory materialization
    # re-runs every build even when the plan shows it as existing.
    cache="auto",
    image=gpu_image,
    secrets=secrets(),
)

selector_env = flyte.TaskEnvironment(
    name="df-selector",
    resources=cpu_resources(),
    env_vars=env_vars(),
    # Off by default in Flyte 2. Without it a factory materialization
    # re-runs every build even when the plan shows it as existing.
    cache="auto",
    image=cpu_image,
)
