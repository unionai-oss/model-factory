"""Data engineering team environments (deploy unit: team_data.py)."""

from __future__ import annotations

import flyte

from ..shared.images import cpu_image, gpu_image, secrets

# GPU env for synthetic task generation (batch inference).
de_gpu_env = flyte.TaskEnvironment(
    name="de-gpu",
    resources=flyte.Resources(cpu=6, memory="24Gi", gpu="A10G:1", disk="100Gi", shm="auto"),
    image=gpu_image,
    secrets=secrets(),
)

de_cpu_env = flyte.TaskEnvironment(
    name="de-cpu",
    resources=flyte.Resources(cpu=2, memory="4Gi"),
    image=cpu_image,
    secrets=secrets(),
    cache="auto",
    depends_on=[de_gpu_env],  # release driver + nightly trigger call synthetic gen
)
