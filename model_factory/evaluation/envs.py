"""Eval team environments (deploy unit: team_eval.py).

eval_gpu_env keeps a GPU so evals still run when the inference service is
unavailable (local-generation fallback); with the service healthy the GPU
sits mostly idle during generation and only scores rewards.
"""

from __future__ import annotations

import flyte

from ..shared.images import cpu_image, gpu_image, secrets

eval_gpu_env = flyte.TaskEnvironment(
    name="eval-gpu",
    resources=flyte.Resources(cpu=6, memory="24Gi", gpu="A10G:1", disk="100Gi", shm="auto"),
    image=gpu_image,
    secrets=secrets(),
)

eval_cpu_env = flyte.TaskEnvironment(
    name="eval-cpu",
    resources=flyte.Resources(cpu=2, memory="4Gi"),
    image=cpu_image,
)
