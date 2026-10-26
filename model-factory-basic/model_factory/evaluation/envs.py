"""Eval team environments (deploy unit: team_eval.py).

eval-gpu generates held-out completions (baseline, candidate, champion);
eval-cpu holds the promotion gate and the champion bookkeeping, which need no
GPU — and must not hold one while a human takes hours to answer the gate.
"""

from __future__ import annotations

import flyte

from ..config import cluster_env_vars, cpu_resources, gpu_resources
from ..shared.images import cpu_image, gpu_image, secrets

eval_gpu_env = flyte.TaskEnvironment(
    name="eval-gpu",
    resources=gpu_resources(),
    env_vars=cluster_env_vars(),
    image=gpu_image,
    secrets=secrets(),
)

eval_cpu_env = flyte.TaskEnvironment(
    name="eval-cpu",
    resources=cpu_resources(),
    env_vars=cluster_env_vars(),
    image=cpu_image,
)
