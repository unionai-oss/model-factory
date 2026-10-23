"""Task environments for RightSizeCPU.

Workloads use RightSizeMem's pinned measurement image (same libraries,
same versions). Measurement is ONE workload per pod at a time: CPU is the
quantity measured, so a neighbor would steal it. Pods are interruptible
(spot c5) and sized to hold an 8-thread workload plus the harness.
"""

from __future__ import annotations

import flyte

from ..bench.envs import bench_gpu_env, bench_image
from ..config import LLM_SERVICE_SECRET, cluster_env_vars
from . import grid

cpu_measure_env = flyte.TaskEnvironment(
    name="rt-cpubench-measure",
    image=bench_image,
    resources=flyte.Resources(cpu=grid.MEASURE_POD_CPU, memory="14Gi", disk="40Gi"),
    interruptible=True,
    env_vars=cluster_env_vars(),
)

cpubench_env = flyte.TaskEnvironment(
    name="rt-cpubench",
    image=bench_image,
    resources=flyte.Resources(cpu=2, memory="8Gi"),
    secrets=[flyte.Secret(key=LLM_SERVICE_SECRET, as_env_var="LLM_SERVICE_API_KEY")],
    env_vars=cluster_env_vars(),
    depends_on=[cpu_measure_env, bench_gpu_env],
)
