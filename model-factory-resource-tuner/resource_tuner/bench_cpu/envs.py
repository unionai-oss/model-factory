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

#: ``big`` releases: two workloads per pod, each pinned to its own 7 cores
#: (``measure.slot_cpus``). The spot c5.4xlarge group tops out at ~5 nodes
#: (rsc-full-1: never more than 5 shards RUNNING), and a 32-CPU pod
#: schedules nowhere, so two slots per 16-core node is the throughput lever.
cpu_measure_pair_env = flyte.TaskEnvironment(
    name="rt-cpubench-measure-pair",
    image=bench_image,
    resources=flyte.Resources(cpu=15, memory="24Gi", disk="60Gi"),
    interruptible=True,
    env_vars=cluster_env_vars(),
)

cpubench_env = flyte.TaskEnvironment(
    name="rt-cpubench",
    image=bench_image,
    resources=flyte.Resources(cpu=2, memory="8Gi"),
    secrets=[flyte.Secret(key=LLM_SERVICE_SECRET, as_env_var="LLM_SERVICE_API_KEY")],
    env_vars=cluster_env_vars(),
    depends_on=[cpu_measure_env, cpu_measure_pair_env, bench_gpu_env],
)
