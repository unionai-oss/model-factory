"""Task environments for RightSizeGPU.

Measurement pass 1 runs on a T4 (g4dn.xlarge: the pool that admits work
here, and the cheapest machine on the ladder). A run that dies with a CUDA
OOM needs a bigger machine by definition, and climbs (round 21, big-2):

    single-GPU code   T4 -> 4xL4 node, one run per L4 -> L40S, one run per GPU
    sharded code      T4 -> one 8xT4 node, the run across all 8 -> one 4xL40S node

The L40S pools admit work slowly (round 20: 42 of 184 L40S shards timed
out queued, and the round-21 probe waited on them), so the L4 and 8xT4
pools, which admit in minutes, take every run they can hold. One workload
per GPU at a time; several per pod on the multi-GPU nodes.
"""

from __future__ import annotations

import os

import flyte

from ..bench.envs import PYTHON, bench_gpu_env, bench_image
from ..config import LLM_SERVICE_SECRET, cluster_env_vars
from ..shared.images import _FLYTE_REPIN_LAYER, _METRICS_LAYER

gpu_workload_image = (
    flyte.Image.from_debian_base(name="rt-gpubench", python_version=PYTHON)
    .with_pip_packages("torch==2.14.0", "numpy==2.5.2", "pandas==3.0.5", "pyarrow==25.0.1",
                       "nvidia-ml-py>=12.560", "psutil>=7.0")
    .with_pip_packages(*_METRICS_LAYER)
    .with_pip_packages(*_FLYTE_REPIN_LAYER)
)

_ENV = cluster_env_vars()  # PyTorch allocator defaults: what users run with

gpu_measure_env = flyte.TaskEnvironment(
    name="rt-gpubench-measure",
    image=gpu_workload_image,
    resources=flyte.Resources(cpu=3, memory="13Gi", gpu="T4:1", disk="60Gi", shm="auto"),
    env_vars=_ENV,
)

GPU_BIG = os.environ.get("RSG_GPU_BIG", "L40s:1")
gpu_measure_big_env = flyte.TaskEnvironment(
    name="rt-gpubench-measure-big",
    image=gpu_workload_image,
    resources=flyte.Resources(cpu=6, memory="44Gi", gpu=GPU_BIG, disk="80Gi", shm="auto"),
    env_vars=_ENV,
)

# Multi-GPU nodes, sized under each node's allocatable (cluster node pools,
# 2026-10-26): g6.12xlarge 47.5 CPU / 174 GiB, g4dn.metal 95 CPU / 346 GiB,
# g6e.12xlarge 47.5 CPU / 351 GiB.
gpu_measure_l4x4_env = flyte.TaskEnvironment(
    name="rt-gpubench-measure-l4x4",
    image=gpu_workload_image,
    resources=flyte.Resources(cpu=44, memory="160Gi", gpu="L4:4", disk="300Gi", shm="auto"),
    env_vars=_ENV,
)
gpu_measure_t4x8_env = flyte.TaskEnvironment(
    name="rt-gpubench-measure-t4x8",
    image=gpu_workload_image,
    resources=flyte.Resources(cpu=88, memory="320Gi", gpu="T4:8", disk="300Gi", shm="auto"),
    env_vars=_ENV,
)
gpu_measure_l40sx4_env = flyte.TaskEnvironment(
    name="rt-gpubench-measure-l40sx4",
    image=gpu_workload_image,
    resources=flyte.Resources(cpu=44, memory="320Gi", gpu="L40s:4", disk="300Gi", shm="auto"),
    env_vars=_ENV,
)

gpubench_env = flyte.TaskEnvironment(
    name="rt-gpubench",
    image=bench_image,
    resources=flyte.Resources(cpu=2, memory="8Gi"),
    secrets=[flyte.Secret(key=LLM_SERVICE_SECRET, as_env_var="LLM_SERVICE_API_KEY")],
    env_vars=cluster_env_vars(),
    depends_on=[gpu_measure_env, gpu_measure_big_env, gpu_measure_l4x4_env, gpu_measure_t4x8_env,
                gpu_measure_l40sx4_env, bench_gpu_env],
)
