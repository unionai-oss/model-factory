"""Task environments for RightSizeGPU.

Measurement pass 1 runs on a T4 (g4dn.xlarge: the pool that admits work
here, and the cheapest device on the ladder). A run that dies with a CUDA
OOM needs a bigger device by definition, so pass 2 re-measures it on an
L40S (48 GB). One workload per GPU at a time.
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

gpubench_env = flyte.TaskEnvironment(
    name="rt-gpubench",
    image=bench_image,
    resources=flyte.Resources(cpu=2, memory="8Gi"),
    secrets=[flyte.Secret(key=LLM_SERVICE_SECRET, as_env_var="LLM_SERVICE_API_KEY")],
    env_vars=cluster_env_vars(),
    depends_on=[gpu_measure_env, gpu_measure_big_env, bench_gpu_env],
)
