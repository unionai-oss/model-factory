"""Images and task environments for RightSize Bench.

The measurement image pins every library a workload imports to an exact
version. A label is "what this code used under these library versions";
an unpinned pandas upgrade between two measurement runs would silently
change ground truth (pandas 3 alone changed the default string dtype).
The same pins are the local dev deps, so a laptop reproduces the cluster.
"""

from __future__ import annotations

import os

import flyte

from ..config import LLM_SERVICE_SECRET, cluster_env_vars
from ..shared.images import _FLYTE_REPIN_LAYER, _METRICS_LAYER

PYTHON = (3, 13)

#: Exact versions: these define the ground truth. Bump deliberately, then
#: re-measure (a new `release` partition), never in place.
WORKLOAD_PINS = (
    "numpy==2.5.2",
    "pandas==3.0.5",
    "pyarrow==25.0.1",
    "polars==2.0.0",
    "duckdb==1.5.6",
    "scikit-learn==1.9.0",
    "scipy==1.18.1",
    "psutil>=7.0",
)

bench_image = (
    flyte.Image.from_debian_base(name="rt-bench", python_version=PYTHON)
    .with_pip_packages(*WORKLOAD_PINS)
    .with_pip_packages("torch==2.14.0", index_url="https://download.pytorch.org/whl/cpu")
    .with_pip_packages(*_METRICS_LAYER)
    .with_pip_packages(*_FLYTE_REPIN_LAYER)
)

# The LLM arms: generative SFT and the decision head share a torso, a
# trainer and this image. Workload libraries are absent on purpose -- no
# LLM arm ever executes a workload.
bench_gpu_image = (
    flyte.Image.from_debian_base(name="rt-bench-gpu", python_version=PYTHON)
    .with_pip_packages(
        "torch==2.14.0",
        "transformers>=4.57",
        "peft>=0.13",
        "accelerate>=1.0",
        "pandas==3.0.5",
        "pyarrow==25.0.1",
        "numpy==2.5.2",
        "scikit-learn==1.9.0",
    )
    .with_pip_packages(*_METRICS_LAYER)
    .with_pip_packages(*_FLYTE_REPIN_LAYER)
)

#: Measurement runs in two passes, sized to what this tenant schedules and
#: to its ~30-concurrent-actions-per-project cap (the binding constraint:
#: rsb-v1-corpus-2 ran 18-27 shards at a time for 288 shards, ~4.5 hours).
#: The c5.4xlarge / i3.xlarge on-demand pools sit at their nodepool limits;
#: the spot c5 pool does not.
#:
#: pass 1 -- every run, MEASURE_CONCURRENCY at a time on an interruptible
#:           12 CPU / 24Gi pod (one spot c5.4xlarge). Workloads cap their own
#:           threads at 3, so 4 x 3 threads never throttle each other.
#:           ru_maxrss is per process, so concurrency does not blur peaks;
#:           each process has a guard, and a workload heading past it is
#:           killed and recorded as censored...
#: pass 2 -- ...and only those re-run alone on a 24Gi pod.
MEASURE_CONCURRENCY = 4
MEASURE_MEMORY_MIB = 24 * 1024
BIG_MEMORY_MIB = 24 * 1024
# floats: they are defaults of `float` task inputs, and flyte will not
# coerce an int literal into one (rsb-pilot-2 lost every shard to that).
MEASURE_GUARD_MIB = float((MEASURE_MEMORY_MIB - 3 * 1024) // MEASURE_CONCURRENCY)
BIG_GUARD_MIB = float(BIG_MEMORY_MIB - 4 * 1024)

measure_env = flyte.TaskEnvironment(
    name="rt-bench-measure",
    image=bench_image,
    resources=flyte.Resources(cpu=12, memory=f"{MEASURE_MEMORY_MIB}Mi", disk="80Gi"),
    interruptible=True,
    env_vars=cluster_env_vars(),
)

measure_big_env = flyte.TaskEnvironment(
    name="rt-bench-measure-big",
    image=bench_image,
    resources=flyte.Resources(cpu=3, memory=f"{BIG_MEMORY_MIB}Mi", disk="60Gi"),
    env_vars=cluster_env_vars(),
)

# Live validation runs ONE workload under an arm's chosen request, so its
# resources are always overridden per call; these are only defaults.
live_env = flyte.TaskEnvironment(
    name="rt-bench-live",
    image=bench_image,
    resources=flyte.Resources(cpu=1, memory="1Gi", disk="40Gi"),
    env_vars=cluster_env_vars(),
)

#: The LLM arms' accelerator, and the pod shape that fits its node.
#: T4:1 (g4dn.xlarge, 16 GB, fp16 + GradScaler -- no bf16 on Turing) is the
#: default because it is the pool that actually admits work here: the L40s:1
#: group sat at "max node group size reached" (rsb-pilot-board-3) and A10G
#: jobs stayed QUEUED for over an hour behind other projects
#: (rsb-v1-board-1), while a T4 probe went straight to a node
#: (rsb-t4-probe-1). Override per deploy with RSB_GPU=A10G:1 / L40s:1.
BENCH_GPU = os.environ.get("RSB_GPU", "T4:1")
_GPU_POD = {  # allocatable on the node each accelerator comes on
    "T4:1": dict(cpu=3, memory="13Gi"),      # g4dn.xlarge: 3.67 CPU / 13.7 GiB
    "A10G:1": dict(cpu=6, memory="24Gi"),
    "L40s:1": dict(cpu=6, memory="44Gi"),    # g6e.2xlarge: 7.66 CPU / 59.1 GiB
}[BENCH_GPU]

bench_gpu_env = flyte.TaskEnvironment(
    name="rt-bench-gpu",
    image=bench_gpu_image,
    resources=flyte.Resources(**_GPU_POD, gpu=BENCH_GPU, disk="80Gi", shm="auto"),
    env_vars={**cluster_env_vars(), "RSB_GPU": BENCH_GPU, "TOKENIZERS_PARALLELISM": "false",
              "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
)

# Orchestration + the classical arms + scoring: CPU only. Awaits GPU and
# measurement children, so it must never itself hold an accelerator. Sized
# to fit a t3a.xlarge (3.67 CPU / 13.7 GiB), the pool that has headroom.
bench_env = flyte.TaskEnvironment(
    name="rt-bench",
    image=bench_image,
    resources=flyte.Resources(cpu=2, memory="10Gi"),
    secrets=[flyte.Secret(key=LLM_SERVICE_SECRET, as_env_var="LLM_SERVICE_API_KEY")],
    env_vars=cluster_env_vars(),
    depends_on=[measure_env, measure_big_env, live_env, bench_gpu_env],
)
