"""Deploy unit for RightSizeGPU Bench stations.

    CFG=.flyte/config.yaml
    uv run flyte --config $CFG deploy bench_gpu.py gpubench_env

    # shake out generate -> measure on one task per family
    uv run flyte --config $CFG run bench_gpu.py gpu_pilot

The factory that wires these into the benchmark is `bench_gpu_factory.py`.
"""

from resource_tuner.bench.gpu_stations import train_llm_arm  # noqa: F401
from resource_tuner.bench_gpu.envs import gpubench_env  # noqa: F401
from resource_tuner.bench_gpu.stations import (  # noqa: F401
    gpu_leaderboard,
    gpu_peek,
    gpu_pilot,
    fit_gpu_arm,
    generate_gpu_runs,
    measure_gpu_bench,
    score_gpu_arm,
)
