"""Deploy unit for RightSizeCPU Bench stations.

    CFG=.flyte/config.yaml
    uv run flyte --config $CFG deploy bench_cpu.py cpubench_env

    # shake out generate -> measure on one task per family
    uv run flyte --config $CFG run bench_cpu.py cpu_pilot

The factory that wires these into the benchmark is `bench_cpu_factory.py`.
"""

from resource_tuner.bench.gpu_stations import train_llm_arm  # noqa: F401
from resource_tuner.bench_cpu.envs import cpubench_env  # noqa: F401
from resource_tuner.bench_cpu.stations import (  # noqa: F401
    cpu_leaderboard,
    cpu_peek,
    cpu_pilot,
    fit_cpu_arm,
    pool_cpu_arms,
    generate_cpu_runs,
    measure_cpu_bench,
    score_cpu_arm,
)
