"""Deploy unit for RightSizeMem Bench stations.

    CFG=.flyte/config.yaml
    uv run flyte --config $CFG deploy bench.py bench_env

    # shake out generate -> measure on a handful of tasks
    uv run flyte --config $CFG run bench.py bench_pilot --tasks_per_family 4

The factory that wires these into the full benchmark is `bench_factory.py`.
"""

from resource_tuner.bench.envs import bench_env, bench_gpu_env, live_env, measure_big_env, measure_env  # noqa: F401
from resource_tuner.bench.stations import (  # noqa: F401
    bench_leaderboard,
    bench_peek,
    bench_pilot,
    fit_bench_arm,
    fit_bench_ensemble,
    generate_bench_runs,
    live_run,
    live_validate,
    measure_bench,
    measure_shard,
    measure_shard_big,
    score_bench_arm,
)
from resource_tuner.bench.gpu_stations import train_llm_arm  # noqa: F401
