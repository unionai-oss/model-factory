"""Deploy unit for the combined RightSize Bench stations.

    CFG=.flyte/config.yaml
    uv run flyte --config $CFG deploy bench_all.py allbench_env

The factory that wires these into the benchmark is `bench_all_factory.py`.
"""

from resource_tuner.bench_all.envs import allbench_env  # noqa: F401
from resource_tuner.bench_all.stations import (  # noqa: F401
    all_leaderboard,
    all_peek,
    fit_all_arm,
    join_corpora,
    pool_all_arms,
    rl_all_policy,
    score_all_arm,
    train_all_policy,
)
