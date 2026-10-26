"""Entrypoint for deploying the combined RightSize Bench as a factory.

    CFG=.flyte/config.yaml
    uv run flyte --config $CFG deploy bench_all.py allbench_env      # stations first
    uv run flyte --config $CFG factory deploy bench_all_factory.py
    uv run flyte --config $CFG factory materialize rightsize-all-bench rt-allbench-leaderboard \
        --partition release=big-1 --partition arm=static,rules,gbt-dist,zeroshot-27b,jev-pointer --wait

The graph lives in `resource_tuner/bench_all/factory.py`.
"""

from resource_tuner.bench_all.factory import rightsize_all_bench  # noqa: F401
