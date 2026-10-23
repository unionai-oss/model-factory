"""Entrypoint for deploying RightSizeMem Bench as a factory.

    CFG=.flyte/config.yaml
    uv run flyte --config $CFG deploy bench.py bench_env        # stations first
    uv run flyte --config $CFG factory deploy bench_factory.py
    uv run flyte --config $CFG factory materialize rightsize-bench rt-bench-leaderboard \
        --partition release=v1 \
        --partition arm=static,vpa,size-rule,gbt,gbt-dist,zeroshot,sft,jev --wait

The graph lives in `resource_tuner/bench/factory.py`.
"""

from resource_tuner.bench.factory import rightsize_bench  # noqa: F401
