"""Entrypoint for deploying RightSizeCPU Bench as a factory.

    CFG=.flyte/config.yaml
    uv run flyte --config $CFG deploy bench_cpu.py cpubench_env      # stations first
    uv run flyte --config $CFG factory deploy bench_cpu_factory.py
    uv run flyte --config $CFG factory materialize rightsize-cpu-bench rt-cpubench-leaderboard \
        --partition release=small-1 --partition arm=static,vpa,gbt,gbt-dist --wait

The graph lives in `resource_tuner/bench_cpu/factory.py`.
"""

from resource_tuner.bench_cpu.factory import rightsize_cpu_bench  # noqa: F401
