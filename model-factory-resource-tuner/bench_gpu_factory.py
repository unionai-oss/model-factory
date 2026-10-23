"""Entrypoint for deploying RightSizeGPU Bench as a factory.

    CFG=.flyte/config.yaml
    uv run flyte --config $CFG deploy bench_gpu.py gpubench_env      # stations first
    uv run flyte --config $CFG factory deploy bench_gpu_factory.py
    uv run flyte --config $CFG factory materialize rightsize-gpu-bench rt-gpubench-leaderboard \
        --partition release=small-1 --partition arm=static,vpa,gbt,gbt-dist --wait

The graph lives in `resource_tuner/bench_gpu/factory.py`.
"""

from resource_tuner.bench_gpu.factory import rightsize_gpu_bench  # noqa: F401
