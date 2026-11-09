"""Entrypoint for deploying RightSize Bench, Flyte and Kubernetes (round 22).

    CFG=.flyte/config.yaml
    uv run flyte --config $CFG deploy bench_all.py allbench_env      # stations first
    uv run flyte --config $CFG factory deploy bench_k8s_factory.py
    uv run flyte --config $CFG factory materialize rightsize-k8s-bench rt-k8sbench-leaderboard \
        --partition release=k8s-1 \
        --partition arm=static,rules,gbt-dist,zeroshot-27b,jev-pointer,jev-pointer-flyte --wait

The graph lives in `resource_tuner/bench_all/factory.py` (`rightsize_k8s_bench`).
"""

from resource_tuner.bench_all.factory import rightsize_k8s_bench  # noqa: F401
