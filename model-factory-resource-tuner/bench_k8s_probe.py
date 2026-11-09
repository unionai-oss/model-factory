"""Run units for the K8s twin's in-cluster probes (round 22).

    CFG=.flyte/config.yaml
    uv run flyte --config $CFG run --name rt-k8sprobe-1 bench_k8s_probe.py staging_probe
    uv run flyte --config $CFG run --name rt-l4timing-1 bench_k8s_probe.py train_timing
    uv run flyte --config $CFG run --name rt-k8sval-1 bench_k8s_probe.py validate --mem <s3 uri> --cpu <s3 uri>

The probes live in `resource_tuner/bench_all/k8s_probe.py`.
"""

from resource_tuner.bench_all.k8s_probe import (  # noqa: F401
    staging_probe, train_timing, upload_inputs, validate, validate_pods, validate_shard)
from resource_tuner.bench_all.paper import paper_export  # noqa: F401,E402
