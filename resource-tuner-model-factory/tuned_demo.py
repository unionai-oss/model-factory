"""The PRD's adoption surface, verbatim: one decorator opts a task in.

    uv run flyte --config .flyte/config.yaml run tuned_demo.py demo_driver

`demo_workload` declares a deliberately generous prior on its env; the
`@tune.resources` decorator intercepts the driver's invocation, asks the
tune service for a proposal built from the task's own source code, and
runs the task via `.override(resources=<proposal>)`. If the service is
unreachable the task runs unchanged on its prior.
"""

from __future__ import annotations

import time

import flyte

from resource_tuner import tune
from resource_tuner.config import cluster_env_vars
from resource_tuner.shared.images import harness_image

demo_env = flyte.TaskEnvironment(
    name="rt-tuned-demo",
    image=harness_image,
    # The author's hard-coded guess — deliberately padded (the PRD's
    # "waste is silent, failure is loud" default behavior).
    resources=flyte.Resources(cpu=4, memory="8Gi"),
    env_vars=cluster_env_vars(),
)


@tune.resources
@demo_env.task(retries=0)
async def demo_workload(rows: int = 400_000, cols: int = 16) -> dict:
    """A modest pandas aggregation that needs ~200MiB, not the 8Gi prior."""
    import resource as _res

    import numpy as np
    import pandas as pd

    deadline = time.monotonic() + 60
    result = {}
    while time.monotonic() < deadline:
        df = pd.DataFrame(np.random.default_rng(0).standard_normal((rows, cols)))
        result = {"rows": int(len(df)), "mean": float(df.mean().mean())}
        del df
    result["peak_rss_mib"] = _res.getrusage(_res.RUSAGE_SELF).ru_maxrss / 1024
    return result


@demo_env.task(timeout=flyte.Timeout(max_runtime=1800))
async def demo_driver() -> dict:
    """Invokes the tuned task exactly like any other task call."""
    out = await demo_workload(rows=400_000, cols=16)
    print(f"workload done; real peak {out.get('peak_rss_mib', 0):.0f}MiB "
          f"(prior asked for 8Gi)")
    return out
