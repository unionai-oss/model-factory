"""Representative tuned workflows — the auto-tuner in action across a
realistic mix, each invoked at several input scales.

    uv run flyte --config .flyte/config.yaml run tuned_workflows_demo.py workflows_demo_driver

Three workflow shapes (data engineering, ML training, batch inference),
all opted in with `@tune.resources` and all declaring the same padded
one-size-fits-all prior (4 CPU / 8Gi — the classic hard-coded guess).
The driver invokes each at small and large scales: the decorator sends
the invocation's actual inputs as the estimation context, so the tuner
can size the SAME task differently per call. Every task returns
`peak_rss_mib`, so outcomes auto-report to the tune service's value
ledger — watch the dashboard at
https://rt-tune-resource-tuner-model-factory-development.apps.demo.hosted.unionai.cloud/
fill in as this runs.
"""

from __future__ import annotations

import time

import flyte

from resource_tuner import tune
from resource_tuner.config import cluster_env_vars
from resource_tuner.shared.images import harness_image

wf_env = flyte.TaskEnvironment(
    name="rt-tuned-workflows",
    image=harness_image,
    # The author's padded guess, shared by every task in this env.
    resources=flyte.Resources(cpu=4, memory="8Gi"),
    env_vars=cluster_env_vars(),
)


def _peak_rss_mib() -> float:
    import resource as _res

    return _res.getrusage(_res.RUSAGE_SELF).ru_maxrss / 1024


@tune.resources
@wf_env.task(retries=0)
async def sessionize_clickstream(rows: int, n_users: int = 5000) -> dict:
    """Data engineering: sessionize a clickstream and aggregate per user."""
    import numpy as np
    import pandas as pd

    deadline = time.monotonic() + 60
    stats = {}
    while time.monotonic() < deadline:
        rng = np.random.default_rng(7)
        events = pd.DataFrame(
            {
                "user": rng.integers(0, n_users, rows),
                "dwell_ms": rng.exponential(1200, rows),
                "page_depth": rng.integers(1, 30, rows),
            }
        )
        sessions = events.groupby("user").agg(
            visits=("dwell_ms", "count"), dwell=("dwell_ms", "sum"), depth=("page_depth", "max")
        )
        stats = {"users": int(len(sessions)), "mean_dwell": float(sessions["dwell"].mean())}
        del events, sessions
    stats["peak_rss_mib"] = _peak_rss_mib()
    return stats


@tune.resources
@wf_env.task(retries=0)
async def train_churn_model(n_samples: int, n_features: int = 32) -> dict:
    """ML training: fit a churn classifier on a synthetic feature matrix."""
    import numpy as np
    from sklearn.ensemble import GradientBoostingClassifier

    deadline = time.monotonic() + 60
    stats = {}
    while time.monotonic() < deadline:
        rng = np.random.default_rng(11)
        X = rng.standard_normal((n_samples, n_features))
        y = (X[:, 0] * 0.7 + X[:, 1] > 0).astype(int)
        model = GradientBoostingClassifier(n_estimators=30, max_depth=3)
        model.fit(X, y)
        stats = {"train_acc": float(model.score(X, y)), "n_samples": n_samples}
        del X, y, model
    stats["peak_rss_mib"] = _peak_rss_mib()
    return stats


@tune.resources
@wf_env.task(retries=0)
async def embed_product_catalog(items: int, dim: int = 768) -> dict:
    """Batch inference: project a product catalog through an embedding head."""
    import numpy as np

    deadline = time.monotonic() + 60
    stats = {}
    head = np.random.default_rng(3).standard_normal((dim, 256)).astype(np.float32)
    while time.monotonic() < deadline:
        catalog = np.random.default_rng(5).standard_normal((items, dim)).astype(np.float32)
        embedded = catalog @ head
        stats = {"items": items, "norm": float(np.linalg.norm(embedded[0]))}
        del catalog, embedded
    stats["peak_rss_mib"] = _peak_rss_mib()
    return stats


@wf_env.task(timeout=flyte.Timeout(max_runtime=3600))
async def workflows_demo_driver() -> dict:
    """Invoke every tuned workflow at two scales — six tuned invocations.

    Each call goes: decorator → tune service (task source + these exact
    inputs) → `.override(resources=<proposal>)` → pod sized by the model
    → outcome auto-reported to the value ledger.
    """
    import asyncio

    invocations = {
        "sessionize/small": sessionize_clickstream(rows=200_000),
        "sessionize/large": sessionize_clickstream(rows=8_000_000),
        "churn/small": train_churn_model(n_samples=20_000),
        "churn/large": train_churn_model(n_samples=400_000, n_features=64),
        "embed/small": embed_product_catalog(items=20_000),
        "embed/large": embed_product_catalog(items=1_500_000),
    }
    results = await asyncio.gather(*invocations.values(), return_exceptions=True)
    out = {}
    for name, res in zip(invocations, results):
        if isinstance(res, BaseException):
            out[name] = {"error": str(res)[:200]}
        else:
            out[name] = {k: round(v, 1) if isinstance(v, float) else v for k, v in res.items()}
        print(f"{name}: {out[name]}")
    return out
