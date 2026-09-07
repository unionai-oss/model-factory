"""fit_ml_baseline: the quantile-GBT estimator as a first-class artifact.

Fits on a corpus's train split, sim-scores itself against the rule
baseline on heldout (the report shows what you're deploying), and
publishes `ml_baseline.joblib` + manifest as the ml-baseline-model
artifact — which the tune service loads and serves as a fallback / A/B
estimator next to the LLM checkpoint.
"""

from __future__ import annotations

import json
import statistics
import tempfile

import flyte
import flyte.io

from .. import pricing
from ..contracts import ARTIFACT_ML_BASELINE, ARTIFACT_TASK_CORPUS, publish
from ..environment.simulator import simulate_episode
from ..shared import assets
from ..shared.reporting import GOOD, Reporter, esc
from .baseline import baseline_proposal, fit_family_baseline
from .envs import driver_env
from .ml_baseline import GPU_MEM_QUANTILE, CPU_QUANTILE, MEM_QUANTILE, MLBaseline


def _sim_row(records_pairs) -> dict:
    ok, costs, wastes = 0, [], []
    for r, p in records_pairs:
        ep = simulate_episode(
            p,
            float(r["true_peak_memory_mib"]),
            float(r["true_cpu_cores"]),
            int(r["duration_s"]),
            true_gpu_mem_mib=float(r.get("true_gpu_mem_mib", 0.0) or 0.0),
        )
        ok += ep.ok
        costs.append(
            pricing.dollars_per_hr(
                ep.requested_cpu, ep.requested_memory_mib,
                ep.requested_gpu_type, ep.requested_gpu,
            )
        )
        if ep.ok:
            wastes.append(
                100 * (ep.requested_memory_mib - ep.peak_memory_mib) / ep.requested_memory_mib
            )
    n = len(records_pairs)
    return {
        "fit": ok / n if n else 0.0,
        "cost_per_task_hr": sum(costs) / n if n else None,
        "median_mem_waste_pct": statistics.median(wastes) if wastes else None,
    }


# Dark-mode wiring mirrors train_tuner: a new corpus refreshes the
# classical estimator too. auto_activate: deploys leave triggers LIVE.
_fit_trigger = flyte.Trigger(
    name="fit-ml-baseline-on-new-corpus",
    automation=flyte.OnArtifact(name=ARTIFACT_TASK_CORPUS),
    inputs={"corpus": flyte.TriggeredArtifact},
    description="New tuning-task-corpus version -> refit quantile-GBT baseline",
    auto_activate=True,
)


@driver_env.task(
    triggers=[_fit_trigger],
    timeout=flyte.Timeout(max_runtime=3600),
    produces_artifacts=True,
    report=True,
)
async def fit_ml_baseline(
    corpus: flyte.io.File | None = None, heldout_sample: int = 512
) -> flyte.io.Dir:
    """Fit the quantile-GBT baseline and publish it as ml-baseline-model."""
    import pandas as pd

    rep = Reporter("ML baseline fit", "quantile gradient-boosted trees")
    rep.p("Resolving corpus…")
    await rep.flush()
    if corpus is None:
        latest = await assets.latest_version(ARTIFACT_TASK_CORPUS)
        if latest is None:
            raise RuntimeError("no tuning-task-corpus artifact to fit against")
        corpus = flyte.io.File.from_existing_remote(latest.path)
    df = pd.read_parquet(await corpus.download())
    train = df[df["split"] == "train"].to_dict("records")
    heldout = df[df["split"] == "heldout"].to_dict("records")[:heldout_sample]

    rep.reset_body().kv({"train rows": len(train), "heldout sample": len(heldout)})
    rep.p("Fitting…")
    await rep.flush()
    ml = MLBaseline().fit(train)

    ml_stats = _sim_row([(r, ml.propose(r)) for r in heldout]) if heldout else {}
    rule = fit_family_baseline(train)
    rule_stats = (
        _sim_row([(r, baseline_proposal(rule, r["family"])) for r in heldout])
        if heldout
        else {}
    )

    out_dir = tempfile.mkdtemp(prefix="ml-baseline-")
    ml.save(out_dir)
    manifest = {
        "estimator": "quantile-gbt",
        "quantiles": {"memory": MEM_QUANTILE, "cpu": CPU_QUANTILE, "gpu_mem": GPU_MEM_QUANTILE},
        "n_train": len(train),
        "n_features": len(ml.feature_names) + len(ml.families),
        "corpus": getattr(corpus, "path", ""),
        "sim_heldout": {"ml": ml_stats, "rule_baseline": rule_stats},
    }
    with open(f"{out_dir}/manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    fmt_pct = lambda v: "-" if v is None else f"{v:.0f}%"  # noqa: E731
    fmt_usd = lambda v: "-" if v is None else f"${v:.4f}"  # noqa: E731
    rep.h("Sim scoring on heldout (what you are deploying)")
    rep.table(
        ["metric", "ML baseline", "rule baseline"],
        [
            [esc("fit"), esc(f"{ml_stats.get('fit', 0):.0%}"), esc(f"{rule_stats.get('fit', 0):.0%}")],
            [esc("$ / task-hr"), esc(fmt_usd(ml_stats.get("cost_per_task_hr"))), esc(fmt_usd(rule_stats.get("cost_per_task_hr")))],
            [esc("median mem waste"), esc(fmt_pct(ml_stats.get("median_mem_waste_pct"))), esc(fmt_pct(rule_stats.get("median_mem_waste_pct")))],
        ],
    )
    rep.p(f"bundle: {len(ml.feature_names)} features + {len(ml.families)} family one-hots", color=GOOD)
    await rep.flush()

    return publish(
        await flyte.io.Dir.from_local(out_dir),
        ARTIFACT_ML_BASELINE,
        description=f"quantile-GBT on {len(train)} rows — heldout fit "
        f"{ml_stats.get('fit', 0):.0%}, {fmt_usd(ml_stats.get('cost_per_task_hr'))}/task-hr",
        kind="model",
    )
