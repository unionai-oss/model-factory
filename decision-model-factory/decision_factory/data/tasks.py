"""Data station: generate one day's labelled decision episodes.

Returns a plain File; the factory declares it as `decision-episodes` and
publishes it. The whole station is CPU-only and takes seconds — the oracle is
a Python function, so there is no model in the data path and no LLM bill for
generating a training set.

The `date` partition is used as the sampling seed, which makes a given day's
dataset reproducible: re-materializing `decision-episodes[2026-10-02]` after
a cache wipe regenerates byte-identical episodes, so a scorecard from last
week still refers to the data it was actually computed on.
"""

from __future__ import annotations

from datetime import datetime

import flyte
import flyte.io
import flyte.report

from ..config import get_profile
from ..harness.observations import sample_episodes
from ..harness.scoring import majority_baseline
from ..shared import reporting
from .envs import data_env


def seed_for(date: datetime | None) -> int:
    """A stable per-day seed, so one day's episodes are reproducible."""
    if date is None:
        return 0
    return int(date.strftime("%Y%m%d"))


@data_env.task(report=True)
async def generate_episodes(
    profile_name: str = "smoke", date: datetime | None = None
) -> flyte.io.File:
    """Sample and label one day's episodes, split into train and eval.

    ``date`` is the partition being built; a factory injects it automatically
    because the parameter is named like the ``Daily`` dimension — and it
    arrives as a ``datetime``, not a string.
    """
    import pandas as pd

    profile = get_profile(profile_name)
    seed = seed_for(date)

    # Train and eval are sampled with DIFFERENT seeds rather than sliced from
    # one pool. Slicing a stratified pool would correlate the splits (the same
    # near-boundary amounts recur), which inflates eval accuracy.
    train = sample_episodes(profile.train_episodes, seed=seed)
    evaluation = sample_episodes(profile.eval_episodes, seed=seed + 1)

    rows = []
    for split, episodes in (("train", train), ("eval", evaluation)):
        for ep in episodes:
            row = ep.to_row()
            row["split"] = split
            row["episode_id"] = f"{split}-{row['episode_id']}"
            rows.append(row)

    df = pd.DataFrame(rows)
    out = "/tmp/decision_episodes.parquet"
    df.to_parquet(out, index=False)

    eval_labels = [r["label_tool"] for r in rows if r["split"] == "eval"]
    baseline = majority_baseline(eval_labels)
    body = reporting.stats_row(
        {
            "train episodes": len(train),
            "eval episodes": len(evaluation),
            "near-boundary (train)": sum(1 for e in train if e.hard),
            "near-boundary (eval)": sum(1 for e in evaluation if e.hard),
            "seed": seed,
            "majority baseline": f"{baseline['accuracy']:.1%} ({baseline['tool']})",
        }
    )

    body += "<h3>Label mix</h3>"
    counts = df[df["split"] == "train"]["label_tool"].value_counts()
    eval_counts = df[df["split"] == "eval"]["label_tool"].value_counts()
    body += reporting.table(
        ["tool", "train", "eval", "train share"],
        [
            [
                tool,
                int(counts.get(tool, 0)),
                int(eval_counts.get(tool, 0)),
                reporting.bar(int(counts.get(tool, 0)) / max(len(train), 1)),
            ]
            for tool in counts.index.union(eval_counts.index)
        ],
    )

    body += "<h3>Sampled episodes (check the labels look right)</h3>"
    sample = df[df["split"] == "train"].sample(min(10, len(train)), random_state=7)
    body += reporting.table(
        ["episode", "message", "intent", "state", "label", "why"],
        [
            [
                r.episode_id,
                r.obs_customer_message,
                r.obs_intent,
                _state_summary(r),
                r.label_json,
                r.rationale,
            ]
            for r in sample.itertuples()
        ],
    )
    await flyte.report.replace.aio(reporting.page("Data card: decision episodes", body))
    await flyte.report.flush.aio()

    if len(df) == 0:
        raise flyte.errors.NonRecoverableError("episode generation produced zero rows")
    return await flyte.io.File.from_local(out)


def _state_summary(row) -> str:
    """The order state fields that actually drove the oracle's decision."""
    parts = []
    if row.obs_order_id:
        parts.append(f"order={row.obs_order_id}")
        parts.append("looked_up" if row.obs_order_looked_up else "NOT looked up")
    if row.obs_order_status:
        parts.append(str(row.obs_order_status))
    if row.obs_order_total_cents is not None and row.obs_order_total_cents == row.obs_order_total_cents:
        parts.append(f"{int(row.obs_order_total_cents)}c")
    if row.obs_days_since_delivery is not None and row.obs_days_since_delivery == row.obs_days_since_delivery:
        parts.append(f"{int(row.obs_days_since_delivery)}d")
    return ", ".join(parts) or "—"
