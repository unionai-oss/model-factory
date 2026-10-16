"""Data station: generate one day's labelled permission-gate episodes.

Returns a plain File; the factory declares it as `permission-episodes` and
publishes it. CPU-only and seconds long — the oracle is a Python function, so
there is no model in the data path and no LLM bill for a training set.

The `date` partition seeds the sampler, so re-materializing a day after a
cache wipe regenerates byte-identical episodes, and a scorecard from last
week still refers to the data it was computed on.

Three splits, each sampled with its own seed rather than sliced from one
pool (slicing a stratified pool correlates the splits):

    train  what the heads learn from
    val    picks the champion and tunes the gate threshold
    test   what every reported number comes from
"""

from __future__ import annotations

from datetime import datetime

import flyte
import flyte.io
import flyte.report

from ..config import get_profile
from ..harness.observations import sample_episodes
from ..harness.policy import DECISIONS
from ..harness.scoring import majority_baseline
from ..shared import reporting
from .envs import data_env


def seed_for(date: datetime | None) -> int:
    """A stable per-day seed, so one day's episodes are reproducible."""
    if date is None:
        return 0
    return int(date.strftime("%Y%m%d"))


def _action_key(row: dict) -> str:
    """What the call *is*, ignoring its workspace: a command, a path, or a URL."""
    return f"{row['obs_tool']}|{row['obs_command'] or ''}|{row['obs_path'] or ''}|{row['obs_url'] or ''}"


def build_rows(train_n: int, val_n: int, test_n: int, seed: int) -> list[dict]:
    """All three splits as parquet-ready rows, with ``seen_in_train`` set.

    The sampler's templates recur, so many test *commands* also occur in
    train (in a different workspace, mode and branch). ``seen_in_train`` lets
    the scorecard report accuracy on calls the model has never seen verbatim.
    """
    rows: list[dict] = []
    for split, n, s in (("train", train_n, seed), ("val", val_n, seed + 1), ("test", test_n, seed + 2)):
        for i, ep in enumerate(sample_episodes(n, seed=s)):
            row = ep.to_row()
            row["split"] = split
            row["episode_id"] = f"{split}-{i:05d}-{row['episode_id']}"
            rows.append(row)
    seen = {_action_key(r) for r in rows if r["split"] == "train"}
    for r in rows:
        r["seen_in_train"] = r["split"] == "train" or _action_key(r) in seen
    return rows


@data_env.task(report=True)
async def generate_episodes(
    profile_name: str = "dev", date: datetime | None = None
) -> flyte.io.File:
    """Sample and label one day's episodes.

    ``date`` is the partition being built; a factory injects it automatically
    because the parameter is named like the ``Daily`` dimension — and it
    arrives as a ``datetime``, not a string.
    """
    import pandas as pd

    profile = get_profile(profile_name)
    seed = seed_for(date)
    rows = build_rows(profile.train_episodes, profile.val_episodes, profile.test_episodes, seed)
    if not rows:
        raise flyte.errors.NonRecoverableError("episode generation produced zero rows")
    df = pd.DataFrame(rows)
    out = "/tmp/permission_episodes.parquet"
    df.to_parquet(out, index=False)

    test = df[df["split"] == "test"]
    baseline = majority_baseline(list(test["label"]))
    body = reporting.stats_row(
        {
            "train": int((df["split"] == "train").sum()),
            "val": int((df["split"] == "val").sum()),
            "test": len(test),
            "adversarial (test)": f"{(test['slice'] == 'adversarial').mean():.0%}",
            "test calls unseen in train": f"{(~test['seen_in_train']).mean():.0%}",
            "seed": seed,
            "majority baseline": f"{baseline['accuracy']:.1%} ({baseline['decision']})",
        }
    )
    body += "<h3>Label mix by slice (train)</h3>"
    tr = df[df["split"] == "train"]
    body += reporting.table(
        ["slice", *DECISIONS, "n"],
        [[s, *[int(((tr["slice"] == s) & (tr["label"] == d)).sum()) for d in DECISIONS], int((tr["slice"] == s).sum())]
         for s in ("standard", "adversarial")],
    )
    body += "<h3>Which rule decided (train)</h3>"
    body += reporting.table(["rule", "n", ""], [[r, int(c), reporting.bar(c / len(tr))]
                                                for r, c in tr["rule"].value_counts().sort_index().items()])
    body += "<h3>Families (train)</h3>"
    fam = tr.groupby(["family", "label"]).size().unstack(fill_value=0)
    body += reporting.table(["family", *DECISIONS],
                            [[f, *[int(fam.loc[f].get(d, 0)) for d in DECISIONS]] for f in fam.index])

    body += "<h3>Adversarial samples — check the labels look right</h3>"
    sample = tr[tr["slice"] == "adversarial"].sample(min(14, len(tr)), random_state=7)
    body += reporting.table(
        ["family", "mode", "call", "label", "why"],
        [[r.family, r.obs_permission_mode,
          r.obs_command or (f"{r.obs_tool} {r.obs_path} -> {r.obs_resolved_path}" if r.obs_path else f"{r.obs_tool} {r.obs_url}"),
          r.label, r.rationale] for r in sample.itertuples()],
    )
    await flyte.report.replace.aio(reporting.page("Data card: permission-gate episodes", body))
    await flyte.report.flush.aio()
    return await flyte.io.File.from_local(out)
