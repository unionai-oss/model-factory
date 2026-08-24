"""Data engineering release driver + dark-mode triggers.

`data_release` is the team's public entrypoint: curate → (synthetic) →
human data-validation gate → publish a new `rl-tasks-dataset` version.
Publishing IS the hand-off: the training team's OnArtifact trigger takes it
from there — data engineering neither imports nor calls training code.
"""

from __future__ import annotations

import flyte
import flyte.io

from ..contracts import ARTIFACT_RL_DATASET, ARTIFACT_SYNTHETIC
from ..shared import assets
from ..shared.gates import gate
from .envs import de_cpu_env
from .synthetic import generate_synthetic_tasks
from .tasks import ingest_and_curate, merge_datasets, publish_dataset


@de_cpu_env.task(report=True, entrypoint=True)
async def data_release(
    profile_name: str = "smoke",
    auto_approve: bool = False,
    include_synthetic: bool = True,
) -> flyte.io.File:
    """Produce and publish one validated dataset release."""
    candidate = await ingest_and_curate(profile_name=profile_name)
    if include_synthetic:
        synthetic = await generate_synthetic_tasks(dataset=candidate, profile_name=profile_name)
        candidate = await merge_datasets(base=candidate, extra=synthetic)

    approved = await gate(
        "approve-dataset",
        "## Data validation gate\n\n"
        "Inspect the **data card report** on `ingest_and_curate` (curation "
        "stats, difficulty mix, random samples) and the synthetic-generation "
        "report.\n\n**Approve this dataset for release?**",
        auto_approve,
    )
    if not approved:
        raise flyte.errors.NonRecoverableError("dataset rejected at data-validation gate")
    return await publish_dataset(dataset=candidate, note=f"profile={profile_name}")


# ── dark-mode triggers ──────────────────────────────────────────────────

_synthetic_merge_trigger = flyte.Trigger(
    name="merge-synthetic-tasks",
    automation=flyte.OnArtifact(name=ARTIFACT_SYNTHETIC),
    inputs={"synthetic": flyte.TriggeredArtifact},
    description="New synthetic batch → merge, gate, publish new dataset version",
    auto_activate=False,
)


@de_cpu_env.task(triggers=[_synthetic_merge_trigger])
async def merge_synthetic_into_dataset(synthetic: flyte.io.File) -> str:
    latest_dataset = await assets.latest(ARTIFACT_RL_DATASET)
    base = flyte.io.File.from_existing_remote(latest_dataset.path)
    merged = await merge_datasets(base=base, extra=synthetic)
    approved = await gate(
        "approve-dataset",
        "## Data validation gate (dark mode)\n\nA synthetic batch was merged. "
        "Inspect the synthetic-generation report. Publish new dataset version?",
        auto_approve=False,
    )
    if not approved:
        return "merge rejected at data gate"
    await publish_dataset(dataset=merged, note="synthetic merge")
    return f"published new `{ARTIFACT_RL_DATASET}` version"


_nightly_synthetic = flyte.Trigger(
    name="nightly-synthetic-generation",
    automation=flyte.Cron("0 6 * * *"),
    inputs={"profile_name": "smoke"},
    description="Nightly synthetic task generation batch",
    auto_activate=False,
)


@de_cpu_env.task(triggers=[_nightly_synthetic])
async def nightly_synthetic_batch(profile_name: str = "smoke") -> str:
    latest_dataset = await assets.latest(ARTIFACT_RL_DATASET)
    dataset = flyte.io.File.from_existing_remote(latest_dataset.path)
    await generate_synthetic_tasks(dataset=dataset, profile_name=profile_name)
    return f"synthetic batch generated; artifact `{ARTIFACT_SYNTHETIC}` updated"
