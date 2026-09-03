"""Factory stations: corpus → train → eval, plus the pipeline driver.

Artifact-wired like basic-model-factory, but the prototype keeps a driver
task as the spine instead of OnArtifact triggers — one readable E2E run
while the loop is being proven. Trigger wiring is a later, mechanical step.
"""

from __future__ import annotations

import tempfile

import flyte
import flyte.io

from ..config import get_profile
from ..contracts import ARTIFACT_TASK_CORPUS, publish
from ..taskgen.corpus import build_corpus
from .envs import driver_env
from .evaluate import eval_tuner
from .grpo import train_tuner


@driver_env.task(produces_artifacts=True)
async def build_task_corpus(profile_name: str = "smoke", seed: int = 0) -> flyte.io.File:
    """Sample the task corpus and publish it as tuning-task-corpus."""
    import pandas as pd

    profile = get_profile(profile_name)
    records = build_corpus(profile.train_contexts, profile.eval_contexts, seed=seed)
    path = tempfile.mktemp(suffix=".parquet")
    pd.DataFrame(records).to_parquet(path, index=False)
    out = await flyte.io.File.from_local(path)
    return publish(
        out,
        ARTIFACT_TASK_CORPUS,
        description=f"{profile.train_contexts} train / {profile.eval_contexts} heldout, seed={seed}",
    )


@driver_env.task(timeout=flyte.Timeout(max_runtime=8 * 3600))
async def tuner_pipeline(
    profile_name: str = "smoke", seed: int = 0, run_cluster_episodes: bool = True
) -> flyte.io.File:
    """E2E: corpus → GRPO train → eval report. Returns the eval report."""
    corpus = await build_task_corpus(profile_name=profile_name, seed=seed)
    checkpoint = await train_tuner(corpus=corpus, profile_name=profile_name)
    return await eval_tuner(
        corpus=corpus,
        checkpoint=checkpoint,
        profile_name=profile_name,
        run_cluster_episodes=run_cluster_episodes,
    )
