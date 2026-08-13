"""Factory driver + dark-mode triggers.

Two ways to run the factory:

1. ``run_factory`` — one explicit loop iteration with human-in-the-loop
   condition gates at the two judgment layers (data validation, checkpoint
   promotion). This is the demo/E2E entrypoint.

2. Dark mode — deployed artifact/cron triggers so stations fire themselves:
   - new `rl-tasks-dataset` version  → training
   - new `policy-checkpoint` version → evaluation (+ auto-gate + HITL gate)
   - new `synthetic-tasks` version   → merge + publish new dataset version
   - nightly cron                    → synthetic generation batch
   Deploy with: ``flyte deploy src/model_factory/pipeline.py factory_env``
"""

from __future__ import annotations

import json
from datetime import timedelta

import flyte
import flyte.io
import flyte.report

from . import reporting
from .config import (
    ARTIFACT_CHECKPOINT,
    ARTIFACT_RL_DATASET,
    ARTIFACT_SYNTHETIC,
    get_profile,
)
from .data import ingest_and_curate, merge_datasets, publish_dataset
from .envs import factory_env
from .evaluate import evaluate_checkpoint, promote_checkpoint
from .synthetic import generate_synthetic_tasks
from .train import train_grpo

GATE_TIMEOUT = timedelta(hours=4)


async def _gate(name: str, prompt_md: str, auto_approve: bool) -> bool:
    """Human-in-the-loop condition gate; auto_approve short-circuits for CI."""
    if auto_approve:
        return True
    condition = await flyte.new_condition.aio(
        name,
        prompt=prompt_md,
        prompt_type="markdown",
        data_type=bool,
        timeout=GATE_TIMEOUT,
    )
    try:
        return bool(await condition.wait.aio())
    except flyte.errors.ConditionTimedoutError:
        return False


@factory_env.task(report=True)
async def run_factory(
    profile_name: str = "smoke",
    auto_approve: bool = False,
    include_synthetic: bool = True,
) -> str:
    """One full factory iteration: data → gate → train → eval → gate → promote."""
    profile = get_profile(profile_name)
    log: list[str] = []

    async def note(msg: str) -> None:
        log.append(msg)
        body = "<ol>" + "".join(f"<li>{reporting.esc(m)}</li>" for m in log) + "</ol>"
        await flyte.report.replace.aio(reporting.page(f"Factory run — {profile.name}", body))
        await flyte.report.flush.aio()

    # ── Station 1: data ────────────────────────────────────────────────
    await note("Ingesting + curating seed dataset…")
    candidate = await ingest_and_curate(profile_name=profile_name)

    # ── Station 1b: synthetic data (batch inference) ───────────────────
    if include_synthetic:
        await note("Generating oracle-verified synthetic tasks (batch inference)…")
        synthetic = await generate_synthetic_tasks(dataset=candidate, profile_name=profile_name)
        candidate = await merge_datasets(base=candidate, extra=synthetic)
        await note("Merged synthetic tasks into candidate dataset.")

    # ── HITL gate 1: data validation ───────────────────────────────────
    await note("Waiting on data-validation gate (see data-card report)…")
    approved = await _gate(
        "approve-dataset",
        "## Data validation gate\n\n"
        "Inspect the **data card report** on the `ingest_and_curate` action "
        "(curation stats, difficulty mix, random samples) and the synthetic "
        "generation report.\n\n**Approve this dataset for training?**",
        auto_approve,
    )
    if not approved:
        await note("Dataset REJECTED at data-validation gate. Stopping.")
        return "rejected: data validation gate"
    dataset = await publish_dataset(dataset=candidate, note=f"profile={profile.name}")
    await note(f"Dataset approved and published as artifact `{ARTIFACT_RL_DATASET}`.")

    # ── Station 2: training ────────────────────────────────────────────
    await note(f"Training GRPO ({profile.base_model}, {profile.max_steps} steps, {profile.gpu})…")
    checkpoint = await train_grpo(dataset=dataset, profile_name=profile_name)
    await note(f"Checkpoint published as artifact `{ARTIFACT_CHECKPOINT}`.")

    # ── Station 3: evaluation ──────────────────────────────────────────
    await note("Evaluating candidate vs base on held-out tasks…")
    eval_report = await evaluate_checkpoint(
        checkpoint=checkpoint, dataset=dataset, profile_name=profile_name
    )
    ev = json.loads(open(await eval_report.download()).read())
    await note(
        f"Eval: candidate {ev['candidate_pass_at_1']:.2%} vs base "
        f"{ev['base_pass_at_1']:.2%} (Δ {ev['delta']:+.2%}); "
        f"auto gate {'PASS' if ev['auto_gate_passed'] else 'FAIL'}."
    )
    if not ev["auto_gate_passed"]:
        await note("Auto quality gate failed — checkpoint NOT promoted.")
        return f"not promoted: delta {ev['delta']:+.2%} below margin"

    # ── HITL gate 2: promotion / vibe check ────────────────────────────
    await note("Waiting on promotion gate (see eval report)…")
    promoted = await _gate(
        "approve-promotion",
        "## Checkpoint promotion gate\n\n"
        f"Candidate pass@1 **{ev['candidate_pass_at_1']:.2%}** vs base "
        f"**{ev['base_pass_at_1']:.2%}** (Δ {ev['delta']:+.2%}) on "
        f"{ev['n_heldout']} held-out tasks.\n\nInspect the eval report "
        "(per-task table + sample completions) for reward hacking or "
        "degenerate outputs.\n\n**Promote this checkpoint?**",
        auto_approve,
    )
    if not promoted:
        await note("Checkpoint REJECTED at promotion gate.")
        return "not promoted: human gate"
    await promote_checkpoint(checkpoint=checkpoint, eval_report=eval_report)
    await note("Checkpoint PROMOTED. Factory iteration complete.")
    return (
        f"promoted: candidate {ev['candidate_pass_at_1']:.2%} vs base "
        f"{ev['base_pass_at_1']:.2%} on {ev['n_heldout']} held-out tasks"
    )


# ── Dark mode: artifact + cron triggers ─────────────────────────────────
# Each trigger binds a new artifact version to the task input, so a published
# dataset kicks off training, a checkpoint kicks off eval, and synthetic
# batches fold themselves into the next dataset version.

_train_trigger = flyte.Trigger(
    name="train-on-new-dataset",
    automation=flyte.OnArtifact(name=ARTIFACT_RL_DATASET),
    inputs={"dataset": flyte.TriggeredArtifact, "profile_name": "smoke"},
    description="New approved dataset version → GRPO training",
    auto_activate=False,  # activate deliberately: flyte update trigger ...
)


@factory_env.task(triggers=[_train_trigger])
async def train_on_new_dataset(dataset: flyte.io.File, profile_name: str = "smoke") -> str:
    checkpoint = await train_grpo(dataset=dataset, profile_name=profile_name)
    return f"trained checkpoint from dataset; artifact `{ARTIFACT_CHECKPOINT}` updated"


_eval_trigger = flyte.Trigger(
    name="eval-on-new-checkpoint",
    automation=flyte.OnArtifact(name=ARTIFACT_CHECKPOINT),
    inputs={"checkpoint": flyte.TriggeredArtifact, "profile_name": "smoke"},
    description="New checkpoint version → evaluation + gates",
    auto_activate=False,
)


@factory_env.task(triggers=[_eval_trigger], report=True)
async def eval_on_new_checkpoint(checkpoint: flyte.io.Dir, profile_name: str = "smoke") -> str:
    """Dark-mode eval: fetch the latest approved dataset, eval, gate, promote."""
    dataset_art = flyte.remote.Artifact.get(ARTIFACT_RL_DATASET)
    dataset = flyte.io.File.from_existing_remote(dataset_art.url)  # latest version
    eval_report = await evaluate_checkpoint(
        checkpoint=checkpoint, dataset=dataset, profile_name=profile_name
    )
    ev = json.loads(open(await eval_report.download()).read())
    if not ev["auto_gate_passed"]:
        return f"not promoted: delta {ev['delta']:+.2%} below margin"
    approved = await _gate(
        "approve-promotion",
        f"## Promotion gate (dark mode)\n\nΔ pass@1 {ev['delta']:+.2%}. Promote?",
        auto_approve=False,
    )
    if not approved:
        return "not promoted: human gate"
    await promote_checkpoint(checkpoint=checkpoint, eval_report=eval_report)
    return "promoted"


_synthetic_merge_trigger = flyte.Trigger(
    name="merge-synthetic-tasks",
    automation=flyte.OnArtifact(name=ARTIFACT_SYNTHETIC),
    inputs={"synthetic": flyte.TriggeredArtifact},
    description="New synthetic batch → merge into dataset (human gate) → publish",
    auto_activate=False,
)


@factory_env.task(triggers=[_synthetic_merge_trigger])
async def merge_synthetic_into_dataset(synthetic: flyte.io.File) -> str:
    dataset_art = flyte.remote.Artifact.get(ARTIFACT_RL_DATASET)
    base = flyte.io.File.from_existing_remote(dataset_art.url)
    merged = await merge_datasets(base=base, extra=synthetic)
    approved = await _gate(
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


@factory_env.task(triggers=[_nightly_synthetic])
async def nightly_synthetic_batch(profile_name: str = "smoke") -> str:
    dataset_art = flyte.remote.Artifact.get(ARTIFACT_RL_DATASET)
    dataset = flyte.io.File.from_existing_remote(dataset_art.url)
    await generate_synthetic_tasks(dataset=dataset, profile_name=profile_name)
    return f"synthetic batch generated; artifact `{ARTIFACT_SYNTHETIC}` updated"
