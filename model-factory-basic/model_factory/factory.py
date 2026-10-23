"""The factory: the whole dark loop as one declared graph of artifacts.

This module is the wiring that used to be spread across OnArtifact triggers,
a release driver and an integration script. It declares what each artifact is
made of and stops there — nothing runs until somebody materializes a target:

    flyte factory deploy factory.py
    flyte factory materialize model-factory mf-promoted-model --partition date=2026-10-02

                    ┌───────────────┐
                    │ mf-seed-tasks │  ingest_and_curate
                    └──────┬────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         │
      ┌───────────────┐                 │
      │mf-synthetic-  │ generate_…      │
      │tasks          │                 │
      └───────┬───────┘                 │
              └────────────┬────────────┘
                           ▼
                  ┌──────────────────┐
                  │mf-rl-tasks-      │  assemble_dataset  [HITL gate 1]
                  │dataset           │
                  └────────┬─────────┘
                           ▼
                  ┌──────────────────┐
                  │mf-policy-        │  train_grpo
                  │checkpoint        │
                  └────────┬─────────┘
                           ▼
                  ┌──────────────────┐
                  │  mf-eval-report  │  evaluate_checkpoint
                  └────────┬─────────┘
                           ▼
                  ┌──────────────────┐
                  │ mf-promoted-model│  promote_checkpoint  [HITL gate 2]
                  └────────┬─────────┘
                           ▼
                  ╭──────────────────╮
                  │   mf-inference   │  serve (app)
                  ╰──────────────────╯

Everything is partitioned by ``date`` — one release per day — declared once on
the first build and inherited downstream by identity mapping. The synthetic
loop that used to be a cycle (dataset -> synthetic -> dataset) is a DAG here:
each day's synthetic batch is mutated from *that day's* curated seed tasks.

Why this replaced the trigger chain
-----------------------------------
The OnArtifact wiring made each station fire on the artifact upstream of it,
which worked but meant: no way to ask for one target, no reuse between runs,
no backfill, and an `integration.py` that duplicated the chain just to test
it. A factory gives all four from one declaration — and the gates still live
inside the station tasks, so `mf-promoted-model` cannot exist without a human
having approved both the data and the promotion.

The two gates are wired to `auto_approve` constants here so a smoke
materialization runs unattended; drop the constants (or override them per run
with `--param mf-rl-tasks-dataset.auto_approve=false`) to arm them.
"""

from __future__ import annotations

import flyte
from flyte.remote import Task
from flyteplugins.union import factory

from .contracts import (
    ARTIFACT_CHECKPOINT,
    ARTIFACT_EVAL_REPORT,
    ARTIFACT_PROMOTED,
    ARTIFACT_RL_DATASET,
    ARTIFACT_SEED_TASKS,
    ARTIFACT_SYNTHETIC,
    ENDPOINT_APP,
    PARTITION_DATE,
)

#: Sizing profile every station runs at. A materialization can override it per
#: build with `--param <artifact>.profile_name=dev`.
DEFAULT_PROFILE = "smoke"

#: Both human gates default to open so an unattended materialization (CI, a
#: cron trigger, a smoke run) completes. See the module docstring.
AUTO_APPROVE = True

FACTORY_NAME = "model-factory"


def _task(name: str):
    """A deployed station task, pinned to whatever version is current."""
    return Task.get(name, auto_version="latest")


def build_graph() -> factory.Factory:
    """Declare the factory. Importable without a cluster connection."""
    ingest_and_curate = _task("de-cpu.ingest_and_curate")
    generate_synthetic_tasks = _task("de-gpu.generate_synthetic_tasks")
    assemble_dataset = _task("de-cpu.assemble_dataset")
    train_grpo = _task("trainer.train_grpo")
    evaluate_checkpoint = _task("eval-gpu.evaluate_checkpoint")
    promote_checkpoint = _task("eval-cpu.promote_checkpoint")

    # The one build that declares the partition dimension; everything
    # downstream inherits `date` by identity mapping. `ingest_and_curate` has a
    # `date` parameter, so it receives the day being built automatically — as a
    # `datetime`, which is why the task annotates it that way.
    seed = factory.build(
        ARTIFACT_SEED_TASKS,
        partitions={PARTITION_DATE: factory.Daily},
        kind="data",
        description="Curated, oracle-verified seed tasks for one release day",
    ).using(ingest_and_curate, profile_name=DEFAULT_PROFILE)

    synthetic = factory.build(
        ARTIFACT_SYNTHETIC,
        kind="data",
        description="Oracle-verified synthetic tasks mutated from the day's seed tasks",
    ).using(generate_synthetic_tasks, dataset=seed, profile_name=DEFAULT_PROFILE)

    dataset = factory.build(
        ARTIFACT_RL_DATASET,
        kind="data",
        description="Human-approved RL coding-task release (seed + synthetic)",
    ).using(assemble_dataset, seed=seed, synthetic=synthetic, auto_approve=AUTO_APPROVE)

    checkpoint = factory.build(
        ARTIFACT_CHECKPOINT,
        kind="model",
        description="LoRA adapter from GRPO with code-execution rewards",
    ).using(train_grpo, dataset=dataset, profile_name=DEFAULT_PROFILE)

    report = factory.build(
        ARTIFACT_EVAL_REPORT,
        kind="data",
        description="Candidate-vs-base pass@1 on the held-out split",
    ).using(
        evaluate_checkpoint,
        checkpoint=checkpoint,
        dataset=dataset,
        profile_name=DEFAULT_PROFILE,
        # Generate in-task, not through the serving app. The app is DOWNSTREAM
        # of this build (it serves `promoted-model`, which is gated on this
        # report), so routing eval through it would make the graph circular —
        # and on the first materialization the app does not exist at all.
        use_service=False,
    )

    promoted = factory.build(
        ARTIFACT_PROMOTED,
        kind="model",
        description="Checkpoint that cleared the auto margin and the human gate",
    ).using(
        promote_checkpoint,
        checkpoint=checkpoint,
        eval_report=report,
        auto_approve=AUTO_APPROVE,
    )

    # Importing the app env is what makes `serve` possible, and it is also why
    # the app's image has to be built before an endpoint is materialized:
    # `flyte factory deploy` builds only the factory's own image, so deploy the
    # app env once (`flyte deploy team_inference.py inference_app_env`) or the
    # app will crash-loop pulling an image that was never pushed.
    from .inference.service import inference_app_env

    endpoint = factory.serve(
        ENDPOINT_APP,
        description="Serves the promoted model for rollouts and evals",
    ).using(inference_app_env, model=promoted)

    return factory.Factory(
        FACTORY_NAME,
        promoted,
        endpoint,
        description="Dark model factory: curate -> synthesize -> GRPO -> eval -> promote -> serve",
        triggers=[
            # Nightly release. When nothing upstream changed this is all cache
            # hits and finishes in seconds, so it is cheap to leave armed.
            factory.on(flyte.Cron("0 6 * * *"), promoted, endpoint, name="nightly-release"),
        ],
    )


#: The factory `flyte factory deploy` picks up from this module.
model_factory = build_graph()


if __name__ == "__main__":
    # `python -m model_factory.factory` prints the graph and any validation
    # problems without touching the cluster beyond task lookups.
    flyte.init_from_config()
    print(model_factory.graph())
    problems = model_factory.validate()
    print("validation:", problems or "ok")
