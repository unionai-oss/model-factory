"""The factory: a closed-loop model factory as one declared graph of artifacts.

It declares what each artifact is made of and stops there — nothing runs until
somebody materializes a target:

    flyte factory deploy factory.py
    flyte factory materialize model-factory mf-promoted-model --partition date=2026-10-14

One release day D:

    built once (no partitions)            per day D (partitioned by date)
    ──────────────────────────            ──────────────────────────────────────
    mf-heldout-tasks ──┬─────────────────▶ mf-seed-tasks[D]  (excludes held-out)
          │            │                        │
          ▼            │   mf-champion[D] ──────┼──▶ mf-synthetic-tasks[D]
    mf-baseline-report │   (source)  │          │      (mutates champion's failures)
          │            │             │          ▼            │
          │            └─────────────┼───▶ mf-rl-tasks-dataset[D] ◀┘  [HITL gate 1 + leak check]
          │                          │          ▼
          │                          ├──▶ mf-policy-checkpoint[D]   (GRPO, warm-started
          │                          │          ▼                     from the champion)
          └──────────────────────────┼──▶ mf-eval-report[D]         (candidate vs champion
                                     │          ▼                     vs base, fixed held-out)
                                     └──▶ mf-promoted-model[D]      (better of the two)
                                                │                    [HITL gate 2]
                                                ├──▶ mf-inference   (serving app)
                                                ▼
              crown_champion (OnArtifact trigger, outside the graph)
                                                ▼
                                      mf-champion[D+1]  ──▶ … day D+1

How the loop closes
-------------------
Day D+1 must build on day D, but a factory graph is a DAG over artifact
names, so `promoted-model -> … -> promoted-model` is rejected as a cycle and
there is no "previous partition" mapping (`.window()` always includes the
consumer's own day). So the cross-day state is its own artifact,
`mf-champion`, which this graph only *reads* (`factory.source`). The eval team's
`crown_champion` task publishes `champion[D+1] = promoted[D]` whenever a
promoted model appears. Three things follow:

- Day D is a pure function of `champion[D]`: re-materializing it is all cache
  hits, and a backfill of D1..D3 runs day by day as champions get crowned.
- A day whose champion does not exist yet stops in seconds, before any GPU
  work, with "source partition not in the registry". Start the loop with
  `seed_champion --date <D0>` (the base model is champion zero).
- The release is champion/challenger: `promoted-model[D]` is the candidate
  only if it beats the champion; otherwise the champion carries forward. A bad
  day never moves the factory backwards and never breaks tomorrow's input.

The two human gates live inside station tasks (`assemble_dataset`,
`promote_checkpoint`) and default to open (`AUTO_APPROVE`) so an unattended
materialization finishes; `--param mf-rl-tasks-dataset.auto_approve=false`
arms one for a run.
"""

from __future__ import annotations

import os

import flyte
import flyte.io
from flyte.remote import Task
from flyteplugins.union import factory

from .contracts import (
    ARTIFACT_BASELINE,
    ARTIFACT_CHAMPION,
    ARTIFACT_CHECKPOINT,
    ARTIFACT_EVAL_REPORT,
    ARTIFACT_HELDOUT,
    ARTIFACT_PROMOTED,
    ARTIFACT_RL_DATASET,
    ARTIFACT_SEED_TASKS,
    ARTIFACT_SYNTHETIC,
    ENDPOINT_APP,
    PARTITION_DATE,
)
from .inference.service import inference_app_env

#: Sizing profile every station runs at — a constant in the deployed graph
#: (and therefore part of every build's cache key). `MF_PROFILE=smoke` at
#: deploy time gives a minutes-long machinery check; one build can be
#: overridden per run with `--param <artifact>.profile_name=...`.
DEFAULT_PROFILE = os.environ.get("MF_PROFILE", "daily")

#: Both human gates default to open so an unattended materialization (a cron
#: trigger, CI) completes. See the module docstring.
AUTO_APPROVE = True

FACTORY_NAME = "model-factory"


def _task(name: str):
    """A deployed station task, pinned to whatever version is current."""
    return Task.get(name, auto_version="latest")


def build_graph(profile: str = DEFAULT_PROFILE) -> factory.Factory:
    """Declare the factory. Importable without a cluster connection."""
    curate_heldout = _task("de-cpu.curate_heldout")
    ingest_and_curate = _task("de-cpu.ingest_and_curate")
    generate_synthetic_tasks = _task("de-gpu.generate_synthetic_tasks")
    assemble_dataset = _task("de-cpu.assemble_dataset")
    train_grpo = _task("trainer.train_grpo")
    evaluate_baseline = _task("eval-gpu.evaluate_baseline")
    evaluate_checkpoint = _task("eval-gpu.evaluate_checkpoint")
    promote_checkpoint = _task("eval-cpu.promote_checkpoint")

    # ── built once: the yardstick ─────────────────────────────────────────
    # No partitions, so every day reads the SAME version of each. That is what
    # makes "day 2 beat day 1" a claim about the model and not about the test.
    heldout = factory.build(
        ARTIFACT_HELDOUT,
        kind="data",
        description="Fixed held-out split; every day is scored on this version",
    ).using(curate_heldout, profile_name=profile)

    baseline = factory.build(
        ARTIFACT_BASELINE,
        kind="data",
        description="Base model pass@1 on the fixed held-out split",
    ).using(evaluate_baseline, heldout=heldout, profile_name=profile)

    # ── the loop's state: read, never built, by this graph ────────────────
    champion = factory.source(
        ARTIFACT_CHAMPION,
        type=flyte.io.Dir,
        partitions={PARTITION_DATE: factory.Daily},
        kind="model",
        description="The model day D starts from and must beat (crowned from promoted[D-1])",
    )

    # ── one release day ───────────────────────────────────────────────────
    # `ingest_and_curate` has a `date` parameter, so it receives the day being
    # built automatically — as a `datetime`. It picks the day's slice of the
    # seed dataset and drops anything in the held-out split.
    seed = factory.build(
        ARTIFACT_SEED_TASKS,
        partitions={PARTITION_DATE: factory.Daily},
        kind="data",
        description="The day's curated, oracle-verified training tasks (disjoint from held-out)",
    ).using(ingest_and_curate, heldout=heldout, profile_name=profile)

    synthetic = factory.build(
        ARTIFACT_SYNTHETIC,
        kind="data",
        description="Verified mutations of the champion's failures (failure-driven curriculum)",
    ).using(generate_synthetic_tasks, dataset=seed, champion=champion, heldout=heldout, profile_name=profile)

    dataset = factory.build(
        ARTIFACT_RL_DATASET,
        kind="data",
        description="Human-approved, leak-checked RL release (seed + synthetic)",
    ).using(assemble_dataset, seed=seed, synthetic=synthetic, heldout=heldout, auto_approve=AUTO_APPROVE)

    checkpoint = factory.build(
        ARTIFACT_CHECKPOINT,
        kind="model",
        description="LoRA adapter: GRPO with code-execution rewards, warm-started from the champion",
    ).using(train_grpo, dataset=dataset, champion=champion, profile_name=profile)

    report = factory.build(
        ARTIFACT_EVAL_REPORT,
        kind="data",
        description="Candidate vs champion vs base pass@1 on the fixed held-out split",
    ).using(
        evaluate_checkpoint,
        checkpoint=checkpoint,
        champion=champion,
        heldout=heldout,
        baseline=baseline,
        profile_name=profile,
    )

    promoted = factory.build(
        ARTIFACT_PROMOTED,
        kind="model",
        description="The day's release: the candidate if it beat the champion, else the champion",
    ).using(
        promote_checkpoint,
        checkpoint=checkpoint,
        champion=champion,
        eval_report=report,
        auto_approve=AUTO_APPROVE,
    )

    # `flyte factory deploy` builds only the factory's own image, so deploy the
    # app env once (`flyte deploy team_inference.py inference_app_env`) or the
    # app will crash-loop pulling an image that was never pushed.
    endpoint = factory.serve(
        ENDPOINT_APP,
        description="Serves the day's release",
    ).using(inference_app_env, model=promoted)

    return factory.Factory(
        FACTORY_NAME,
        promoted,
        endpoint,
        description="Closed-loop model factory: curate -> curriculum -> GRPO -> eval vs champion -> release -> serve",
        triggers=[
            # Nightly release of today's partition. It does GPU work only when
            # today's champion exists (yesterday's release was crowned); when
            # nothing changed it is all cache hits.
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
