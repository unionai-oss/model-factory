"""The decision-arm factory: a suite of decision models over one corpus.

    flyte factory deploy factory.py
    flyte factory materialize resource-tuner rt-decision-champion \
        --partition arm=balanced,oom-averse,oom-paranoid

        ┌──────────────────────┐
        │ tuning-task-corpus   │   factory.source  (published OUTSIDE the
        │   (source)           │                    factory, by the existing
        └──────────┬───────────┘                    build_task_corpus station)
                   │                         one instance per `arm` value
       ┌───────────┼───────────┐
       ▼           ▼           ▼
  ┌─────────┐ ┌─────────┐ ┌─────────┐
  │rt-      │ │rt-      │ │rt-      │  fit_decision_model        [arm]
  │decision-│ │decision-│ │decision-│
  │model    │ │model    │ │model    │
  │[balanced│ │[oom-av…]│ │[oom-pa…]│
  └────┬────┘ └────┬────┘ └────┬────┘
       ▼           ▼           ▼
  ┌─────────┐ ┌─────────┐ ┌─────────┐  score_decision_model      [arm]
  │rt-decision-scorecard × 3        │
  └────┬────┘ └────┬────┘ └────┬────┘
       └───────────┼───────────┘
                   ▼  .all("arm")
        ┌──────────────────────┐
        │ rt-decision-champion │  select_decision_tuner    (no partitions)
        └──────────────────────┘

Why a `source` and not a build
------------------------------
The corpus is produced by `build_task_corpus`, which still publishes
`tuning-task-corpus` itself and still fires the existing OnArtifact triggers
that drive the LLM/GRPO arm. Declaring it as `factory.source(...)` means this
factory *reads* that artifact rather than owning it — so adding the whole
decision-model arm required **no change to the LLM path**: not to
`build_task_corpus`, not to `train_tuner`, not to the trigger wiring, not to
the tune service.

It also unlocks the trigger shape a factory cannot get from its own builds:
`factory.on(corpus)` fires a fresh suite whenever a new corpus version is
published, which is exactly the "new corpus -> refit" automation the LLM and
GBT arms already have, expressed once instead of per station.

Why `arm` is a partition
------------------------
`under_penalty` — how many times worse an OOM is than the same-size
over-provision — is the knob that decides the whole policy's character, and
nobody knows the right value a priori. Making it a partition value means each
setting is trained, scored and compared as its own artifact instance, the
frontier is a table rather than an argument, and re-running after changing one
arm rebuilds that arm only.

Partition naming
----------------
These artifacts are `rt-`-prefixed because an artifact's partition schema is
fixed in the registry by its first version: `tuner-checkpoint`,
`ml-baseline-model` and friends already exist UNPARTITIONED from earlier
rounds, so a partitioned build under those names is rejected at deploy. The
source keeps its original name precisely because it is unpartitioned.
"""

from __future__ import annotations

import flyte
from flyte.remote import Task
from flyteplugins.union import factory

from .contracts import ARTIFACT_TASK_CORPUS
from .training.decision_model import DEFAULT_ARMS

FACTORY_NAME = "resource-tuner"

# ── artifact names (partitioned; see the module docstring) ──────────────
ARTIFACT_DECISION_MODEL = "rt-decision-model"
ARTIFACT_DECISION_SCORECARD = "rt-decision-scorecard"
ARTIFACT_DECISION_CHAMPION = "rt-decision-champion"

#: The partition dimension the suite fans out over: a `DecisionArm` name.
PARTITION_ARM = "arm"


def _task(name: str):
    return Task.get(name, auto_version="latest")


def build_graph() -> factory.Factory:
    """Declare the factory. Importable without a cluster connection."""
    fit_decision_model = _task("rt-decision.fit_decision_model")
    score_decision_model = _task("rt-decision.score_decision_model")
    select_decision_tuner = _task("rt-decision.select_decision_tuner")

    # Type and partitions come from the registry at deploy. The existing
    # corpus is unpartitioned, so this source carries no dimensions — which
    # is why `arm` has to be declared on the build below.
    corpus = factory.source(
        ARTIFACT_TASK_CORPUS,
        description="Tuning task corpus, published by the build_task_corpus station",
    )

    model = factory.build(
        ARTIFACT_DECISION_MODEL,
        partitions={PARTITION_ARM: str},
        kind="model",
        description="Multi-head decision model over the discretized resource grid",
    ).using(fit_decision_model, corpus=corpus, arm=factory.partition(PARTITION_ARM))

    scorecard = factory.build(
        ARTIFACT_DECISION_SCORECARD,
        kind="data",
        description="Simulated fit rate / OOM rate / waste / $-per-task-hr for one arm",
    ).using(
        score_decision_model,
        model_dir=model,
        corpus=corpus,
        arm=factory.partition(PARTITION_ARM),
    )

    champion = factory.build(
        ARTIFACT_DECISION_CHAMPION,
        kind="model",
        description="Best decision arm: highest fit rate, cheapest as tie-break",
    ).using(
        select_decision_tuner,
        models=model.all(PARTITION_ARM),
        scorecards=scorecard.all(PARTITION_ARM),
    )

    return factory.Factory(
        FACTORY_NAME,
        champion,
        description=(
            "Resource-tuner decision arm: fit a suite of cost-asymmetry "
            f"configurations over the discretized action grid ({', '.join(DEFAULT_ARMS)})"
        ),
        triggers=[
            # The shape a factory can only get from a source: a new corpus
            # version refits the whole suite. All cache hits when nothing
            # changed, so it is cheap to leave armed.
            factory.on(corpus, champion, name="on-new-corpus"),
        ],
    )


#: The factory `flyte factory deploy` picks up from this module.
resource_tuner_factory = build_graph()


def materialize_args() -> dict[str, object]:
    """The partition selector a full suite run needs."""
    return {PARTITION_ARM: list(DEFAULT_ARMS)}


if __name__ == "__main__":
    flyte.init_from_config()
    print(resource_tuner_factory.graph())
    print("validation:", resource_tuner_factory.validate() or "ok")
    print("suite:", DEFAULT_ARMS)
