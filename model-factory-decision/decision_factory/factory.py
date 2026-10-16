"""The decision-model factory: a coding agent's permission gate.

Generate oracle-labelled tool calls, fine-tune every torso x head in the
suite, score them, promote the cheapest-to-be-wrong one, and serve it.

    flyte factory deploy factory.py
    flyte factory materialize decision-factory permission-champion --partition date=2026-10-15 \
        --partition model=qwen2.5-0.5b-pointer,qwen2.5-0.5b-sft,...
    flyte factory materialize decision-factory permission-gate --partition date=2026-10-15 ...

              ┌────────────────────────┐
              │  permission-episodes   │  generate_episodes          [date]
              │   (oracle-labelled)    │
              └───────────┬────────────┘
                          │                      one instance per suite member
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
    ┌───────────┐   ┌───────────┐   ┌───────────┐
    │permission-│   │permission-│   │permission-│  finetune_decision_model
    │  model    │   │  model    │   │  model    │            [date, model]
    │[…-pointer]│   │ […-sft]   │   │   […]     │
    └─────┬─────┘   └─────┬─────┘   └─────┬─────┘
          ▼               ▼               ▼
    ┌───────────┐   ┌───────────┐   ┌───────────┐
    │permission-│   │permission-│   │permission-│  score_model
    │ scorecard │   │ scorecard │   │ scorecard │            [date, model]
    └─────┬─────┘   └─────┬─────┘   └─────┬─────┘
          └───────────────┼───────────────┘
                          ▼  .all("model")  — collapses the suite
              ┌────────────────────────┐
              │  permission-champion   │  select_champion            [date]
              └───────────┬────────────┘
                          ▼
              ╭────────────────────────╮
              │    permission-gate     │  serve (app)
              ╰────────────────────────╯

The shape worth noticing is the fan-out and the collapse. `model` is a string
partition dimension whose values are the suite's slugs, so the two middle
builds exist once per model — separately cached, separately retried,
separately visible in the console — and `select_champion` takes
`.all("model")` to see the whole suite at once. "Train N models, compare them,
promote the best" is therefore a declaration, not a loop inside a task that
has to fetch its siblings' outputs and that re-runs everything whenever one
model changes.

Adding a model to the suite changes the graph (one more partition value), so
it needs a redeploy — unlike the per-build constants, which `--param` can
override per materialization.
"""

from __future__ import annotations

import flyte
from flyte.remote import Task
from flyteplugins.union import factory

from .config import DEFAULT_PROFILE, get_profile
from .contracts import (
    ARTIFACT_CHAMPION,
    ARTIFACT_EPISODES,
    ARTIFACT_MODEL,
    ARTIFACT_SCORECARD,
    ENDPOINT_APP,
    PARTITION_DATE,
    PARTITION_MODEL,
)

FACTORY_NAME = "decision-factory"


def _task(name: str):
    return Task.get(name, auto_version="latest")


def build_graph(profile_name: str = DEFAULT_PROFILE) -> factory.Factory:
    """Declare the factory. Importable without a cluster connection."""
    profile = get_profile(profile_name)

    generate_episodes = _task("df-data.generate_episodes")
    finetune_decision_model = _task("df-trainer.finetune_decision_model")
    score_model = _task("df-scorer.score_model")
    select_champion = _task("df-selector.select_champion")

    # Declares the `date` dimension. `generate_episodes` has a `date`
    # parameter, so it receives the day being built automatically — as a
    # `datetime`, which is why the task annotates it that way.
    episodes = factory.build(
        ARTIFACT_EPISODES,
        partitions={PARTITION_DATE: factory.Daily},
        kind="data",
        description="Oracle-labelled coding-agent tool calls (allow / ask_user / deny) for one day",
    ).using(generate_episodes, profile_name=profile_name)

    # `model` is a dimension no input carries, which is exactly what
    # `partitions=` is for — and it must also re-declare `date`, which the
    # episodes input does carry. `factory.partition("model")` passes the slug
    # being built to the task, so one declaration covers the whole suite.
    candidate = factory.build(
        ARTIFACT_MODEL,
        partitions={PARTITION_DATE: factory.Daily, PARTITION_MODEL: str},
        kind="model",
        description="One torso x head (pointer or sft) fine-tuned on the day's episodes",
    ).using(
        finetune_decision_model,
        episodes=episodes,
        model=factory.partition(PARTITION_MODEL),
        profile_name=profile_name,
    )

    scorecard = factory.build(
        ARTIFACT_SCORECARD,
        kind="data",
        description="Accuracy, unsafe-allow, calibration, latency and the autonomy/safety frontier for one fine-tune",
    ).using(
        score_model,
        candidate=candidate,
        episodes=episodes,
        model=factory.partition(PARTITION_MODEL),
        profile_name=profile_name,
    )

    # The collapse: `.all("model")` hands the task a LIST covering every suite
    # member for this day, and the output drops back to `[date]` only.
    champion = factory.build(
        ARTIFACT_CHAMPION,
        kind="model",
        description="Lowest val expected-cost suite member, with its tuned gate threshold",
    ).using(
        select_champion,
        candidates=candidate.all(PARTITION_MODEL),
        scorecards=scorecard.all(PARTITION_MODEL),
        profile_name=profile_name,
    )

    # Importing the app env is what makes `serve` possible. It is also why the
    # app image must exist first: `flyte factory deploy` builds only the
    # factory's own image, so deploy the app env once
    # (`flyte deploy app.py permission_app_env`) or the app crash-loops pulling
    # an image that was never pushed.
    from .serving.service import permission_app_env

    endpoint = factory.serve(
        ENDPOINT_APP,
        description="Serves the champion as a permission gate (/decide, /oracle)",
    ).using(permission_app_env, model=champion)

    return factory.Factory(
        FACTORY_NAME,
        champion,
        endpoint,
        description=(
            "Coding-agent permission gate: oracle-labelled tool calls -> fine-tune torso x head "
            f"-> score -> serve the winner (suite: {', '.join(profile.models)})"
        ),
        triggers=[
            # A fresh suite every morning. When nothing changed this is all
            # cache hits and finishes in seconds, so it is cheap to leave on.
            factory.on(flyte.Cron("0 5 * * *"), champion, endpoint, name="nightly-suite"),
        ],
    )


#: The factory `flyte factory deploy` picks up from this module.
decision_factory = build_graph()

#: The partition selector a materialization needs: the suite is a list of
#: `model` values, so a full run is
#: `--partition date=<day> --partition model=<slug>,<slug>,...`.
#: `materialize_args()` builds that for the declared profile.
def materialize_args(profile_name: str = DEFAULT_PROFILE) -> dict[str, object]:
    profile = get_profile(profile_name)
    return {PARTITION_MODEL: list(profile.models)}


if __name__ == "__main__":
    flyte.init_from_config()
    print(decision_factory.graph())
    print("validation:", decision_factory.validate() or "ok")
    print("suite:", get_profile(DEFAULT_PROFILE).models)
