"""RightSize Bench as a Union factory.

    flyte factory deploy bench_factory.py
    flyte factory materialize rightsize-bench rt-bench-leaderboard \
        --partition release=v1 \
        --partition arm=static,vpa,size-rule,gbt,gbt-dist,zeroshot,sft,jev

    rt-bench-runs        [release]        generate_bench_runs   code + inputs
         │
    rt-bench-corpus      [release]        measure_bench         ~N pods: run + measure
         │                         one instance per `arm`
         ├──────────┬─────────┬──── … ────┐
         ▼          ▼         ▼           ▼
    rt-bench-arm   [release, arm]          fit_bench_arm   CPU, hosted LLM, or GPU
         ▼          ▼         ▼           ▼
    rt-bench-scorecard [release, arm]      score_bench_arm
         └──────────┴─────────┴──── … ────┘
                              ▼  .all("arm")
    rt-bench-leaderboard [release]         bench_leaderboard
    rt-bench-live        [release]         live_validate   (arm.all -> real pods)

Why this shape:

- **`release` is the corpus's identity.** Measurements are ground truth only
  for one set of library pins and one generator; a new release is a new
  partition, never an overwrite, so every leaderboard says exactly which
  measurements it was computed from.
- **`arm` is a partition, not a loop.** Each arm is cached, retried and
  visible on its own. Adding an arm adds a partition value; a change to
  one arm re-materializes that arm and the leaderboard and reuses the
  expensive part -- thousands of measurement pods -- untouched.
- **The leaderboard reads `.all("arm")`**, so it is computed from exactly the
  scorecards that exist for this release, and lineage records each one.
"""

from __future__ import annotations

import flyte
from flyte.remote import Task
from flyteplugins.union import factory

FACTORY_NAME = "rightsize-bench"

ARTIFACT_RUNS = "rt-bench-runs"
ARTIFACT_CORPUS = "rt-bench-corpus"
ARTIFACT_ARM = "rt-bench-arm"
ARTIFACT_SCORECARD = "rt-bench-scorecard"
ARTIFACT_LEADERBOARD = "rt-bench-leaderboard"
ARTIFACT_LIVE = "rt-bench-live"

PARTITION_RELEASE = "release"
PARTITION_ARM = "arm"

#: Build constants. Override per materialization with
#: `--param rt-bench-runs.tasks_per_family=40` etc.
TASKS_PER_FAMILY = 200
TEACHER = "qwen35-397b"


def _task(name: str):
    return Task.get(name, auto_version="latest")


def build_graph() -> factory.Factory:
    generate = _task("rt-bench.generate_bench_runs")
    measure = _task("rt-bench.measure_bench")
    fit = _task("rt-bench.fit_bench_arm")
    score = _task("rt-bench.score_bench_arm")
    leaderboard = _task("rt-bench.bench_leaderboard")
    live = _task("rt-bench.live_validate")

    runs = factory.build(
        ARTIFACT_RUNS,
        partitions={PARTITION_RELEASE: str},
        kind="data",
        description="RightSize Bench workloads: real task code + input specs, one row per run",
    ).using(generate, release=factory.partition(PARTITION_RELEASE), tasks_per_family=TASKS_PER_FAMILY)

    corpus = factory.build(
        ARTIFACT_CORPUS,
        kind="data",
        description="Every run executed and measured (peak RSS, CPU, time to peak), labelled and split",
    ).using(measure, runs=runs)

    arm = factory.build(
        ARTIFACT_ARM,
        partitions={PARTITION_RELEASE: str, PARTITION_ARM: str},
        kind="model",
        description="One right-sizing arm, fit on train (knob tuned on val), predictions for val/test/ood",
    ).using(fit, corpus=corpus, arm=factory.partition(PARTITION_ARM), teacher=TEACHER)

    scorecard = factory.build(
        ARTIFACT_SCORECARD,
        kind="data",
        description="Cost to SUCCEEDED, OOM rate, regret vs oracle, calibration -- per split and slice",
    ).using(score, corpus=corpus, arm_dir=arm, arm=factory.partition(PARTITION_ARM))

    board = factory.build(
        ARTIFACT_LEADERBOARD,
        kind="data",
        description="All arms side by side for one release",
    ).using(leaderboard, scorecards=scorecard.all(PARTITION_ARM))

    live_check = factory.build(
        ARTIFACT_LIVE,
        kind="data",
        description="A stratified sample re-run in real pods under each arm's request: real vs simulated OOMs",
    ).using(live, corpus=corpus, arm_dirs=arm.all(PARTITION_ARM))

    return factory.Factory(
        FACTORY_NAME,
        board,
        live_check,
        description="RightSize Bench: rules vs GBT vs autoregressive LLMs vs a Jev-style decision model",
    )


#: The factory `flyte factory deploy` picks up.
rightsize_bench = build_graph()


if __name__ == "__main__":
    flyte.init_from_config()
    print(rightsize_bench.graph())
    print("validation:", rightsize_bench.validate() or "ok")
