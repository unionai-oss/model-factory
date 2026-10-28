"""RightSizeCPU Bench as a Union factory.

    flyte factory deploy bench_cpu_factory.py
    flyte factory materialize rightsize-cpu-bench rt-cpubench-leaderboard \
        --partition release=small-1 \
        --partition arm=static,vpa,code-rule,gbt,gbt-dist,zeroshot-27b,jev-pointer

    rt-cpubench-runs        [release]        generate_cpu_runs
    rt-cpubench-corpus      [release]        measure_cpu_bench   one workload per pod at a time
    rt-cpubench-arm         [release, arm]   fit_cpu_arm         CPU, hosted LLM, or T4
    rt-cpubench-scorecard   [release, arm]   score_cpu_arm
    rt-cpubench-leaderboard [release]        cpu_leaderboard     .all("arm")

The same shape as RightSizeMem's ``rightsize-bench`` factory, under its own
artifact names (``rt-cpubench-*``): an artifact's partition schema is
immutable, and these are new artifacts.
"""

from __future__ import annotations

import flyte
from flyte.remote import Task
from flyteplugins.union import factory

FACTORY_NAME = "rightsize-cpu-bench"

ARTIFACT_RUNS = "rt-cpubench-runs"
ARTIFACT_CORPUS = "rt-cpubench-corpus"
ARTIFACT_ARM = "rt-cpubench-arm"
ARTIFACT_SCORECARD = "rt-cpubench-scorecard"
ARTIFACT_LEADERBOARD = "rt-cpubench-leaderboard"
ARTIFACT_POOL = "rt-cpubench-pool"
ARTIFACT_POOL_SCORECARD = "rt-cpubench-pool-scorecard"

PARTITION_RELEASE = "release"
PARTITION_ARM = "arm"


def _task(name: str):
    return Task.get(name, auto_version="latest")


def build_graph() -> factory.Factory:
    runs = factory.build(ARTIFACT_RUNS, partitions={PARTITION_RELEASE: str}, kind="data",
                         description="RightSizeCPU workloads: task code + input specs, one row per run",
                         ).using(_task("rt-cpubench.generate_cpu_runs"), release=factory.partition(PARTITION_RELEASE))
    corpus = factory.build(ARTIFACT_CORPUS, kind="data",
                           description="Every run executed alone in a 9-CPU pod; CPU sampled per second from the cgroup",
                           ).using(_task("rt-cpubench.measure_cpu_bench"), runs=runs)
    arm = factory.build(ARTIFACT_ARM, partitions={PARTITION_RELEASE: str, PARTITION_ARM: str}, kind="model",
                        description="One arm, fit on train (knob tuned on val), predictions for val/test/ood",
                        ).using(_task("rt-cpubench.fit_cpu_arm"), corpus=corpus, arm=factory.partition(PARTITION_ARM))
    scorecard = factory.build(ARTIFACT_SCORECARD, kind="data",
                              description="Peak within margin, under-prediction, throttling cost -- per split and slice",
                              ).using(_task("rt-cpubench.score_cpu_arm"), corpus=corpus, arm_dir=arm,
                                      arm=factory.partition(PARTITION_ARM))
    # The pooled hybrids, built from every arm of the release: lineage records
    # which tree and LLM arm versions each pool combined.
    pool = factory.build(ARTIFACT_POOL, kind="model",
                         description="gbt-dist x every distributional LLM variant, log pool, weight tuned on val",
                         ).using(_task("rt-cpubench.pool_cpu_arms"), corpus=corpus, arm_dirs=arm.all(PARTITION_ARM))
    pool_card = factory.build(ARTIFACT_POOL_SCORECARD, kind="data", description="The pooled hybrids' scorecard",
                              ).using(_task("rt-cpubench.score_cpu_arm"), corpus=corpus, arm_dir=pool, arm="pool")
    board = factory.build(ARTIFACT_LEADERBOARD, kind="data", description="All arms side by side for one release",
                          ).using(_task("rt-cpubench.cpu_leaderboard"), scorecards=scorecard.all(PARTITION_ARM),
                                  pool_scorecard=pool_card)
    return factory.Factory(FACTORY_NAME, board,
                           description="RightSizeCPU Bench: predict peak CPU from code and input metadata")


rightsize_cpu_bench = build_graph()


if __name__ == "__main__":
    flyte.init_from_config()
    print(rightsize_cpu_bench.graph())
    print("validation:", rightsize_cpu_bench.validate() or "ok")
