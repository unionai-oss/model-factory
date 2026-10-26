"""RightSize Bench (all three benches at once) as a Union factory.

    flyte factory deploy bench_all_factory.py
    flyte factory materialize rightsize-all-bench rt-allbench-leaderboard \
        --partition release=big-1 \
        --partition arm=static,rules,gbt-dist,zeroshot-27b,jev-pointer

    rt-bench-corpus[v2]   rt-cpubench-corpus[release]   rt-gpubench-corpus[release]     sources
            └───────────────────────┼──────────────────────────────┘
    rt-allbench-corpus      [release]        join_corpora
            ├────────────┬──── … ────┐
    rt-allbench-arm         [release, arm]   fit_all_arm        CPU, hosted LLM, or T4
            │            └─── .all("arm") ──► rt-allbench-rl [release]   rl_all_policy (T4)
    rt-allbench-scorecard   [release, arm]   score_all_arm      rt-allbench-rl-scorecard
            └──── .all("arm") ───────────────────────┬────────────┘
    rt-allbench-leaderboard [release]        all_leaderboard

The three measured corpora are **sources**: the benches' own factories
publish them, and this factory reads them. `release` names the CPU and GPU
releases joined (``big-1`` joins ``rt-cpubench-corpus[release=big-1]`` and
``rt-gpubench-corpus[release=big-1]``); RightSizeMem is pinned to its
standing release, v2. The RL node is built from the arms (``.all("arm")``)
and picks the supervised decision model out of them, so lineage records
which supervised model every RL policy started from.
"""

from __future__ import annotations

import flyte
from flyte.remote import Task
from flyteplugins.union import factory

FACTORY_NAME = "rightsize-all-bench"

SOURCE_MEM = "rt-bench-corpus"
SOURCE_CPU = "rt-cpubench-corpus"
SOURCE_GPU = "rt-gpubench-corpus"
MEM_RELEASE = "v2"

ARTIFACT_CORPUS = "rt-allbench-corpus"
ARTIFACT_ARM = "rt-allbench-arm"
ARTIFACT_RL = "rt-allbench-rl"
ARTIFACT_SCORECARD = "rt-allbench-scorecard"
ARTIFACT_RL_SCORECARD = "rt-allbench-rl-scorecard"
ARTIFACT_LEADERBOARD = "rt-allbench-leaderboard"

PARTITION_RELEASE = "release"
PARTITION_ARM = "arm"


def _task(name: str):
    return Task.get(name, auto_version="latest")


def build_graph() -> factory.Factory:
    mem = factory.source(SOURCE_MEM, partitions={PARTITION_RELEASE: str}, kind="data",
                         description="RightSizeMem's measured corpus")
    cpu = factory.source(SOURCE_CPU, partitions={PARTITION_RELEASE: str}, kind="data",
                         description="RightSizeCPU's measured corpus")
    gpu = factory.source(SOURCE_GPU, partitions={PARTITION_RELEASE: str}, kind="data",
                         description="RightSizeGPU's measured corpus")
    corpus = factory.build(ARTIFACT_CORPUS, kind="data",
                           description="All three benches' runs, labelled on every axis each one measured",
                           ).using(_task("rt-allbench.join_corpora"), mem=mem.select(release=MEM_RELEASE),
                                   cpu=cpu, gpu=gpu)
    arm = factory.build(ARTIFACT_ARM, partitions={PARTITION_RELEASE: str, PARTITION_ARM: str}, kind="model",
                        description="One arm's whole request (memory, CPU, GPU) for val/test/ood",
                        ).using(_task("rt-allbench.fit_all_arm"), corpus=corpus, arm=factory.partition(PARTITION_ARM))
    rl = factory.build(ARTIFACT_RL, kind="model",
                       description="The supervised decision model, RL-tuned on the joint cost",
                       ).using(_task("rt-bench-gpu.rl_all_policy"), corpus=corpus, arm_dirs=arm.all(PARTITION_ARM))
    scorecard = factory.build(ARTIFACT_SCORECARD, kind="data",
                              description="Joint regret, first-attempt success, per-axis accuracy -- per split and source",
                              ).using(_task("rt-allbench.score_all_arm"), corpus=corpus, arm_dir=arm,
                                      arm=factory.partition(PARTITION_ARM))
    rl_card = factory.build(ARTIFACT_RL_SCORECARD, kind="data", description="The RL policy's scorecard",
                            ).using(_task("rt-allbench.score_all_arm"), corpus=corpus, arm_dir=rl, arm="rl-pointer")
    board = factory.build(ARTIFACT_LEADERBOARD, kind="data", description="All arms side by side for one release",
                          ).using(_task("rt-allbench.all_leaderboard"), scorecards=scorecard.all(PARTITION_ARM),
                                  rl_scorecard=rl_card)
    return factory.Factory(FACTORY_NAME, board,
                           description="RightSize Bench: the whole request -- memory, CPU and GPU -- under one cost model")


rightsize_all_bench = build_graph()


if __name__ == "__main__":
    flyte.init_from_config()
    print(rightsize_all_bench.graph())
    print("validation:", rightsize_all_bench.validate() or "ok")
