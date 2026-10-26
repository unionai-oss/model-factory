"""RightSizeGPU Bench as a Union factory.

    flyte factory deploy bench_gpu_factory.py
    flyte factory materialize rightsize-gpu-bench rt-gpubench-leaderboard \
        --partition release=small-1 \
        --partition arm=static,vpa,size-rule,gbt,gbt-dist,zeroshot-27b,jev-pointer

    rt-gpubench-runs        [release]        generate_gpu_runs
    rt-gpubench-corpus      [release]        measure_gpu_bench   T4 pass; CUDA OOMs re-run on an L40S
    rt-gpubench-arm         [release, arm]   fit_gpu_arm         CPU, hosted LLM, or T4
    rt-gpubench-scorecard   [release, arm]   score_gpu_arm
    rt-gpubench-leaderboard [release]        gpu_leaderboard     .all("arm")

The same shape as RightSizeMem's ``rightsize-bench`` factory, under its own
artifact names (``rt-gpubench-*``): an artifact's partition schema is
immutable, and these are new artifacts.
"""

from __future__ import annotations

import flyte
from flyte.remote import Task
from flyteplugins.union import factory

FACTORY_NAME = "rightsize-gpu-bench"

ARTIFACT_RUNS = "rt-gpubench-runs"
ARTIFACT_CORPUS = "rt-gpubench-corpus"
ARTIFACT_ARM = "rt-gpubench-arm"
ARTIFACT_SCORECARD = "rt-gpubench-scorecard"
ARTIFACT_LEADERBOARD = "rt-gpubench-leaderboard"

PARTITION_RELEASE = "release"
PARTITION_ARM = "arm"


def _task(name: str):
    return Task.get(name, auto_version="latest")


def build_graph() -> factory.Factory:
    runs = factory.build(ARTIFACT_RUNS, partitions={PARTITION_RELEASE: str}, kind="data",
                         description="RightSizeGPU workloads: PyTorch task code + input specs, one row per run",
                         ).using(_task("rt-gpubench.generate_gpu_runs"), release=factory.partition(PARTITION_RELEASE))
    corpus = factory.build(ARTIFACT_CORPUS, kind="data",
                           description="Every run on a T4 (CUDA OOMs again on an L40S); NVML memory and utilization every 0.5 s",
                           ).using(_task("rt-gpubench.measure_gpu_bench"), runs=runs)
    arm = factory.build(ARTIFACT_ARM, partitions={PARTITION_RELEASE: str, PARTITION_ARM: str}, kind="model",
                        description="One arm, fit on train (knob tuned on val), predictions for val/test/ood",
                        ).using(_task("rt-gpubench.fit_gpu_arm"), corpus=corpus, arm=factory.partition(PARTITION_ARM))
    scorecard = factory.build(ARTIFACT_SCORECARD, kind="data",
                              description="Device cost with OOM retries, memory and utilization accuracy -- per split and slice",
                              ).using(_task("rt-gpubench.score_gpu_arm"), corpus=corpus, arm_dir=arm,
                                      arm=factory.partition(PARTITION_ARM))
    board = factory.build(ARTIFACT_LEADERBOARD, kind="data", description="All arms side by side for one release",
                          ).using(_task("rt-gpubench.gpu_leaderboard"), scorecards=scorecard.all(PARTITION_ARM))
    return factory.Factory(FACTORY_NAME, board,
                           description="RightSizeGPU Bench: predict GPU memory and utilization from code and input metadata")


rightsize_gpu_bench = build_graph()


if __name__ == "__main__":
    flyte.init_from_config()
    print(rightsize_gpu_bench.graph())
    print("validation:", rightsize_gpu_bench.validate() or "ok")
