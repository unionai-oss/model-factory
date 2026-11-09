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

Round 22 builds the same graph a second time as ``rightsize-k8s-bench``
(artifacts ``rt-k8sbench-*``): the standing corpora pinned, each run joined
with its Kubernetes twin (``k8s``), a larger training budget. Its arms
include ``jev-pointer-flyte``, the decision model trained on the Flyte rows
only, and it has no stacked arm. It adds ``rt-k8sbench-gate`` (``gating``):
novelty gates over the trained arms and the Flyte -> K8s adapter, no
retraining.

Round 21 adds the hybrids (``bench.hybrid``): the ``stack-jev-pointer`` arm
(the trees' estimate in the decision model's prompt), a second RL node
from it (``rt-allbench-rl-stack``), and ``rt-allbench-pool`` -- gbt-dist
pooled with every decision model, supervised, stacked and RL -- each with
its scorecard on the leaderboard. Since big-2 the CPU corpus is pinned
(``CPU_RELEASE``) and ``release`` names the GPU release joined.

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
#: big-2 joins a new GPU corpus with the standing CPU one (big-1), not a re-measured CPU corpus.
CPU_RELEASE = "big-1"

#: Artifacts are named ``<prefix>-<suffix>``: rt-allbench-corpus, rt-k8sbench-leaderboard, ...

PARTITION_RELEASE = "release"
PARTITION_ARM = "arm"


def _task(name: str):
    return Task.get(name, auto_version="latest")


def build_graph(name: str = FACTORY_NAME, prefix: str = "rt-allbench", gpu_release: str | None = None,
                platforms: str = "flyte", stack: bool = True, arm_knobs: dict | None = None, gate: bool = False,
                description: str = "RightSize Bench: the whole request -- memory, CPU and GPU -- under one cost model",
                ) -> factory.Factory:
    """The combined bench's graph. ``gpu_release`` pins the GPU corpus (None:
    the target's ``release`` names it); ``platforms`` is the corpus's
    (``"flyte,k8s"`` adds the Kubernetes twins); ``stack`` adds the stacked
    decision model's RL node; ``arm_knobs`` reach ``fit_all_arm`` (training
    budget, context length); ``gate`` adds the post-hoc arms (``gating``)."""
    def art(suffix: str) -> str:
        return f"{prefix}-{suffix}"

    mem = factory.source(SOURCE_MEM, partitions={PARTITION_RELEASE: str}, kind="data",
                         description="RightSizeMem's measured corpus")
    cpu = factory.source(SOURCE_CPU, partitions={PARTITION_RELEASE: str}, kind="data",
                         description="RightSizeCPU's measured corpus")
    gpu = factory.source(SOURCE_GPU, partitions={PARTITION_RELEASE: str}, kind="data",
                         description="RightSizeGPU's measured corpus")
    corpus_desc = "All three benches' runs, labelled on every axis each one measured"
    if "k8s" in platforms:
        corpus_desc += ", and each run's Kubernetes twin"
    # partitioned by release even when every source is pinned (round 22), so a release names its corpus
    corpus = factory.build(art("corpus"), partitions={PARTITION_RELEASE: str}, kind="data", description=corpus_desc,
                           ).using(_task("rt-allbench.join_corpora"), mem=mem.select(release=MEM_RELEASE),
                                   cpu=cpu.select(release=CPU_RELEASE),
                                   gpu=gpu.select(release=gpu_release) if gpu_release else gpu, platforms=platforms)
    arm = factory.build(art("arm"), partitions={PARTITION_RELEASE: str, PARTITION_ARM: str}, kind="model",
                        description="One arm's whole request (memory, CPU, GPU) for val/test/ood",
                        ).using(_task("rt-allbench.fit_all_arm"), corpus=corpus, arm=factory.partition(PARTITION_ARM),
                                release=factory.partition(PARTITION_RELEASE), **(arm_knobs or {}))
    rl = factory.build(art("rl"), kind="model",
                       description="The supervised decision model, RL-tuned on the joint cost",
                       ).using(_task("rt-allbench-train.rl_all_policy"), corpus=corpus, arm_dirs=arm.all(PARTITION_ARM))
    scorecard = factory.build(art("scorecard"), kind="data",
                              description="Joint regret, first-attempt success, per-axis accuracy -- per split and source",
                              ).using(_task("rt-allbench.score_all_arm"), corpus=corpus, arm_dir=arm,
                                      arm=factory.partition(PARTITION_ARM))
    rl_card = factory.build(art("rl-scorecard"), kind="data", description="The RL policy's scorecard",
                            ).using(_task("rt-allbench.score_all_arm"), corpus=corpus, arm_dir=rl, arm="rl-pointer")
    pool_inputs = {"corpus": corpus, "arm_dirs": arm.all(PARTITION_ARM), "rl_dir": rl}
    board_inputs = {"scorecards": scorecard.all(PARTITION_ARM), "rl_scorecard": rl_card}
    if stack:
        rl_stack = factory.build(art("rl-stack"), kind="model",
                                 description="The stacked decision model (trees' estimate in its prompt), RL-tuned on the joint cost",
                                 ).using(_task("rt-allbench-train.rl_all_policy"), corpus=corpus,
                                         arm_dirs=arm.all(PARTITION_ARM), base_arm="stack-jev-pointer")
        board_inputs["rl_stack_scorecard"] = factory.build(
            art("rl-stack-scorecard"), kind="data", description="The stacked RL policy's scorecard",
        ).using(_task("rt-allbench.score_all_arm"), corpus=corpus, arm_dir=rl_stack, arm="stack-rl-pointer")
        pool_inputs["rl_stack_dir"] = rl_stack
    pool = factory.build(art("pool"), kind="model",
                         description="gbt-dist x every decision model (supervised, stacked, RL), log pool per axis",
                         ).using(_task("rt-allbench.pool_all_arms"), **pool_inputs)
    board_inputs["pool_scorecard"] = factory.build(
        art("pool-scorecard"), kind="data", description="The pooled hybrids' scorecard",
    ).using(_task("rt-allbench.score_all_arm"), corpus=corpus, arm_dir=pool, arm="pool")
    if gate:
        gate_arm = factory.build(art("gate"), kind="model",
                                 description="Novelty gates (pooled champion on familiar runs, RL policy on novel "
                                             "ones) and the Flyte -> K8s adapter, over the trained arms",
                                 ).using(_task("rt-allbench.gate_all_arms"), corpus=corpus, arm_dirs=arm.all(PARTITION_ARM),
                                         rl_dir=rl, pool_dir=pool)
        board_inputs["counterfactual"] = factory.build(
            art("counterfactual"), kind="data",
            description="Sampled runs as Jobs under all four stagings, and with neutral names: does each approach read the manifest?",
        ).using(_task("rt-allbench-train.staging_counterfactual"), corpus=corpus, arm_dirs=arm.all(PARTITION_ARM),
                rl_dir=rl, pool_dir=pool)
        board_inputs["gate_scorecard"] = factory.build(
            art("gate-scorecard"), kind="data", description="The gates' and adapters' scorecard",
        ).using(_task("rt-allbench.score_all_arm"), corpus=corpus, arm_dir=gate_arm, arm="gate")
    board = factory.build(art("leaderboard"), kind="data", description="All arms side by side for one release",
                          ).using(_task("rt-allbench.all_leaderboard"), **board_inputs)
    return factory.Factory(name, board, description=description)


rightsize_all_bench = build_graph()

#: Round 22: every run of the standing corpora (RightSizeMem v2, RightSizeCPU
#: big-1, RightSizeGPU big-2) and its Kubernetes twin (``k8s``), every
#: training row the trainer's 12 h allows (L4, rt-l4timing-2: 0.9 s per training sample and
#: 0.24 s per scored row at 3,072 tokens -> 5,000 rows per source and platform, one epoch).
K8S_FACTORY_NAME = "rightsize-k8s-bench"
GPU_RELEASE_K8S = "big-2"
#: ``seed`` seeds the decision models (init, batch order; the rows are the same
#: draw): k8s-1 trained at 0 (twice: rsk-k8s-1, rsk-k8s-1-gate), k8s-2 at 1. A replicate
#: release ``k8s-2-s<k>`` trains the same corpus at seed k (``stations.seed_of``).
K8S_ARM_KNOBS = {"per_source": 5000, "epochs": 1, "max_len": 3072, "seed": 1}
#: The arms of a K8s release (``--partition arm=...``).
K8S_ARMS = ("static", "rules", "rules-sized", "gbt-dist", "gbt-dist-flyte", "gbt-dist-k8s", "zeroshot-27b",
            "zeroshot-27b-json", "jev-pointer", "jev-pointer-flyte", "jev-pointer-k8s")
rightsize_k8s_bench = build_graph(
    K8S_FACTORY_NAME, prefix="rt-k8sbench", gpu_release=GPU_RELEASE_K8S, platforms="flyte,k8s", stack=False,
    arm_knobs=K8S_ARM_KNOBS, gate=True,
    description="RightSize Bench, Flyte and Kubernetes: every run as a Flyte task and as a plain K8s Job")


if __name__ == "__main__":
    flyte.init_from_config()
    for f in (rightsize_all_bench, rightsize_k8s_bench):
        print(f.graph())
        print("validation:", f.validate() or "ok")
