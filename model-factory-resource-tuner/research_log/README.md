# Research log

Audit trail of resource-tuner experiments: what ran, where it ran, what it
showed, and what changed because of it. One entry file per experiment
round, newest last in the index.

Conventions:
- Every cluster run gets its console URL. Base:
  `https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/<run>`
- Failed runs stay in the log — a failure that changed the code is a
  result. Each failure links the commit that fixed it.
- Metrics come from the eval-report artifacts (also served by the
  [lineage dashboard](https://rt-lineage-resource-tuner-model-factory-development.apps.demo.hosted.unionai.cloud)
  and its `/api/lineage` JSON).
- New entries: copy the structure of an existing entry — context, run
  table, findings, code changes, links. Add the entry here.

## Entries

| date | entry | tl;dr |
|---|---|---|
| 2026-09-02 | [design-and-scaffold](2026-09-02-design-and-scaffold.md) | Research + design phase; project scaffold, sim-first env, PR #6 |
| 2026-09-03 | [round-1-smoke-experiments](2026-09-03-round-1-smoke-experiments.md) | First cluster runs: env works, thinking-budget + batch-divisibility + unschedulable-proposal bugs found and fixed; stage-A saturates |
| 2026-09-03 | [round-2-dark-loop-and-teacher](2026-09-03-round-2-dark-loop-and-teacher.md) | Triggers live (dark runs), synthetic pipeline via qwen38-27b + execution oracle, dev-scale training moves waste 83%→47% |
| 2026-09-03 | [round-3-flyte-2.7-metrics](2026-09-03-round-3-flyte-2.7-metrics.md) | flyte 2.7 + public metrics plugin; teacher auth via LLM_SERVICE_API_KEY; Cloudflare 1010 UA-ban diagnosed; per-task live reports |
| 2026-09-04 | [round-4-archetype-100k](2026-09-04-round-4-archetype-100k.md) | 100k-pipeline corpus via teacher archetypes × oracle-calibrated instantiation; hash-salt reproducibility fix |
| 2026-09-04 | [round-5-last-mile-tune-service](2026-09-04-round-5-last-mile-tune-service.md) | Tune service + @tune.resources; FINAL A/B: OOM 13%→0%, fit 87%→100%, waste 54%→47% vs hard-coded prior; serve-best + batcher-await lessons |
| 2026-09-04 | [round-6-value-ledger](2026-09-04-round-6-value-ledger.md) | Append-only Dir value ledger + rt-tune dashboard (cumulative savings, task registry); 3-workflow × 2-scale demo; TunedTask duck-typing + weakref-prior fixes; uniform-4Gi OOMs embed/large |
| 2026-09-04 | [round-7-reward-shaping](2026-09-04-round-7-reward-shaping.md) | Reward-shape menu + \$ pricing + GPU estimation; all dev-scale arms LOSE money; c-cost least bad; GPU proposals 0% valid |
| 2026-09-05 | [round-8-scale-arms](2026-09-05-round-8-scale-arms.md) | Same arms at 250k/4096-ctx/300-step scale: ALL arms save \$52-55/1k task-hrs (+31%); shapes barely separate; gate still fails on waste/fit |
| 2026-09-04 | [round-9-checkpointing](2026-09-04-round-9-checkpointing.md) | Intra-task + artifact checkpoints + warm start, all proven via chaos runs (2 bugs caught); Qwen3.5 RL unblocked; 500k corpus + ambitious 4B run launched |
| 2026-09-05 | [round-10-ml-baseline](2026-09-05-round-10-ml-baseline.md) | Quantile-GBT baseline: 99% fit vs policy 55%; TIED at ~\$0.22/successful task-hr — the LLM has not yet earned its GPU |
| 2026-09-06 | [round-11-format-capacity-composition](2026-09-06-round-11-format-capacity-composition.md) | Graded JSON reward -> 100% validity on every arm; **FIRST GATE PASS** (Qwen3.5-4B, +\$40.52/1k task-hrs); GBT-hint hits 93% fit but copies the anchor's padding; full FT > LoRA on fit |
| 2026-09-09 | [round-12-14-corpus-at-scale](2026-09-09-round-12-14-corpus-at-scale.md) | Corpus quality gates (2 gate bugs, recovered via `flyte fork`) + stack axis + teacher tiers; **the 1M release shipped 31,704 rows and reported success** — post-mortem: deadline unenforceable inside a wave, retries feeding dead endpoints, 7 pods to reject broken code, ~17 min/pod at 1/10 the available fan-out |
| 2026-09-16 | [round-15-fullft-1day](2026-09-16-round-15-fullft-1day.md) | Artifact cards on every published artifact; **the 1M corpus target is met** (1,016,896 rows); 1-day Qwen3-4B FULL fine-tune launched after 3 dead runs that each fixed something real — `flyte run` re-registers trainer resources, 10⁶ rows can't be read with pandas, preflight was blind to full-FT optimizer state |
| 2026-10-15 | [round-17-rightsize-bench](2026-10-15-round-17-rightsize-bench.md) | **RightSize Bench**: 21,323 runs of real task code measured in pods; rules vs quantile GBT vs zero-shot 27B vs SFT vs Jev-style decision heads, one cost model, 600 live pods. 4B decision model 25.8% memory regret vs GBT 34.0%; GBT x decision pool **20.8% at 1.7% OOM**; decision head beats generation at 1/9-1/17 the latency; unseen families still 79-91% for all |
| 2026-10-02 | [round-16-decision-arm-and-factory](2026-10-02-round-16-decision-arm-and-factory.md) | Third arm: a multi-head **decision model** that picks a grid cell directly, trained with a cost-asymmetry loss; the arm suite is a Union **factory** fanning out over an `under_penalty` partition with the corpus as a `factory.source` (LLM path untouched). The knob is monotone across all 3 arms (fit 54%→71%→89%) but only the paranoid arm beats the rule baseline on fit, and at +5% cost — 3 dead runs first (1M rows OOM the station, `Reporter` API, `baseline_proposal` arity) |
| 2026-10-23 | [round-18-rightsize-cpu-gpu](2026-10-23-round-18-rightsize-cpu-gpu.md) | RightSize Bench renamed **RightSizeMem**; **RightSizeCPU** (peak cores in Union's 15 s monitoring window, cgroup-measured) and **RightSizeGPU** (peak GPU memory -> device on the T4/L4/L40S ladder, mean utilization; NVML at DCGM's 30 s cadence) small releases, one factory materialization each. CPU: zero-shot 27B 79%/93% within ±25%, code-rule 81%/95% (fails the GIL), 1.7B decision head 71%/83% from 116 rows. GPU: no arm above 55% within ±20% at 62 training rows; utilization saturates (88% of runs 90-100%). Factory deploy pins station versions (rsg-small-3 re-ran old code) |
| 2026-10-24 | [round-19-rightsize-cpu-gpu-full](2026-10-23-round-19-rightsize-cpu-gpu-full.md) | Full releases (880 CPU / 640 GPU runs; 3-run shards, `full-*` sizing, 4B decision arm). **CPU ordering flips**: 4B decision head 89.6% / 88.7% within ±25% (test / unseen), trees 89.6% / 76.2%, zero-shot 27B falls to 74.6% / 83.8% and still fails phased code (8%). **GPU memory stays near 50%** (gbt-dist 50.5% test, SFT 37.3% unseen); 26 encoder runs unmeasurable (outputs accumulate on device); utilization still saturated |
| 2026-10-26 | [round-20-rightsize-big-and-combined](2026-10-24-round-20-rightsize-big-and-combined.md) | **big** releases at ~10^4 runs, plus a combined memory+CPU+GPU bench with a GRPO-style RL arm. CPU big-1 (rsc-big-1, 23 h): 1.7B decision head 97.9% within ±25% on test, but unseen families stay at 84-86% (4B 86.1%, 27B zero-shot 83.7%, gbt-dist 68.4%), no better than full-1. Combined shakedown (rsa-full-3): RL head argmax 12.6% test regret vs 23.6% supervised. **Combined big-1 (rsa-big-1)**: with 10,500 supervised runs the decision model reaches 9.7% and RL argmax 8.0% (lead 11 -> 1.7 points); on unseen families RL is worse (37.3% vs 34.1%) and `rules` is best (32.5%). **GPU big-1 (rsg-big-3)**: gbt-dist 71.3% within ±20% on test, 4B decision model 50.5% on unseen (trees 25.7%); accuracy decoding never leaves the T4 (6.4% OOM), argmax reads cost 13-16% |
| 2026-10-28 | [round-21-gpu-big-2-hybrids](2026-10-26-round-21-gpu-big-2-hybrids.md) | **GPU big-2** on the cluster's whole machine ladder (T4 -> 4xL40S; sharded lm_finetune / llm_generate / recsys_train): 74.7% of runs fit a T4 (big-1: 94.8%), but 4xL4 / 4xL40S capacity never came and 1,569 large runs are censored. **Trees x LLM hybrids** (stacked hints, log pools) on CPU, GPU and combined: best in distribution everywhere (CPU 98.5%, GPU 74.8%, combined **3.2% regret** for RL from the stacked model), worst learned arm on unseen families (combined 77-126% vs rl-pointer 35.5%): the trust in the trees is learned where the trees are right. Only the combined pool beats its LLM on both splits |

## Standing results — round 21: big releases with trees x LLM hybrids (as of 2026-10-28)

From [rsc-big-2](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rsc-big-2)
(CPU big-1, test n=1,634 / unseen n=1,800),
[rsg-big2-1](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rsg-big2-1)
(GPU big-2, new workloads over the whole machine ladder, n=1,202 / 2,390)
and [rsa-big-2](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rsa-big-2)
(combined: v2 + CPU big-1 + GPU big-2, n=5,095 / 6,416). The GPU big-2
numbers are not comparable with GPU big-1 below.

| arm | CPU: within ±25% (test / unseen) | GPU big-2: within ±20% (test / unseen) | combined: regret (test / unseen) |
|---|---|---|---|
| rules (code-rule / size-rule) | 73.6% / 79.4% | 37.1% / 31.6% | 39.4% / 48.0% |
| gbt-dist | 94.6% / 68.4% | 71.9% / 9.8% | 15.4% / 90.8% |
| jev-pointer (1.7B) | 97.7% / 85.6% | 58.2% / 33.6% | 14.0% / 58.8% |
| jev-pointer-4b | 97.9% / **86.3%** | 55.5% / **42.1%** | -- |
| rl-pointer / -argmax | -- | -- | 20.3% / **35.5%** ; 9.8% / 47.8% |
| stack-jev-pointer | **98.5%** / 71.7% | 73.4% / 20.5% | 10.1% / 94.6% |
| stack-rl-pointer-argmax | -- | -- | **3.2%** / 102.0% |
| pool-jev-pointer | 97.6% / 73.1% | 73.7% / 27.7% | 6.9% / 52.2% |
| best pool | pool-jev-pointer-4b 98.3% / 82.2% | pool-stack-jev-pointer **74.8%** / 16.2% | pool-jev-pointer 6.9% / 52.2% |

Details: [round-21](2026-10-26-round-21-gpu-big-2-hybrids.md).

## Standing results — RightSizeCPU / RightSizeGPU big-1 (as of 2026-10-26)

Test n=1,634 / 1,413, unseen n=1,800 / 2,411 (zeroshot-27b: ~300 per
split). From
[rsc-big-1](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rsc-big-1)
and [rsg-big-3](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rsg-big-3).

| arm | CPU: peak within ±25% (test / unseen) | GPU: peak memory within ±20% (test / unseen) |
|---|---|---|
| static | 31.0% / 26.7% | 8.7% / 10.3% |
| vpa | 64.6% / 60.6% | 35.3% / 33.1% |
| code-rule (CPU) / size-rule (GPU) | 73.6% / 79.4% | 37.1% / 34.8% |
| gbt-dist | 94.6% / 68.4% | **71.3%** / 25.7% |
| zeroshot-27b | 68.4% / 83.7% | 37.5% / 37.3% |
| sft-greedy (1.7B) | 94.9% / 75.2% | 58.5% / 44.5% |
| jev-pointer (1.7B) | **97.9%** / 83.9% | 63.3% / 45.0% |
| jev-pointer-4b | 97.6% / **86.1%** | 64.9% / **50.5%** |

Combined bench (memory + CPU + GPU, mean regret vs the oracle, test /
unseen, [rsa-big-1](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rsa-big-1)):
rules 34.0% / **32.5%**, gbt-dist 14.1% / 33.2%, jev-pointer 9.7% / 34.1%,
**rl-pointer-argmax 8.0%** / 37.3%.

Details and caveats: [round-20](2026-10-24-round-20-rightsize-big-and-combined.md);
full releases in [round-19](2026-10-23-round-19-rightsize-cpu-gpu-full.md),
small in [round-18](2026-10-23-round-18-rightsize-cpu-gpu.md).

## Standing results — RightSizeMem Bench (as of 2026-10-17; called RightSize Bench then)

Measured labels (`ru_maxrss` in pods), scheduler-visible inputs only, memory
regret vs oracle (lower is better), v2 corpus (14,234 runs), from
[rsb-v2-board-1](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rsb-v2-board-1).

| arm | test | OOM | unseen families | per decision |
|---|---|---|---|---|
| static | 255.6% | 8.9% | 231.8% | -- |
| size-rule | 118.7% | 23.0% | 81.3% | -- |
| gbt-dist | 34.0% | 2.5% | 84.9% | ~1 ms |
| zeroshot-27b | 154.5% | 9.7% | 192.6% | ~630 tokens, ~93 s |
| sft-sampled (1.7B) | 39.1% | 2.2% | 91.5% | 2,087 ms (T4) |
| jev-pointer (1.7B) | 33.2% | 3.2% | 90.8% | 126 ms, 0 tokens |
| jev-pointer-4b | 25.8% | 3.0% | 86.6% | 256 ms, 0 tokens |
| **gbt-dist x jev-pointer, pooled** | **20.8%** | **1.7%** | **79.0%** | both |

Full tables, live validation and caveats: [round-17](2026-10-15-round-17-rightsize-bench.md).
Results page: `model-factory-resource-tuner/benchmarks/rightsize-mem-bench.html`;
every bench's pages are indexed in [`benchmarks/README.md`](../benchmarks/README.md).

The rounds 1-16 tables below were scored against analytic footprints, not
measurements, and are superseded by the bench.

## Standing results — LLM arm (as of 2026-09-03)

Eval reports across checkpoints (policy vs rule-based baseline, held-out split):

| producing run | profile/steps | validity | fit vs baseline | median waste vs baseline | gate |
|---|---|---|---|---|---|
| [utxxsdc7529ngc855rxg](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/utxxsdc7529ngc855rxg) | smoke/10 (pre-no_think) | 0% | 0% vs 69% | – vs 27% | fail |
| [udc23ee7b0f53c7d2](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/udc23ee7b0f53c7d2) | dark eval | 100% | 88% vs 75% | 84% vs 28% | fail |
| [us4xhjj2z48kshwkkdpd](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/us4xhjj2z48kshwkkdpd) | smoke-composite/30 | 100% | 91% vs 75% | 83% vs 28% | fail |
| [uafafffa71dc77534](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/uafafffa71dc77534) | dark eval | 100% | 91% vs 75% | 83% vs 28% | fail |
| [u5f66af9fdeb76356](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/u5f66af9fdeb76356) | dark eval | 100% | 81% vs 84% | 83% vs 22% | fail |
| [u76392a10c7169269](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/u76392a10c7169269) | dark eval | 100% | 75% vs 84% | 83% vs 22% | fail |
| [u5e6d929fd544e39f](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/u5e6d929fd544e39f) | dark eval | 100% | 94% vs 84% | **46% vs 22%** | fail |
| [u859bl7b9xp7xsnl899v](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/u859bl7b9xp7xsnl899v) | **dev/150** | 100% | **98% vs 64%** | **47% vs 25%** | fail |

Reading: the reward curriculum moved each metric in order — validity
0→100%, fit 0→98%, waste 83→47%. The gate still fails on waste vs the
baseline (47% vs 25%); closing that gap is a training-scale problem
(next: longer runs / larger corpus / stage-B weight tuning), not a
machinery problem.

## Standing results — decision arm (as of 2026-10-02)

Heldout split, 512 workloads, corpus `uhrmbq9th9pw` (1,016,896 rows; 60,000
sampled for training). Rule baseline = per-family median, fitted on the same
train rows. From [rt-decision-r4](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rt-decision-r4).

| arm | `under_penalty` | fit | OOM | median waste | grid floor | $/task-hr | $ saved / 1k task-hr |
|---|---|---|---|---|---|---|---|
| **oom-paranoid** (champion) | 40 | **89.1%** | 10.9% | 69.1% | 36.1% | 0.1558 | **-7.68** |
| oom-averse | 12 | 70.9% | 29.1% | 61.8% | 36.1% | 0.1495 | -1.34 |
| balanced | 2 | 54.3% | 45.7% | 38.4% | 36.1% | 0.1418 | +6.34 |
| *rule baseline* | - | *76.0%* | *24.0%* | *54.1%* | - | *0.1481* | - |

Reading: the `under_penalty` knob is **monotone on every metric** across all
three arms, which is what the arm was built to demonstrate. But only the
paranoid arm beats the rule baseline on fit rate, and it costs ~5% more per
task-hour — so like the LLM arm in round 10, the decision arm has not yet
earned its place on cost. The grid itself forces 36.1% median
over-provisioning, so roughly half the champion's waste is the action space
rather than the model; the unrun `fine` action space is the next experiment
and is one extra partition value.

## Key links

- Product context: [AI Resource Tuning PRD](https://app.notion.com/p/AI-Resource-Tuning-3cb8cc06513d81f4b381c8294419a920)
  ([markdown source](https://github.com/unionai/prds/blob/main/product_prds/ai_resource_tuning/prd.md))
- Design doc: [../docs/DESIGN.md](../docs/DESIGN.md)
- Dashboard (lineage graph + eval charts):
  <https://rt-lineage-resource-tuner-model-factory-development.apps.demo.hosted.unionai.cloud>
- Cluster project: [resource-tuner-model-factory / development](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs)
- Metrics plugin: [flyteplugins-union @ niels/get-metrics](https://github.com/unionai/flyteplugins-union/tree/niels/get-metrics)
- Teacher LLMs: [qwen38-27b](https://demo.hosted.unionai.cloud/v2/domain/development/project/llm-service/apps/qwen38-27b) ·
  [minimax-m3](https://demo.hosted.unionai.cloud/v2/domain/development/project/llm-service/apps/minimax-m3) ·
  [qwen35-397b](https://demo.hosted.unionai.cloud/v2/domain/development/project/llm-service/apps/qwen35-397b) ·
  [llm-service source](https://github.com/unionai/internal-union-apps/tree/main/llm-service)
  (`glm-5-2` retired from the roster 2026-09-15; rounds 12-14 corpora were
  generated with it in the pool)
- PRs: [#6 project + reorg](https://github.com/unionai-oss/model-factory/pull/6) ·
  [#7 experiments + dark loop + dashboard](https://github.com/unionai-oss/model-factory/pull/7)
- Upstream blockers for Qwen3.5 RL: [trl#5269](https://github.com/huggingface/trl/issues/5269) ·
  [vllm#39993](https://github.com/vllm-project/vllm/issues/39993)
