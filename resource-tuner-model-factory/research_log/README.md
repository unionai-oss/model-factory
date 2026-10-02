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
| 2026-10-02 | [round-16-decision-arm-and-factory](2026-10-02-round-16-decision-arm-and-factory.md) | Third arm: a multi-head **decision model** that picks a grid cell directly, trained with a cost-asymmetry loss; the arm suite is a Union **factory** fanning out over an `under_penalty` partition with the corpus as a `factory.source` (LLM path untouched). The knob is monotone across all 3 arms (fit 54%→71%→89%) but only the paranoid arm beats the rule baseline on fit, and at +5% cost — 3 dead runs first (1M rows OOM the station, `Reporter` API, `baseline_proposal` arity) |

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
