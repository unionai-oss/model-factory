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

## Standing results (as of 2026-09-03)

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

## Key links

- Product context: [AI Resource Tuning PRD](https://app.notion.com/p/AI-Resource-Tuning-3cb8cc06513d81f4b381c8294419a920)
  ([markdown source](https://github.com/unionai/prds/blob/main/product_prds/ai_resource_tuning/prd.md))
- Design doc: [../docs/DESIGN.md](../docs/DESIGN.md)
- Dashboard (lineage graph + eval charts):
  <https://rt-lineage-resource-tuner-model-factory-development.apps.demo.hosted.unionai.cloud>
- Cluster project: [resource-tuner-model-factory / development](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs)
- Metrics plugin: [flyteplugins-union @ niels/get-metrics](https://github.com/unionai/flyteplugins-union/tree/niels/get-metrics)
- Teacher LLMs: [qwen38-27b](https://demo.hosted.unionai.cloud/v2/domain/development/project/llm-service/apps/qwen38-27b) ·
  [glm-5-2](https://demo.hosted.unionai.cloud/v2/domain/development/project/llm-service/apps/glm-5-2) ·
  [llm-service source](https://github.com/unionai/internal-union-apps/tree/main/llm-service)
- PRs: [#6 project + reorg](https://github.com/unionai-oss/model-factory/pull/6) ·
  [#7 experiments + dark loop + dashboard](https://github.com/unionai-oss/model-factory/pull/7)
- Upstream blockers for Qwen3.5 RL: [trl#5269](https://github.com/huggingface/trl/issues/5269) ·
  [vllm#39993](https://github.com/vllm-project/vllm/issues/39993)
