# 2026-09-03 — round 2: dark loop, synthetic teacher, dev-scale training

Goal: wire the factory dark (artifact triggers), add the teacher-LLM
synthetic pipeline with an execution oracle, and run dev-scale training
to see whether the composite reward moves waste.

## Triggers (dark mode)

`train-on-new-corpus` (OnArtifact `tuning-task-corpus` → `train_tuner`)
and `eval-on-new-checkpoint` (OnArtifact `tuner-checkpoint` →
`eval_tuner`, corpus resolved from the latest artifact) — deployed
`auto_activate=False`, activated via the API.

- First dark eval observed: [u2d94dc66f80db08f](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/u2d94dc66f80db08f)
  (`RUN_SOURCE_ARTIFACT_TRIGGER`; aborted — it ran the pre-clamp task
  version).
- **Operational rule learned twice: a redeploy resets triggers to
  inactive AND re-binds them to the new task version.** Reactivation is
  part of the deploy ritual.
- Loop is cycle-free by construction: corpus → train → checkpoint → eval
  → report; nothing republishes a corpus.
- Dark runs verified end-to-end: [u50e29a7f692ceb75](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/u50e29a7f692ceb75)
  (train, from the synthetic merge), [u3e59ab9e22af72a5](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/u3e59ab9e22af72a5)
  (train, from the dev corpus), [u5e6d929fd544e39f](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/u5e6d929fd544e39f)
  (eval: fit 94%, waste 46%). 19 dark-tagged runs on the dashboard by
  end of day.

## Synthetic pipeline (teacher = qwen38-27b on llm-service)

Design: teacher writes novel workloads → AST safety screen → **execution
oracle** (harness pod, generous resources, rusage peak + avg CPU) labels
footprints — never the teacher's own guess. Curated rows publish
`synthetic-task-corpus` + a merged `tuning-task-corpus` (which dark-fires
training).

| run | outcome |
|---|---|
| [um67nxwg2bl5br4hkxxd](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/um67nxwg2bl5br4hkxxd) | ❌ `ImportError: llm_client` — the code bundler walks the **static import graph**; modules imported only inside task bodies aren't bundled |
| [u7sfp8trbxv7qbfsxhl9](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/u7sfp8trbxv7qbfsxhl9) | ❌ same (deployed bundle predates the fix) |
| [ucbd65hdc5nxlbphscz5](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/ucbd65hdc5nxlbphscz5) | aborted (leftover bare name after refactor) |
| [uj7mjgbdtxwqj2gpb6kz](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/uj7mjgbdtxwqj2gpb6kz) | ❌ teacher answered with **empty content 6/6** — hybrid-thinking server (reasoning_effort=medium) spent the whole budget in `reasoning_content`. Fix: per-request `chat_template_kwargs={"reasoning_effort":"none"}` + 4096 budget |
| [u5qfhvnqttdlxdkkr4zp](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/u5qfhvnqttdlxdkkr4zp) | ❌ teacher unreachable 900s — scale-from-zero + fresh L40S node takes 15+ min; ready deadline raised to 1800s |
| [umgtmn7jswljbv8jrktr](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/umgtmn7jswljbv8jrktr) | ✅ **5/6 yield** — the oracle rejected one sample where the teacher used sklearn's removed `multi_class` kwarg (execution failed → curated out). But the synthetic artifact silently didn't version: `publish()` only creates a version when the wrapped value is **returned** from a task |
| [uq2v5tpkxt4kxvmqh2wr](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/uq2v5tpkxt4kxvmqh2wr) | ❌ svc DNS gone — the platform **unassigned** the app ("Service marked for deletion") after failed wake cycles; fixed with `activate_app` |
| [uvwz9rwtgzdtr87xb74k](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/uvwz9rwtgzdtr87xb74k) | ✅ artifact publishes (child-task publish fix); merged corpus dark-fired [ud2c9ac29ada302ed](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/ud2c9ac29ada302ed) (✅ succeeded — the first training whose corpus includes teacher-generated tasks) |

llm-service fixes along the way (project `llm-service`, apps by
[Niels](https://demo.hosted.unionai.cloud/v2/domain/development/project/llm-service/apps/qwen38-27b)):
the project's 2Ti `limits.memory` quota was exhausted by idle giants —
deactivated `kimi-k3` (1.2Ti) and `inkling` (820Gi) to let `qwen38-27b`
(52Gi) wake. In-cluster callers must use svc DNS
(`http://qwen38-27b.llm-service-development.svc.cluster.local`) — the
public URL is OIDC-gated.

## Dev-scale training (the headline result)

[u859bl7b9xp7xsnl899v](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/u859bl7b9xp7xsnl899v)
— `tuner_pipeline --profile_name dev`: 150 composite GRPO steps,
Qwen3-1.7B fp16 LoRA on T4:1, 512 contexts.

**fit 98% vs baseline 64% · median waste 83% → 47% (baseline 25%) ·
validity 100%.** The composite reward reduces overprovisioning while
success improves. Gate still fails on waste vs baseline — next increment
is training scale, not machinery.

## Dashboard

Rebuilt in Union dark tokens with the ported React Flow lineage graph,
EvalBadges on checkpoint/report cards (linked via `checkpoint_path` now
embedded in reports), aggregate eval charts, dark-run tags, and a pinned
`{app}-{project}-{domain}` subdomain:
<https://rt-lineage-resource-tuner-model-factory-development.apps.demo.hosted.unionai.cloud>

Two bugs the dashboard surfaced by disagreeing with expectations:
the unpublished synthetic artifact (above) and dark-run detection
(pb2 `source` is an enum int — resolve names via the protobuf
descriptor).

## Everything merged in

PR [#7 — rt-tests-and-first-experiments](https://github.com/unionai-oss/model-factory/pull/7)
(94 unit tests).
