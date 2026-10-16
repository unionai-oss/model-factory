# Research log

Audit trail of model-factory-basic experiments. Same conventions as
[model-factory-resource-tuner/research_log](../../model-factory-resource-tuner/research_log/README.md):
one entry per experiment round (`YYYY-MM-DD-<slug>.md`) with a run ledger
(Union console URLs, failures included and linked to fixes), findings,
and code changes; keep the entries table below current.

Run URL base:
`https://demo.hosted.unionai.cloud/v2/domain/development/project/model-factory/runs/<run>`

## Standing results (as of 2026-10-16)

**Round 3: the loop compounds.** Fixed held-out split of 500 KodCode tasks
(fingerprint `fdf17e9063c13a57`), greedy pass@1, profile `daily-r3`
(Qwen2.5-Coder-0.5B-Instruct; fresh LoRA on the merged champion each day; 80
GRPO steps × 8 prompts × 8 rollouts on colocated vLLM; 7B teacher for the
curriculum).

| day | run | champion | base | champion | candidate | Δ vs champion [95% CI] | release |
|---|---|---|---|---|---|---|---|
| 2026-10-01 | [bmf-r3-d1c](https://demo.hosted.unionai.cloud/v2/domain/development/project/model-factory/runs/bmf-r3-d1c) | base | 42.0% | 42.0% | **47.6%** | **+5.6 [+2.8, +8.6]** | gen 1 |
| 2026-10-02 | [bmf-r3-d2](https://demo.hosted.unionai.cloud/v2/domain/development/project/model-factory/runs/bmf-r3-d2) | gen 1 | 42.0% | 47.6% | **50.8%** | **+3.2 [+0.6, +5.8]** | gen 2 |
| 2026-10-03 | [bmf-r3-d3](https://demo.hosted.unionai.cloud/v2/domain/development/project/model-factory/runs/bmf-r3-d3) | gen 2 | 42.0% | 50.8% | 50.8% | 0.0 [−2.2, +2.2] | champion kept |

Best model: **generation 2, 50.8% vs 42.0% base (+8.8, CI [+5.6, +12.0])**,
served by `mf-inference` ([bmf-r3-serve-gen2](https://demo.hosted.unionai.cloud/v2/domain/development/project/model-factory/runs/bmf-r3-serve-gen2)).

Round 2 (`daily` profile, 200-task split `9911df20d92736a4`, 50 HF-generate
steps, one LoRA continued): 35.0% → 42.5% on day 1, then three days within
noise of that champion (42.0, 41.0, 41.0). See the round-2 entry.

## Entries

| date | entry | tl;dr |
|---|---|---|
| — | (historical runs predate this log; notable ones are recorded in the repo's PR descriptions, e.g. [#1](https://github.com/unionai-oss/model-factory/pull/1), [#3](https://github.com/unionai-oss/model-factory/pull/3), [#4](https://github.com/unionai-oss/model-factory/pull/4)) | |
| 2026-10-02 | [round-1-factory-refactor](2026-10-02-round-1-factory-refactor.md) | The OnArtifact chain + `integration.py` become one `factory.Factory`; full loop green end to end (`bmf-smoke-r1`, 6 built) and serving the bound promoted model; 5 gotchas cost 3 runs — `datetime` partitions, unbuilt app images, root-path `mount=`, immutable partition schemas (hence `mf-` names), circular eval-via-app |
| 2026-10-15 | [round-2-closed-loop](2026-10-15-round-2-closed-loop.md) | The loop closes through a partitioned `mf-champion` source written by an OnArtifact crown (a same-artifact cross-day edge is a cycle); fixed held-out + baseline built once; failure-driven curriculum; champion/challenger release. Day 1: 35.0% → **42.5%** pass@1 (CI excludes 0). Days 2–4 within noise of it, champion kept. Synthetic parser kept 0/48 until fixed; GPU cache hits wait in the GPU queue; `--version` pin serves a past release without rebuilding |
| 2026-10-16 | [round-3-compounding](2026-10-16-round-3-compounding.md) | 500-task held-out; fresh LoRA on merged champion; vLLM rollouts (2× throughput, 3.2× prompts/day); 7B vLLM teacher (38% yield vs 3%). **Compounds**: 42.0 → 47.6 → 50.8% pass@1, both steps with CIs above 0; day 3 a dead heat. A redefined profile name silently reused round-2's held-out (profiles are now immutable names); FlashInfer sampler needs nvcc; the served 27B teacher is reachable but too slow today |
