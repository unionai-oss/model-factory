# Research log

Audit trail of model-factory-basic experiments. Same conventions as
[model-factory-resource-tuner/research_log](../../model-factory-resource-tuner/research_log/README.md):
one entry per experiment round (`YYYY-MM-DD-<slug>.md`) with a run ledger
(Union console URLs, failures included and linked to fixes), findings,
and code changes; keep the entries table below current.

Run URL base:
`https://demo.hosted.unionai.cloud/v2/domain/development/project/model-factory/runs/<run>`

## Standing results (as of 2026-10-16)

Fixed held-out split `mf-heldout-tasks` (200 KodCode-Light-RL-10K tasks,
fingerprint `9911df20d92736a4`), greedy pass@1, `daily` profile
(Qwen2.5-Coder-0.5B-Instruct, GRPO + LoRA, 50 steps/day on one A10G).

| day | run | champion | base | champion | candidate | Δ vs champion [95% CI] | release |
|---|---|---|---|---|---|---|---|
| 2026-10-13 | [bmf-r2-daily-d1](https://demo.hosted.unionai.cloud/v2/domain/development/project/model-factory/runs/bmf-r2-daily-d1) | base | 35.0% | 35.0% | **42.5%** | **+7.5 [+3.0, +12.0]** | candidate (gen 1) |
| 2026-10-14 | [bmf-r2-daily-d2](https://demo.hosted.unionai.cloud/v2/domain/development/project/model-factory/runs/bmf-r2-daily-d2) | gen 1 | 35.0% | 42.5% | 42.0% | −0.5 [−5.0, +4.0] | champion kept |
| 2026-10-15 | [bmf-r2-daily-d3](https://demo.hosted.unionai.cloud/v2/domain/development/project/model-factory/runs/bmf-r2-daily-d3) | gen 1 | 35.0% | 42.5% | 41.0% | −1.5 [−6.5, +3.0] | champion kept |
| 2026-10-16 | [bmf-r2-daily-d4](https://demo.hosted.unionai.cloud/v2/domain/development/project/model-factory/runs/bmf-r2-daily-d4) | gen 1 | 35.0% | 42.5% | 41.0% | −1.5 [−6.5, +3.5] | champion kept |

Best model: generation 1, **42.5% vs 35.0% base**, served by `mf-inference`
([bmf-r2-serve-d3b](https://demo.hosted.unionai.cloud/v2/domain/development/project/model-factory/runs/bmf-r2-serve-d3b)).
The loop closes and runs unattended, but it **does not compound yet**: warm-started
days stay within noise of the day-1 champion.

## Entries

| date | entry | tl;dr |
|---|---|---|
| — | (historical runs predate this log; notable ones are recorded in the repo's PR descriptions, e.g. [#1](https://github.com/unionai-oss/model-factory/pull/1), [#3](https://github.com/unionai-oss/model-factory/pull/3), [#4](https://github.com/unionai-oss/model-factory/pull/4)) | |
| 2026-10-02 | [round-1-factory-refactor](2026-10-02-round-1-factory-refactor.md) | The OnArtifact chain + `integration.py` become one `factory.Factory`; full loop green end to end (`bmf-smoke-r1`, 6 built) and serving the bound promoted model; 5 gotchas cost 3 runs — `datetime` partitions, unbuilt app images, root-path `mount=`, immutable partition schemas (hence `mf-` names), circular eval-via-app |
| 2026-10-15 | [round-2-closed-loop](2026-10-15-round-2-closed-loop.md) | The loop closes through a partitioned `mf-champion` source written by an OnArtifact crown (a same-artifact cross-day edge is a cycle); fixed held-out + baseline built once; failure-driven curriculum; champion/challenger release. Day 1: 35.0% → **42.5%** pass@1 (CI excludes 0). Days 2–4 within noise of it, champion kept. Synthetic parser kept 0/48 until fixed; GPU cache hits wait in the GPU queue; `--version` pin serves a past release without rebuilding |
