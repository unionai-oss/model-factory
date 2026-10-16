# Research log

Audit trail of basic-model-factory experiments. Same conventions as
[resource-tuner-model-factory/research_log](../../resource-tuner-model-factory/research_log/README.md):
one entry per experiment round (`YYYY-MM-DD-<slug>.md`) with a run ledger
(Union console URLs, failures included and linked to fixes), findings,
and code changes; keep the entries table below current.

Run URL base:
`https://demo.hosted.unionai.cloud/v2/domain/development/project/model-factory/runs/<run>`

## Standing results (as of 2026-10-02)

| run | target | profile | outcome |
|---|---|---|---|
| [bmf-smoke-r1](https://demo.hosted.unionai.cloud/v2/domain/development/project/model-factory/runs/bmf-smoke-r1) | `mf-promoted-model[2026-10-02]` | smoke | 6 artifacts built, ~12 min; promotion gate auto-approved (smoke margin is -1.0, so the quality gate is deliberately off) |
| [bmf-serve-r1](https://demo.hosted.unionai.cloud/v2/domain/development/project/model-factory/runs/bmf-serve-r1) | `mf-inference[2026-10-02]` | smoke | 6 reused + endpoint built; app ACTIVE and generating with the adapter on |
| [u883f0f834236535d](https://demo.hosted.unionai.cloud/v2/domain/development/project/model-factory/runs/u883f0f834236535d) | `mf-promoted-model`, **by the nightly cron** | smoke | fired unattended at 06:00 UTC; all 6 stations cache-hit in **9.1s** |

No quality claim yet: `smoke` sets `promotion_margin=-1.0`, so
`bmf-smoke-r1` proves the machinery, not that the checkpoint beats base.
A `dev`-profile materialization is what would put a pass@1 number here.

## Entries

| date | entry | tl;dr |
|---|---|---|
| — | (historical runs predate this log; notable ones are recorded in the repo's PR descriptions, e.g. [#1](https://github.com/unionai-oss/model-factory/pull/1), [#3](https://github.com/unionai-oss/model-factory/pull/3), [#4](https://github.com/unionai-oss/model-factory/pull/4)) | |
| 2026-10-02 | [round-1-factory-refactor](2026-10-02-round-1-factory-refactor.md) | The OnArtifact chain + `integration.py` become one `factory.Factory`; full loop green end to end (`bmf-smoke-r1`, 6 built) and serving the bound promoted model; 5 gotchas cost 3 runs — `datetime` partitions, unbuilt app images, root-path `mount=`, immutable partition schemas (hence `mf-` names), circular eval-via-app |
