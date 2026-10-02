# model-factory

Model factory proofs-of-concept on [Flyte v2](https://www.union.ai/docs/v2/)
/ Union, one subdirectory per factory.

| project | what it is |
|---|---|
| [`basic-model-factory/`](basic-model-factory/) | A dark model factory: RL from verifiable rewards (GRPO + sandboxed unit tests), the whole loop declared as one factory — curate → synthesize → train → eval → promote → serve, partitioned by release day. |
| [`decision-model-factory/`](decision-model-factory/) | Fine-tunes a **suite** of open-weights decision models on synthetic tool-calling data, scores them against an exact oracle, and serves the winner. The suite is a partition dimension; `.all("model")` collapses it into a champion. |
| [`resource-tuner-model-factory/`](resource-tuner-model-factory/) | Right-sizes `flyte.Resources` for Flyte tasks (the AI Resource Tuning PRD's RL track). Three arms — an LLM policy (GRPO), a quantile-GBT baseline, and a decision model that picks a grid cell directly — scored through one simulator. |

All three use [Union factories](https://www.union.ai/docs/v2/union/user-guide/factories/)
(preview; `flyte>=2.10.0` + `flyteplugins-union>=0.12.0`) to declare their
graph of partitioned [artifacts](https://www.union.ai/docs/v2/union/user-guide/artifacts/)
and end in a served app, and each one uses a different part of the API:

| | partitions | fan-out | ends in |
|---|---|---|---|
| basic-model-factory | `date: Daily` | — | `factory.serve` → `mf-inference` |
| decision-model-factory | `date: Daily`, `model: str` | one fine-tune per suite member, collapsed with `.all("model")` | `factory.serve` → `decision-api` |
| resource-tuner | `arm: str` | one arm per cost-asymmetry setting, collapsed with `.all("arm")` | the existing tune service |

The resource-tuner is the one that reads its input as a
[`factory.source`](resource-tuner-model-factory/resource_tuner/factory.py) —
an artifact published outside the factory — which is how its decision arm was
added without touching the LLM path, and which unlocks `factory.on(source)`
for "new corpus → refit the suite".

Each project is self-contained (its own `pyproject.toml`, `uv.lock`,
`.flyte/config.yaml`, tests, and docs); run its commands from inside its
directory:

```bash
cd basic-model-factory
uv sync
uv run pytest
```

Every project keeps a `research_log/` — the audit trail of what ran, where,
what it showed, and what changed because of it. Failed runs stay in the log;
a failure that changed the code is a result.

CI (`.github/workflows/ci.yml`) runs each project's tests on every push/PR
and deploys it to its Flyte cluster on green `main`.
