# model-factory

A **dark model factory** proof-of-concept on [Flyte v2](https://www.union.ai/docs/v2/)
/ Union: a mostly-autonomous loop that trains a small coding agent with RL
from verifiable rewards (GRPO + sandboxed unit-test execution), where every
asset is a versioned artifact, artifact events drive the next factory
station, and humans stay in the loop only at the judgment layers — data
validation and checkpoint promotion.

Source context: the "Effective dark model factory patterns" conference
abstract and poolside's model-factory blog series. See
[docs/SPEC.md](docs/SPEC.md) for the architecture and small/medium/large
scopes, and [docs/research/](docs/research/) for the research notes.

## The loop

```
seed data (HF: KodCode) ──► curate + oracle-verify ──► [HITL gate 1: data validation]
        ▲                                                        │
        │                                             publish [rl-tasks-dataset]
 synthetic tasks (batch                                          │ (OnArtifact trigger)
 inference + execution                                           ▼
 oracle)  ◄──────────────────────────────────── GRPO training (TRL + LoRA, A10G)
                                                                 │
                                                    [policy-checkpoint] artifact
                                                                 │ (OnArtifact trigger)
                                                                 ▼
                                              eval: candidate vs base pass@1
                                                                 │
                                            auto quality gate + [HITL gate 2: promote?]
                                                                 ▼
                                                     [promoted-model] artifact
```

## Quickstart

```bash
uv sync

# unit tests (sandbox, rewards, parsing)
uv run pytest

# one factory iteration on the cluster, gates auto-approved (CI/smoke):
uv run flyte --config ~/.flyte/config-model-factory.yaml run run.py factory \
    --profile_name smoke --auto_approve true

# with live human gates (approve in the Union UI, or:
#   flyte signal condition <run-name> <action-name> true)
uv run flyte --config ~/.flyte/config-model-factory.yaml run run.py factory \
    --profile_name smoke

# deploy dark-mode triggers (artifact-event + nightly cron; activate deliberately)
uv run flyte --config ~/.flyte/config-model-factory.yaml deploy run.py factory_env

# deploy the lineage app (the factory's "Podium")
uv run flyte --config ~/.flyte/config-model-factory.yaml deploy \
    model_factory/lineage_app.py lineage_app_env
```

Profiles (`model_factory/config.py`): `smoke` (0.5B model, 96 tasks, 10 GRPO
steps — minutes), `dev` (1.5B, ~2K tasks, 100 steps), `full` (10K tasks,
500+ steps).

## Secrets

Optional but recommended — see [TODO.md](TODO.md). Without them the loop
still runs (public models/datasets; W&B disabled).

## Layout

| path | what |
|---|---|
| `model_factory/config.py` | profiles, artifact names, secret wiring |
| `model_factory/sandbox.py` | resource-limited execution of generated code vs tests |
| `model_factory/rewards.py` | reward stack: format + compile + all-tests-pass + anti-hack guards |
| `model_factory/data.py` | ingest, curate, oracle-verify, publish dataset artifact |
| `model_factory/synthetic.py` | batch-inference synthetic task generation (DynamicBatcher) |
| `model_factory/train.py` | GRPO training (TRL + LoRA) with live report + W&B |
| `model_factory/evaluate.py` | candidate-vs-base eval, promotion |
| `model_factory/pipeline.py` | factory driver + HITL gates + dark-mode triggers |
| `model_factory/lineage_app.py` | AppEnvironment: global lineage visualization |
