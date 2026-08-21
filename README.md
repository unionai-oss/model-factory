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

The factory is decomposed into four team-owned subsystems wired ONLY by
artifacts + OnArtifact triggers (see docs/SPEC.md §6): data engineering
(`team_data.py`), model training (`team_training.py`), model eval
(`team_eval.py`), and inference (`team_inference.py`).

```bash
uv sync

# unit tests (sandbox, rewards, parsing)
uv run pytest

# deploy each team's subsystem (envs + dark-mode triggers)
CFG=~/.flyte/config-model-factory.yaml
uv run flyte --config $CFG deploy team_data.py de_cpu_env
uv run flyte --config $CFG deploy team_training.py trainer_env
uv run flyte --config $CFG deploy team_eval.py eval_gpu_env
uv run flyte --config $CFG deploy team_inference.py inference_app_env   # serving app
uv run flyte --config $CFG deploy team_inference.py inference_ops_env   # refresh trigger
uv run flyte --config $CFG deploy app.py lineage_app_env                # lineage "Podium"

# run one dataset release (data engineering's entrypoint); with artifact
# events live, publishing fires training -> eval/inference automatically
uv run flyte --config $CFG run team_data.py data_release --profile_name smoke

# cross-team E2E (stand-in event bus until the backend ships artifact events)
uv run flyte --config $CFG run integration.py factory_chain \
    --profile_name smoke --auto_approve
```

Profiles (`model_factory/config.py`): `smoke` (0.5B model, 96 tasks, 10 GRPO
steps — minutes), `dev` (1.5B, ~2K tasks, 100 steps), `full` (10K tasks,
500+ steps).

## Secrets

Optional but recommended — see [TODO.md](TODO.md). Without them the loop
still runs (public models/datasets; W&B disabled).

## Layout

| path | team / role |
|---|---|
| `model_factory/contracts.py` | the inter-team interface: artifact names, schemas, publish() |
| `model_factory/shared/` | platform libs: sandbox, rewards, reporting, assets, gates, images, inference client |
| `model_factory/data_engineering/` | data eng: curate, filter, oracle-verify, synthetic gen, release + data gate |
| `model_factory/training/` | training: GRPO (TRL + LoRA), OnArtifact(rl-tasks-dataset) trigger |
| `model_factory/evaluation/` | eval: candidate-vs-base, gates, promotion, OnArtifact(policy-checkpoint) trigger |
| `model_factory/inference/` | inference: serving app (adapter toggle per request) + weight-rollout ops |
| `model_factory/lineage_app.py` | platform: global lineage AppEnvironment |
| `integration.py` | cross-team E2E driver (plays the event bus until artifact events ship) |
