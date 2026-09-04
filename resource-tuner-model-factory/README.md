# resource-tuner-model-factory

RL fine-tuning of a small LLM that right-sizes `flyte.Resources` for Flyte
tasks — the prototype for the **AI Resource Tuning** PRD's RL track. The
policy reads a task's source code + input profile and emits the kwargs for
`flyte.Resources` (e.g. `{"cpu": 2, "memory": "4Gi"}`); the reward is
"the task succeeded and didn't waste the request".

See [research_log/](research_log/) for the experiment audit trail (run
links, findings, standing results), and
[docs/DESIGN.md](docs/DESIGN.md) for the full design (MDP framing,
sim-first environment, reward curriculum, model choice rationale).

## The loop

```
teacher LLM (qwen38-27b / glm-5-2      build_task_corpus (templates,
on llm-service, in-cluster svc DNS)     analytic footprints)
        │                                       │
synthetic_data_release: generate ─► AST screen ─► EXECUTION ORACLE
(harness pod measures real peak RSS + avg CPU — labels never come
from the teacher)                               │
        ▼                                       ▼
[synthetic-task-corpus] ──── merge ────► [tuning-task-corpus]
                                                │  OnArtifact trigger
                                                ▼  (train-on-new-corpus)
                    train_tuner (GRPO + LoRA, T4) ──► [tuner-checkpoint]
                            rewards from the SIMULATOR  │  OnArtifact trigger
                                                        ▼  (eval-on-new-checkpoint)
                    eval_tuner ──► [tuner-eval-report] ──► rt-lineage app
                        │     policy vs rule-based baseline  (aggregate charts
                        └───► REAL episodes: harness pods     across checkpoints)
                              sized by the policy's proposals
```

Both triggers deploy `auto_activate=False`; activate them (console or
`flyte trigger activate`) and publishing a corpus IS the request to train,
a checkpoint IS the request to evaluate. A trigger keeps firing the task
version it was deployed with — re-deploy after fixes dark mode should see.

## Quickstart

```bash
uv sync                       # unit tests only
uv run pytest

CFG=.flyte/config.yaml        # demo.hosted, project resource-tuner-model-factory
uv run flyte --config $CFG deploy main.py driver_env       # stations + triggers
uv run flyte --config $CFG deploy app.py lineage_app_env   # dashboard

# E2E smoke: corpus -> 10 GRPO steps on a T4 -> eval incl. real episodes
uv run flyte --config $CFG run main.py tuner_pipeline --profile_name smoke

# teacher-generated corpus (wakes the 27B llama.cpp app from zero; the
# merged corpus fires train-on-new-corpus when triggers are active)
uv run flyte --config $CFG run main.py synthetic_data_release --n_tasks 10
```

## Teachers (synthetic data)

`llm-service` project apps, llama.cpp with OpenAI-compatible `/v1`:
`qwen38-27b` (default — cheapest, single L40S) and `glm-5-2`. In-cluster
tasks reach them via internal service DNS (the public app URL sits behind
OIDC and answers pods with a login redirect); locally set `RT_TEACHER_URL`.
The teacher only writes code — footprint labels always come from the
execution oracle, because a teacher's guess about resource needs is
exactly the bias this factory exists to remove.

Profiles (`resource_tuner/config.py`): `smoke` (64 contexts, 10 steps,
stage-A reward — proves reward goes up), `dev` (512 contexts, 150 steps,
composite reward), `full` (4K contexts, 500 steps, QLoRA-ready).

Model is parameterized: `RT_MODEL=Qwen/Qwen3-0.6B` for plumbing tests,
default `Qwen/Qwen3-1.7B`; Qwen3.5 rungs live in `MODEL_LADDER` behind
upstream TRL support (see DESIGN.md §3).

## Metrics plugin

Pod-level utilization cross-checks use `flyteplugins-union>=0.10.0`
(public on PyPI as of 2026-09-03; previously a private branch). It's a
normal dependency — installed by `uv sync` and baked into the gpu/driver
task images. Note: 0.10.0's PyPI metadata still caps `flyte<2.7.0`, so
the project carries a uv `override-dependencies` and the images re-pin
flyte after the plugin layer (see `shared/images.py`); drop both once a
plugins release declares flyte 2.7 support. Episode scoring still
degrades gracefully to harness rusage if pod metrics answer errors.

## Secrets

| where | name | purpose |
|---|---|---|
| Flyte cluster (project-scoped) | `HUGGINGFACE_TOKEN`, `WANDB_API_KEY` | model pulls / W&B, attached with `RT_USE_SECRETS=1` |
| GitHub Actions | `DEMO_HOSTED_FLYTE_API_KEY` | CI deploys (shared with basic-model-factory) |

(`RT_GH_TOKEN` is no longer used — the repo secret can be deleted.)
