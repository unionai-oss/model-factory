# resource-tuner-model-factory

RL fine-tuning of a small LLM that right-sizes `flyte.Resources` for Flyte
tasks — the prototype for the **AI Resource Tuning** PRD's RL track. The
policy reads a task's source code + input profile and emits the kwargs for
`flyte.Resources` (e.g. `{"cpu": 2, "memory": "4Gi"}`); the reward is
"the task succeeded and didn't waste the request".

See [docs/DESIGN.md](docs/DESIGN.md) for the full design (MDP framing,
sim-first environment, reward curriculum, model choice rationale).

## The loop

```
build_task_corpus ──► [tuning-task-corpus]      synthetic-but-realistic Flyte
        │                                       tasks w/ analytic footprints
        ▼
train_tuner (GRPO + LoRA, T4) ──► [tuner-checkpoint]
        │                         rewards from the SIMULATOR (cheap loop)
        ▼
eval_tuner ──► [tuner-eval-report]
        │      policy vs rule-based baseline, held-out split
        └────► REAL episodes: harness pods sized by the policy's proposals
               (resources=.override(...)), rusage + pod metrics ground truth
```

## Quickstart

```bash
uv sync                       # unit tests only
uv run pytest

CFG=.flyte/config.yaml        # demo.hosted, project resource-tuner-model-factory
uv run flyte --config $CFG deploy main.py driver_env

# E2E smoke: corpus -> 10 GRPO steps on a T4 -> eval incl. real episodes
uv run flyte --config $CFG run main.py tuner_pipeline --profile_name smoke
```

Profiles (`resource_tuner/config.py`): `smoke` (64 contexts, 10 steps,
stage-A reward — proves reward goes up), `dev` (512 contexts, 150 steps,
composite reward), `full` (4K contexts, 500 steps, QLoRA-ready).

Model is parameterized: `RT_MODEL=Qwen/Qwen3-0.6B` for plumbing tests,
default `Qwen/Qwen3-1.7B`; Qwen3.5 rungs live in `MODEL_LADDER` behind
upstream TRL support (see DESIGN.md §3).

## Metrics plugin (optional, private repo)

Pod-level utilization cross-checks use `flyteplugins-union` at branch
[`niels/get-metrics`](https://github.com/unionai/flyteplugins-union/tree/niels/get-metrics)
(private). Locally:

```bash
uv sync --extra metrics       # needs GitHub credentials for unionai/flyteplugins-union
```

For task images, deploy with `RT_WITH_METRICS=1 RT_GH_TOKEN=<token>` so the
remote builder can install it (short-lived read-only token; it is visible
in image pip metadata — prototype-grade wiring, see shared/images.py).
Everything degrades gracefully to harness rusage when the plugin is absent.

## Secrets

| where | name | purpose |
|---|---|---|
| Flyte cluster (project-scoped) | `HUGGINGFACE_TOKEN`, `WANDB_API_KEY` | model pulls / W&B, attached with `RT_USE_SECRETS=1` |
| GitHub Actions + local deploy-time env | `RT_GH_TOKEN` | private metrics plugin in image builds; CI sets `RT_WITH_METRICS=1` automatically when it is present |
| GitHub Actions | `DEMO_HOSTED_FLYTE_API_KEY` | CI deploys (shared with basic-model-factory) |
