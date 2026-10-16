# resource-tuner-model-factory

Right-sizing `flyte.Resources` for Flyte tasks — the prototype for the
**AI Resource Tuning** PRD's RL track.

Three arms now decide the request, all scored through the same simulator and
pricing so their numbers are directly comparable:

| arm | how it decides | trained by | code |
|---|---|---|---|
| **LLM policy** | reads source + input profile, emits `{"cpu": 2, "memory": "4Gi"}` as text, snapped onto the action grid | GRPO on a shaped reward | `training/grpo.py` |
| **GBT baseline** | regresses a high quantile per resource, then buckets | quantile loss | `training/ml_baseline.py` |
| **decision model** | **chooses a grid cell directly** (3 heads: memory / CPU / GPU) | cost-weighted cross-entropy | `training/decision_model.py` |

The decision arm is the newest and is wired as a **Union factory** (see
[Decision arm](#decision-arm-a-factory-over-the-action-grid) below); the LLM
arm keeps its OnArtifact trigger chain unchanged.

See [research_log/](research_log/) for the experiment audit trail (run
links, findings, standing results), and
[docs/DESIGN.md](docs/DESIGN.md) for the full design (MDP framing,
sim-first environment, reward curriculum, model choice rationale).

## The loop

```
teacher LLM (qwen38-27b / minimax-m3    build_task_corpus (templates,
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

## Decision arm: a factory over the action grid

`policy/actions.py` has always bucketed proposals onto a finite grid
(log-scale memory, fixed CPU increments, typed GPU counts). The LLM arm emits
numbers and gets snapped onto it; the GBT baseline regresses numbers and
rounds them. The decision model skips the number entirely and **picks the
cell**, which lets the loss say the one thing a regression loss cannot:

- one memory bucket too small → the task OOMs; the run is lost, plus a retry
- one bucket too big → you pay for memory you did not use; a few percent

So the loss is cross-entropy weighted by a cost matrix where
under-provisioning costs a flat `under_penalty` and over-provisioning costs
the extra resource actually paid for. **`under_penalty` is the arm's
hyperparameter, and it is a partition dimension** — each setting is trained,
scored and compared as its own artifact instance:

```
  tuning-task-corpus  (factory.source — published by build_task_corpus,
          │                              OUTSIDE this factory)
          │                        one instance per `arm`
   ┌──────┼──────┐
   ▼      ▼      ▼
 rt-decision-model × 3        [arm]   fit_decision_model
   ▼      ▼      ▼
 rt-decision-scorecard × 3    [arm]   score_decision_model
   └──────┼──────┘
          ▼  .all("arm")
 rt-decision-champion                 select_decision_tuner
```

The corpus is a **source**, not a build — which is exactly why this arm went
in without touching `grpo.py`, `build_task_corpus`, the trigger wiring or the
tune service. It also gives the trigger shape a factory cannot get from its
own builds: `factory.on(corpus)` refits the whole suite on every new corpus
version.

```bash
CFG=.flyte/config.yaml
uv run flyte --config $CFG deploy decision.py decision_env
uv run flyte --config $CFG factory deploy factory.py

# the full frontier, three arms in parallel
uv run flyte --config $CFG factory materialize resource-tuner rt-decision-champion \
    --partition arm=balanced,oom-averse,oom-paranoid --wait

# one arm; or pin the corpus version instead of taking the latest
uv run flyte --config $CFG factory materialize resource-tuner rt-decision-scorecard \
    --partition arm=oom-averse
uv run flyte --config $CFG factory materialize resource-tuner rt-decision-champion \
    --partition arm=balanced,oom-averse,oom-paranoid \
    --version tuning-task-corpus=<version>
```

Results so far (512 heldout workloads,
[rt-decision-r4](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rt-decision-r4)):
the knob is **monotone on every metric** — fit 54.3% → 70.9% → 89.1% as
`under_penalty` goes 2 → 12 → 40 — but only the paranoid arm beats the rule
baseline's 76.0% fit, and it costs ~5% more per task-hour. The grid itself
forces 36.1% median over-provisioning, so roughly half the waste is the action
space rather than the model. Full numbers and caveats:
[research_log/round-16](research_log/2026-10-02-round-16-decision-arm-and-factory.md).

The action space is configurable (`policy/action_space.py`): `default`,
`coarse` and `fine` are registered, and `ActionSpace` is serialized into every
checkpoint's manifest because a model is undecodable with the wrong grid.

## Artifacts

Stations hand off through versioned artifacts — publishing one IS the
request to run the next station (the triggers bind to these names):

| artifact | kind | what it is | produced by |
|---|---|---|---|
| `tuning-task-corpus` | data | parquet of workloads + measured footprints; the training/eval corpus | `build_task_corpus`, `synthetic_data_release`, `archetype_data_release` |
| `synthetic-task-corpus` | data | the oracle-verified teacher rows on their own | `publish_synthetic_corpus` |
| `tuner-checkpoint` | model | GRPO-trained resource-proposal policy (PEFT adapter + tokenizer + manifest) | `train_tuner` |
| `tuner-checkpoint-intermediate` | model | mid-training snapshot; deliberately a separate name so it fires no eval | `publish_intermediate_checkpoint` |
| `tuner-eval-report` | data | held-out metrics + gate verdict for one checkpoint | `eval_tuner` |
| `ml-baseline-model` | model | quantile-GBT floor the policy has to beat | `fit_ml_baseline` |
| `tuning-ab-report` | data | tuned vs hard-coded prior on real pods — the auditable savings record | `tune_ab_experiment` |

Every one of them publishes an **artifact card** (markdown, rendered in
the console) plus flat `attrs` for filtering: schema and composition for
corpora, base model / reward stage / trajectory for checkpoints, verdicts
for reports. The renderers live in `resource_tuner/shared/cards.py` and
are pure functions — the producing task already holds every number, so a
consumer never has to download a payload to find out what it is. Cards
are best-effort: a card failure logs and the artifact still publishes.

## Teachers (synthetic data)

`llm-service` project apps, llama.cpp with OpenAI-compatible `/v1`:
`qwen38-27b` (default — cheapest, single L40S), plus the frontier-class
`minimax-m3` and `qwen35-397b`. In-cluster tasks reach them via internal
service DNS (the public app URL sits behind OIDC and answers pods with a
login redirect); locally set `RT_TEACHER_URL`.
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
