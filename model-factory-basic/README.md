# model-factory

A **dark model factory** proof-of-concept on [Flyte v2](https://www.union.ai/docs/v2/)
/ Union: a mostly-autonomous loop that trains a small coding agent with RL
from verifiable rewards (GRPO + sandboxed unit-test execution), where every
asset is a versioned artifact, the whole loop is one declared
[Union factory](https://www.union.ai/docs/v2/union/user-guide/factories/), and
humans stay in the loop only at the judgment layers — data validation and
checkpoint promotion.

Source context: the "Effective dark model factory patterns" conference
abstract and poolside's model-factory blog series. See
[docs/SPEC.md](docs/SPEC.md) for the architecture and small/medium/large
scopes, and [docs/research/](docs/research/) for the research notes.

## The loop

The whole loop is **one declared factory** (`model_factory/factory.py`) — a
graph of partitioned artifacts, not a chain of triggers. Nothing runs until
you materialize a target:

```
mf-seed-tasks        ingest_and_curate        [date: Daily]
      │   (HF: KodCode, curated + oracle-verified)
      ├──────────────────────┐
      ▼                      │
mf-synthetic-tasks           │  generate_synthetic_tasks (batch inference
      │                      │                           + execution oracle)
      └──────────┬───────────┘
                 ▼
mf-rl-tasks-dataset          assemble_dataset     [HITL gate 1: data validation]
                 ▼
mf-policy-checkpoint         train_grpo (TRL + LoRA, A10G; sandboxed
                 ▼                       unit tests ARE the reward)
mf-eval-report               evaluate_checkpoint (candidate vs base pass@1)
                 ▼
mf-promoted-model            promote_checkpoint  [auto margin + HITL gate 2]
                 ▼
mf-inference  (app)          serve — the app's `model` parameter is bound to
                                    the promoted-model version the
                                    materialization resolved
```

Everything is partitioned by `date` — one release per day — declared once on
the first build and inherited downstream. The synthetic loop that used to be a
cycle (dataset → synthetic → dataset) is a DAG here: each day's synthetic
batch is mutated from *that day's* curated seed tasks.

Both human gates still exist, inside the station tasks: `mf-promoted-model`
cannot be published without someone approving the data and the promotion. They
default to open (`AUTO_APPROVE = True`) so an unattended materialization
finishes; set it False, or pass
`--param mf-rl-tasks-dataset.auto_approve=false`, to arm them.

### What this replaced

The loop used to be wired by `OnArtifact` triggers plus an `integration.py`
that re-implemented the chain so it could be tested. That worked, but gave no
way to ask for one target, no reuse between runs, no backfill, and a test
harness that duplicated production. The factory gives all four from one
declaration — see
[research_log/round-1](research_log/2026-10-02-round-1-factory-refactor.md)
for the port and the five gotchas it cost.

## Quickstart

```bash
uv sync
uv run pytest                 # 68 tests (sandbox, rewards, parsing, graph)

CFG=~/.flyte/config-model-factory.yaml    # or .flyte/config.yaml

# 1. the station tasks (one per team)
uv run flyte --config $CFG deploy team_data.py de_cpu_env
uv run flyte --config $CFG deploy team_training.py trainer_env
uv run flyte --config $CFG deploy team_eval.py eval_cpu_env

# 2. the serving app env — this is what BUILDS the app image.
#    On a cold registry it then fails at "Failed to materialize artifact
#    mf-promoted-model@latest". Expected and harmless: the image is built by
#    then, and `flyte factory deploy` does NOT build app images.
uv run flyte --config $CFG deploy team_inference.py inference_app_env

# 3. the factory, and the lineage dashboard
uv run flyte --config $CFG factory deploy factory.py
uv run flyte --config $CFG deploy app.py lineage_app_env

# 4. one release day, end to end (builds what is missing, reuses the rest)
uv run flyte --config $CFG factory materialize model-factory mf-promoted-model \
    --partition date=2026-10-02 --wait

# 5. roll it out to the serving app
uv run flyte --config $CFG factory materialize model-factory mf-inference \
    --partition date=2026-10-02 --wait
```

Steps 1–3 are one-time per code change; step 4 is the loop.

### Other things the graph gives you

```bash
# what would be rebuilt, without launching anything
uv run flyte --config $CFG factory plan model-factory mf-promoted-model \
    --partition date=2026-10-02

# just the data release, or just one station's output
uv run flyte --config $CFG factory materialize model-factory mf-rl-tasks-dataset \
    --partition date=2026-10-02

# backfill a week; one action per day, run in parallel
uv run flyte --config $CFG factory materialize model-factory mf-promoted-model \
    --partition date=2026-09-25..2026-10-02

# roll the endpoint back to an earlier promoted model
uv run flyte --config $CFG factory materialize model-factory mf-inference \
    --partition date=2026-10-02 --version mf-promoted-model=<version>
```

Profiles (`model_factory/config.py`): `smoke` (0.5B model, 96 tasks, 10 GRPO
steps — minutes), `smoke-plus`, `dev` (1.5B, ~2K tasks, 100 steps), `full`
(10K tasks, 500+ steps). The profile is a build constant, so
`--param mf-policy-checkpoint.profile_name=dev` overrides it per run.

### Why the artifacts are `mf-`-prefixed

An artifact's partition schema is fixed in the registry by its first version
and cannot change afterwards. The pre-factory trigger chain had already
published `rl-tasks-dataset`, `policy-checkpoint`, `eval-report` and
`promoted-model` **unpartitioned**, so declaring them `{date: Daily}` is
rejected at deploy. The prefix keeps the daily dimension (and with it per-day
reuse and backfill) and leaves the earlier rounds' lineage in place.

## Clusters

Tenants differ in node pools and org policy, so everything cluster-specific
(GPU device, GPU task sizing, whether apps may be anonymous) lives in
`ClusterProfile` in `model_factory/config.py`. Select one at deploy time
with `MF_CLUSTER`; it defaults to `demo`.

| | `demo` (default) | `playground` |
|---|---|---|
| endpoint | `demo.hosted.unionai.cloud` | `playground.canary.unionai.cloud` |
| config | `~/.flyte/config-model-factory.yaml` | `~/.flyte/config-playground.yaml` |
| accelerator (train/synth/eval) | `A10G:1` | `T4:1` |
| accelerator (serving app) | `L4:1` | `T4:1` |
| GPU envs: cpu / memory / disk | 6 / 24Gi / 100Gi | 2 / 10Gi / 100Gi |
| CPU envs: cpu / memory | 2 / 4Gi | 12 / 24Gi |
| apps | public (`requires_auth=False`) | authenticated (org sets `app.disallow_anonymous`) |

`flyte.Resources(memory=...)` is **host** memory, not VRAM — VRAM comes with
the accelerator named in `gpu`, so `V100:4` requests four V100s and their
memory along with them.

`playground` has no A10G or L4. Its GPU pools are V100 (`p3.8xlarge` /
`p3.16xlarge`) and T4 (`g4dn.xlarge`). On-demand V100 capacity is scarce
there — `V100:4` needs a whole free p3 node and sat queued for 90+ minutes
with the node group at max size — so the profile targets `g4dn.xlarge`
(1x T4, 3670m CPU / 14000Mi allocatable). CPU envs at 12 / 24Gi target the
`c5.4xlarge` pool (15640m / 26900Mi).

An unschedulable pod does not fail the run, it just queues, so the parent
reports `running` indefinitely. Check the action's K8s events rather than
waiting.

Deploying there also needs an explicit `--project`, since that config file
defaults to `flytesnacks`:

```bash
export MF_CLUSTER=playground
CFG=~/.flyte/config-playground.yaml
P="--project model-factory --domain development"
uv run flyte --config $CFG deploy $P team_data.py de_cpu_env
# ... same for the other units, then:
uv run flyte --config $CFG run $P team_data.py data_release --profile_name smoke --auto_approve
```

Individual fields can be overridden without adding a profile: `MF_GPU`,
`MF_INFERENCE_GPU`, `MF_GPU_CPU`, `MF_GPU_MEMORY` (host memory), `MF_GPU_DISK`,
`MF_CPU`, `MF_CPU_MEMORY`, `MF_REQUIRE_AUTH`, `MF_ORG`.

## Secrets

`HUGGINGFACE_TOKEN` and `WANDB_API_KEY` exist on the demo tenant (project
`model-factory`, domain `development`); attach them by deploying/running with
`MF_USE_SECRETS=1` (CI does). Without them the loop still runs (public
models/datasets; W&B disabled). See [TODO.md](TODO.md).

## Layout

| path | team / role |
|---|---|
| `model_factory/factory.py` | **the graph**: what each artifact is made of, its partitions, the serve step |
| `model_factory/contracts.py` | the inter-team interface: artifact names, partition dimension, payload schemas |
| `model_factory/shared/` | platform libs: sandbox, rewards, reporting, assets, gates, images, inference client |
| `model_factory/data_engineering/` | data eng: curate + oracle-verify, synthetic gen, release assembly + data gate |
| `model_factory/training/` | training: GRPO (TRL + LoRA) |
| `model_factory/evaluation/` | eval: candidate-vs-base pass@1, auto margin + promotion gate |
| `model_factory/inference/` | inference: serving app, model bound by the factory (adapter toggle per request) |
| `model_factory/lineage_app.py` | platform: global lineage AppEnvironment |
| `factory.py` | deploy entrypoint for the graph (`flyte factory deploy factory.py`) |
| `team_*.py` | deploy units for each team's task environments |

Station tasks return **plain** `File`/`Dir`. The factory declares and publishes
each artifact — name, kind, partition values and parent versions — by calling
the task with `produces_artifacts=True` inside a `flyte.artifacts.produces(...)`
block, so a task must not wrap its own output in `flyte.artifacts.new()`.
Lineage comes for free: every version carries `flyte.io/materialization`,
`flyte.io/factory` and a `flyte.io/consumed/<param>` entry per input.
