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
| `model_factory/contracts.py` | the inter-team interface: artifact names, schemas, publish() |
| `model_factory/shared/` | platform libs: sandbox, rewards, reporting, assets, gates, images, inference client |
| `model_factory/data_engineering/` | data eng: curate, filter, oracle-verify, synthetic gen, release + data gate |
| `model_factory/training/` | training: GRPO (TRL + LoRA), OnArtifact(rl-tasks-dataset) trigger |
| `model_factory/evaluation/` | eval: candidate-vs-base, gates, promotion, OnArtifact(policy-checkpoint) trigger |
| `model_factory/inference/` | inference: serving app (adapter toggle per request) + weight-rollout ops |
| `model_factory/lineage_app.py` | platform: global lineage AppEnvironment |
| `integration.py` | cross-team E2E driver (plays the event bus until artifact events ship) |
