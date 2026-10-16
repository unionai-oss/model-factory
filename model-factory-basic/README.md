# basic-model-factory

A **closed-loop model factory** on [Flyte v2](https://www.union.ai/docs/v2/) /
Union, small enough to read in an afternoon. Every day it:

1. curates a fresh slice of coding problems,
2. writes new practice problems aimed at what yesterday's model got wrong,
3. trains a small coding model (Qwen2.5-Coder-0.5B) with RL — GRPO, where
   **running the unit tests is the reward** — starting from yesterday's best model,
4. scores it against yesterday's best and the base model on a fixed held-out
   exam,
5. releases whichever is better, serves it, and hands it to tomorrow.

The whole loop is one declared [Union factory](https://www.union.ai/docs/v2/union/user-guide/factories/):
a graph of versioned, partitioned artifacts. Nothing runs until you ask for an
artifact; the platform works out what has to be built and reuses the rest.

Everything is public and self-contained: KodCode-Light-RL-10K for problems,
Qwen2.5-Coder 0.5B (student) and 1.5B (problem writer) from Hugging Face, a
local subprocess sandbox for the tests. No external LLM APIs, no gated models.

**Results so far** — see [research_log/](research_log/README.md):
on a fixed 200-task held-out split, one A10G day of GRPO takes
Qwen2.5-Coder-0.5B-Instruct from **35.0% to 42.5% pass@1** (+7.5 points, paired
95% CI [+3, +12], 19 tasks gained / 4 lost). Three further warm-started days
landed at 42.0%, 41.0% and 41.0%, within noise of that champion, so the
champion/challenger gate kept the day-1 model each time. The loop runs
unattended end to end; it does not yet *compound* at this budget (see
[round 2](research_log/2026-10-15-round-2-closed-loop.md) for why and what's
next).

## The loop, one day at a time

```
  built once (no partitions)              per release day D  (partitioned by date)
  ──────────────────────────              ────────────────────────────────────────
  mf-heldout-tasks ─┬───────────────────▶ mf-seed-tasks[D]       curate the day's slice
        │           │                          │
        ▼           │   mf-champion[D] ────────┼──▶ mf-synthetic-tasks[D]   mutate the champion's
  mf-baseline-      │   (source) │             ▼          │                  failures (1.5B teacher)
  report            └────────────┼──▶ mf-rl-tasks-dataset[D] ◀┘   leak check + HITL gate 1
        │                        │             ▼
        │                        ├──▶ mf-policy-checkpoint[D]     GRPO, warm-started from the champion
        │                        │             ▼
        └────────────────────────┼──▶ mf-eval-report[D]           candidate vs champion vs base
                                 │             ▼
                                 └──▶ mf-promoted-model[D]        the better of the two + HITL gate 2
                                               │
                                               ├──▶ mf-inference (app)      serve it
                                               ▼
                       crown_champion (artifact trigger) ──▶ mf-champion[D+1]
```

| artifact | station task | what it is, and why it is an artifact |
|---|---|---|
| `mf-heldout-tasks` | `curate_heldout` | 200 verified problems from rows 0–599 of the dataset. **No partition**: it is built once and every day reads the same version, which is what makes day-over-day numbers comparable. |
| `mf-baseline-report` | `evaluate_baseline` | The base model's pass@1 on that exam. Also built once; every day's eval reads it instead of re-generating it. |
| `mf-champion[D]` | *(published outside the factory)* | The model day D starts from and must beat. Day one: the base model (`seed_champion`). After that: yesterday's release (`crown_champion`). |
| `mf-seed-tasks[D]` | `ingest_and_curate` | ~256 problems from a window of the dataset that rotates by date (new day, new problems), schema-mapped, deduped, and **oracle-verified**: the reference solution must pass its own tests in the sandbox. Anything in the held-out set is dropped. |
| `mf-synthetic-tasks[D]` | `generate_synthetic_tasks` | The **failure-driven curriculum**. The champion carries the list of training problems its RL rollouts kept failing; a 1.5B teacher writes variants of those (topping up with random ones), and only variants whose own solution passes their own tests survive; the failures it *sometimes* solved are also replayed as-is. Batched GPU generation via `DynamicBatcher`. |
| `mf-rl-tasks-dataset[D]` | `assemble_dataset` | Seed + synthetic. Refuses to publish if any task is in the held-out set. **Human gate 1** (data validation) lives here. |
| `mf-policy-checkpoint[D]` | `train_grpo` | TRL GRPO + LoRA on one A10G, continuing the champion's adapter. The reward is format (0.1) + parses (0.1) + all hidden tests pass (1.0), with anti-reward-hacking guards. Records which tasks it could not solve. |
| `mf-eval-report[D]` | `evaluate_checkpoint` | Greedy pass@1 of candidate, champion and base on the fixed exam, paired deltas with bootstrap 95% CIs. |
| `mf-promoted-model[D]` | `promote_checkpoint` | Champion/challenger: the candidate if it beats the champion by ≥1 point and **human gate 2** approves; otherwise the champion carries forward. A bad day never moves the factory backwards. |
| `mf-inference` | *(app)* | A FastAPI app serving the release; the factory deploys it with the exact version it just built. |

Every station returns a plain `File`/`Dir`. The factory declares the artifact
(name, kind, partition, parents) and publishes it, and stamps each version with
`flyte.io/materialization` and one `flyte.io/consumed/<param>` entry per input —
so "which dataset trained the model we serve?" is a lookup, not an investigation.

### How the loop closes (and why it looks like this)

Day D+1 has to build on day D. The obvious wiring — "train from yesterday's
`promoted-model`" — is a cycle (`promoted-model → … → promoted-model`), and a
factory graph is a DAG over artifact names, so `factory deploy` rejects it.
There is also no "previous partition" mapping: `.window()` always includes the
consumer's own day.

So the state that crosses days is its own artifact, `mf-champion`, which the
graph only *reads* (`factory.source`). One small task outside the graph writes
it:

```python
crown_on_promotion = flyte.Trigger(
    name="crown-champion",
    automation=flyte.OnArtifact(name="mf-promoted-model"),
    inputs={"promoted": flyte.TriggeredArtifact, "date": flyte.TriggeredPartition("date")},
)

@eval_cpu_env.task(triggers=[crown_on_promotion], produces_artifacts=True)
async def crown_champion(promoted: Dir, date: datetime) -> Dir:
    return flyte.artifacts.new(promoted, champion_metadata(date + timedelta(days=1), ...))
```

What that buys:

- **Day D is a pure function of `champion[D]`.** Re-materializing a day is all
  cache hits; a cache hit publishes no new version, so it does not re-crown.
- **A day without a champion never trains.** Blocking is per dependency: the
  day's seed tasks still get curated (they do not read the champion), then
  everything downstream of `mf-champion[D]` is blocked with `source
  partition(s) not in the registry`. That is also why a backfill runs day by
  day: each day's champion appears when the previous day is released.
- **Warm start and curriculum are both just inputs.** `train_grpo` and
  `generate_synthetic_tasks` take `champion=champion` like any other artifact,
  and the lineage of every checkpoint names the champion it grew from.

## What the factory gives you

```bash
CFG=.flyte/config.yaml
F="uv run flyte --config $CFG factory"

# plan: what would be built or reused, launching nothing
$F materialize model-factory mf-promoted-model --partition date=2026-10-15 --plan

# materialize: one release day, end to end (builds what is missing, reuses the rest)
$F materialize model-factory mf-promoted-model --partition date=2026-10-15 --wait

# reuse: ask again and every station is a cache hit (seconds, no GPU)
$F materialize model-factory mf-promoted-model --partition date=2026-10-15 --wait

# backfill: a range is one action per day; with a champion chain it runs day by day
$F materialize model-factory mf-promoted-model --partition date=2026-10-16..2026-10-18

# serve: deploy the app bound to that day's release
$F materialize model-factory mf-inference --partition date=2026-10-15 --wait

# roll the endpoint back to an earlier release
$F materialize model-factory mf-inference --partition date=2026-10-15 \
    --version mf-promoted-model=<version>

# just the data release, or rebuild one station on purpose
$F materialize model-factory mf-rl-tasks-dataset --partition date=2026-10-15
$F materialize model-factory mf-promoted-model --partition date=2026-10-15 --rebuild trainer.train_grpo
```

**Triggers.** A `flyte.Cron` on the factory materializes today's release
every night at 06:00 UTC, and the `crown-champion` artifact trigger closes the
loop. Both are live as soon as they are deployed. The nightly run only
trains if today's champion exists, i.e. if yesterday was released; otherwise
it curates the day's seed tasks and stops.

**Human gates.** Both gates are `flyte` conditions inside station tasks, so
`mf-promoted-model` cannot exist without them. They default open
(`AUTO_APPROVE = True` in `model_factory/factory.py`) so an unattended night
finishes; `--param mf-rl-tasks-dataset.auto_approve=false` arms one per run.

## Quickstart

```bash
uv sync
uv run pytest                    # graph shape, loop wiring, held-out isolation, rewards, sandbox

CFG=.flyte/config.yaml
# 1. station tasks, one deploy unit per team (also arms the crown trigger)
MF_USE_SECRETS=1 uv run flyte --config $CFG deploy --all team_data.py
MF_USE_SECRETS=1 uv run flyte --config $CFG deploy team_training.py trainer_env
MF_USE_SECRETS=1 uv run flyte --config $CFG deploy --all team_eval.py

# 2. the serving app env — this BUILDS the app image. On a cold registry it
#    then fails at "Failed to materialize artifact mf-promoted-model@latest";
#    expected: `flyte factory deploy` does not build app images, this does.
uv run flyte --config $CFG deploy team_inference.py inference_app_env

# 3. the factory, and the lineage dashboard
uv run flyte --config $CFG factory deploy factory.py
uv run flyte --config $CFG deploy app.py lineage_app_env

# 4. start the loop: the base model is the champion of the first day
uv run flyte --config $CFG run team_eval.py seed_champion --date 2026-10-13

# 5. day one, then day two (warm-started from day one's release)
uv run flyte --config $CFG factory materialize model-factory mf-promoted-model --partition date=2026-10-13 --wait
uv run flyte --config $CFG factory materialize model-factory mf-promoted-model --partition date=2026-10-14 --wait
```

Profiles (`model_factory/config.py`):

- `daily-r3` (the default): 500 held-out tasks; 384 fresh tasks/day; 80 GRPO
  steps of 8 prompts × 8 rollouts on colocated vLLM; a fresh LoRA each day on
  the champion's merged weights; a 7B teacher (vLLM) for the curriculum.
- `daily`: round 2's day (200 held-out tasks, 50 HF-generate steps, one LoRA
  continued across days).
- `smoke`: tiny everything, minutes. Its promotion margin is -1, so it proves
  plumbing, never quality.

The profile is a constant in the deployed graph: `MF_PROFILE=smoke uv run flyte
... factory deploy factory.py` deploys a smoke factory. **Never change what a
profile name means.** The name is part of every build's cache key but its
contents are not, so a redefined profile silently reuses artifacts built under
the old settings. Add a new name instead.

## Layout

| path | team / role |
|---|---|
| `model_factory/factory.py` | **the graph**: what each artifact is made of, its partitions, the serve step, the nightly trigger |
| `model_factory/contracts.py` | the inter-team interface: artifact names, partition dimension, payload schemas, held-out leak checks |
| `model_factory/data_engineering/` | held-out split, daily curation, synthetic curriculum, release assembly + data gate |
| `model_factory/training/` | GRPO (TRL + LoRA), warm start, hard-task bookkeeping |
| `model_factory/evaluation/` | baseline, candidate-vs-champion eval, release decision, `crown_champion` / `seed_champion` |
| `model_factory/inference/` | the serving app (model bound by the factory) |
| `model_factory/shared/` | sandbox, rewards, reporting, gates, images, registry lookups |
| `model_factory/lineage_app.py` | lineage dashboard (every version of every artifact, and the run that made it) |
| `factory.py`, `team_*.py`, `app.py` | deploy entrypoints |

## Clusters

Everything cluster-specific (GPU type, task sizing, app auth) lives in
`ClusterProfile` in `model_factory/config.py`, selected with `MF_CLUSTER`
(default `demo`). On demo, training/eval run on `A10G:1` and serving on `T4:1`
(`L4` never schedules there). `flyte.Resources(memory=...)` is host memory;
VRAM comes with the accelerator. An unschedulable pod does not fail a run, it
queues — check the action's events rather than waiting.

## Secrets

`HUGGINGFACE_TOKEN` and `WANDB_API_KEY` exist on the demo tenant; attach them
by deploying with `MF_USE_SECRETS=1`. Without them the loop still runs (all
models and data are public; W&B disabled).

## Why the artifacts are `mf-`-prefixed

An artifact's partition schema is fixed by its first version. An earlier,
trigger-chained version of this project published `rl-tasks-dataset`,
`policy-checkpoint` etc. **unpartitioned**, so declaring them `{date: Daily}`
is rejected at deploy; the prefix keeps the daily dimension and leaves the old
lineage in place. See [research_log/round-1](research_log/2026-10-02-round-1-factory-refactor.md).
