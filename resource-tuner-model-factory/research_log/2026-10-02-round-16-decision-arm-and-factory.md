# Round 16 — a decision-model arm, and the suite as a factory (2026-10-02)

## Context

Two arms existed: the LLM policy (GRPO, emits `{"cpu","memory","gpu"}` as text
and gets snapped onto the grid) and the quantile-GBT baseline (regresses a
high quantile per resource, then buckets). Round 10 left them tied on
$/successful task-hr; round 11 got the first gate pass with a 4B full
fine-tune.

This round adds a **third** arm with a different target, and wires the arm
suite as a Union factory. The LLM path is untouched — not refactored, not
re-pinned, not re-triggered — which was a hard constraint on the design.

The new arm's premise: the thing being optimized is a choice among finitely
many requests, and `policy/actions.py` already defines that finite set
(log-scale memory grid, fixed CPU increments, typed GPU counts). The GBT
baseline predicts a *number* and rounds it. A decision model can predict the
*action* directly — and then the loss can express the one thing a regression
loss cannot:

- one bucket too small → the task OOMs; the run is lost, plus a retry
- one bucket too big → you pay for memory you did not use; a few percent

So: a multi-head classifier (memory / CPU / GPU) over the grid, trained with
cross-entropy weighted by a cost matrix in which under-provisioning costs a
flat `under_penalty` and over-provisioning costs the extra resource actually
paid for. `under_penalty` is the arm's hyperparameter, and sweeping it is
supposed to trace the OOM-rate / waste frontier.

## Run ledger

Base: `https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/<run>`

| run | what | result |
|---|---|---|
| [rt-decision-r1](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rt-decision-r1) | first suite materialization | **FAILED** — all 3 arms `OOMKilled` (exit 137) |
| [rt-decision-r2](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rt-decision-r2) | after streaming reader + row caps | **FAILED** — `'Reporter' object has no attribute 'stats'` (3 arms) |
| [rt-decision-r3](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rt-decision-r3) | after fixing the report API | **3 models built**; 3 scorecards FAILED — `baseline_proposal() missing 1 required positional argument: 'family'` |
| [rt-decision-r4](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rt-decision-r4) | after fitting the family baseline first | **1 source + 3 models + 3 scorecards + 1 champion**, 7m12s |
| [uhsxjcwkhndldcgd9mcx](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/uhsxjcwkhndldcgd9mcx) | throwaway task to read the scorecards in-cluster | succeeded |

`rt-decision-r4`: three `fit_decision_model` actions ran in parallel (5m38s,
6m11s, 6m47s), three `score_decision_model` in parallel (~15s each),
`select_decision_tuner` 3.9s.

## Results

Heldout split, 512 workloads, corpus `uhrmbq9th9pw` (1,016,896 rows; 60,000
sampled for training per the new cap). Rule baseline is the per-family median
(`fit_family_baseline`), fitted on the same train rows.

| arm | `under_penalty` | fit | OOM | median waste | grid floor | $/task-hr | $ saved / 1k task-hr |
|---|---|---|---|---|---|---|---|
| **oom-paranoid** (champion) | 40 | **89.1%** | 10.9% | 69.1% | 36.1% | 0.1558 | **−7.68** |
| oom-averse | 12 | 70.9% | 29.1% | 61.8% | 36.1% | 0.1495 | −1.34 |
| balanced | 2 | 54.3% | 45.7% | 38.4% | 36.1% | 0.1418 | +6.34 |
| *rule baseline* | — | *76.0%* | *24.0%* | *54.1%* | — | *0.1481* | — |

**The knob works, and the arm does not yet beat the baseline.** Both halves
matter:

- `under_penalty` is **monotone across all three arms** on fit rate, OOM rate,
  waste and cost. That is the frontier the arm was built to trace, and it
  traced it on the first successful run. The ranking the champion selector
  produced (`oom-paranoid > oom-averse > balanced`) is exactly the penalty
  ordering.
- Only the paranoid arm clears the rule baseline on fit (89.1% vs 76.0%), and
  it does so **at higher cost** (0.1558 vs 0.1481 $/task-hr — it *loses*
  $7.68 per 1k task-hours). The balanced arm is cheaper than the baseline but
  much worse on fit. So on this corpus, at this training scale, the decision
  model buys +13pp fit for +5% cost. It does not dominate.

Caveats that bound the claim, all of them fixable:

- 60,000 of 1,016,896 train rows (the new cap), 40 epochs, a 256x128 trunk.
  This is the smallest version of the arm, not a tuned one.
- **The grid itself forces 36.1% median over-provisioning.** So the paranoid
  arm's 69.1% waste is roughly half grid and half model, and the baseline's
  54.1% is *below* the grid floor only because it is not confined to the grid
  the same way. A `fine` action space is already registered and unrun; that is
  the first experiment to run next, and it is one extra partition value.
- Selection is lexicographic (fit first, cost as tie-break), which is why a
  more expensive arm won. That ordering is deliberate — an OOM costs the whole
  run plus a retry, and no weighting of a blended objective expresses that
  cliff honestly — but it does mean the champion is not the cheapest arm.

## Findings

**1. `factory.source` is what let this be additive.** The corpus is produced
by the existing `build_task_corpus` station, which still publishes
`tuning-task-corpus` itself and still fires the OnArtifact triggers that drive
the LLM and GBT arms. Declaring it as a source means the factory *reads* that
artifact rather than owning it — so the whole arm went in with **zero changes**
to `grpo.py`, `build_task_corpus`, the trigger wiring or the tune service.

It also unlocks the trigger shape a factory cannot get from its own builds:
`factory.on(corpus)` fires a fresh suite on every new corpus version. The
sibling basic-model-factory has no source and therefore no such trigger.

**2. A million-row corpus cannot be read with `pd.read_parquet`.** Round 15
hit this in the trainer; it bit again here, from a different direction — the
station OOMed before any model was built (`rt-decision-r1`, exit 137 on all
three arms at 4Gi). Two causes, both fixed:

- the whole frame, including two source-code columns, materialized at once.
  Now `pyarrow.ParquetFile.iter_batches` with an explicit column projection
  (`harness_code` is a second full copy of every workload's source and nothing
  reads it), stopping once the row caps are met. Station memory is now a
  function of the caps, not of the corpus.
- `torch.tensor(featurize(...))` built a 60k x 43 Python list-of-lists first,
  where every value is a boxed float. Now a preallocated tensor is filled row
  by row.

`feature_names` also stopped scanning every row to discover a key set that is
identical for all of them.

**3. Two API-shape bugs cost two full runs, and both were avoidable locally.**
`Reporter` has `kv`/`h`/`table`/`p`, not `stats`, and `table` takes no
`title=`. `baseline_proposal(baselines, family)` is a *lookup* into a baseline
that `fit_family_baseline(train)` has to build first, not a function of one
record. Both live inside async flyte tasks, so both surfaced only on the
cluster — after every model in the suite had already trained for six minutes.

Added two guard tests: one reflects over the station's `rep.*` calls and
asserts each exists on `Reporter` (plus that `table` has no `title`
parameter), and one runs the whole scoring composition —
`read_corpus` → `fit` → `_simulate` → `fit_family_baseline` →
`baseline_proposal` → `_simulate` — on a tiny synthetic parquet outside flyte.
The second would have caught `rt-decision-r3` in 3 seconds.

**4. The image layers were pinned a minor behind the SDK.** `images.py` carried
`flyte>=2.7.0,<2.8.0` as a re-pin layer from round 3. With the project moved
to `flyte>=2.10.0` for factories, that would have run tasks on a runtime that
cannot read a factory task spec. Floor now tracks the deploying SDK, with the
reason written down.

**5. Partitioned artifacts needed new names.** `tuner-checkpoint`,
`ml-baseline-model` and friends already exist unpartitioned, and a registry
partition schema is immutable after the first version — so the arm's
artifacts are `rt-`-prefixed. The *source* keeps its original name precisely
because it is unpartitioned, which is what makes it readable as a source at
all.

## Code changes

- **new** `policy/action_space.py` — the discretization as an explicit,
  configurable object: `ActionSpace` with three heads
  (`memory_grid_mib`, `cpu_grid`, `gpu_options`), `encode_labels` (the
  cheapest cell covering the ground truth — the training target), `decode`
  (→ `Proposal` → `flyte.Resources(cpu=…, memory=…, gpu=…)`), and
  `grid_headroom` (the waste floor the grid alone imposes). Three registered
  variants: `default`, `coarse`, `fine`.
  Three heads rather than one joint head because the joint space is 10x8x4 =
  320 classes, most of which never occur; the factored version shares the
  trunk and keeps each head's labels dense.
- **new** `training/decision_model.py` — the multi-head classifier,
  `cost_matrix`/`gpu_cost_matrix`, `expected_cost_loss` (the expectation of
  the cost matrix under the predicted distribution, so the gradient pushes
  mass toward the *cheap* side of the right answer), `DecisionArm` + the
  `ARMS` registry, save/load carrying the grid and feature layout (a
  checkpoint is undecodable without both).
- **new** `training/decision_station.py` — `fit_decision_model`,
  `score_decision_model`, `select_decision_tuner`, plus the streaming
  `read_corpus`. Scoring reuses `simulate_episode` and `pricing`, the same
  instruments the LLM arm's eval uses, so all three arms land on one
  comparable set of numbers.
- **new** `resource_tuner/factory.py` + `factory.py` + `decision.py`.
- `shared/images.py`: `decision_image` (CPU torch — the arm never needs an
  accelerator, and torch on `driver_image` would pad every orchestration
  action's pull), metrics/flyte floors corrected.
- `training/envs.py`: `decision_env`, explicitly at 12Gi.
- Features are reused verbatim from `ml_baseline.extract_features`, so the GBT
  and decision arms are compared on identical information and any difference
  is attributable to the target and the loss.

Tests: 312 pass (was 263) — 49 new across the action space, the arm and the
factory graph. The load-bearing one is
`test_higher_under_penalty_reduces_under_provisioning`, which asserts the
frontier claim locally; `rt-decision-r4` then confirmed it on the cluster.

## Next

- Run the `fine` action space as a fourth arm. The 36.1% grid floor is the
  single biggest term in the waste number, and it is a hyperparameter.
- Lift the train cap and see whether the arm's cost crosses under the
  baseline; 60k/1M is the smallest version of this experiment.
- Put the decision arm into the tune service as a third A/B estimator
  alongside the LLM checkpoint and the GBT baseline. `rt-decision-champion` is
  published and ready to be served; nothing consumes it yet.
- `factory.on(corpus)` is declared but has not fired — the next corpus
  release should trigger a suite refit by itself.
