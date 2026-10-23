# 2026-09-16 — round 15: artifact cards, and a 1-day full fine-tune on the 1M corpus

## Context

Two threads, one branch.

**Artifact cards.** Every station here hands off through a versioned
artifact, but a published artifact carried only a name, a one-line
description and a `kind`. Everything a consumer needs to trust it — the
schema, the split sizes, which teacher wrote the rows, which corpus
trained a checkpoint, the gate verdict — was either re-derived by
downloading the payload (the lineage dashboard's `_artifact_card`, at view
time, per viewer) or not recoverable at all (the seed, the profile, the
hypothesis). The SDK has supported `flyte.artifacts.Card` since 2.7;
nothing here used it.

**The 1-day full fine-tune.** Round 11's `r11-fullft-4b` beat LoRA on
reliability (79% vs 53% fit, 31% vs 34% waste) on a 300-step budget, but
every arm to date still fails the gate's waste clause, and the only gate
PASS has been the QLoRA `ambitious` arm. The question this round asks: what
does a full fine-tune do with a real training budget on a real corpus?

The corpus makes that question worth asking now.
[`tuning-task-corpus/uhrmbq9th9pwvcnr2vfb-a0-1`](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/artifacts/tuning-task-corpus/uhrmbq9th9pwvcnr2vfb-a0-1)
(published 2026-09-13) is **templates(16,896) + archetypes(1,000,000)** via
`qwen35-397b,minimax-m3,qwen38-27b`, seed 31 — **the 1M target that round
12-14 left open is met**, at 1,016,896 rows.

## Run ledger

Base URL: `https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/`

| run | what | outcome |
|---|---|---|
| [usnnr9xmpnnq4ftjqgzn](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/usnnr9xmpnnq4ftjqgzn) | `ambitious-fullft-4b-1d`, first launch | ❌ **OOMKilled (host RAM)**, 4 attempts, ~40s each, before a step ran |
| [umnzhngvwbf9d6fbhwfz](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/umnzhngvwbf9d6fbhwfz) | relaunch with the streamed corpus read | ❌ **CUDA OOM on the L40S**, 4 attempts, ~8 min each — reached training, died in the step |
| [u74ftlrgr5fvt7sfpkmh](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/u74ftlrgr5fvt7sfpkmh) | relaunch at batch 2 × accum 2 | ✅ trained clean on attempt 1 — **aborted by us at 839 steps**, having measured the arm at 2.92 s/step |
| [udrtv5nj7dcth86qdfn6](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/udrtv5nj7dcth86qdfn6) | the real run: 26,000 steps, ~21.7h of stepping | _in flight_ |

Corpus provenance for that run:
[uhrmbq9th9pwvcnr2vfb](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/uhrmbq9th9pwvcnr2vfb)
(the 1M archetype release, 2026-09-13).

Deployed with `RT_TRAIN_GPU=L40s:1 RT_TRAIN_MEMORY=48Gi RT_TRAIN_DISK=100Gi
RT_TRAIN_CPU=4` — the trainer env's resources are a DEPLOY-time property,
so this is also the shape every trigger-fired dark training run now gets
until the next deploy changes it.

## How the run was sized (measurement, not guesswork)

Every number below came from a prior run before a single GPU-hour was
committed:

- **Step time**: run `unv2hq4csvqzqv4tnpg2` (this same model, full FT, one
  L40S) logged 49 steps at **mean 3.57 s/step** (median 3.27, range
  2.45-6.77). Reserving ~1.8h of the day for model load, 18 intra-task
  saves and the final ~8GB checkpoint upload leaves ~80,000s of stepping —
  20,000 steps at 4.0 s/step. **18,000** keeps ~10% headroom, so the run
  lands inside a day anywhere in the 3.6-4.8 s/step range.
- **Model rung**: Qwen3-4B on ONE L40S, chosen over the bigger full-FT
  rungs on PROVISIONING, not on capability. `L40s:1` has scheduled in ~4
  minutes every time; the `L40s:4` (14B) and `L4:4` (8B) groups failed to
  provision on most round-11 attempts, and a 1-day budget spent in a queue
  is a day lost. The 14B is also ~10x the cost per step (~36 s/step derived
  from `uk9pj886sxmw9sqnhgrt`'s wall clock) — 24h buys ~2,400 steps there
  against ~18,000 here.
- **Group size 4, not 8** — the one knob NOT inherited from
  `r11-fullft-4b`. That arm ran batch 8 at the default 1,024-token prompt
  budget; this corpus carries real library code and `ambitious` already
  raised the budget to 1,536 for it. Paying for the longer prompt out of
  the GROUP rather than out of the PROMPT is the trade `ambitious` made:
  a clipped prompt drops real code out of the middle of the workload,
  whereas 4 completions per group is an ordinary GRPO configuration with a
  noisier advantage estimate. `vram_estimate_gib` agrees — batch 4 ×
  1,664 tokens × Qwen3's 151,936 vocab is ~3.8 GiB of fp32 logits (~17 GiB
  by the calibrated multiplier) on top of ~24 GiB of bf16 weights + grads +
  8-bit Adam, clearing one L40S's ~44 GiB. Batch 8 at 1,536 would not.

## Code changes

- **`resource_tuner/shared/cards.py`** (new): pure markdown renderers for
  every artifact type + `upload()`. All seven publish sites now attach a
  card and flat `attrs`; `publish()` in `contracts.py` grew `attrs`/`card`
  parameters. `CORPUS_COLUMN_DOCS` turns the corpus schema comments into
  data so the card renders the same column meanings the code enforces, and
  an undocumented new column shows as *undocumented* rather than
  unexplained. Cards are best-effort by construction — `upload()` and
  `corpus_stats_safe()` log and continue, because losing documentation is
  never a reason to lose an artifact at the tail of a 19-hour run.
  `tests/test_cards.py` covers the renderers on partial state and asserts
  at AST level that no publish site ships without a card.
- **`publish_intermediate_checkpoint`** now takes `base_model`/`use_lora`/
  `max_steps` so an intermediate's card names the model it adapts instead
  of assuming LoRA.
- **`train_tuner` reads the corpus without `harness_code`.** Nothing in
  training touches that column — it is the executable twin used by episode
  pods, which eval runs, not this task — and on a 10⁶-row corpus it is
  gigabytes of trainer RAM for a column the task never opens.
- **`glm-5-2` retired** from the teacher roster (`shared/llm_client.py`);
  the roster is now `qwen38-27b`, `minimax-m3`, `qwen35-397b`. The corpus
  this round trains on was already generated without it.
- **`AMBITIOUS_FULLFT_4B_1D`** profile added: Qwen3-4B full FT, 26,000
  steps, group 4, prompt 1,536, batch 2 × accum 2 on one L40S. The sizing
  arithmetic and both OOM calibration points are recorded in its comment.
- **`_sample_train_records`** replaces the whole-corpus pandas read
  (finding 2); **`gradient_accumulation_steps`** added to `TunerProfile`
  and wired into `GRPOConfig`, which never set it (finding 3);
  **`vram_estimate_gib`/`preflight_vram`** now take the model's real
  parameter count and model full-FT state (finding 3).
- Tests: `tests/test_corpus_sampling.py` (8 cases) pins the bounded read —
  including, precisely, that `pd.read_parquet` is never called;
  `tests/test_round11_arms.py` gains the TRL divisibility + VRAM-fit
  checks and a regression that the batch-4 config can never pass preflight
  again.

## Findings

_The training run is in flight; nothing about full FT vs LoRA is claimed
until it lands and its eval report does._

What the three dead runs DID establish, all of it load-bearing for the
next person:

### 1. Trainer resources come from the DEPLOY-time environment, and `flyte run` re-registers them

`train_resources()` reads `RT_TRAIN_GPU`/`RT_TRAIN_MEMORY` at import, so
the task environment is serialized from whatever process registers it —
and **`flyte run` registers it too, not just `flyte deploy`**. Setting the
vars on the deploy alone put `usnnr9xmpnnq4ftjqgzn` on the 10Gi/T4
defaults while the deployment said L40S/48Gi. It OOMKilled in 40s on the
corpus read; had it survived that, it would have failed VRAM preflight on
a T4. **Export the `RT_TRAIN_*` vars for both commands.** The profile's
`train_gpu`/`train_memory` are advisory — preflight compares against them,
they do not request anything.

### 2. A 10⁶-row corpus cannot be read with `pd.read_parquet`

`train_tuner` read the whole corpus into pandas and then sampled it down.
Fine at 10⁵ rows, fatal at 10⁶ — and pure waste either way: GRPO consumes
ONE prompt per step, so this run touches 26,000 of 1,016,896 rows.
Replaced by `_sample_train_records`: pass 1 reads only the `split` column
to choose k rows with a fixed seed, pass 2 re-reads in row-group batches
keeping just those. Memory scales with `train_contexts`, not corpus size,
and `harness_code` — the executable twin used by EVAL's episode pods,
never by training — is not read at all.

### 3. `preflight_vram` was blind to the thing that makes full FT expensive

It models the fp32 logits tensor and compares it against free VRAM *at
preflight time*. For a full fine-tune the gradients (bf16, 2 B/param) and
the 8-bit Adam moments (2 B/param) — ~15 GiB for a 4B model — allocate on
the first optimizer step, long after the check has passed. So
`umnzhngvwbf9d6fbhwfz` sailed through preflight and died in training,
four times, eight minutes apart.

The estimator now adds that state and a **1.25× margin measured, not
padded**: the observed failure wanted ~45.8 GiB, ~37.8 GiB of it outside
the resident weights, while logits×4.5 + state explains only ~84% of that.
The residual is activations and the generation KV cache, which the
function still does not model — the margin is that residual, stated as
what it is. It now rejects the config that died (39.9 GiB vs 36.5 free)
and passed the fixed one with 7.2 GiB to spare, confirmed on the live
device: `L40S: 36.5 GiB free of 44.4 · step needs ~29.3 GiB`.

### 4. Step time tracks sequences per optimizer step, not nominal batch size

The 18,000-step budget was extrapolated from `unv2hq4csvqzqv4tnpg2`'s
3.57 s/step — measured at batch 8 / ngen 8. This arm puts FOUR sequences
through a step, and measured **2.92 s/step** (median 2.69, recent 3.01,
839 steps). 18,000 steps was therefore ~16h, not the intended day; the
budget is now 26,000. Extrapolating a step time across a different
batch/group shape is a guess wearing measurement's clothes.

### 5. The 1M-row corpus target is met

`uhrmbq9th9pwvcnr2vfb` published 1,016,896 rows on 2026-09-13, closing the
"1M target is not met" item from round 12-14. `udrtv5nj7dcth86qdfn6` is
the first run to train on it.

## Still open

- The run itself: does full FT + 26,000 steps + 1M rows move median
  overprovision below the rule baseline's 25-28%? That clause is the one
  every arm has failed.
- Artifact cards are **unverified on a live artifact** as of this entry.
  The renderers and wiring are unit-tested, but the first card actually
  published to the control plane will be this run's `tuner-checkpoint`.
  (The three failed runs published nothing — they died before the tail.)
- Artifact versions published before this round carry no card; nothing
  retrofits them. The dashboard's download-and-derive `_artifact_card`
  remains the fallback for those, and was left in place.
- `ARTIFACT_PROMOTED` (`promoted-tuner`) is declared in `contracts.py` but
  never published by any task — so it has no artifact and no card. Decide
  whether promotion was meant to land as an artifact.
- The Qwen3-14B full-FT checkpoint from `uk9pj886sxmw9sqnhgrt` is STILL
  unevaluated (carried over from round 12-14).
