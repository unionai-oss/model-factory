# Round 3: trying to make the loop compound (2026-10-16)

## Context

Round 2 closed the loop. Day 1 lifted held-out pass@1 by 7.5 points, but three
warm-started days stayed within noise of that champion (−0.5, −1.5, −1.5 on a
200-task split with ±5-point CIs). Its next-steps list, worked here in priority
order:

1. **A held-out split of ≥500 tasks**, so a real ~+2-point day can clear the
   paired CI. New unpartitioned held-out version, rebaselined once.
2. **Warm-start hyperparameters**: a fresh LoRA each day on the champion's
   *merged* weights (not one adapter continued forever), a lower LR, and
   enough steps per day to matter.
3. **vLLM rollouts** to buy those steps inside the time budget.
4. **A stronger synthetic generator**, since round 2's 1.5B teacher kept 3 of 96.

A fresh champion chain starts at 2026-10-01 (base model, `bmf-r3-seed`), so
round 3's days never mix with round 2's champions, and the final crown lands
on a past date that no nightly will ever materialize.

## Design changes

- **Profile `daily-r3`** (now the factory default):
  - 500 held-out tasks from rows 0–1999 of KodCode-Light-RL-10K; training
    windows start at row 2000.
  - 384 fresh tasks/day.
  - 80 GRPO steps × 8 prompts × 8 rollouts (640 prompt visits/day, 3.2×
    round 2), LR 2e-5 (round 2: 3e-5).
  - Colocated vLLM rollouts.
  - Qwen2.5-Coder-**7B**-Instruct teacher on vLLM.
- **Fresh LoRA on merged champion** (`shared/checkpoints.py`). Checkpoints
  are now `weights: base | lora | full`. Every day loads the champion, folds
  in a round-2 adapter if there is one, trains a fresh LoRA, and saves the
  merge as full weights. Generation N is one model directory, and each day's
  optimizer starts clean. Training, eval and serving all load through the
  same helper.
- **vLLM image for the trainer and the teacher** (`trainer_image`:
  `vllm==0.31.0`, which pins the torch 2.13 the stack already uses; TRL 1.10
  only *warns* above the vLLM 0.26 it was tested with).
  `VLLM_USE_FLASHINFER_SAMPLER=0` (finding 4).
- **Teacher choice is a profile field.** `teacher_app="<app>@<project>"`
  switches the curriculum to `generate_synthetic_tasks_served`, a CPU task
  that calls an OpenAI-compatible app over in-cluster DNS. The model id comes
  from `/v1/models` and reasoning before `</think>` is stripped. Without it,
  `generate_synthetic_tasks` runs the teacher locally, on vLLM when the image
  has it (one batched `LLM.chat` call). The curriculum logic itself
  (`_curriculum`) is shared.
- **Profiles are immutable names** (finding 1): round 2's settings keep the
  name `daily`, and round 3's are `daily-r3`.

## Run ledger

Base: `https://demo.hosted.unionai.cloud/v2/domain/development/project/model-factory/runs/<run>`

| run | what | outcome |
|---|---|---|
| [bmf-r3-teacher-probe](https://demo.hosted.unionai.cloud/v2/domain/development/project/model-factory/runs/bmf-r3-teacher-probe) | can a task reach `qwen38-27b-vllm@llm-service`? | yes via `http://qwen38-27b-vllm.llm-service-development.svc.cluster.local`; `/v1/models` → `qwen38-27b`. **118 s for 111 tokens**; reasoning text precedes `</think>` despite `/no_think` |
| [bmf-r3-vllm-probe](https://demo.hosted.unionai.cloud/v2/domain/development/project/model-factory/runs/bmf-r3-vllm-probe) | `train_grpo`, profile `probe-r3` (daily shape, 6 steps), vLLM colocate | works: **64 completions/step at ~31 s/step** (round 2's HF generate: 32 per 32 s), about 2× rollout throughput. 80 steps chosen to fit the day |
| [bmf-r3-teacher-curriculum](https://demo.hosted.unionai.cloud/v2/domain/development/project/model-factory/runs/bmf-r3-teacher-curriculum) | `generate_synthetic_tasks_served` with the 27B app, 128 prompts | **ABORTED** by me after 80+ min with no result: the shared app is too slow for a daily budget. Switched to a local 7B teacher on vLLM |
| [bmf-r3-seed](https://demo.hosted.unionai.cloud/v2/domain/development/project/model-factory/runs/bmf-r3-seed) | `seed_champion --date 2026-10-01` | champion[10-01] = base |
| [bmf-r3-d1](https://demo.hosted.unionai.cloud/v2/domain/development/project/model-factory/runs/bmf-r3-d1) | day 1, first try | **ABORTED** by me: `curate_heldout` was a 0.1 s cache hit on **round 2's 200-task** held-out version (finding 1). Fixed by naming round-3's profile `daily-r3` |
| [bmf-r3-d1b](https://demo.hosted.unionai.cloud/v2/domain/development/project/model-factory/runs/bmf-r3-d1b) | day 1, second try | **FAILED**: 7B teacher vLLM engine init, `Could not find nvcc` (FlashInfer sampler JIT; finding 4). The new 500-task held-out, the day's seed tasks and the 500-task baseline were built and reused by the retry |
| [bmf-r3-d1c](https://demo.hosted.unionai.cloud/v2/domain/development/project/model-factory/runs/bmf-r3-d1c) | **day 1** (2026-10-01), cold start | promoted; crowned ([u8edef6fd110e4f4d](https://demo.hosted.unionai.cloud/v2/domain/development/project/model-factory/runs/u8edef6fd110e4f4d)). Teacher kept **49/128** mutations. 2h16m wall: ~1h of it waiting for an A10G, and 19 min for the baseline *cache hit* to get a GPU slot |
| [bmf-r3-d2](https://demo.hosted.unionai.cloud/v2/domain/development/project/model-factory/runs/bmf-r3-d2) | **day 2** (2026-10-02), from gen 1 | **promoted (gen 2)**: 50.8% vs champion 47.6%, **CI excludes 0**. 1h23m wall (curation 7m, teacher 17m, training 50m, eval 9m; baseline a 0.4 s cache hit). Crowned ([u3d5ef7533c8b044f](https://demo.hosted.unionai.cloud/v2/domain/development/project/model-factory/runs/u3d5ef7533c8b044f)) |
| [bmf-r3-d3](https://demo.hosted.unionai.cloud/v2/domain/development/project/model-factory/runs/bmf-r3-d3) | **day 3** (2026-10-03), from gen 2 | 50.8% vs champion 50.8%: **champion kept**. 62.5 min wall. Crowned ([uc26497d06efaf519](https://demo.hosted.unionai.cloud/v2/domain/development/project/model-factory/runs/uc26497d06efaf519)) → champion[10-04] = gen 2 |
| [bmf-r3-serve-gen2](https://demo.hosted.unionai.cloud/v2/domain/development/project/model-factory/runs/bmf-r3-serve-gen2) | `mf-inference[2026-10-02] --version mf-promoted-model=124da423…` | upstream `skipped`, endpoint built; `/health` → `weights: full, generation: 2`; `/generate` returned a correct `first_unique` |
| `bmf-r3-read-manifests` | in-cluster read of the three checkpoint manifests | see finding 8 |

Report reads (in-cluster scratch task): `bmf-r3-read-d1`, `bmf-r3-read-d2`, `bmf-r3-read-d3`, `bmf-r3-read-manifests`.

## Results: fixed held-out split, 500 tasks, fingerprint `fdf17e9063c13a57`

| day | champion | base | champion | candidate | Δ vs champion [95% CI] | W/L | Δ vs base [95% CI] | release |
|---|---|---|---|---|---|---|---|---|
| 2026-10-01 | base (gen 0) | 42.0% | 42.0% | **47.6%** | +5.6 [+2.8, +8.6] | 43/15 | +5.6 [+2.8, +8.6] | candidate → gen 1 |
| 2026-10-02 | gen 1 | 42.0% | 47.6% | **50.8%** | **+3.2 [+0.6, +5.8]** | 28/12 | +8.8 [+5.6, +12.0] | candidate → gen 2 |
| 2026-10-03 | gen 2 | 42.0% | 50.8% | 50.8% | 0.0 [−2.2, +2.2] | 16/16 | +8.8 [+5.2, +12.4] | champion kept (gen 2) |

**Compounding, day over day:** base 42.0% → gen 1 **47.6%** (+5.6, CI
[+2.8, +8.6]) → gen 2 **50.8%** (+3.2, CI [+0.6, +5.8]). Both steps clear a
paired 95% CI. The third day matched gen 2 exactly (16 wins, 16 losses), so the
gate held, and the chain stands at +8.8 points over base after three days.

Releases: 10-01 = 384 seed + 49 synthetic (7B teacher, random seeds, since champion
zero has no failures on record). 10-02 = 384 + 32 mutations of gen 1's
failures + 64 replayed frontier tasks. 10-03 = 384 + 36 mutations of gen 2's
failures + 64 replayed. 0 held-out overlap (by text and by id) in all three releases.

The base pass@1 is higher here (42.0%) than round 2's 35.0% because this is a
different, larger split (rows 0–1999 vs 0–599), not a different model.

## Findings

**1. A profile name is in the cache key; its contents are not.** The factory
passes `profile_name="daily"` as a constant, and `curate_heldout`'s task
version did not change (same function body). So after `daily` was redefined
(500 held-out tasks), `mf-heldout-tasks` was a cache hit on round 2's
200-task version. Nothing errored. Day 1 would have been scored on the wrong
split if the 0.1 s hit had not stood out. **Profiles are now immutable names**
(`daily` = round 2, `daily-r3` = round 3), documented in `config.py` and the
README. The general rule: anything a build reads that is not one of its
inputs or its own code is invisible to the factory's cache.

**2. vLLM colocated rollouts work in TRL 1.10 with vLLM 0.31** (outside TRL's
tested range, warning only) and roughly double rollout throughput for the
0.5B student. The step is no longer dominated by generation. Scoring 64
completions in the sandbox and the backward pass now share it.

**3. The served 27B teacher is reachable but not usable at this budget.**
In-cluster DNS works where the public URL does not (it redirects to auth).
At ~1 token/s under shared load, 128 mutation prompts did not finish in 80
min. The served path stays in the code as a profile option
(`teacher_app=...`) for when the app has headroom.

**4. vLLM in a plain debian image needs `VLLM_USE_FLASHINFER_SAMPLER=0`.**
Engine warm-up picks the FlashInfer top-k/top-p sampler, which JIT-compiles a
CUDA kernel and needs `nvcc`. It failed only for the teacher, after a 3.5 min
7B weight load. TRL's rollouts sample differently and never hit it.

**5. A 7B teacher makes the curriculum real.** 49 of 128 mutations passed
their own tests (38%), against 3 of 96 for round 2's 1.5B. Seeded from champion failures on days 2
and 3, it kept 32 and 36 (the seeds are harder, so the yield is lower). With
replay, ~20% of each warm-started day's release targets the champion's
weaknesses.

**6. GPU-queue latency dominates wall clock, not compute.** Day 1's compute
was ~1h20 (curation 8 + teacher 4 + training 53 + eval ~20), while its wall
clock was 2h16. Even the baseline's cache hit waited 19 min for a GPU slot
(round-2 finding 6 again).

**7. The round-2 plateau was a budget problem, not a loop problem.** With
3.2× the prompt visits per day, a fresh LoRA on merged weights each day, and a
curriculum aimed at the champion's failures, day 2 beat day 1's champion by a
margin a 500-task split can resolve. That did not happen once in three tries in
round 2. These changes were made together, so this round does not attribute
the gain to any one of them. An ablation (same budget, continued LoRA, no
curriculum) is the obvious next experiment.

**8. Gains taper fast at this model size.** Day 3 was a dead heat with gen 2
(0.0, CI ±2.2). Its last logged training step had a 70% rollout pass rate (day 1: 23%; a single step, so noisy). The 0.5B student may be near what 80 steps a day
can extract from KodCode-scale problems. As a curiosity, all three days'
`hard_tasks` lists have exactly 304 rows, but the task ids and pass-rate
histograms differ (`bmf-r3-read-manifests`), so the per-day bookkeeping is
real. I have no explanation for the equal counts.

**9. Pinning serves without rebuilding, again.** `bmf-r3-serve-gen2`
skipped every upstream node and deployed generation 2's merged weights to the
T4 app. Full-weight releases need no adapter at serve time.

## Code changes

On `round-2-basic-closed-loop` (same branch as round 2):

- `058a090` round 3: 500-task held-out, fresh LoRA on merged champion, vLLM rollouts,
  7B vLLM teacher: `shared/checkpoints.py` (base/lora/full); trainer merges
  and saves full weights; eval and serving load via the helper;
  `trainer_image` with vLLM; the served-teacher task + `teacher_url` +
  reasoning strip; the batched `_curriculum` core; `probe-r3`.
- `fea2c13` round-3 settings get their own profile name (`daily-r3`) (finding 1).
- `77989e2` vLLM images use the PyTorch sampler (finding 4).
- `be38e4b` replay label fix, README profiles, this log, index.

Tests: 106 pass (round 2: 98). New: checkpoint shapes, teacher URL, reasoning
strip on the real 27B output, profile wiring (served vs local teacher), replay
label idempotence.

## What is still open

- **Attribution.** Four levers changed at once (split size, warm-start
  scheme, steps via vLLM, teacher). The compounding claim is about the loop,
  not about which lever.
- **The plateau after gen 2.** Next: a decaying per-generation LR, more
  steps per day (the scorer is now a large share of the step, so raise
  `_SCORE_WORKERS` or move scoring off the GPU node), or a 1.5B student.
- **Wall clock is ~1h–2h15 per day, mostly GPU-queue time** on the shared
  project. Compute per day is ~1h05–1h20.
- **The served 27B teacher** (`teacher_app=...`) works mechanically but
  was too slow on the shared app today. Untested at production load.
- The `hard_tasks` count coincidence (finding 8).
- Each daily release is now ~1 GB of full weights (round 2: ~35 MB
  adapters). Fine at this scale; worth a retention policy for a real nightly.

## Operational state at the end of the round

- **No unattended training is armed.** Round-3 champions exist for
  2026-10-01..10-04 (base, gen 1, gen 2, gen 2) and round-2 champions for
  10-13..10-16. There is no champion for 2026-10-17 or later, so the 06:00 UTC
  nightly (`daily-r3` profile) will curate that day's seed tasks (~8 min CPU),
  wait for a GPU slot for the baseline cache hit, and block on
  `mf-champion[D] not in the registry`. It will not train. Seeding a champion
  for the current date, or materializing a day dated today, would arm it.
- `crown-champion` is active (`auto_activate=True`). It only fires on a new
  `mf-promoted-model` version.
- `mf-inference` serves **generation 2** (`mf-promoted-model@124da423…`,
  merged full weights) and scales to zero.
- Scratch tasks `bmf-probe.*`, `bmf-reader*`, `bmf-teacher-probe` remain
  deployed (no triggers active).
