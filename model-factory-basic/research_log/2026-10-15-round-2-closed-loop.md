# Round 2: closing the loop (2026-10-15/16)

## Context

Round 1 ported the loop to a Union factory and got it green, but it made no
quality claim (smoke's promotion margin was -1), and it was not actually
closed-loop: every day trained from the base model, and nothing a day learned
reached the next. This round asked three questions:

1. Can a factory express "day D builds on day D-1" declaratively, and what is
   the cleanest way to do it with the API as it ships?
2. Does the fine-tuned checkpoint beat the base model on a held-out split
   that stays fixed across days, under an honest gate?
3. Does the loop compound, with day 2 beating day 1's champion?

Short answers: (1) yes, through a partitioned `factory.source` that an
artifact trigger writes, because a same-artifact cross-day edge is a cycle;
(2) yes, +7.5 pass@1 points over base with a paired 95% CI that excludes 0;
(3) **no**. Three warm-started days (with a broken curriculum, a
repaired one, and frontier replay) all landed within noise of the day-1
champion (−0.5, −1.5, −1.5 pts), so the gate kept generation 1 each time. The
loop's machinery is shown working end to end; compounding is not.

## What the factory API supports (flyteplugins-union 0.15.1 / 0.17.0)

Read from `flyteplugins/union/factory/` before designing anything:

- **`ArtifactHandle.window(date=TimeRange(days=n))`**: a trailing window
  that *ends at the consumer's own partition* (`window_values` is inclusive
  of `end`). Neither `days=0` nor a negative value gives an offset. There is
  no "previous partition" mapping.
- **`.select(**pins)`**: fixes dimensions to constants. **`.all(dim)`**:
  every observed value. Neither expresses "D-1".
- **Cycles**: `Graph.topological_nodes` walks *artifact names*, so
  `promoted-model → … → policy-checkpoint → promoted-model` is rejected
  whatever the partition mapping.
- **Sources**: `validate_structure` rejects a name that is both a
  `factory.source` and built by the factory, so "read yesterday's promoted
  model as a source" is also rejected.
- **Missing source partitions**: the driver probes the registry and marks a
  missing source instance `missing`. Its consumers are `blocked`; nodes that
  do not depend on it still build.
- **0.17.0 is byte-identical to 0.15.1** in `factory/` (diffed the wheel),
  and it caps `flyte<2.11`, so flyte 2.11.x is not usable with factories yet.
  Nothing newer helps here.

So the state that crosses days has to be an artifact the graph only reads.
Design, in `model_factory/factory.py` and `evaluation/tasks.py`:

```
champion = factory.source("mf-champion", partitions={"date": factory.Daily}, type=Dir, kind="model")
synthetic = build(...).using(generate_synthetic_tasks, ..., champion=champion)   # curriculum
checkpoint = build(...).using(train_grpo, dataset=dataset, champion=champion)    # warm start
report = build(...).using(evaluate_checkpoint, ..., champion=champion, baseline=baseline)
promoted = build(...).using(promote_checkpoint, ..., champion=champion)          # champion/challenger

crown_champion: OnArtifact("mf-promoted-model") -> publishes champion[D+1] = promoted[D]
seed_champion:  run once                         -> publishes champion[D0] = base model
```

Alternatives considered and rejected:

- **Unpartitioned `mf-champion` ("the current champion")**: simpler, but
  then a day is not a pure function of its inputs. Re-materializing day D
  after its own release was crowned would read the *new* champion, miss the
  cache, and retrain day D from its own output. A backfill would read one
  champion for every day.
- **`factory.on(champion, promoted)`** (trigger the next day off the
  crown): a runaway chain. Each crown would materialize the next day
  immediately, forever into future dates, at about 1 A10G-hour each.
- **`.window(days=2)` on a source**: includes day D itself, which does not
  exist until day D is released, so day D is always blocked.

Probe factory first (throwaway `bmf-probe`, CPU-only, ~1 min per run) to check
each mechanism against the real control plane before touching the real
factory. See the probe runs in the ledger.

## Other design changes

- **A fixed held-out split.** `mf-heldout-tasks` and `mf-baseline-report`
  are builds with **no partitions**, so every day reads the same version.
  The held-out split is rows 0–599 of KodCode-Light-RL-10K (200 curated
  tasks). Each day's training data is a window over rows 600–9999 that
  rotates by date. Isolation is checked three times: `ingest_and_curate`
  excludes held-out questions, `generate_synthetic_tasks` drops mutations that
  reproduce one, and `assemble_dataset` refuses to publish a release with any
  overlap. `train_grpo` never receives the held-out artifact. Every eval
  report carries `heldout_fingerprint` (a hash of the sorted task ids) and
  refuses to run if the baseline's fingerprint differs.
- **Failure-driven curriculum.** `train_grpo` counts per-task rollout
  passes and writes `hard_tasks.parquet` (train tasks with rollout pass rate
  ≤ 0.5) into the checkpoint. Once that checkpoint is crowned, the next day's
  `generate_synthetic_tasks` mutates those first (teacher:
  Qwen2.5-Coder-1.5B-Instruct) and tops up with random seeds.
- **Champion/challenger release.** `promote_checkpoint` returns the
  candidate only if it beats the champion by ≥ 1 point *and* the human gate
  approves. Otherwise it returns the champion. A losing day therefore still
  publishes a release, and the crown always fires, so the chain never breaks.
- **Paired comparisons.** The eval reports pass@1 for candidate, champion
  and base on the same 200 tasks, with paired-bootstrap 95% CIs and
  win/loss counts.
- **Profiles.** `smoke` (machinery, margin -1) and `daily` (0.5B student,
  200 held-out, 256 fresh tasks/day, 50 GRPO steps × 4 prompts × 8 rollouts,
  margin +0.01). `smoke-plus`/`dev`/`full` are gone; `dev`/`full` asked for
  vLLM, which the image never had.
- **Dead code from the trigger era:** `shared/inference_client.py` and its
  tests, the eval-through-the-serving-app path (`use_service`), the app's
  `/reload` and registry fallback, stale `depends_on` coordinator envs.
- **Images pin `flyte>=2.10.6,<2.11`.** They pinned `>=2.6.0`, which
  resolves to whatever is newest on PyPI and does not track the deploying
  SDK.

## Run ledger

Base: `https://demo.hosted.unionai.cloud/v2/domain/development/project/model-factory/runs/<run>`

Probe (throwaway `bmf-probe` factory, CPU-only):

| run | what | outcome |
|---|---|---|
| [bmfprobe-seed-r2](https://demo.hosted.unionai.cloud/v2/domain/development/project/model-factory/runs/bmfprobe-seed-r2) | task publishing `bmfprobe-champ[2026-09-01]` via `flyte.artifacts.new` | **succeeded but published nothing**: the task lacked `produces_artifacts=True` |
| [bmfprobe-d1-r2](https://demo.hosted.unionai.cloud/v2/domain/development/project/model-factory/runs/bmfprobe-d1-r2) | materialize day 1 | **FAILED** as expected: `1 source partition(s) not in the registry. Missing: bmfprobe-champ[2026-09-01]` |
| [bmfprobe-seed-r2b](https://demo.hosted.unionai.cloud/v2/domain/development/project/model-factory/runs/bmfprobe-seed-r2b) | same, with `produces_artifacts=True` | champ[09-01] published |
| [bmfprobe-d1-r2b](https://demo.hosted.unionai.cloud/v2/domain/development/project/model-factory/runs/bmfprobe-d1-r2b) | materialize day 1 | 1 built; unpartitioned upstream reused; crown fired in **<1 s** ([u8aa079fa533b2267](https://demo.hosted.unionai.cloud/v2/domain/development/project/model-factory/runs/u8aa079fa533b2267)) → champ[09-02] |
| [bmfprobe-d12-r2](https://demo.hosted.unionai.cloud/v2/domain/development/project/model-factory/runs/bmfprobe-d12-r2) | materialize 09-01..09-02 | 1 reused, 1 built from champ[09-02]; **only one** new crown ([u3c873d8e677e127d](https://demo.hosted.unionai.cloud/v2/domain/development/project/model-factory/runs/u3c873d8e677e127d)): a cache hit publishes no new version |

Smoke profile (`MF_PROFILE=smoke` factory):

| run | what | outcome |
|---|---|---|
| [bmf-r2-seed-smoke](https://demo.hosted.unionai.cloud/v2/domain/development/project/model-factory/runs/bmf-r2-seed-smoke) | `seed_champion --date 2026-09-10` | champion[09-10] = base |
| [bmf-r2-smoke-d1](https://demo.hosted.unionai.cloud/v2/domain/development/project/model-factory/runs/bmf-r2-smoke-d1) | `mf-promoted-model[2026-09-10]` | **green first try**, 8 built, 13 min; crowned ([u2f6b770ea22e92bb](https://demo.hosted.unionai.cloud/v2/domain/development/project/model-factory/runs/u2f6b770ea22e92bb)) |
| [bmf-r2-smoke-d2](https://demo.hosted.unionai.cloud/v2/domain/development/project/model-factory/runs/bmf-r2-smoke-d2) | day 2 (warm start) | **ABORTED** by me after `evaluate_baseline`, a cache hit, sat `queued` for 70 min (see finding 6) |
| [bmf-r2-smoke-d2b](https://demo.hosted.unionai.cloud/v2/domain/development/project/model-factory/runs/bmf-r2-smoke-d2b) | day 2 again | green; warm start and the champion-adapter eval path exercised; crowned ([ue3eeaff8fd2aded0](https://demo.hosted.unionai.cloud/v2/domain/development/project/model-factory/runs/ue3eeaff8fd2aded0)) |

Daily profile (the real runs):

| run | what | outcome |
|---|---|---|
| [bmf-r2-seed-daily](https://demo.hosted.unionai.cloud/v2/domain/development/project/model-factory/runs/bmf-r2-seed-daily) | `seed_champion --date 2026-10-13` | champion[10-13] = base (gen 0) |
| [bmf-r2-plan-d1](https://demo.hosted.unionai.cloud/v2/domain/development/project/model-factory/runs/bmf-r2-plan-d1) | `--plan` for 10-13 | reports partitions as "existing" (round 1's nightly cron had built smoke versions); see finding 8 |
| [bmf-r2-daily-d1](https://demo.hosted.unionai.cloud/v2/domain/development/project/model-factory/runs/bmf-r2-daily-d1) | **day 1**, cold start | 97 min wall (≈50 min of it waiting for a GPU). Candidate 42.5% vs base 35.0%: **promoted**, crowned ([uf275db606406e29](https://demo.hosted.unionai.cloud/v2/domain/development/project/model-factory/runs/uf275db606406e29)). **Bug: 0 of 48 synthetic tasks kept** |
| [bmf-r2-daily-d2](https://demo.hosted.unionai.cloud/v2/domain/development/project/model-factory/runs/bmf-r2-daily-d2) | **day 2**, warm start from gen 1 | 55.6 min wall. Candidate 42.0% vs champion 42.5%: **champion kept**. Still 0 synthetic (ran before the fix) |
| [bmf-r2-daily-d3](https://demo.hosted.unionai.cloud/v2/domain/development/project/model-factory/runs/bmf-r2-daily-d3) | **day 3**, gen-1 champion, repaired curriculum ([2a31a5c](#code-changes)) | 61 min wall. Candidate 41.0% vs champion 42.5%: **champion kept**. Curriculum now non-empty: 3 mutations kept, all from champion failures. Crowned ([u465104b9307baac2](https://demo.hosted.unionai.cloud/v2/domain/development/project/model-factory/runs/u465104b9307baac2)) → champion[10-16] |
| [bmf-r2-serve-d3](https://demo.hosted.unionai.cloud/v2/domain/development/project/model-factory/runs/bmf-r2-serve-d3) | `mf-inference[2026-10-15]` | **ABORTED** by me: after the `de-gpu` redeploy it was rebuilding day 3 (finding 11) |
| [bmf-r2-serve-d3b](https://demo.hosted.unionai.cloud/v2/domain/development/project/model-factory/runs/bmf-r2-serve-d3b) | `mf-inference[2026-10-15] --version mf-promoted-model=f6611f86…` | everything upstream `skipped`, endpoint built. App `spec.inputs` pins `mf-promoted-model@f6611f868f60…`; `/health` → `loaded, adapter: true, generation: 1`; `/generate` returned a correct `is_anagram` |
| [bmf-r2-daily-d4](https://demo.hosted.unionai.cloud/v2/domain/development/project/model-factory/runs/bmf-r2-daily-d4) | **day 4**, gen-1 champion, + frontier replay (`frontier_replay`) | Candidate 41.0% vs champion 42.5%: **champion kept**. 48 replayed frontier tasks, 0 mutations kept. Crown trigger paused by me for this run (see "Operational state") |
| [u41d15dc4e7494428](https://demo.hosted.unionai.cloud/v2/domain/development/project/model-factory/runs/u41d15dc4e7494428), [ubb4d3acd7e0d6f9b](https://demo.hosted.unionai.cloud/v2/domain/development/project/model-factory/runs/ubb4d3acd7e0d6f9b) | **nightly cron**, 2026-10-16 06:00 UTC (promoted + inference targets) | both **FAILED as designed**: `1 built, 2 reused, 5 blocked … Missing: mf-champion[2026-10-16]` (68 min, mostly waiting for a GPU slot to resolve a cache hit) and `0 built, 3 reused, 6 blocked` (25 min) |

Report-reading helper runs (in-cluster, scratch task): `bmf-r2-read-d1`,
`bmf-r2-read-d1b`, `bmf-r2-read-d2`, `bmf-r2-read-d3`, `bmf-r2-read-d4`.

Day-1 station timings (wall clock includes GPU queueing): `curate_heldout`
4m, `ingest_and_curate` 5m, `generate_synthetic_tasks` 34m (about 1.5 min of
generation; the rest was waiting for an A10G), `train_grpo` 46m (26.9 min of
training at 32 s/step), `evaluate_baseline` 2.6m, `evaluate_checkpoint` 6m.
Day 2 (warm start): 55.6 min end to end. The baseline was a cache hit (but
4 min in the GPU queue), and evaluation took 14 min because it also generated
with the champion's adapter.

## Results (fixed held-out split: 200 tasks, fingerprint `9911df20d92736a4`)

| day | champion (gen) | base | champion | candidate | Δ vs champion [95% CI] | W/L vs champion | Δ vs base [95% CI] | release |
|---|---|---|---|---|---|---|---|---|
| 2026-10-13 | base (0) | 35.0% | 35.0% | **42.5%** | +7.5 [+3.0, +12.0] | 19/4 | +7.5 [+3.0, +12.0] | candidate → gen 1 |
| 2026-10-14 | gen 1 | 35.0% | 42.5% | 42.0% | −0.5 [−5.0, +4.0] | 10/11 | +7.0 [+2.0, +12.0] | champion kept (gen 1) |
| 2026-10-15 | gen 1 | 35.0% | 42.5% | 41.0% | −1.5 [−6.5, +3.0] | 11/14 | +6.0 [+1.0, +11.5] | champion kept (gen 1) |
| 2026-10-16 | gen 1 | 35.0% | 42.5% | 41.0% | −1.5 [−6.5, +3.5] | 13/16 | +6.0 [+1.0, +11.0] | champion kept (gen 1) |

Release composition: 10-13 and 10-14 were 256 seed + 0 synthetic (the parser
bug). 10-15 was 256 + 3 mutations of champion failures. 10-16 was 256 + 48
replayed frontier tasks + 0 mutations.

The held-out fingerprint is identical in every report, and the baseline
report is the same artifact version (`e60c766b…`) across all three days.
Leak check from in-cluster reads of the released datasets: 0 held-out
questions and 0 held-out ids in `mf-rl-tasks-dataset` for all four days.

## Findings

**1. A factory cannot express a same-artifact cross-day dependency.** The
cycle check runs over artifact names, and `.window` always includes the
consumer's own partition. The supported shape is a partitioned
`factory.source` that something outside the graph publishes, and an
`OnArtifact` trigger on the release is the natural publisher. It reads well:
the graph is "one day", and the one arrow from today into tomorrow is an
artifact event.

**2. OnArtifact triggers fire on factory-published versions, within a
second, and pass the partition as a typed value.**
`flyte.TriggeredPartition("date")` arrives as a `datetime`. A cache-hit
materialization publishes no new version and fires nothing, so re-running a
day does not re-crown it. That is what keeps `champion[D+1]` stable and day
D+1 cacheable.

**3. A task that publishes its own artifact needs
`@env.task(produces_artifacts=True)`.** Without it, `flyte.artifacts.new(...)`
returns normally and the run succeeds, but nothing is published. Cost one
probe run.

**4. Unpartitioned builds upstream of partitioned ones work, and they are
the right home for a yardstick.** `mf-heldout-tasks` / `mf-baseline-report`
were built once and reused by every daily materialization (0.1 s cache hits
on CPU). Comparability across days follows from the graph's shape.

**5. Day 1 moves pass@1 by a real margin; day 2 does not.** GRPO + LoRA on
Qwen2.5-Coder-0.5B, 50 steps, takes held-out pass@1 from 35.0% to 42.5% (19
tasks gained, 4 lost; the paired CI excludes zero). Warm-starting a second day
on 256 fresh tasks ends where it started (42.0%, 10 gained, 11 lost). The
candidate's format compliance is already 100% after day 1, so the cheap part
of the gain (answer in one fenced block, correct signature) has been
harvested, and 200 more prompt visits at a decaying LR do not find more. Days 3 and 4 (curriculum repaired, then frontier replay added) land at 41.0% both times. The
champion/challenger gate did what it is for: day 2's release is still gen 1.

**6. On this tenant, a cache hit on a GPU station still waits for a GPU
admission slot.** `evaluate_baseline` in `bmf-r2-smoke-d2` sat `queued` for
70+ min with no pod while the sibling decision-factory's six A10G jobs ran in
the same project. In `bmf-r2-smoke-d2b` it then completed in 0.7 s with no
logs, and `train_grpo` was a 389 s "queued" cache hit with no running time.
The round-1 "9.1 s nightly" holds only when the GPU queue is empty. Reuse is
free in compute, not in latency, on a shared project.

**7. A missing champion blocks per dependency, not per run.** The 06:00 UTC
nightly for 2026-10-16 (no `champion[10-16]` then) still curated that day's
seed tasks (5 min CPU) and queued for the baseline's GPU cache hit before
blocking everything downstream of the champion. Docs that said "stops before
any GPU work" were corrected. The seed tasks it built were not wasted: day 4
(`bmf-r2-daily-d4`) reused `mf-seed-tasks[10-16]` from that failed nightly as
a 0.1 s cache hit.

**8. `--plan` reports per partition, not per version.** For 2026-10-13 it
listed every node as "existing", because round 1's nightly cron had built
smoke-profile versions of those partitions, and any version of an
unpartitioned artifact counts. The materialization then rebuilt everything,
because reuse is decided by the cache key (task version + inputs +
constants). Read `--plan` as "what partitions exist", not "what will be
reused".

**9. `factory deploy` pins each station's task version.** Redeploying
`de-gpu` with the parser fix did nothing until the factory was redeployed too.
Redeploy a single env (`deploy team_data.py de_gpu_env`), not the whole unit,
so the other stations' versions, and with them the cache keys of
`mf-heldout-tasks` and `mf-baseline-report`, stay put.

**10. The synthetic stage was silently producing nothing.** Day 1 kept 0
of 48 mutations. Running the teacher locally showed why: it writes
`### QUESTION:` headers (the strict regex wanted `QUESTION:\n`), writes tests
that call the function without importing it (`NameError` under pytest), and
writes tests as bare module-level `assert`s with no `test_` function. The
parser is now section-based and tolerant; tests get `from solution import *`
when they forget it, and bare asserts get wrapped in one test function. The
oracle still decides. Local re-check: 6/8 parse (was 0/3), 1/8 passes the
oracle before the assert wrapping. Each seed now gets two draws. On the cluster the repaired stage kept 3 of
96 draws on day 3 and 0 of 96 on day 4: non-zero, but a 1.5B teacher is too
weak a problem writer to matter at this scale.

**11. A station redeploy changes the cache key of every past day.**
`mf-inference[2026-10-15]` after the `de-gpu` redeploy started rebuilding day 3
(new synthetic code version, so every downstream version changed). To serve
or roll back a past release after code changes, pin it:
`--version mf-promoted-model=<version>` marks the release `pinned` and every
upstream node `skipped`, so only the endpoint is built.

**12. The served artifact path is the app's working directory.**
`flyte.app.get_parameter("model")` for a bound `Dir` returned the cwd
(`/home/flyte`), with `manifest.json` and the adapter in it. The app treats
whatever path it gets as the model directory, which works.

**13. Pausing an OnArtifact trigger drops events; it does not queue them.**
The crown was deactivated for day 4 and re-activated after it, and no crown ran
for `promoted-model[10-16]`. Anyone pausing the crown to stop the nightly
loop should know this: re-activating does not catch up, so the next day blocks
until someone runs `seed_champion` or a new release appears.

## Code changes

On branch `round-2-basic-closed-loop`:

- `8281656`: the closed loop. `mf-champion` source + `crown_champion`
  trigger + `seed_champion`; `mf-heldout-tasks` / `mf-baseline-report`
  built once; the rotating daily window; leak checks; warm start; hard-task
  bookkeeping; candidate-vs-champion eval with paired CIs; champion/challenger
  release; `smoke`/`daily` profiles; trigger-era dead code removed; image
  flyte pin; 91 tests.
- `2a31a5c`: tolerant synthetic parser, assert wrapping, two draws per seed
  (finding 10).
- `875888b`: replay the champion's learnable failures (day 4's hypothesis; not supported).
- (final commit) research log, README walkthrough, docs corrected per finding 7.

Tests: 98 pass (round 1: 68). New: graph-shape tests for the loop wiring
(champion is the only source, it feeds synthetic/train/eval/promote by
identity, nothing reads the release back, the yardstick has no partitions,
training never receives the held-out split), held-out isolation and window
tests, curriculum and decision logic, and parser regressions taken from real
teacher output.

## What is still open

- **Compounding is not demonstrated.** Day 1 harvests ~7 points, and further
  days at the same budget stay within ±1.5 points of that, inside a ±5-point
  CI. What I would try next, in order: (1) **more GRPO steps per day**; generation
  is plain HF `generate` at 32 s/step, so vLLM colocated rollouts (3–5× faster)
  would buy 3–5× the steps inside the same hour. (2) **A lower learning rate
  for warm starts**. Every warm start re-ran the 3e-5 → 0 schedule, and all
  three candidates landed slightly below the champion. (3) **A larger held-out
  split** (500+), so a real +2-point day can clear a CI. (4) **A stronger
  teacher** (7B) for the mutations. (5) Seed the 1.5B student, where the base
  has more headroom per step.
- The nightly trigger is armed. Once a champion exists for "today", every
  night trains a real day (~1 A10G-hour). See TODO.md.
- Leak checks are exact-text; KodCode contains near-duplicate problem
  variants, so a near-duplicate check (MinHash) would make the isolation
  claim stronger.
- `--plan` cannot say which builds will be cache hits (finding 8).

## Operational state at the end of the round

- Champions: `mf-champion[10-13]` = base, `[10-14]`, `[10-15]` and `[10-16]` =
  generation 1 (the day-1 adapter). There is no `[10-17]`, so tomorrow's 06:00 UTC
  nightly will curate seed tasks for 10-17 and block. **No unattended training
  is armed.** Running `seed_champion` or materializing a day dated today
  would arm it.
- `crown-champion` is active, as the code declares (`auto_activate=True`).
- `mf-inference` serves `mf-promoted-model@f6611f86…` (generation 1) and
  scales to zero.
- The throwaway probe factory `bmf-probe` and its `bmfprobe-*` artifacts
  remain; its crown trigger is deactivated.
