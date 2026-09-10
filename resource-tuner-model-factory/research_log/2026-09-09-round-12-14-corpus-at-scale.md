# Rounds 12–14: the synthetic corpus at scale

## Context

Round 11 got the first gate pass, and its own reading was that the
remaining gap is a **training-scale** problem. That makes the corpus the
bottleneck: the shipped corpus was ~250k rows built from a few dozen
archetypes, and the question for these rounds was whether we can produce a
**1,000,000-row corpus that is actually diverse, representative and
honestly labelled** — not 1M rows of the same thirty pipelines.

Three sub-questions, one per round:

- **Round 12** — can we *measure* corpus quality, and gate a release on it?
- **Round 13** — does the corpus span the libraries real workloads use, and
  do multiple teacher tiers buy diversity?
- **Round 14** — why did the 1M release ship 31,704 rows, and what has to
  change for the target to be reachable at all?

## Run ledger

| run | what it was | outcome |
|---|---|---|
| [umpfpp92w7mrvm96f4sc](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/umpfpp92w7mrvm96f4sc) | first quality-gated release | **failed the gates — on two gate bugs, not on data** (below) |
| [ufbv2bjp9mcnszd2qxd7](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/ufbv2bjp9mcnszd2qxd7) | `flyte fork` recovery of umpfpp92 with fixed gates | succeeded — traced teacher calls replayed, thousands of measured pods reused |
| [ujwttv7dxsbkt78zrc2c](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/ujwttv7dxsbkt78zrc2c) | first 1M attempt | stuck unschedulable: driver asked 8751m CPU / 25087Mi, more than a t3a.xlarge has |
| [uxc4twfdrncmvkhgj8kk](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/uxc4twfdrncmvkhgj8kk) | 1M attempt, right-sized driver | aborted deliberately to pick up the round-13 stack axis |
| [upnz22h5cdt6gdgrtshv](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/upnz22h5cdt6gdgrtshv) | 1M release, round-13 code | **SUCCEEDED at 31,704 rows of 1,000,000** — the round-14 post-mortem |
| [uqjsqswf4fntdjrq9qzq](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/uqjsqswf4fntdjrq9qzq) | 1M release, round-14 code | launched 2026-09-09 |

## Findings

### 1. A quality gate that measures candidates instead of shipped data is a broken gate

Run `umpfpp92` failed its own gates twice over, and both were bugs in the
gates rather than in the corpus:

- **Label error counted rejected archetypes.** The median label error was
  computed over every archetype the oracle measured, including the ones the
  holdout check had just thrown out for lying. Rejected archetypes
  contribute no rows, so the corpus-level gate was reporting 30% while
  every archetype that would actually ship was at ≤25%.
- **Concentration had an unreachable floor.** The gate asked that no
  archetype hold more than 2% of rows. With N archetypes every one holds at
  least 1/N, so below 50 archetypes a *perfectly even* corpus fails: 33
  kept archetypes failed at 3.0% while being as flat as arithmetic allows.
  The limit is now `max(0.02, 2/n_archetypes)` — twice fair share, never
  below the 2% target.

The recovery is itself the result worth recording: `flyte fork` (from
`flyteplugins-union`) replayed the run with the fixed gates and reused
every already-measured calibration pod. That only works because the teacher
call is wrapped in `@flyte.trace` — traces record inputs and outputs as
literals, so the stochastic step replays instead of re-rolling. Without it
a fork re-samples every archetype, every calibration input changes, and not
one measured pod can be reused.

### 2. Corpus diversity has to be constructed, not hoped for

Round 13 added a **stack axis**: the teacher is *told* which library family
to write against, sampled per archetype, rather than left to reach for
pandas every time. This matters because the libraries differ in exactly the
dimension the policy is learning — Arrow-backed columnar (polars),
out-of-core with disk spill (duckdb, dask), their own histogram structures
(xgboost/lightgbm), a separate allocator (jax), and agent/RAG shapes where
memory lives in documents, embeddings and vector indexes rather than
frames. Four teacher tiers across three model families (qwen35-397b,
minimax-m3, glm-5-2, qwen38-27b) generate concurrently, and every row
records which model wrote it.

### 3. The 1M release shipped 3% of its target, and the machinery reported success

`upnz22h5` asked for 1,000,000 rows and produced **31,704**: 78 surviving
archetypes × 400 variants + 504 measured calibration rows. It **succeeded**,
because `require_target=False` turned shipping short into a warning. Three
mechanisms failed together.

**The wave loop could not stop itself.** Wave 1 ran 19.5 hours against a
15-hour generation deadline. The deadline was only tested *after*
`asyncio.gather` returned, so a wave that overruns cannot be interrupted —
the run generated once, spent the entire budget in a single un-adapted
wave, and never reached wave 2. The adaptive sizing that was supposed to
respond to the observed keep rate never executed at all.

**Retrying a down service is a slow way to fail.** 277 archetypes died on
HTTP 504 *after exhausting all three retries*, and the retry histogram is
the tell: 291 first retries, 290 second, 288 third — essentially every
retry failed too. These endpoints were persistently unavailable, not
hiccuping. Each failure slept 5+10+20s first, so ~290 archetypes burned
**~2.8 hours of pure sleeping** against the concurrency limit for zero
recovered work, while the release kept dutifully routing its fixed share of
archetypes to a dead gateway.

**Broken code cost 7 pods to discover.** The largest single rejection
bucket was `curated out: execution failed` — **462 pods** — code that
passed the AST screen, imported fine, and raised at runtime. This is new
since round 13: admitting a dozen libraries with one line of guidance each
names the library but pins none of the call signatures that decide whether
a module runs.

Full rejection histogram: 462 `execution failed`; 277 teacher HTTP 504;
157 `pod failed`; 34 duration out of bounds; 21 bad JSON; 12 peak out of
bounds; 10 teacher request failed; 10 forbidden import `os`; 4 no
module-level PARAMS; 1 syntax error; 1 PARAMS not literal. 326 of 600
attempts never reached the oracle at all.

### 4. The binding constraint is pod wall clock, and it is mostly not the workload

The arithmetic that actually decides whether 1M is reachable: `upnz22h5`
ran roughly **1,550 oracle pods in 19.5 hours at concurrency 24** — about
**17 minutes of wall clock per pod**, of which only ~1–2 minutes is the
workload itself. The rest is node scale-up and pulling the harness image,
which grew a dozen libraries when the stack axis landed. That cost is not
going away, so the levers are width and pod count, and the round-13
configuration was using a *tenth* of the available fan-out (24 concurrent
pods against a 3–250 node t3a.xlarge pool, one pod per node).

This reframes the target. 1M rows was never blocked by teacher quality —
it was blocked by spending 7 pods on archetypes that one pod could reject,
at a fixed 17-minute unit cost, 24 at a time.

## Code changes

Round 14 (this entry's branch — `round14-throughput`):

- **Deadline enforced inside waves.** Checked in `build_archetype`, not
  only between waves. Wave sizing is bounded by the time actually left, and
  `plan_wave` is extracted as a pure function so the sizing rules are unit
  tested rather than inferred from a 19-hour run. Wave 1 is deliberately
  small (150) so there IS a keep-rate observation to size wave 2 with.
- **Per-teacher circuit breaker** (`TeacherPool`). Round-robins over
  healthy endpoints only; trips after N consecutive *transport* failures
  (content rejections don't count — that's a prompt problem, and dropping
  the endpoint wouldn't fix it); never trips the last one standing;
  half-open probe each wave, since these apps scale from zero and do come
  back. Per-call retries 3/5s → 2/2s.
- **Viability probe.** The cheapest calibration point runs first and stops
  the archetype if the code cannot execute. It costs nothing when it passes
  — that point is a measurement we needed anyway. It fires only on
  execution/pod failure, never on a point merely landing out of bounds:
  that's a property of the parameter point, not the code.
- **Per-stack API contracts** (`STACK_PITFALLS`). A narrow contract per
  library, listing only the calls observed to crash pods — polars
  `group_by`, lightgbm's removed `verbose_eval`, networkx
  `from_numpy_array`, faiss float32 contiguity, langgraph recursion limits,
  lightning's writes-to-cwd defaults, transformers `hidden_size %
  num_heads`. The prompt now also states outright that the module is
  executed at its range *lows* first and discarded if it raises there.
- **Sizing from the measured unit cost**: `oracle_concurrency` 24 → 160,
  `variants_per_archetype` 400 → 1200 (~834 archetypes for 1M — 10× what
  round 13 shipped, and well inside both gates that bound this axis, since
  near-duplication is measured over archetype code and variants don't
  affect it), 150s peak-hold dropped from the rotation, generation deadline
  16h → 20h, task runtime 24h → 36h so a release that generates to its
  deadline still has time to ship.
- 23 tests over the wave planner, the breaker, the probe classifier and the
  stack contracts (`tests/test_wave_throughput.py`).

Earlier in these rounds: quality metrics + enforced gates
(`taskgen/quality.py`), power-law `ParamFit` with holdout label-error
rejection, streaming parquet writes (so 10⁶ rows fit the driver), teacher
provenance through to the corpus rows and report, wave-based target-driven
scale, and retry-with-backoff on teacher gateway errors (commit `105d862`
— superseded in part by the circuit breaker above, which is the right
response to a *persistent* failure).

## Still open

- **The 1M target is not met.** `uqjsqswf4fntdjrq9qzq` is the first attempt
  with the round-14 fixes; the projections above (~14–16h of generation)
  are estimates from one prior run's unit cost, not measurements.
- The successful Qwen3-14B full-FT checkpoint from
  [uk9pj886sxmw9sqnhgrt](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/uk9pj886sxmw9sqnhgrt)
  trained but was never evaluated.
- The L40s:8/32B and L40s:4/14B full-FT rungs still await nodes.
