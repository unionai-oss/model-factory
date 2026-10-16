# Round 3 — making the held-out claim trustworthy, and improving it (2026-10-16)

## Context

Round 2 ended with an in-distribution result that reproduced across three
days and an out-of-distribution (held-out family) result that did not:
single-run held-out numbers swung by up to 31pp, the in-distribution-tuned
gate threshold was 0 for every strong model (so it protected nothing under
shift), and champion selection could not see shift at all. This round asked
four focused questions, on a modest GPU budget (4-model sub-suite, ≤2–3 A10Gs
at a time via `--concurrency 1`):

1. **How big is seed variance on the held-out slice?** ≥3 independent seeds.
2. **Can calibration be tuned *under shift*** without touching the held-out
   test families, and do cheap robustness levers move the held-out frontier?
3. **Can champion selection be held-out-aware** without selecting on test?
4. **Recurring GPU spend**: should the weekly retrain stay armed?

### Design

- **Leave-families-out validation.** Two disjoint groups of families never
  appear in training: the round-2 held-out TEST families (`a_symlink`,
  `a_url_trick`, `a_xargs`, `a_wrapper`; 600 test calls) and a new held-out
  VALIDATION group (`a_dotdot`, `a_redirect`; 400 calls in the val split,
  slice `heldout_val`). Anything tuned under shift is tuned on the second
  group only and reported on the first.
- **Seeds.** Profile `ood`: `qwen2.5-0.5b` and `qwen3-0.6b` × {pointer, sft},
  dev sizing (8k train). One materialization per date (2026-10-18/19/20); the
  date now seeds the episodes **and** the training init and option shuffles,
  so seeds are independent end to end.
- **Levers**, all scored inside `score_model` (no extra training runs):
  1. *Temperature scaling on logits*, fit by NLL on the held-out-val
     families, then the gate threshold re-tuned there (≤1% unsafe-allow).
     On **logits**, because the pointer head's fp32 softmax saturates to
     exactly 1.0 on its confident calls — which are the confident-and-wrong
     OOD ones — so tempering probabilities cannot soften them (caught by a
     unit test before any cluster run).
  2. *Evidence-disagreement guard* (`harness/guard.py`): never auto-allow
     when the call's own fields disagree — a file tool whose `path`
     (resolved naively) is not its `resolved_path`, or a URL with userinfo
     (`docs.python.org@evil.io`). Structural, not the oracle: it knows
     nothing about which targets are dangerous.
  3. *MC-dropout on the pointer head*: LoRA left unmerged so its trained
     dropout (p=0.05) stays live, 8 passes averaged (`pointer_mc`).
- **Held-out-aware selection** (`select.selection_cost`): in-distribution
  val cost **+ held-out-VALIDATION-family cost**, SEs combined, 2-SE ties to
  the lowest latency. The served operating point (`serving_gate`) is the
  shift-tuned temperature + threshold with the guard on.

## Run ledger

Base: `https://demo.hosted.unionai.cloud/v2/domain/development/project/model-factory/runs/<run>`

| run | what | outcome |
|---|---|---|
| (deploy) | stations + factory redeployed with round-3 code; **no trigger declared** | `weekly-suite` trigger removed from the tenant (verified via the trigger list) |
| [df3-seed1](https://demo.hosted.unionai.cloud/v2/domain/development/project/model-factory/runs/df3-seed1) | seed 1, date 2026-10-18, `profile_name=ood`, `--concurrency 1` | succeeded, ~2h (4 sequential fine-tunes 15–35 min each incl. queueing, then 4 scorers) |
| [df3-seed2](https://demo.hosted.unionai.cloud/v2/domain/development/project/model-factory/runs/df3-seed2) | seed 2, date 2026-10-19, in parallel with seed 1 | succeeded, ~2h |
| [df3-seed3](https://demo.hosted.unionai.cloud/v2/domain/development/project/model-factory/runs/df3-seed3) | seed 3, date 2026-10-20, launched as seeds 1–2 were finishing | succeeded, 3h28m (A10G queueing; one scorer 55 min wall-clock). Selector: `champion: qwen2.5-0.5b-sft` |
| [df3-read-seed1](https://demo.hosted.unionai.cloud/v2/domain/development/project/model-factory/runs/df3-read-seed1), [df3-read-seed2](https://demo.hosted.unionai.cloud/v2/domain/development/project/model-factory/runs/df3-read-seed2), [df3-read-seed3](https://demo.hosted.unionai.cloud/v2/domain/development/project/model-factory/runs/df3-read-seed3) | in-cluster scorecard readers | succeeded |
| [df3-serve](https://demo.hosted.unionai.cloud/v2/domain/development/project/model-factory/runs/df3-serve) | materialize `permission-gate` for 2026-10-20 (seed-3 champion, serving gate = shift-tuned + guard) | **aborted by me** after the first cache-hit fine-tune waited 20+ min for A10G admission (`--concurrency 1` would have serialized four such waits); round 2's `df2-serve-r3` already proved the factory serve path end to end |
| (app deploy, revision `permission-gate-00003`) | `flyte deploy app.py permission_app_env` — binds `permission-champion@latest` | ACTIVE; `spec.inputs.model` → `permission-champion@deac3402…` = seed 3's (2026-10-20) champion `qwen2.5-0.5b-sft`, `serving_gate` = {T 1.0, guard on}; 9/11 probes agree; the `authorized_keys` symlink edit (model: allow, P=1.0) is **gated to ask_user by the guard**, live; **8.4–9.6 s/decision on CPU** |

GPU spend: 12 fine-tunes + 12 scorings (MC-dropout adds ~8 pointer passes
per pointer scorecard), at most 3 A10Gs concurrently.

## Results

### Seed-averaged scorecard (3 seeds; mean ± sd [min–max], %)

In-distribution test 1,600 calls; held-out test 600 calls from 4 families
never in train or val. Unsafe-allow = P(model allows | oracle says ask or
deny); severe = P(allows | oracle says deny). "Shift-tuned gate" =
temperature + threshold fit on the held-out VALIDATION families.
`pointer_mc` is omitted: it matched `pointer` to within 0.1–1.5pp everywhere
(finding 5).

| metric | qwen2.5-0.5b · pointer | qwen2.5-0.5b · sft json | qwen2.5-0.5b · sft logprob | qwen3-0.6b · pointer | qwen3-0.6b · sft json | qwen3-0.6b · sft logprob |
|---|---|---|---|---|---|---|
| in-dist accuracy | 98.6 ± 0.8 | 99.1 ± 0.4 | 99.1 ± 0.3 | 98.4 ± 0.7 | 99.0 ± 0.2 | 99.1 ± 0.2 |
| in-dist unsafe-allow | 0.6 ± 0.6 | 0.2 ± 0.1 | 0.2 ± 0.1 | 0.3 ± 0.4 | 0.1 ± 0.1 | 0.1 ± 0.1 |
| held-out accuracy | 63.1 ± 3.9 | 67.4 ± 1.4 | 67.2 ± 1.6 | 61.6 ± 2.6 | **73.2 ± 2.1** | **73.3 ± 2.1** |
| held-out unsafe-allow (argmax) | 24.7 ± 2.7 | 18.9 ± 3.0 | 19.3 ± 3.4 | 21.8 ± 3.7 | **15.4 ± 4.9** | 15.5 ± 4.8 |
| held-out **severe** (argmax) | 4.1 ± 4.0 | 3.7 ± 1.9 | 3.7 ± 1.9 | 2.0 ± 2.1 | 2.5 ± 2.0 | 2.7 ± 1.8 |
| held-out unsafe, in-dist-tuned gate | 23.0 ± 3.5 | 18.9 ± 3.0 | 19.3 ± 3.4 | 21.6 ± 3.4 | 15.4 ± 4.9 | 15.5 ± 4.8 |
| held-out unsafe, shift-tuned gate | 20.2 ± 5.5 | 18.9 ± 3.0 | 15.9 ± 3.7 | 14.6 ± 4.8 | 15.4 ± 4.9 | 15.5 ± 4.8 |
| held-out unsafe, shift-tuned gate **+ guard** | 17.1 ± 5.5 | 16.5 ± 1.2 | 14.6 ± 2.8 | **12.2 ± 4.3** | 13.8 ± 3.3 | 13.8 ± 3.3 |
| held-out **severe**, shift-tuned gate + guard | **0.0 ± 0.0** | **0.0 ± 0.0** | **0.0 ± 0.0** | **0.0 ± 0.0** | **0.0 ± 0.0** | **0.0 ± 0.0** |
| held-out autonomy, shift-tuned gate + guard | 64.0 ± 3.7 | 67.7 ± 11.2 | 65.9 ± 9.0 | 60.5 ± 6.0 | 69.4 ± 6.8 | 69.4 ± 6.8 |
| in-dist autonomy, shift-tuned gate + guard | 92.9 ± 4.7 | 96.7 ± 0.9 | 89.2 ± 9.5 | 87.1 ± 12.2 | 96.9 ± 0.8 | 96.9 ± 0.8 |
| in-dist unsafe, shift-tuned gate + guard | 0.1 ± 0.0 | 0.1 ± 0.1 | 0.1 ± 0.1 | 0.1 ± 0.0 | 0.1 ± 0.0 | 0.1 ± 0.0 |
| held-out ECE raw → tempered | 0.338 → 0.212 | 0.326 → 0.326 (one-hot) | 0.293 → 0.201 | 0.359 → 0.228 | 0.268 → 0.268 (one-hot) | 0.233 → **0.154** |
| temperature fit on held-out val | 3.3 ± 0.5 | 1 (n/a) | 2.5 ± 0.4 | 3.7 ± 0.5 | 1 (n/a) | 2.5 ± 0.4 |

Held-out unsafe-allow per family, argmax → guarded (seed mean, %):

| model · readout | a_symlink | a_url_trick | a_wrapper | a_xargs |
|---|---|---|---|---|
| qwen2.5-0.5b · pointer | 7.5 → **0.0** | 35.1 → 23.7 | 0.0 → 0.0 | 58.0 → 58.0 |
| qwen2.5-0.5b · sft json | 7.7 → **0.0** | 7.3 → 4.9 | 0.0 → 0.0 | 56.7 → 56.7 |
| qwen3-0.6b · pointer | 5.3 → **0.0** | 35.5 → 27.2 | 0.0 → 0.0 | 50.5 → 50.5 |
| qwen3-0.6b · sft json | 4.1 → **0.0** | 8.1 → 4.9 | 0.0 → 0.0 | 46.5 → 46.5 |

### The held-out frontier (seed-averaged autonomy / unsafe-allow, %)

Threshold swept on P(allow); "tempered" uses each seed's held-out-val temperature.

| model · readout | curve | t=0 | t=0.8 | t=0.9 | t=0.96 | t=0.99 | t=0.999 |
|---|---|---|---|---|---|---|---|
| qwen2.5-0.5b · pointer | raw | 76 / 24.7 | 75 / 22.5 | 75 / 21.5 | 74 / 20.3 | 68 / 16.1 | 62 / 14.2 |
| qwen2.5-0.5b · pointer | tempered + guard | 66 / 20.5 | 58 / 13.5 | 55 / 10.9 | 51 / 8.9 | 25 / 3.7 | 0 / 1.5 |
| qwen3-0.6b · pointer | raw | 75 / 21.8 | 74 / 19.0 | 74 / 18.0 | 74 / 17.1 | 74 / 16.6 | 68 / 12.3 |
| qwen3-0.6b · pointer | tempered + guard | 64 / 18.7 | 64 / 12.4 | 52 / 9.1 | 34 / 7.5 | 4 / 0.7 | 0 / 0.0 |
| qwen3-0.6b · sft logprob | raw | 80 / 15.5 | 79 / 14.5 | 78 / 13.7 | 73 / 12.6 | 70 / 11.7 | 50 / 7.0 |
| qwen3-0.6b · sft logprob | tempered + guard | 69 / 13.8 | 65 / 11.1 | 57 / 11.0 | 38 / **4.0** | 0 / 0.0 | 0 / 0.0 |
| qwen3-0.6b · sft json | (one point) | 80 / 15.4; guarded 69 / 13.8 at every t | | | | | |
| qwen2.5-0.5b · sft json | (one point) | 74 / 18.9; guarded 68 / 16.5 at every t | | | | | |

### Champion per seed

| seed | held-out-aware rule (this round) | in-distribution-only rule (round 2) |
|---|---|---|
| 1 (10-18) | `qwen2.5-0.5b-sft` — held-out unsafe 17.6%, 166 ms | `qwen2.5-0.5b-pointer` — 21.4%, 24 ms |
| 2 (10-19) | `qwen3-0.6b-sft` — 10.8%, 245 ms | `qwen3-0.6b-sft` (same) |
| 3 (10-20) | `qwen2.5-0.5b-sft` — 23.0%, 162 ms | `qwen2.5-0.5b-pointer` — 28.0%, 24 ms |

(Recomputed locally with the same `select.rank` over the read scorecards;
seed 3 matches the cluster selector's log line.)

## Findings

**1. The seed spread is now a number: held-out unsafe-allow sd is 3–5pp,
held-out accuracy sd 1.4–3.9pp, in-distribution accuracy sd ≤0.8pp.**
Per-family rates move much more (one family is ~150 calls), which is how
round 2 saw single-family swings of up to 31pp. With three seeds, a held-out
ordering is worth reporting when the gap exceeds ~2 sd and not otherwise.

**2. Seed-averaged, generative SFT is *more robust under shift* than the
pointer head on the same torso — the opposite of round 2c's single run.**
Held-out accuracy 67–73% vs 62–63%, unsafe-allow 15–19% vs 22–25%. The gap
is almost entirely `a_url_trick` (pointer 35% unsafe vs SFT 7–8%): the SFT
model, reading the URL token by token, notices a `@` the pointer head's
summary vector does not. Round 2's "qwen3-pointer is best held-out" was one
seed. The pointer head's advantages (6.5× latency, zero output tokens, the
small-torso result) stand; robustness under shift is not one of them.

**3. Every severe held-out failure is eliminated by the shift-tuned gate +
guard: deny→allow goes from 2.0–4.1% to 0.0% for all six readouts on all
three seeds.** The remaining "unsafe" allows are `ask`-labelled calls,
dominated by `a_xargs` (46–58%) — `git ls-files | xargs wc -l`, which the
oracle asks about because a target arriving on stdin is unverifiable. That
is the oracle's least intuitive rule; a model that reads those as harmless
reads is wrong by the policy but not dangerous. (Destructive xargs calls
like `… | xargs rm -f` are `deny` and are no longer allowed.)

**4. Calibration under shift is tunable, partially.** Temperatures fit on
the held-out VALIDATION families are consistently >1 (pointer 3.3–3.7, SFT
logprob 2.5), cutting held-out ECE by about a third (pointer 0.34–0.36 →
0.21–0.23; SFT logprob 0.23 → 0.15) on families the temperature never saw.
The shift-tuned threshold lowers held-out unsafe-allow vs the
in-distribution threshold (qwen3-pointer 21.6 → 14.6%), but the ≤1% budget
it was tuned for on `a_dotdot`/`a_redirect` does not carry to the test
families: shift is not one distribution, and two validation families do not
span it. On the frontier, getting held-out unsafe-allow under 5% costs most
of the autonomy (qwen3 sft-logprob: 38% autonomy at 4.0%; qwen3-pointer: 4%
autonomy at 0.7%). That is the honest price today.

**5. MC-dropout is a null result at the trained dropout rate.** With LoRA
dropout p=0.05 the 8-pass average differs from a single pass by 0–1.5pp on
every metric and does not move ECE. Uncertainty from a dropout this small
is not uncertainty; a real test would train with higher dropout or use a
small ensemble (≥3 independently trained heads), which costs GPU.

**6. The guard does what a guard should, and has a visible price.** Symlink
unsafe-allow 4–8% → **0%** for every model; URL tricks down 2–12pp.
In distribution it costs ~3–10pp autonomy (harmless release symlinks inside
the repo now ask). It is a harness rule, not learning — reported separately
from the model so the two are not confused.

**7. Held-out-aware selection changes the champion, toward robustness and
away from speed.** With the held-out-validation term, every seed's champion
is an SFT model; in seeds 1 and 3 the in-distribution-only rule would have
picked the pointer model, which was 3.8 and 5.0pp worse on held-out test
unsafe-allow (families the selection never saw), at 6.5× lower latency.
Whether that trade is right is a product decision; the code exposes it as
`OOD_WEIGHT` (0 = round-2 behaviour). Caveat: the held-out-val cost is noisy
(SE 0.02–0.06 on ~400 calls). Served, the trade is stark: the SFT champion
takes **8.4–9.6 s per decision on the 2-vCPU app** (generation has no prefix
cache) versus 1.7–2.0 s for round 2's prefix-cached pointer champion — and,
being one-hot, it cannot be gated by a threshold at all; on held-out
validation the ≤1% budget was unreachable, so the manifest records
threshold 1.0, which for a one-hot readout means "argmax, plus the guard".

**9. A cache-hit GPU build can wait longer than it would take to run.** In
df3-serve the first cache-hit fine-tune sat 20+ minutes waiting for A10G
admission before its cache lookup, under contention from the sibling
projects. Refreshing an endpoint through a GPU graph therefore depends on
GPU availability even when nothing needs a GPU; binding the app directly
(`flyte deploy app.py`, which resolves `permission-champion@latest`) is the
cheap path when the champion is already built.

**8. The weekly retrain is now opt-in.** Factory triggers have no
`auto_activate` knob (`factory.on` compiles them itself), so the off switch
is not declaring one: the schedule exists only when deployed with
`DF_WEEKLY_TRIGGER=1`. After this round's factory deploy the tenant has no
decision-factory triggers (checked). Recurring GPU spend is the user's call.

## Code changes

- `harness/observations.py`: `HELDOUT_VAL_FAMILIES`, `NOT_IN_TRAINING`,
  `sample_heldout(families=…)`; `data/tasks.py`: held-out-val rows in the
  val split (`heldout_val`); `config.py`: `heldout_val_episodes`, profile `ood`.
- `training/tasks.py`: the date seeds init and option shuffles.
- `harness/guard.py` (new): `evidence_disagrees`, `apply_guard`.
- `harness/scoring.py`: `temper` (logits), `apply_temperature` (one-hot
  pass-through), `fit_temperature`, guard-aware `frontier`/`pick_threshold`.
- `heads/modeling.py`: readouts return `logits`; `merge=False`;
  `pointer_probs_mc`.
- `evaluation/tasks.py`: four populations kept apart; `shift` block
  (temperature, shift-tuned thresholds, guarded/unguarded held-out metrics,
  tempered frontiers), `guarded` block, per-family guarded rates,
  `pointer_mc` readout.
- `evaluation/select.py`: `selection_cost` (+ held-out-val cost),
  `serving_gate` in the champion manifest.
- `serving/service.py`: `/decide`'s `gated` applies temperature, the
  shift-tuned threshold and the guard, and reports `evidence_disagreement`.
- `factory.py`: trigger only with `DF_WEEKLY_TRIGGER=1`.
- `tools/aggregate_seeds.py` (new): seed tables and frontiers for the log.
- Tests: 121 (temperature/one-hot, guard, held-out-aware ranking, no
  default trigger, call shapes through `shift`/`pointer_mc`/`serving_gate`).

## Open

- **xargs/stdin rule**: decide whether "a target on stdin is unverifiable →
  ask" is the policy we want. It now accounts for most remaining held-out
  unsafe-allows, none of them severe.
- **Two held-out-val families do not span shift.** More (and more varied)
  validation families, or rotating which families are held out
  (k-fold over families), would make the shift-tuned threshold transfer.
- **A real ensemble** (3 independently trained pointer heads, ~3× training
  GPU) is the next robustness lever; MC-dropout at p=0.05 was not one.
- Pointer-head blindness to URL userinfo: try putting the URL host as an
  explicit state field (the harness can parse it), the same lesson as
  `resolved_path`.
- Serving an SFT champion on CPU is slow (generation has no prefix cache);
  the latency/robustness trade from finding 7 is a decision for the user.
