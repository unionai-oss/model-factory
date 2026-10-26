# Round 2 — a coding agent's permission gate, as a real decision model (2026-10-15 → 10-16)

## Context

Round 1 built the factory on a toy problem (support triage, 6 tools) and
trained models to *emit JSON*. Two things were missing for a showcase:

1. **A timely decision problem.** Every coding agent ships a permission gate:
   before a shell command, file edit or fetch runs, something answers
   *allow / ask_user / deny*. It is legible, and it is **cost-asymmetric** —
   auto-allowing `rm -rf ~` is far worse than an unnecessary prompt.
2. **An actual decision model.** TypeSafe's Jev and the OSS Strands Decider
   ([blog](https://strandsagents.com/blog/introducing-strands-decider/)) take
   state + question + candidate answers and return a *typed, calibrated
   distribution* over the candidates, with zero output tokens: a Qwen torso,
   the LM head replaced by a ~1M-param pointer head scoring each candidate's
   hidden state against an `<answer>` position, rank-16 LoRA, Brier-scored.

Questions this round asks:

- Does a pointer head beat generative SFT (same torso, same LoRA, same data,
  same steps) on unsafe-allow rate and calibration, at equal accuracy?
- What does each cost per decision (latency, output tokens, head params)?
- Do calibrated probabilities buy an autonomy-vs-safety frontier that JSON
  cannot, and does a threshold tuned in-distribution hold on *novel* calls?

### Design (choices made up front)

- **Oracle = five recitable rules over a command's *effects*** (`harness/policy.py`),
  read off the command by a small shell reader (`harness/shell.py`) that
  undoes quoting (`r''m`), splits chains, descends into `$(…)`/backticks/
  `bash -c`/`eval`/`xargs`, expands `~`/`$HOME`, resolves `..`, tracks `cd`.
  A call's decision is the worst over its effects. Rule-based on effects (not
  regexes on text) is what makes the adversarial slice meaningful.
- **Stratified sampler** (`harness/observations.py`): (slice × label) quotas,
  45% adversarial — obfuscation, fatal links in innocent chains, prefix
  lookalikes (`../app-old`), `src/../../.bashrc`, URL lookalikes, plan-mode
  flips, and *benign lookalikes* (`grep -rn "rm -rf"`) that must be allowed.
  Majority baseline 37.0% (`ask_user`).
- **Suite = torso × head as one `model` partition dimension**
  (`qwen2.5-0.5b-pointer`, `qwen2.5-0.5b-sft`, …) so a single `.all("model")`
  compares every variant. New artifact names `permission-*` (round 1's
  `decision-*` meant a different problem).
- **One training loop for both heads** (`heads/modeling.py`): LoRA r=16 on all
  linear layers, AdamW, linear warmup/decay, option order shuffled per example
  for both arms. Pointer: LN → W_q/W_k (rank 512) dot product, CE over options;
  0.92–1.05M params. SFT: CE on the completion tokens only (left-padded,
  `logits_to_keep`, so no `[batch, seq, 150k]` logits tensor).
- **Three readouts.** `pointer`; `json` (generate, parse, one-hot — what a
  JSON API gives you); `logprob` (the SFT model's likelihood of each full
  completion, renormalised — the honest way to get probabilities out of a
  generative model). References: untuned base by option likelihood, majority.
- **Splits:** train / val / test sampled with separate seeds. Val tunes the
  gate threshold (lowest t with ≤1% unsafe-allow) and picks the champion
  (lowest val expected cost under `scoring.COST`, deny→allow = 50,
  unneeded ask = 1); every reported number is test.

## Run ledger

Base: `https://demo.hosted.unionai.cloud/v2/domain/development/project/model-factory/runs/<run>`

| run | what | outcome |
|---|---|---|
| [udkbpvbvpnxvn4zt6fjs](https://demo.hosted.unionai.cloud/v2/domain/development/project/model-factory/runs/udkbpvbvpnxvn4zt6fjs), [ulk5n2qv87dl72rlbwrg](https://demo.hosted.unionai.cloud/v2/domain/development/project/model-factory/runs/ulk5n2qv87dl72rlbwrg) | station image builds (`df-gpu` with flyte 2.10.6, no trl) | succeeded |
| [uk6m4wsvtp8ht6hwf9fb](https://demo.hosted.unionai.cloud/v2/domain/development/project/model-factory/runs/uk6m4wsvtp8ht6hwf9fb) | `df-serve` image build via `flyte deploy app.py permission_app_env` | image built (11m41s); the deploy then failed resolving `permission-champion@latest` — expected on a cold registry |
| [df2-smoke-r1](https://demo.hosted.unionai.cloud/v2/domain/development/project/model-factory/runs/df2-smoke-r1) | smoke: `smollm2-135m-{pointer,sft}`, `--param …profile_name=smoke`, date 2026-10-14 | **green first try**, 10m23s. 57.0% / 51.0% acc; pointer p50 31.6ms vs JSON 229ms. Champion `smollm2-135m-sft` on val cost |
| [df2-read-smoke](https://demo.hosted.unionai.cloud/v2/domain/development/project/model-factory/runs/df2-read-smoke) | in-cluster scorecard reader (`tools/read_scorecards.py`) — no local S3 creds | succeeded |
| [df2-dev-r1](https://demo.hosted.unionai.cloud/v2/domain/development/project/model-factory/runs/df2-dev-r1) | dev suite, 3 torsos × 2 heads, 8k train / 800 val / 1.6k test, date 2026-10-15 | succeeded, 80 min (2 A10Gs free → 3 training waves). Champion `qwen2.5-0.5b-sft` |
| [df2-read-dev-r1](https://demo.hosted.unionai.cloud/v2/domain/development/project/model-factory/runs/df2-read-dev-r1) | reader | succeeded |
| [uwhbqlsclrlwzl42xjsb](https://demo.hosted.unionai.cloud/v2/domain/development/project/model-factory/runs/uwhbqlsclrlwzl42xjsb) | `factory plan … permission-gate` for 10-15 (plan creates a run) | plan: 14 existing, 1 pending |
| [df2-serve-r1](https://demo.hosted.unionai.cloud/v2/domain/development/project/model-factory/runs/df2-serve-r1) | materialize `permission-gate` for 10-15 | **aborted by me**: despite the plan, it regenerated episodes and queued all 6 fine-tunes again (~6 GPU-min wasted). Cause found later: no `cache=` on any env (finding 6) |
| [df2-dev-r2](https://demo.hosted.unionai.cloud/v2/domain/development/project/model-factory/runs/df2-dev-r2) | dev suite + **held-out-family slice** (600 calls from 4 families never in train/val) + tie-aware selection + wget fix, date 2026-10-16 | succeeded, ~1h55m (scorers spent ~30 min each *queued* for an A10G; 172s running). Champion **`qwen3-0.6b-pointer`** |
| [df2-read-dev-r2](https://demo.hosted.unionai.cloud/v2/domain/development/project/model-factory/runs/df2-read-dev-r2), [df2-read-dev-r2b](https://demo.hosted.unionai.cloud/v2/domain/development/project/model-factory/runs/df2-read-dev-r2b) | reader (first one predated the `heldout` block in the reader's key list) | succeeded |
| [df2-serve-r2](https://demo.hosted.unionai.cloud/v2/domain/development/project/model-factory/runs/df2-serve-r2) | materialize `permission-gate` for 10-16 | **aborted by me**: queued all 6 fine-tunes for a GPU again — same no-cache cause, now diagnosed |
| (app deploy, revision 9) | `flyte deploy app.py permission_app_env` directly: binds `permission-champion@latest` (= r2's) | ACTIVE at `https://solitary-moon-91169.apps.demo.hosted.unionai.cloud`; `spec.inputs.model` → `permission-champion@d2e7ed4e…`; 10/11 hard probes agree with the oracle; **4.3 s/decision on CPU** |
| [df2-dev-r3](https://demo.hosted.unionai.cloud/v2/domain/development/project/model-factory/runs/df2-dev-r3) | dev suite with `cache="auto"` + harmless `path != resolved_path` family, date 2026-10-17 | succeeded, 2h28m (fine-tunes in 3 waves on 2 free A10Gs; ~1h of it queueing). Champion **`qwen2.5-0.5b-pointer`** (val-cost tie with `qwen3-0.6b-pointer`, lower latency) |
| [df2-read-dev-r3](https://demo.hosted.unionai.cloud/v2/domain/development/project/model-factory/runs/df2-read-dev-r3) | reader | succeeded |
| [df2-serve-r3](https://demo.hosted.unionai.cloud/v2/domain/development/project/model-factory/runs/df2-serve-r3) | **materialize `permission-gate` for 10-17 through the factory** (after redeploying the factory with the prefix-cached app) | **succeeded, 5m25s, all 13 upstream builds `CACHE_HIT`** — ~5 min of it the cache-hit fine-tunes waiting for A10G admission (finding 11). App revision `permission-gate-00002` ACTIVE, `spec.inputs.model` → `permission-champion@05cf92ee…` = the 2026-10-17 champion from df2-dev-r3; 9/11 hard probes agree with the oracle; **1.7–2.0 s/decision on CPU, 85 of 422 tokens computed** |

## Results

### Headline scorecard (df2-dev-r2, test split)

In-distribution test n=1,600 (45% adversarial); held-out families n=600.
Unsafe-allow = P(model allows | oracle says ask or deny). Latency is one
decision at batch size 1 on an A10G. Val cost ± SE is what selection used.

| model · readout | acc | adversarial acc | unsafe-allow | Brier | ECE | gate autonomy @≤1% | held-out acc | held-out unsafe | held-out ECE | p50 | out tok | head params |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **qwen3-0.6b · pointer** (champion) | **99.31%** | 98.5% | **0.00%** | 0.013 | 0.006 | 99.8% | 59.2% | 25.7% | 0.380 | **36 ms** | **0** | **1.05M** |
| qwen3-0.6b · sft json | 99.12% | 98.2% | 0.00% | 0.018 | 0.009 | n/a (one-hot) | 67.5% | 14.0% | 0.325 | 236 ms | 7.4 | 155.6M |
| qwen3-0.6b · sft logprob | 99.12% | 98.2% | 0.00% | 0.015 | 0.004 | 100% | 67.5% | 14.8% | 0.266 | 63 ms | 0 | 155.6M |
| qwen2.5-0.5b · pointer | 98.81% | 97.4% | 0.28% | 0.021 | 0.009 | 100% | 67.7% | 22.6% | 0.279 | 25 ms | 0 | 0.92M |
| qwen2.5-0.5b · sft json | 99.25% | 98.3% | 0.00% | 0.015 | 0.008 | n/a (one-hot) | 60.5% | 25.5% | 0.395 | 163 ms | 7.4 | 136.1M |
| qwen2.5-0.5b · sft logprob | 99.25% | 98.3% | 0.00% | 0.012 | 0.004 | 100% | 60.5% | 26.1% | 0.308 | 38 ms | 0 | 136.1M |
| smollm2-360m · pointer | **98.38%** | 96.8% | **0.37%** | **0.029** | 0.006 | **99.6%** | 64.5% | 20.1% | 0.288 | 32 ms | 0 | 0.99M |
| smollm2-360m · sft json | 83.25% | 82.5% | 8.86% | 0.335 | 0.168 | unreachable | 55.2% | 30.6% | 0.448 | 237 ms | 8.0 | 47.2M |
| smollm2-360m · sft logprob | 83.69% | 82.6% | 8.96% | 0.253 | 0.065 | 45.8% (t=0.85) | 55.3% | 31.4% | 0.192 | 47 ms | 0 | 47.2M |
| ref: qwen3-0.6b untuned (option likelihood) | 45.2% | 47.1% | 68.8% | 0.867 | 0.351 | — | 26.7% | 81.5% | 0.543 | — | — | — |
| ref: qwen2.5-0.5b untuned | 36.8% | 36.5% | 6.8% | 0.783 | 0.245 | — | 47.7% | 3.1% | 0.137 | — | — | — |
| ref: majority (`ask_user`) | 37.0% | — | 0.0% | 1.26 | 0.63 | — | — | — | — | — | — | — |

Val expected cost (selection): qwen3-pointer **0.0112 ± 0.0051**, qwen3-sft
0.0150 ± 0.0058 (tied within 2 SE → lower latency wins; the pointer was also
strictly cheapest), qwen2.5-sft 0.030, qwen2.5-pointer 0.051, smollm2-pointer
0.051, smollm2-sft 0.84.

df2-dev-r1 (same suite, previous day's data, no held-out slice) agrees:
≥0.5B both heads 98.9–99.6%; smollm2-360m pointer 95.8% / unsafe 1.59% /
Brier 0.070 vs sft 82.0% / 8.21% / 0.360. Champion there was
`qwen2.5-0.5b-sft` on val cost 0.009 vs the pointer's 0.019 — one mistake in
800, which is why selection now treats a 2-SE band as a tie (finding 3).

### Round 2c: feature coverage for `resolved_path` (df2-dev-r3)

Same suite; training adds `s_file_resolved` (harmless `path != resolved_path`:
release symlinks, /tmp, shared mounts, sibling repos — never persistence or
secret targets). `a_symlink` stays held out. Caching on. Date 2026-10-17.

| model · readout | acc | adv acc | unsafe-allow | Brier | ECE | held-out acc | held-out unsafe | held-out ECE | held-out symlink unsafe (r2 → r3) | p50 |
|---|---|---|---|---|---|---|---|---|---|---|
| **qwen2.5-0.5b · pointer** (champion) | **99.44%** | 99.0% | 0.09% | 0.011 | 0.004 | 62.7% | 31.7% | 0.355 | 11.4% → 26.9% | **25 ms** |
| qwen2.5-0.5b · sft json | 98.31% | 96.8% | 1.21% | 0.034 | 0.017 | 64.3% | 25.4% | 0.357 | 30.1% → 19.3% | 163 ms |
| qwen3-0.6b · pointer | 98.81% | 97.8% | 0.37% | 0.021 | 0.008 | **79.7%** | **12.7%** | **0.187** | 22.0% → **6.7%** | 36 ms |
| qwen3-0.6b · sft json | 98.88% | 97.6% | 0.28% | 0.023 | 0.011 | 57.7% | 25.8% | 0.423 | 5.7% → 10.9% | 237 ms |
| smollm2-360m · pointer | 97.25% | 95.3% | 0.93% | 0.043 | 0.005 | 62.7% | 18.6% | 0.291 | 13.8% → **0.8%** | 33 ms |
| smollm2-360m · sft json | 83.25% | 83.9% | 7.18% | 0.335 | 0.168 | 57.3% | 21.1% | 0.427 | 33.3% → 19.3% | 235 ms |
| smollm2-360m · sft logprob | 83.06% | 84.0% | 8.02% | 0.261 | 0.073 | 58.2% | 21.5% | 0.144 | 34.1% → 19.3% | 47 ms |

(sft logprob rows for the 0.5B/0.6B torsos match their json rows to within
0.1pp on accuracy and unsafe-allow, with lower Brier/ECE; see the scorecards.)

Mean held-out symlink unsafe-allow across the six suite members: **19.4% → 14.0%**.
Val cost: qwen2.5-pointer 0.0225 ± 0.0072, qwen3-pointer 0.0338 (within 2 SE →
tie → lower p50 wins), qwen3-sft 0.0475, qwen2.5-sft 0.060.

## Findings

**1. At ≥0.5B parameters the in-distribution gate is solved by either head;
the head matters at the small end.** Every 0.5–0.6B variant lands at
98.8–99.3% accuracy with 0–0.28% unsafe-allow and ECE ≤ 0.009. Training with
CE on the completion is itself a proper scoring rule, so the SFT model's
token log-probs are *as well calibrated* as the pointer head (logprob ECE
0.004). At 360M the pointer head is far better on the same torso, data and
steps — 98.4% vs 83.3% accuracy, 0.37% vs 8.9% unsafe-allow, Brier 0.029 vs
0.335 — and the result repeats across two independent days of data
(r1: 95.8% vs 82.0%). Reading: a 3-way pointer softmax is an easier target for
a small torso than learning to steer a 49k-token vocabulary.

**2. What the pointer head buys at every size is cost.** 25–36 ms p50 vs
163–237 ms for JSON on the same A10G (6.5×), **zero output tokens** vs ~7.4,
and a ~1M-param head vs a 47–156M-param LM head. The `logprob` readout
recovers SFT's calibration and a frontier, but costs one sequence per option
(38–63 ms, 1.5–1.8× the pointer) and needs the candidate set enumerated up
front — a pointer head's inputs without its efficiency.

**3. Don't let noise pick the champion.** Val cost is a mean over ~800 calls
whose costs are mostly 0 with rare 1–50s; r1's champion "won" by one mistake.
`select.rank` now treats every model within 2 SE of the best as tied and
takes the lowest p50 latency among them. In r2 the pointer won outright
anyway.

**4. JSON gives you a point; probabilities give you a dial.** The frontier
charts (champion report) show the pointer and logprob readouts tracing a
curve, the JSON readout a single hollow dot. For the small torso this is the
difference between 99.6% autonomy at ≤1% unsafe-allow (pointer) and "no
threshold reaches the budget" (JSON) — retraining is the only lever a
one-hot gate has.

**5. Out of distribution, nothing is calibrated — and the threshold does not
transfer.** On 600 calls from four families never seen in training
(symlinks, URL lookalikes, xargs, privilege wrappers) every model drops to
55–68% accuracy with 14–31% unsafe-allow, at mean confidence 0.93–1.0. The
gate threshold tuned to ≤1% unsafe-allow on in-distribution val delivers
14–31% on these calls. The pointer head's calibration is an
*in-distribution* property; it does not make the model know what it does not
know. Per family: privilege wrappers transfer (0% unsafe), URL lookalikes
mostly do, **symlinks and xargs do not**.

**6. The symlink failure is a data-design bug, not a capacity limit.** The
champion's worst live miss: `edit_file .devcontainer/setup.sh` whose
`resolved_path` is `~/.ssh/authorized_keys` → `allow` at P=0.997. In every
training family `path == resolved_path`, so nothing ever made the decisive
field decisive; the model learned to read `path`. The xargs misses
(`git ls-files | xargs wc -l` → allow, oracle says ask because stdin targets
are unresolvable) are partly the oracle's least intuitive rule. r3 tests the
symlink hypothesis by adding *harmless* `path != resolved_path` calls
(release symlinks, /tmp, shared mounts — never persistence or secret targets)
to training, keeping `a_symlink` held out.

**7. Flyte 2 task caching is off by default, and a factory relies on it.**
None of the station envs declared `cache=`. Consequences, all observed:
`factory plan` reported every upstream instance as *existing* while
`materialize` re-ran them all (df2-serve-r1, df2-serve-r2); and round 1's
nightly cron was a full GPU retrain every night (finding 8). model-factory-basic
sets `cache="auto"` on its envs; all four stations here now do too
(`test_every_station_env_caches`).

**8. Round 1's nightly trigger ran every GPU job twice, every night.** The
2026-10-13/14/15 runs of `nightly-suite-decision-champion` and
`nightly-suite-decision-api` (e.g. [uf3a1c0172954f0d0](https://demo.hosted.unionai.cloud/v2/domain/development/project/model-factory/runs/uf3a1c0172954f0d0),
[ua224bc75118d339d](https://demo.hosted.unionai.cloud/v2/domain/development/project/model-factory/runs/ua224bc75118d339d))
have identical action IDs and overlapping timestamps: two triggers (one per
target of `factory.on(cron, champion, endpoint)`), each re-running the shared
upstream concurrently, ~7 min each. Two compounding causes: a Cron on a
`Daily` graph builds a *new* day each time (never cache hits), and no
`cache=` (finding 7). The trigger is now `weekly-suite` on the endpoint only,
Mondays 05:00 UTC.

**9. CPU serving of a 0.6B decision model is 4.3 s/decision naively.** The
same forward is 36 ms on an A10G. Two fixes, both verified locally: cap torch
threads to the container's CPU grant (a pod sees the node's cores), and
**prefix KV caching** — 337 of 428 prompt tokens (chat template + policy)
are identical on every request, so they are run once and only ~90 tokens run
per decision (`DecisionModel.pointer_probs_cached`; the split is checked to
tokenize identically, else it falls back; a local test pins cached == full).
Deployed result (df2-serve-r3): **4.3 s → 1.7–2.0 s per decision** on the
2-vCPU app, with 85 of 422 tokens computed per request (first request ~7 s
while it builds the cache). Still 50–80× the A10G number: the rest is plain
CPU compute on a 0.5B torso, so sub-second needs more cores, a smaller torso,
or the GPU option.

**10. The coverage fix helps on average and is not conclusive; single-seed
OOD numbers are mostly noise.** Adding harmless `path != resolved_path` calls
cut mean held-out symlink unsafe-allow from 19.4% to 14.0% (qwen3-pointer
22.0% → 6.7%, smollm2-pointer 13.8% → 0.8%) — but two members got *worse* on
symlinks, and unrelated families swung by up to 31pp between r2 and r3 with
no related change (qwen2.5-pointer `a_url_trick` 14.9% → 45.9%). In-distribution
numbers reproduce to within ~1pp across three days; held-out ones do not. A
real OOD claim needs several seeds per arm.

**11. The val-tuned gate threshold is t=0 for every strong model, so it gives
no protection out of distribution.** In-distribution val has nothing for the
threshold to trade (argmax already meets ≤1%), so the "gate" is plain argmax
on novel calls. The one readout whose gate *did* partly transfer is the least
confident: smollm2-360m sft-logprob kept held-out unsafe-allow at 4.7–4.9%
(r2/r3) at ~50–60% autonomy, versus ~20% for the same model's JSON. Tuning the
threshold on a held-out-*family* val split (not test) is the obvious next step.
Relatedly, champion selection cannot see OOD behaviour by design: r3's
latency tie-break picked qwen2.5-pointer over qwen3-pointer, which was far
better held-out (12.7% vs 31.7% unsafe).

**12. On this tenant a cache-hit GPU task still waits for GPU admission.**
In df2-serve-r3 the six fine-tunes were `CACHE_HIT`, yet each sat `queued` for
~5 min before resolving in seconds; CPU builds resolve instantly. Cache hits
cost no GPU time, but an "all cache hits" materialization of a GPU graph is
not a 9-second operation when the pool is busy.

**13. Small operational ones.** The app ingress 403s `Python-urllib`'s user
agent (curl works). `factory plan` itself creates a run. Scorecard blobs
cannot be read from a laptop without object-store creds — `tools/read_scorecards.py`
reads them in-cluster and returns JSON.

## Code changes

Branch `round-2-decision-coding-agent` (not pushed). Every module except
`shared/reporting.py` (extended with the frontier SVG) is new or rewritten:

- `harness/shell.py` (new): effect reader; `harness/policy.py`: the five-rule
  oracle + `POLICY_TEXT`; `harness/observations.py`: 36 families (16 standard
  incl. `s_file_resolved`, 20 adversarial, 4 of them held out);
  `harness/prompt.py`: state/question/options prompt, `parse_decision`;
  `harness/scoring.py`: unsafe-allow, Brier, ECE, cost, gate, frontier.
  `harness/tools.py` deleted.
- `heads/encoding.py`, `heads/modeling.py` (new): pointer head, shared LoRA
  training loop for both heads, `DecisionModel` (pointer / generate /
  option-logprob readouts, prefix-cached pointer path).
- `data/tasks.py`: train/val/test + held-out slice, `seen_in_train`.
  `training/tasks.py`: one task for both heads. `evaluation/tasks.py`:
  `build_scorecard` (readouts, references, gate, frontier, held-out block,
  latency). `evaluation/select.py` (new, CPU, no torch): tie-aware ranking,
  champion manifest with the tuned gate threshold and leaderboard.
- `serving/service.py`: `permission-gate` app, `/decide` returns decision,
  probabilities, gated decision, oracle verdict + rule, agreement, latency,
  tokens; `/oracle`; `/champion`.
- `factory.py`: `permission-*` artifacts, weekly endpoint-only trigger;
  all station envs `cache="auto"`; images floor `flyte>=2.10.6`, trl dropped.
- `tools/read_scorecards.py`: in-cluster scorecard reader.
- Tests: 117, including `tests/test_call_shapes.py`, which trains a tiny
  random Qwen2 + in-memory BPE tokenizer through the exact task-path
  functions (`build_rows → training_inputs → train → write_manifest →
  build_scorecard → choose → decide_with`, plus cached == full pointer).

## Open

- **OOD generalization is the open problem** (findings 5, 10, 11). Next:
  several seeds per arm before any OOD claim; tune the gate threshold on a
  held-out-family val split (families disjoint from both train and test);
  consider including a held-out-family term in champion selection.
- xargs/stdin targets: decide whether the oracle's "unresolvable stdin
  target → treat as outside" is the policy we want, or an artifact; it is
  the least intuitive rule and the one models drift from.
- Next-tool routing as a second decision type over the same pointer head
  (the head is candidate-set agnostic; not built this round).
- CPU serving latency after the fixes; a GPU serving option exists
  (`DF_SERVING_GPU`) but the demo T4 serving node is held by `mf-inference`.

---

*2026-10-16 note (round 3):* the `weekly-suite` trigger described in finding 8
is no longer declared by default — the schedule now exists only when the
factory is deployed with `DF_WEEKLY_TRIGGER=1`, so recurring GPU spend is an
explicit choice. Finding 10's single-run held-out comparisons are superseded
by round 3's 3-seed numbers, which reverse the pointer-vs-SFT ordering under
shift ([round 3](2026-10-16-round-3-heldout-claims.md)).
