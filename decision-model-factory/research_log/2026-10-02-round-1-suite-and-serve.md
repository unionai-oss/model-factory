# Round 1 — the suite as a partition, and the winner as an app (2026-10-02)

## Context

New project. The goal was a factory whose shape is "train several models,
compare them, deploy the best", on a decision problem small enough that the
whole loop runs in minutes and legible enough that a reader can check the
labels by hand.

Design choices made up front, and why:

- **The ground truth is a Python function.** A support-triage policy in seven
  rules (`harness/policy.py`), so an episode is a sampled observation plus its
  exactly-correct label. That removes the LLM from data generation (seconds,
  free, perfectly labelled) and the judge from evaluation (exact match).
- **The suite is a partition dimension, not a loop.** `model` is a string
  dimension whose values are suite slugs, so each fine-tune is a separate
  artifact instance — separately cached, retried and visible — and
  `select_champion` reads `.all("model")`.
- **SFT, not RL.** There is a supervised target. GRPO (the sibling
  basic-model-factory) earns its complexity when the reward is only computable
  by *running* something; here policy agreement **is** the label.
- **Ungated models only** in the default suite. Verified against the Hub API:
  SmolLM2-135M/360M-Instruct, Qwen2.5-0.5B/1.5B-Instruct and Qwen3-0.6B are
  `gated=false`; `gemma-3-270m-it` and `Llama-3.2-1B-Instruct` are
  `gated="manual"` and went to `GATED_MODELS` as opt-in. A gated default makes
  the loop fail for anyone without a token.

## Run ledger

Base: `https://demo.hosted.unionai.cloud/v2/domain/development/project/model-factory/runs/<run>`

| run | target | result |
|---|---|---|
| [df-smoke-r1](https://demo.hosted.unionai.cloud/v2/domain/development/project/model-factory/runs/df-smoke-r1) | `decision-champion`, `date=2026-10-02`, `model=smollm2-135m,qwen2.5-0.5b` | **1 episodes + 2 models + 2 scorecards + 1 champion built**, 7m00s total |
| [df-serve-r1](https://demo.hosted.unionai.cloud/v2/domain/development/project/model-factory/runs/df-serve-r1) | `decision-api` | endpoint built, app **never scheduled** — `node(s) had untolerated taint(s)` |
| [df-serve-r2](https://demo.hosted.unionai.cloud/v2/domain/development/project/model-factory/runs/df-serve-r2) | `decision-api`, after moving serving to CPU | 5 reused + endpoint built; app ACTIVE |
| [u6ptgd6n28b44f8dbxzr](https://demo.hosted.unionai.cloud/v2/domain/development/project/model-factory/runs/u6ptgd6n28b44f8dbxzr) | throwaway task to read the scorecards from inside the cluster | succeeded |

`df-smoke-r1` timings: `generate_episodes` **6.7s** (720 labelled episodes, no
model in the data path), the two fine-tunes 3m31s and 4m49s **in parallel**,
the two scorecards 56s and 1m55s in parallel, `select_champion` 5.6s.

## Results

See [the standing-results table](README.md#standing-results) for the full
scorecards. Headline: **qwen2.5-0.5b wins at 82.5% exact** (82.5% tool, 100%
valid JSON) against a 20.0% majority baseline and its own untuned 32.5%;
smollm2-135m reaches 31.7%.

Served endpoint (`https://odd-wave-b09c1.apps.demo.hosted.unionai.cloud`),
exercised live:

```
GET /health   -> {"loaded":true,"champion":"qwen2.5-0.5b",
                  "base_model":"Qwen/Qwen2.5-0.5B-Instruct","exact_accuracy":0.825}

POST /decide  (delivered, 3 days ago, 4200c -> refundable)
  decision: issue_refund {amount_cents: 4200, order_id: A12345}
  oracle:   issue_refund {order_id: A12345, amount_cents: 4200}
  agrees_with_oracle: true

POST /decide  (delivered, 3 days ago, 45000c -> over the 10000c limit)
  decision: issue_refund {amount_cents: 45000, order_id: A99}
  oracle:   escalate_to_human
  agrees_with_oracle: false
```

The second call is the scorecard's top confusion
(`escalate_to_human → issue_refund`) reproducing in production, on demand.
Serving `/oracle` next to `/decide` turned out to be the most useful thing in
the project: the model's actual failure mode is visible without running an
eval.

## Findings

**1. Fan-out and collapse behave exactly as declared.** `--partition
model=a,b` produced two `decision-model` and two `decision-scorecard`
instances, run concurrently, and one `decision-champion`. The `model`
partition value reaches the task through `factory.partition("model")` —
confirmed in the action inputs (`'model': 'qwen2.5-0.5b'`).

**2. `.all("model")` validates against the deployed task's types at deploy.**
`factory deploy` checks that a list-valued mapping lands on a list-typed
parameter. That is the one piece the offline graph tests cannot cover, and it
passed first try.

**3. String partitions and the time partition live in different places on the
spec.** `spec.partitions.value.<key>.staticValue` for `model`,
`spec.timePartition.value.timeValue` for `date`. Any code reading partitions
back has to look in both.

**4. Separating `json_valid_rate` from `exact_accuracy` was worth it.** Without
it, smollm2-135m's result reads as "31.7%, bad model". With it, the real story
is visible: fine-tuning took it from *unable to emit a tool call at all* (0%
valid JSON) to 97.5% well-formed, and the decision barely moved. Those are
different failures needing different fixes.

**5. A GPU serving app is the wrong default here, and the tenant enforced it.**
`df-serve-r1`'s app requested `T4:1` and never scheduled —
`mf-inference` (the sibling factory's app) holds the single T4 serving node.
Serving a sub-1B model emitting ~30 tokens per request does not need an
accelerator, so `serving_resources()` now omits `gpu=` unless
`DF_SERVING_GPU` is set, and the app runs on a CPU-torch image
(`serving_image`) rather than the multi-GB CUDA build it would pull on every
cold start. This is a better design that a scheduling failure forced into
view.

**6. A Dir artifact bound as an app parameter materializes into the app's
working directory root.** `get_parameter("model")` returned `/home/flyte`,
with `manifest.json` directly inside it. Works, but `os.path.isdir` is the
right check, not "is this a subdirectory".

## Code changes

Whole project is new. Structure and rationale are in
[README.md](../README.md); the parts worth calling out:

- `harness/policy.py` — the oracle, seven rules, with the refund conjunction
  as the hard case. `system_prompt()` states both thresholds, because a policy
  is only learnable if the numbers it turns on are visible to the model.
- `harness/observations.py` — stratified by oracle tool (uniform sampling
  buries `issue_refund` and floods `lookup_order`), with 40% of episodes drawn
  *near* the numeric boundaries on both sides. A dataset of obvious cases lets
  a model score well having learned "big number → escalate".
- `harness/scoring.py` — balanced-brace JSON extraction (a "first `{` to first
  `}`" slice truncates every real call, since `args` is itself an object),
  strict about the content: a tool outside the catalog is a wrong decision, not
  a parse to salvage.
- `evaluation/tasks.py` — `select_champion` matches candidate Dirs to
  scorecards by the slug in each manifest, not by list position, so a
  reordering cannot promote the wrong model.
- Train and eval episodes are sampled with **different seeds** rather than
  sliced from one stratified pool; slicing correlates the splits (the same
  near-boundary amounts recur) and inflates eval accuracy.

Tests: 61, all offline (43 domain + 18 graph/config).

## Next

- `qwen3-0.6b` and `smollm2-360m` are in the `dev` profile but unrun; adding
  them is two more partition values.
- The champion's refund over-approval is the obvious target: more epochs, or
  episodes weighted toward the conjunction. Near-boundary accuracy being
  *higher* than overall says proximity to the threshold is not the whole story.
- `min_exact_accuracy` is 0.0 in `smoke`, so nothing has yet exercised the
  "no model cleared the floor, deploy nothing" path.
- The nightly `flyte.Cron` trigger is declared but has not fired.
