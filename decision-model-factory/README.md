# decision-model-factory

A **model factory for decision models**, built on [Union
factories](https://www.union.ai/docs/v2/union/user-guide/factories/): generate
a synthetic tool-calling dataset for a small, legible decision problem,
fine-tune a *suite* of open-weights models on it, score them against an exact
oracle, and deploy the winner behind an app — all declared as one graph of
partitioned artifacts.

The factory is the point. The suite is a **partition dimension**, not a loop:

```
flowchart
  decision-episodes   [date]          generate_episodes
        │
        ├──────────────┬──────────────┐      one instance per `model` value
        ▼              ▼              ▼
  decision-model  decision-model  decision-model     [date, model]
   [smollm2-135m]  [qwen2.5-0.5b]      […]           finetune_decision_model
        ▼              ▼              ▼
  decision-scorecard  …              …               [date, model]
        └──────────────┴──────────────┘               score_model
                       ▼  .all("model")
             decision-champion         [date]        select_champion
                       ▼
                 decision-api          (app)         serve
```

Because `model` is a dimension, each fine-tune is **separately cached,
separately retried, and separately visible**; adding a model to the suite adds
partition values rather than editing a loop; and `select_champion` sees the
whole suite at once via `.all("model")`. Re-materializing after one model
changes rebuilds that model and the champion, and reuses everything else.

## The decision problem

A customer-support triage agent with six tools (`lookup_order`,
`check_shipping`, `issue_refund`, `escalate_to_human`, `search_faq`, `reply`).
Given an observation — the customer's message, an intent, and what is known
about the order — the model must emit exactly one tool call as JSON.

What makes it a good factory subject is that **the ground truth is a Python
function** (`harness/policy.py`), stated as seven rules a person could recite.
So:

- Data generation needs no LLM: sample an observation, label it with the
  oracle. Thousands of perfectly-labelled episodes in seconds, for free.
- Evaluation needs no judge: exact match on tool name and arguments.
- The served app can expose `/oracle` next to `/decide`, so you can watch the
  model agree or disagree with the rule it was trained on, live.

The rule that carries the difficulty is the refund one — a conjunction of
three conditions, two of them numeric thresholds (`amount <= 10000 cents`,
`days_since_delivery <= 30`). Episodes are deliberately sampled so ~40% sit
*near* those boundaries on both sides, because a dataset of obvious cases lets
a model score well having learned "big number → escalate". The scorecard
reports near-boundary accuracy separately for exactly this reason.

## Metrics

Three numbers, kept apart on purpose:

| metric | what it measures |
|---|---|
| `json_valid_rate` | did it emit a well-formed call at all — a *formatting* skill fine-tuning fixes almost immediately |
| `tool_accuracy` | did it pick the right tool |
| `exact_accuracy` | right tool **and** right arguments — what the champion is chosen on |

Every scorecard also carries the **majority-class baseline** (~20% on this
mix) and the **untuned base model's** score, so a result can be read as "the
fine-tune taught it the policy" rather than "this model already knew" or "the
split was skewed".

## Quickstart

```bash
uv sync --all-extras
uv run pytest                      # 61 tests: oracle, sampler, parser, graph

CFG=.flyte/config.yaml

# 1. the station tasks
uv run flyte --config $CFG deploy stations.py df_data_env
uv run flyte --config $CFG deploy stations.py trainer_env
uv run flyte --config $CFG deploy stations.py scorer_env
uv run flyte --config $CFG deploy stations.py selector_env

# 2. the app env — this is what BUILDS the app image.
#    On a cold registry it then fails at "Failed to materialize artifact
#    decision-champion@latest". That is expected and harmless: the image is
#    built by that point, and `flyte factory deploy` does NOT build app images.
uv run flyte --config $CFG deploy app.py decision_app_env

# 3. the factory
uv run flyte --config $CFG factory deploy factory.py

# 4. one full suite run for one day
uv run flyte --config $CFG factory materialize decision-factory decision-champion \
    --partition date=2026-10-02 --partition model=smollm2-135m,qwen2.5-0.5b --wait

# 5. deploy the winner
uv run flyte --config $CFG factory materialize decision-factory decision-api \
    --partition date=2026-10-02 --partition model=smollm2-135m,qwen2.5-0.5b --wait
```

Steps 1–3 are one-time (per code change). After that, step 4 is the loop.

### Other things the graph gives you for free

```bash
# what would be rebuilt, without launching anything
uv run flyte --config $CFG factory plan decision-factory decision-champion \
    --partition date=2026-10-02 --partition model=smollm2-135m,qwen2.5-0.5b

# one model's scorecard only (builds just that branch)
uv run flyte --config $CFG factory materialize decision-factory decision-scorecard \
    --partition date=2026-10-02 --partition model=qwen2.5-0.5b

# backfill a week; one action per day, run in parallel
uv run flyte --config $CFG factory materialize decision-factory decision-champion \
    --partition date=2026-09-25..2026-10-02 --partition model=smollm2-135m,qwen2.5-0.5b

# roll the endpoint back to an earlier champion
uv run flyte --config $CFG factory materialize decision-factory decision-api \
    --partition date=2026-10-02 --version decision-champion=<version>
```

## The suite

`config.MODEL_SUITE`. All defaults are **ungated** on the Hub (verified
2026-10-02), because a model needing a licence acceptance makes the factory
fail for anyone without `HUGGINGFACE_TOKEN` — and the point is a loop that
runs unattended.

| slug | HF id | params |
|---|---|---|
| `smollm2-135m` | `HuggingFaceTB/SmolLM2-135M-Instruct` | 135M |
| `smollm2-360m` | `HuggingFaceTB/SmolLM2-360M-Instruct` | 360M |
| `qwen2.5-0.5b` | `Qwen/Qwen2.5-0.5B-Instruct` | 494M |
| `qwen3-0.6b` | `Qwen/Qwen3-0.6B` | 596M |
| `qwen2.5-1.5b` | `Qwen/Qwen2.5-1.5B-Instruct` | 1.5B |

`gemma-3-270m-it` and `Llama-3.2-1B-Instruct` are `gated="manual"`, so they
live in `GATED_MODELS` as opt-in (needs `DF_USE_HF_TOKEN=1` and the secret).

Slugs never contain `/` because a slug **is** a partition value, and an HF
repo id cannot be one.

Profiles (`config.PROFILES`): `smoke` (2 models, 600 train episodes, 1 epoch —
minutes), `dev` (4 models, 4k episodes, 2 epochs), `full` (5 models, 20k
episodes, 3 epochs). The suite is part of the graph's *structure*, so changing
the profile needs a `factory deploy`; per-build constants can be overridden per
run with `--param`.

## Why SFT and not RL

The oracle gives an exact label for every observation, so there is a
supervised target. GRPO — what the sibling `basic-model-factory` uses — earns
its complexity when the reward is only computable by *running* something
(sandboxed unit tests there). Here, policy agreement **is** the label, and SFT
on labels is both cheaper and a tighter fit.

Training is completion-only: the prompt is long and the target
(`{"tool": …, "args": {…}}`) is short, so training on the concatenation would
spend most of the loss on tokens the model is never asked to produce.

## The app

`decision-api` gets its model as an app **parameter** bound to the
`decision-champion` artifact, so it never looks a model up itself — a
materialization deploys it with the exact version it resolved, and Flyte
downloads the Dir before the container starts.

```bash
curl -s $ENDPOINT/health
curl -s $ENDPOINT/champion            # which model won, and what it beat
curl -s -X POST $ENDPOINT/decide -H 'content-type: application/json' -d '{
  "observation": {"customer_message": "refund please", "intent": "refund",
                  "order_id": "A12345", "order_looked_up": true,
                  "order_status": "delivered", "order_total_cents": 4200,
                  "days_since_delivery": 3}}'
```

`/decide` returns the model's call, the oracle's call, and whether they agree.

## Gotchas worth knowing

Learned the hard way; each one cost a failed run. See `research_log/`.

- **An implicitly-injected time partition arrives as a `datetime`, not a
  `str`.** A task parameter named like a `factory.Daily` dimension receives
  the value automatically, typed `datetime`. Annotating it `str` fails the
  build with `Type conversion failed for variable 'date'`.
- **`flyte factory deploy` does not build app images.** Deploy the app env
  first (step 2 above), or the endpoint crash-loops pulling a missing image.
- **`Parameter(mount=...)` at a filesystem-root path crash-loops the app.**
  Use `flyte.app.get_parameter(name)`, which returns the downloaded path.
- **An artifact's partition schema is immutable once its first version is
  published.** Declaring new dimensions on a name that already exists
  unpartitioned is rejected at deploy; publish under a new name.
- **Factory-driven tasks return plain `File`/`Dir`.** The factory declares and
  publishes the artifact itself (name, kind, partitions, parents), so a task
  must not wrap its output in `flyte.artifacts.new()`.

## Layout

```
decision_factory/
  config.py          model suite, profiles, cluster sizing
  contracts.py       artifact names, partition dimensions, payload schemas
  factory.py         THE GRAPH
  harness/
    tools.py         the tool catalog (prompt + oracle + scorer read one copy)
    policy.py        the oracle: observation -> tool call, in seven rules
    observations.py  stratified, near-boundary-aware episode sampling
    scoring.py       parse model output, grade it, aggregate
  data/tasks.py      generate_episodes          -> decision-episodes
  training/tasks.py  finetune_decision_model    -> decision-model
  evaluation/tasks.py score_model               -> decision-scorecard
                     select_champion            -> decision-champion
  serving/service.py decision-api app
factory.py           deploy entrypoint for the graph
stations.py          deploy unit for the task envs
app.py               deploy unit for the app
```
