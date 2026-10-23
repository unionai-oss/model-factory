"""Artifact names, partition dimensions, and payload schemas.

The factory (`decision_factory/factory.py`) is declared against these names.
Nothing here publishes anything: the factory's driver calls each build's task
with `produces_artifacts=True` inside a `flyte.artifacts.produces(...)` block
and declares the artifact itself, so a station task returns a plain
`File`/`Dir` and never wraps it in `flyte.artifacts.new()`.

The graph, and what each partition dimension means:

    decision-episodes      [date]           one labelled dataset per day
    decision-model         [date, model]    one fine-tune per suite member
    decision-scorecard     [date, model]    one scorecard per fine-tune
    decision-champion      [date]           the winner, collapsing `model`
    decision-api           (endpoint)       serves the champion

`model` is the interesting one. It is a *string* dimension whose values are
the suite's slugs, so the two middle builds fan out to one instance per
model, and `decision-champion` collapses them with `.all("model")` — which is
how "train a suite, compare, deploy the best" becomes a declaration rather
than a loop someone has to write and babysit.
"""

from __future__ import annotations

# ── artifact registry ───────────────────────────────────────────────────
ARTIFACT_EPISODES = "decision-episodes"
ARTIFACT_MODEL = "decision-model"
ARTIFACT_SCORECARD = "decision-scorecard"
ARTIFACT_CHAMPION = "decision-champion"

#: The factory's `serve` node: a running app, not an artifact. Never published
#: to or read from the registry. Must be a DNS label (lowercase, digits, '-').
ENDPOINT_APP = "decision-api"

# ── partition dimensions ────────────────────────────────────────────────
#: One episode set / suite run per day.
PARTITION_DATE = "date"
#: One fine-tune and scorecard per suite member; the value is a model slug
#: from `config.MODEL_SUITE`. Slugs never contain '/' for this reason.
PARTITION_MODEL = "model"

# ── payload schemas ─────────────────────────────────────────────────────
# decision-episodes: parquet File. `obs_*` columns are the observation
# fields; `label_json` is the training target verbatim.
EPISODE_COLUMNS = [
    "episode_id",
    "label_tool",
    "label_json",
    "rationale",
    "hard",
    "split",  # "train" | "eval"
    "obs_customer_message",
    "obs_intent",
    "obs_order_id",
    "obs_order_looked_up",
    "obs_order_status",
    "obs_order_total_cents",
    "obs_days_since_delivery",
    "obs_tracking_number",
]

# decision-model / decision-champion: Dir with a PEFT adapter, tokenizer
# files, and manifest.json carrying at least these keys. The serving app
# reads `base_model` to know what to load the adapter onto, so a Dir without
# it is unservable.
MODEL_MANIFEST_KEYS = ["model_slug", "base_model", "profile", "train_episodes"]

# decision-scorecard: JSON File with at least these keys. `exact_accuracy` is
# what the champion is selected on.
SCORECARD_KEYS = [
    "model_slug",
    "base_model",
    "n",
    "json_valid_rate",
    "tool_accuracy",
    "exact_accuracy",
    "majority_baseline",
]

# decision-champion's manifest adds the selection record on top of
# MODEL_MANIFEST_KEYS, so the served Dir says why it won.
CHAMPION_MANIFEST_KEYS = MODEL_MANIFEST_KEYS + ["selected_from", "exact_accuracy"]
