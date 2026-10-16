"""Artifact names, partition dimensions, and payload schemas.

The factory (`decision_factory/factory.py`) is declared against these names.
Nothing here publishes anything: the factory's driver calls each build's task
with `produces_artifacts=True` inside a `flyte.artifacts.produces(...)` block
and declares the artifact itself, so a station task returns a plain
`File`/`Dir` and never wraps it in `flyte.artifacts.new()`.

The graph, and what each partition dimension means:

    permission-episodes    [date]           one oracle-labelled dataset per day
    permission-model       [date, model]    one fine-tune per torso x head
    permission-scorecard   [date, model]    one scorecard per fine-tune
    permission-champion    [date]           the winner, collapsing `model`
    permission-gate        (endpoint)       serves the champion

Round 1 published `decision-*` artifacts for a different problem (support
triage). These are new names on purpose: an artifact name is a contract about
what its versions *mean*, and a permission gate is not a triage router — the
lineage graph should not show one as a newer version of the other.
"""

from __future__ import annotations

# ── artifact registry ───────────────────────────────────────────────────
ARTIFACT_EPISODES = "permission-episodes"
ARTIFACT_MODEL = "permission-model"
ARTIFACT_SCORECARD = "permission-scorecard"
ARTIFACT_CHAMPION = "permission-champion"

#: The factory's `serve` node: a running app, not an artifact. Must be a DNS
#: label (lowercase, digits, '-').
ENDPOINT_APP = "permission-gate"

# ── partition dimensions ────────────────────────────────────────────────
#: One episode set / suite run per day.
PARTITION_DATE = "date"
#: One fine-tune and scorecard per suite member; the value is a torso x head
#: slug from `config.MODEL_SUITE` (e.g. `qwen2.5-0.5b-pointer`).
PARTITION_MODEL = "model"

# ── payload schemas ─────────────────────────────────────────────────────
# permission-episodes: parquet File, one row per proposed tool call.
EPISODE_COLUMNS = [
    "episode_id",
    "split",  # "train" | "val" | "test"
    "label",  # allow | ask_user | deny
    "rule",  # which oracle rule fired (R1..R5)
    "rationale",
    "family",  # sampler family, e.g. a_quote_obf
    "slice",  # standard | adversarial
    "seen_in_train",  # test/val rows: was this exact command/path/url in train?
    "obs_tool",
    "obs_command",
    "obs_path",
    "obs_resolved_path",
    "obs_url",
    "obs_workspace_root",
    "obs_cwd",
    "obs_home",
    "obs_git_branch",
    "obs_permission_mode",
    "obs_user_task",
]

# permission-model / permission-champion: Dir with a PEFT adapter, tokenizer
# files, (pointer only) `pointer_head.pt`, and manifest.json carrying at least
# these keys. The serving app reads `base_model` and `head` to know how to
# load it, so a Dir without them is unservable.
MODEL_MANIFEST_KEYS = ["model_slug", "base_model", "head", "profile", "train_episodes"]

# permission-scorecard: JSON File with at least these keys. `primary` is the
# readout the champion is selected on (expected cost under scoring.COST).
SCORECARD_KEYS = ["model_slug", "base_model", "head", "primary", "readouts", "reference"]

# permission-champion's manifest adds the selection record and the tuned gate
# threshold, so the served Dir says why it won and how to gate with it.
CHAMPION_MANIFEST_KEYS = MODEL_MANIFEST_KEYS + ["selected_from", "gate_threshold", "test_metrics"]
