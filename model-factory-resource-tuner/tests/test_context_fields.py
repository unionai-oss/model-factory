"""PRD §8 context wiring: author prior + ledger run history → the prompt."""

import json

from resource_tuner.policy.prompts import parse_context_fields, render_messages
from resource_tuner.taskgen.corpus import build_corpus
from resource_tuner.training.grpo import _record_to_row
from resource_tuner.tune_store import history_of


def test_parse_context_fields_degrades_to_cold_start():
    assert parse_context_fields("", "") == (None, None)
    assert parse_context_fields(None, None) == (None, None)
    assert parse_context_fields("{not json", "[broken") == (None, None)
    assert parse_context_fields("{}", "[]") == (None, None)  # empty ≠ signal
    prior, history = parse_context_fields(
        '{"cpu": 4, "memory": "8Gi"}', '[{"resources": {"cpu": 1}, "peak": "800MiB", "ok": true}]'
    )
    assert prior == {"cpu": 4, "memory": "8Gi"}
    assert history[0]["ok"] is True


def test_render_messages_includes_prior_and_history_lines():
    msgs = render_messages(
        "def f(): ...",
        "input: 1M rows",
        prior={"cpu": 4, "memory": "8Gi"},
        history=[{"resources": {"cpu": 2, "memory": "2Gi"}, "peak": "812MiB", "ok": True}],
    )
    user = msgs[1]["content"]
    assert "Author-declared prior: {'cpu': 4, 'memory': '8Gi'}" in user
    assert "Recent runs:" in user and "812MiB" in user and "success=True" in user


def test_corpus_rows_carry_deterministic_context_fields():
    a = build_corpus(60, 20, seed=3)
    b = build_corpus(60, 20, seed=3)
    assert a == b  # prior/history sampling is task-keyed, not process-keyed
    with_prior = [r for r in a if r["prior_json"]]
    with_history = [r for r in a if r["history_json"]]
    cold = [r for r in a if not r["prior_json"] and not r["history_json"]]
    # all three regimes must exist: hinted, historied, and cold-start
    assert with_prior and with_history and cold
    priors = [json.loads(r["prior_json"]) for r in with_prior]
    assert all({"cpu", "memory"} <= set(p) for p in priors)
    entries = [e for r in with_history for e in json.loads(r["history_json"])]
    assert all({"resources", "peak", "ok"} <= set(e) for e in entries)


def test_dataset_prompts_render_the_context():
    records = [r for r in build_corpus(60, 0, seed=3) if r["prior_json"]][:2]
    user_msgs = [_record_to_row(r)["prompt"][1]["content"] for r in records]
    assert all("Author-declared prior" in m for m in user_msgs)


def test_history_of_reads_the_ledger_shape():
    records = [
        {"kind": "proposal", "ts": 1, "task_id": "wf.x"},
        {"kind": "outcome", "ts": 2, "task_id": "wf.x",
         "requested": {"cpu": 2, "memory": "2Gi"}, "peak_rss_mib": 812.3, "ok": True},
        {"kind": "outcome", "ts": 3, "task_id": "wf.other",
         "requested": {"cpu": 1}, "peak_rss_mib": 100.0, "ok": True},
        {"kind": "outcome", "ts": 4, "task_id": "wf.x",
         "requested": {"cpu": 2, "memory": "1Gi"}, "peak_rss_mib": None, "ok": False},
    ]
    h = history_of(records, "wf.x")
    assert len(h) == 2
    assert h[0] == {"resources": {"cpu": 2, "memory": "2Gi"}, "peak": "812MiB", "ok": True}
    assert h[1]["peak"] == "unknown" and h[1]["ok"] is False
    assert history_of(records, "wf.unseen") == []
    # limit keeps prompts bounded
    many = [
        {"kind": "outcome", "ts": i, "task_id": "wf.x", "requested": {}, "peak_rss_mib": i, "ok": True}
        for i in range(10)
    ]
    assert len(history_of(many, "wf.x", limit=3)) == 3
