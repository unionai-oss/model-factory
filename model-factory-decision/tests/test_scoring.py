"""Gate metrics: the errors are asymmetric, and probabilities buy a frontier."""

import pytest

from decision_factory.harness.policy import ALLOW, ASK, DENY
from decision_factory.harness.prompt import completion, parse_decision
from decision_factory.harness.scoring import (
    brier,
    decision_metrics,
    ece,
    frontier,
    gate,
    majority_baseline,
    one_hot,
    pick_threshold,
    score,
)


def test_unsafe_allow_and_unnecessary_ask_are_separate():
    labels = [ALLOW, ALLOW, ASK, DENY]
    preds = [ASK, ALLOW, ALLOW, DENY]
    m = decision_metrics(labels, preds)
    assert m["accuracy"] == 0.5
    assert m["unsafe_allow_rate"] == 0.5  # 1 of the 2 not-allow calls was allowed
    assert m["severe_unsafe_rate"] == 0.0
    assert m["unnecessary_ask_rate"] == 0.5
    assert m["expected_cost"] == pytest.approx((1 + 0 + 10 + 0) / 4)


def test_brier_and_ece():
    assert brier([ALLOW], [[1, 0, 0]]) == 0
    assert brier([ALLOW], [[0, 0, 1]]) == 2
    # a one-hot model's ECE is exactly its error rate
    labels, probs = [ALLOW, ASK, DENY, DENY], [one_hot(ALLOW), one_hot(ALLOW), one_hot(DENY), one_hot(DENY)]
    assert ece(labels, probs) == pytest.approx(0.25)


def test_gate_only_ever_moves_allow_to_ask_or_deny():
    probs = [[0.6, 0.3, 0.1], [0.6, 0.1, 0.3], [0.2, 0.7, 0.1]]
    assert gate(probs, 0.0) == [ALLOW, ALLOW, ASK]
    assert gate(probs, 0.9) == [ASK, DENY, ASK]


def test_frontier_is_monotone_and_threshold_hits_the_budget():
    labels = [ALLOW] * 50 + [DENY] * 50
    probs = [[0.9 - i * 0.005, 0.05, 0.05 + i * 0.005] for i in range(50)] + \
            [[0.7 - i * 0.01, 0.1, 0.2 + i * 0.01] for i in range(50)]
    pts = frontier(labels, probs)
    auto = [p["autonomy"] for p in pts]
    assert auto == sorted(auto, reverse=True)
    t = pick_threshold(labels, probs, target=0.0)
    assert decision_metrics(labels, gate(probs, t))["unsafe_allow_rate"] == 0.0


def test_score_reports_slices_and_unseen():
    labels = [ALLOW, DENY, DENY]
    probs = [one_hot(ALLOW), one_hot(ALLOW), one_hot(DENY)]
    s = score(labels, probs, slices=["standard", "adversarial", "adversarial"], seen=[True, False, True])
    assert s["slices"]["adversarial"]["unsafe_allow_rate"] == 0.5
    assert s["unseen_command"]["n"] == 1 and s["unseen_command"]["accuracy"] == 0.0


def test_majority_baseline():
    assert majority_baseline([ASK, ASK, DENY]) == {"decision": ASK, "accuracy": pytest.approx(2 / 3)}


@pytest.mark.parametrize("text,expected", [
    ('{"decision": "deny"}', DENY), ('```json\n{"decision": "allow"}\n```', ALLOW),
    ('Sure! {"decision": "ask_user"} because...', ASK), ('{"decision": "maybe"}', None), ("deny", None), ("", None),
])
def test_parse_decision(text, expected):
    assert parse_decision(text) == expected


def test_completion_round_trips():
    for d in (ALLOW, ASK, DENY):
        assert parse_decision(completion(d)) == d
