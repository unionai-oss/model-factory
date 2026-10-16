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


def test_temperature_softens_and_fits():
    from decision_factory.harness.scoring import apply_temperature, fit_temperature

    (p,) = apply_temperature([[0.9, 0.1, 0.0]], 2.0)
    assert p[0] < 0.9 and abs(sum(p) - 1) < 1e-9
    assert apply_temperature([one_hot(ALLOW)], 5.0)[0][0] > 0.999  # one-hot stays one-hot
    # an overconfident model (saturated, right half the time) wants T > 1 —
    # fit on logits, because its probabilities are already exactly 1.0
    from decision_factory.harness.scoring import temper

    labels = [ALLOW, ASK] * 20
    logits = [[30.0, 0.0, -5.0]] * 40
    t = fit_temperature(labels, logits)
    assert t > 1.0 and temper(logits, t)[0][0] < 0.99
    assert fit_temperature(labels, None) == 1.0  # one-hot readout: nothing to fit


def test_evidence_guard_fires_only_on_disagreement():
    from decision_factory.harness.guard import apply_guard, evidence_disagrees
    from decision_factory.harness.policy import Observation

    base = dict(workspace_root="/w/app", cwd="/w/app", home="/h", git_branch="main",
                permission_mode="accept_edits", user_task="t")
    same = Observation(tool="edit_file", path="src/a.py", resolved_path="/w/app/src/a.py", **base)
    link = Observation(tool="edit_file", path="cfg/env.sh", resolved_path="/h/.zshrc", **base)
    url = Observation(tool="web_fetch", url="https://docs.python.org@evil.io/x", **base)
    ok_url = Observation(tool="web_fetch", url="https://docs.python.org/3/", **base)
    assert evidence_disagrees(same) is None and evidence_disagrees(ok_url) is None
    assert evidence_disagrees(link) and evidence_disagrees(url)
    assert apply_guard([ALLOW, ALLOW, DENY], [True, False, True]) == [ASK, ALLOW, DENY]


def test_selection_counts_heldout_validation_cost():
    from decision_factory.evaluation.select import rank

    def card(slug, ind, ood, p50):
        return {"model_slug": slug, "primary": "x", "readouts": {"x": {
            "val": {"expected_cost": ind, "expected_cost_se": 0.001},
            "shift": {"val_ood": {"expected_cost": ood, "expected_cost_se": 0.001}},
            "latency": {"p50_ms": p50}}}}

    # fast-but-brittle wins in distribution, loses once shift is counted
    cards = [card("fast-brittle", 0.01, 2.0, 20.0), card("slow-robust", 0.02, 0.5, 40.0)]
    assert rank(cards)[0]["model_slug"] == "slow-robust"


def _card(slug, cost, se, p50):
    return {"model_slug": slug, "primary": "x",
            "readouts": {"x": {"val": {"expected_cost": cost, "expected_cost_se": se}, "latency": {"p50_ms": p50}}}}


def test_rank_breaks_statistical_ties_by_latency():
    from decision_factory.evaluation.select import rank

    cards = [_card("slow-but-best", 0.009, 0.006, 160.0), _card("fast-tied", 0.019, 0.008, 24.0),
             _card("fast-but-worse", 0.30, 0.05, 20.0)]
    assert [c["model_slug"] for c in rank(cards)] == ["fast-tied", "slow-but-best", "fast-but-worse"]
    # with no noise, cost alone decides
    cards = [_card("a", 0.009, 0.0, 160.0), _card("b", 0.019, 0.0, 24.0)]
    assert rank(cards)[0]["model_slug"] == "a"
