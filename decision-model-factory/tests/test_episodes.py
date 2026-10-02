"""The synthetic generator: labels must be correct, and the mix must be usable.

The generator is the dataset, so a bug here is indistinguishable from a bad
model. The load-bearing property is the first test: every label comes from
the oracle, so no episode can ever be mislabelled even if a builder drifts.
"""

import json

from decision_factory.harness.observations import (
    HARD_FRACTION,
    TOOL_MIX,
    sample_episodes,
)
from decision_factory.harness.policy import (
    AUTO_REFUND_LIMIT_CENTS,
    RETURN_WINDOW_DAYS,
    decide,
)
from decision_factory.harness.tools import TOOL_NAMES, ToolCall


def test_every_label_is_what_the_oracle_says():
    # The whole point: the generator cannot mislabel, because the label is
    # recomputed from the observation by the same function eval grades against.
    for ep in sample_episodes(400, seed=1):
        assert ep.label == decide(ep.observation), ep.episode_id


def test_labels_are_valid_catalog_calls():
    for ep in sample_episodes(200, seed=2):
        call = ToolCall.from_obj(json.loads(ep.label.to_json()))
        assert call is not None
        assert call.tool in TOOL_NAMES


def test_sampling_is_deterministic_in_the_seed():
    a = [e.to_row() for e in sample_episodes(100, seed=7)]
    b = [e.to_row() for e in sample_episodes(100, seed=7)]
    assert a == b
    c = [e.to_row() for e in sample_episodes(100, seed=8)]
    assert a != c


def test_requested_count_is_exact():
    for n in (1, 7, 50, 333):
        assert len(sample_episodes(n, seed=3)) == n
    assert sample_episodes(0) == []
    assert sample_episodes(-5) == []


def test_every_tool_is_represented():
    # A tool with no episodes is a tool the model cannot learn and the
    # scorecard cannot measure.
    episodes = sample_episodes(600, seed=4)
    seen = {e.label.tool for e in episodes}
    assert seen == set(TOOL_NAMES), f"missing: {set(TOOL_NAMES) - seen}"


def test_the_mix_is_roughly_the_target_and_not_dominated():
    episodes = sample_episodes(1200, seed=5)
    n = len(episodes)
    for tool, target in TOOL_MIX.items():
        share = sum(1 for e in episodes if e.label.tool == tool) / n
        # Generous tolerance: builders retry toward a target but the oracle has
        # the last word, so the realized mix drifts. What matters is that no
        # class collapses and none dominates.
        assert 0.3 * target <= share <= target + 0.18, f"{tool} share {share:.3f}"


def test_hard_episodes_exist_and_cluster_at_the_boundaries():
    episodes = sample_episodes(800, seed=6)
    hard = [e for e in episodes if e.hard]
    assert len(hard) / len(episodes) > HARD_FRACTION * 0.5

    # Among hard refund/escalate episodes, totals should sit near the limit —
    # that is what makes them hard rather than decidable from magnitude.
    near = [
        e
        for e in hard
        if e.observation.order_total_cents is not None
        and e.label.tool in ("issue_refund", "escalate_to_human")
        and abs(e.observation.order_total_cents - AUTO_REFUND_LIMIT_CENTS) <= 1_500
    ]
    assert near, "no hard episode landed near the refund limit"


def test_both_sides_of_each_threshold_appear():
    # Only-under or only-over would let a model learn a constant instead of a
    # comparison.
    episodes = sample_episodes(1500, seed=9)
    totals = [
        e.observation.order_total_cents
        for e in episodes
        if e.observation.order_total_cents is not None and e.observation.intent == "refund"
    ]
    assert any(t <= AUTO_REFUND_LIMIT_CENTS for t in totals)
    assert any(t > AUTO_REFUND_LIMIT_CENTS for t in totals)

    days = [
        e.observation.days_since_delivery
        for e in episodes
        if e.observation.days_since_delivery is not None and e.observation.intent == "refund"
    ]
    assert any(d <= RETURN_WINDOW_DAYS for d in days)
    assert any(d > RETURN_WINDOW_DAYS for d in days)


def test_rows_are_flat_and_carry_the_training_target():
    from decision_factory.contracts import EPISODE_COLUMNS

    row = sample_episodes(1, seed=11)[0].to_row()
    # Every declared column except `split` (added by the station task) must be
    # produced here, or the parquet schema drifts from the contract.
    for col in EPISODE_COLUMNS:
        if col == "split":
            continue
        assert col in row, col
    # label_json round-trips to a real call: this string IS the SFT target.
    assert ToolCall.from_obj(json.loads(row["label_json"])) is not None
    for value in row.values():
        assert not isinstance(value, (dict, list)), "rows must stay parquet-flat"


def test_episode_ids_are_unique():
    episodes = sample_episodes(500, seed=12)
    assert len({e.episode_id for e in episodes}) == len(episodes)
