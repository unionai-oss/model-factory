"""The sampler and the data station's row building."""

from collections import Counter

from decision_factory.contracts import EPISODE_COLUMNS
from decision_factory.data.tasks import build_rows
from decision_factory.harness.observations import (
    ADVERSARIAL,
    ADVERSARIAL_FAMILIES,
    FAMILIES,
    LABEL_MIX,
    sample_episodes,
    sample_one,
)
from decision_factory.harness.policy import DECISIONS, Observation, decide


def test_every_label_comes_from_the_oracle():
    for ep in sample_episodes(300, seed=3):
        assert decide(ep.obs).decision == ep.label


def test_stratified_label_mix():
    eps = sample_episodes(2000, seed=1)
    counts = Counter(e.label for e in eps)
    for d, p in LABEL_MIX.items():
        assert abs(counts[d] / len(eps) - p) < 0.02, (d, counts)
    adv = sum(1 for e in eps if e.slice == ADVERSARIAL) / len(eps)
    assert 0.4 < adv < 0.5


def test_sampling_is_reproducible():
    a = [e.to_row() for e in sample_episodes(50, seed=9)]
    b = [e.to_row() for e in sample_episodes(50, seed=9)]
    assert a == b


def test_every_family_runs_and_most_adversarial_ones_cross_labels():
    import random

    rng = random.Random(0)
    for fam in FAMILIES:
        labels = {sample_one(rng, fam).label for _ in range(60)}
        assert labels <= set(DECISIONS)
        # an adversarial family that only ever yields one label has no near-miss in it
        if fam in ADVERSARIAL_FAMILIES and fam not in ("a_benign_lookalike",):
            assert len(labels) >= 2, (fam, labels)


def test_rows_have_the_contract_columns_and_three_splits():
    rows = build_rows(60, 20, 30, seed=4)
    assert set(EPISODE_COLUMNS) <= set(rows[0])
    assert Counter(r["split"] for r in rows) == {"train": 60, "val": 20, "test": 30}
    assert all(r["seen_in_train"] for r in rows if r["split"] == "train")
    assert len({r["episode_id"] for r in rows}) == len(rows)


def test_rows_round_trip_to_observations():
    for r in build_rows(30, 5, 5, seed=2):
        assert decide(Observation.from_row(r)).decision == r["label"]
