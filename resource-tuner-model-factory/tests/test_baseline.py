"""Rule-based baseline: the bar every learned model must clear."""

import pytest

from resource_tuner.taskgen.corpus import build_corpus
from resource_tuner.training.baseline import baseline_proposal, fit_family_baseline


def test_fit_covers_every_family_and_buckets_up():
    records = build_corpus(40, 0, seed=5)
    baselines = fit_family_baseline(records)
    assert set(baselines) == {r["family"] for r in records}
    for fam, proposal in baselines.items():
        peaks = [r["true_peak_memory_mib"] for r in records if r["family"] == fam]
        median_peak = sorted(peaks)[len(peaks) // 2]
        # Margin + round-up bucketing: the median task must fit.
        assert proposal.memory_mib >= median_peak


def test_unseen_family_falls_back_conservatively():
    baselines = fit_family_baseline(build_corpus(20, 0, seed=5))
    fallback = baseline_proposal(baselines, "quantum_widgets")
    assert fallback.memory_mib == max(p.memory_mib for p in baselines.values())


def test_empty_baseline_fails_loudly():
    with pytest.raises(ValueError):
        baseline_proposal({}, "etl")
