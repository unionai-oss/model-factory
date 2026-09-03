"""Rule-based baseline estimator (the PRD's Phase 1 benchmark).

Every learned model must beat this or it has no reason to exist. Cold-start
rule: per family, take the median analytic footprint of the TRAIN split,
add a safety margin, bucket. With run history it would become
percentile-of-observed + margin; the corpus is cold-start, so family
medians are the honest equivalent.
"""

from __future__ import annotations

from statistics import median

from ..policy.actions import Proposal, bucket_cpu, bucket_memory_mib

SAFETY_MARGIN = 0.25


def fit_family_baseline(train_records: list[dict]) -> dict[str, Proposal]:
    """family → proposal derived from the train split's median footprint."""
    by_family: dict[str, list[dict]] = {}
    for r in train_records:
        by_family.setdefault(r["family"], []).append(r)
    out: dict[str, Proposal] = {}
    for family, rows in by_family.items():
        peak = median(r["true_peak_memory_mib"] for r in rows)
        cpu = median(r["true_cpu_cores"] for r in rows)
        out[family] = Proposal(
            cpu=bucket_cpu(cpu),
            memory_mib=bucket_memory_mib(peak * (1 + SAFETY_MARGIN)),
        )
    return out


def baseline_proposal(baselines: dict[str, Proposal], family: str) -> Proposal:
    """Median-family proposal; unseen family falls back to the global max
    (the conservative direction — fail closed toward waste, not OOM)."""
    if family in baselines:
        return baselines[family]
    if not baselines:
        raise ValueError("baseline has no families fitted")
    return max(baselines.values(), key=lambda p: p.memory_mib)
