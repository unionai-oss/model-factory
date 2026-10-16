"""Eval arithmetic: pass@1, paired comparison, the promotion decision.

No flyte imports here: unit-testable pure Python.
"""

from __future__ import annotations

import random


def pass_at_1(passed: list[bool]) -> float:
    return sum(map(bool, passed)) / max(len(passed), 1)


def paired_delta(a: list[bool], b: list[bool], n_boot: int = 2000, seed: int = 0) -> dict:
    """pass@1(a) - pass@1(b) on the same tasks, with a paired bootstrap 95% CI.

    Pairing matters: both models answer the SAME held-out tasks, so resampling
    tasks (not models) gives a much tighter interval than two independent ones.
    """
    if len(a) != len(b):
        raise ValueError(f"paired comparison needs equal lengths, got {len(a)} and {len(b)}")
    n = len(a)
    diffs = [int(x) - int(y) for x, y in zip(a, b)]
    delta = sum(diffs) / max(n, 1)
    rng = random.Random(seed)
    boots = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot)) if n else [0.0]
    return {
        "delta": delta,
        "ci95": [boots[int(0.025 * (len(boots) - 1))], boots[int(0.975 * (len(boots) - 1))]],
        "wins": sum(1 for d in diffs if d > 0),
        "losses": sum(1 for d in diffs if d < 0),
    }


def promotion_decision(delta_vs_champion: float, margin: float, approved: bool) -> tuple[str, str]:
    """Which model day D releases: ``("candidate" | "champion", reason)``.

    Champion/challenger: the release is always a model — the candidate only
    when it beats the champion by ``margin`` AND a human (or the auto-approve
    switch) signs off; otherwise the champion is carried forward. So a bad day
    never takes the factory backwards and never breaks tomorrow's input.
    """
    if delta_vs_champion < margin:
        return "champion", f"candidate {delta_vs_champion:+.2%} vs champion, below the {margin:+.2%} margin"
    if not approved:
        return "champion", "candidate cleared the margin but was rejected at the human gate"
    return "candidate", f"candidate {delta_vs_champion:+.2%} vs champion, cleared the {margin:+.2%} margin"
