"""Reward curriculum for the resource-tuner policy.

Stage A ("success") — the fundamental-first reward: did the proposal parse,
and did the task fit? Binary, unhackable, and the signal we must see go up
before anything fancier is worth training (all-or-nothing per DeepCoder /
basic-model-factory: partial credit on the success axis is what reward
hacking feeds on).

Stage B ("composite") — the PRD formula:

    reward = w1·run_success − w2·mean_m(percent_m_overprovisioned)
             − w3·oom_penalty − w4·throttle_penalty

The OOM penalty is deliberately asymmetric: a failed run costs more than
proportional waste, because that is how users experience the two errors
(failure is loud, waste is silent). Waste only counts on SUCCESSFUL runs —
punishing waste on a failed run would teach the policy that failing cheaply
beats succeeding generously.

Every component is returned individually (model-factory style) so training
curves show *which* term moved, not just the sum.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..environment.simulator import EpisodeResult

# Format shaping: a parseable, schema-valid proposal earns a small bonus even
# before episode outcomes — the policy must first learn to speak kwargs.
FORMAT_REWARD = 0.1
SUCCESS_REWARD = 1.0

# Stage B weights. Success dominates; full-scale waste (≈100% of both
# metrics idle) costs about half a success; an OOM lands the total well
# below any successful-but-wasteful episode.
W_SUCCESS = 1.0
W_WASTE = 0.5
W_OOM = 0.5
W_THROTTLE = 0.05

MAX_REWARD = FORMAT_REWARD + SUCCESS_REWARD


@dataclass(frozen=True)
class RewardBreakdown:
    """Per-component scores; `total` is what the trainer optimizes."""

    format: float
    success: float
    waste_penalty: float
    oom_penalty: float
    throttle_penalty: float
    total: float


def overprovision_fraction(requested: float, peak: float) -> float:
    """(requested − peak) / requested, clipped to [0, 1].

    1.0 = the whole request sat idle; 0.0 = right-sized (or starved).
    """
    if requested <= 0:
        return 0.0
    return min(max((requested - peak) / requested, 0.0), 1.0)


def invalid_proposal_reward() -> RewardBreakdown:
    """The model failed to emit valid kwargs: floor everything. No episode
    ran, so no outcome components apply."""
    return RewardBreakdown(0.0, 0.0, 0.0, 0.0, 0.0, total=0.0)


def stage_a_reward(episode: EpisodeResult) -> RewardBreakdown:
    success = SUCCESS_REWARD if episode.ok else 0.0
    return RewardBreakdown(
        format=FORMAT_REWARD,
        success=success,
        waste_penalty=0.0,
        oom_penalty=0.0,
        throttle_penalty=0.0,
        total=FORMAT_REWARD + success,
    )


def stage_b_reward(episode: EpisodeResult) -> RewardBreakdown:
    success = W_SUCCESS * SUCCESS_REWARD if episode.ok else 0.0
    oom = W_OOM if episode.oom else 0.0
    waste = 0.0
    if episode.ok:
        waste = W_WASTE * (
            overprovision_fraction(episode.requested_memory_mib, episode.peak_memory_mib)
            + overprovision_fraction(episode.requested_cpu, episode.peak_cpu)
        ) / 2
    throttle = W_THROTTLE if (episode.ok and episode.throttled) else 0.0
    return RewardBreakdown(
        format=FORMAT_REWARD,
        success=success,
        waste_penalty=-waste,
        oom_penalty=-oom,
        throttle_penalty=-throttle,
        total=FORMAT_REWARD + success - waste - oom - throttle,
    )


REWARD_STAGES = {"success": stage_a_reward, "composite": stage_b_reward}


def score_episode(stage: str, episode: EpisodeResult) -> RewardBreakdown:
    try:
        return REWARD_STAGES[stage](episode)
    except KeyError:
        raise ValueError(f"unknown reward stage {stage!r}; choose from {sorted(REWARD_STAGES)}")
