"""Execution-ceiling clamp (the PRD's quota clamp, learned the hard way:
an unclamped 16-CPU/64Gi proposal queued a harness pod forever)."""

from resource_tuner.environment.episodes import (
    EPISODE_MAX_CPU,
    EPISODE_MAX_MEMORY_MIB,
    clamp_for_execution,
)
from resource_tuner.policy.actions import Proposal


def test_in_bounds_proposals_pass_through():
    p = Proposal(cpu=2, memory_mib=4096)
    clamped, was_clamped = clamp_for_execution(p)
    assert clamped == p and not was_clamped


def test_over_ask_clamps_down_onto_the_grid():
    clamped, was_clamped = clamp_for_execution(Proposal(cpu=16, memory_mib=65536))
    assert was_clamped
    assert clamped.cpu <= EPISODE_MAX_CPU
    assert clamped.memory_mib <= EPISODE_MAX_MEMORY_MIB
    # floor, not bucket-up: rounding up would bounce above the ceiling
    assert clamped.memory_mib == 8192
    assert clamped.cpu == 6
