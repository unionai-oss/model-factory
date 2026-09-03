"""Simulator semantics: memory is incompressible, CPU is compressible."""

from resource_tuner.environment.simulator import MEMORY_JITTER, simulate_episode
from resource_tuner.policy.actions import Proposal


def test_memory_shortfall_is_an_oom():
    ep = simulate_episode(Proposal(cpu=1, memory_mib=512), 800.0, 1.0, 60)
    assert not ep.ok and ep.oom


def test_memory_at_exact_peak_still_ooms():
    """Request == peak leaves no allocator slack; the jitter margin makes
    the boundary honest instead of knife-edge."""
    ep = simulate_episode(Proposal(cpu=1, memory_mib=1024), 1024.0, 1.0, 60)
    assert ep.oom
    ok = simulate_episode(
        Proposal(cpu=1, memory_mib=int(1024 * (1 + MEMORY_JITTER) + 1)), 1024.0, 1.0, 60
    )
    assert ok.ok


def test_cpu_shortfall_throttles_but_succeeds():
    ep = simulate_episode(Proposal(cpu=1, memory_mib=2048), 800.0, 4.0, 60)
    assert ep.ok and not ep.oom
    assert ep.throttled
    assert ep.duration_s == 240  # 4 cores of demand on 1 core: 4x wall-clock


def test_result_shape_is_cluster_compatible():
    """Sim and cluster episodes must stay interchangeable for the rewards."""
    ep = simulate_episode(Proposal(cpu=2, memory_mib=4096), 800.0, 1.0, 60)
    assert ep.simulated
    assert ep.requested_memory_mib == 4096
    assert ep.peak_memory_mib == 800.0
    assert ep.peak_cpu == 1.0
