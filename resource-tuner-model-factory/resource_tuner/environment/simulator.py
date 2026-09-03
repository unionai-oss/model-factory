"""Offline episode simulator — the cheap loop.

Scores a proposal against a task's analytic footprint without touching a
cluster. Training runs almost entirely here (PRD: "real runs are slow and
expensive, so the loop starts synthetic"); the on-cluster episode runner
(episodes.py) exists to validate that simulator truth tracks pod truth.

Semantics mirror Kubernetes:
- memory is incompressible: peak demand above the request → OOMKilled.
- cpu is compressible: demand above the request throttles (the task still
  succeeds; we record the throttle ratio, it never fails the episode).
"""

from __future__ import annotations

from dataclasses import dataclass
from random import Random

from ..policy.actions import Proposal

# A pod whose limit sits exactly at the workload's peak still gets OOM-killed
# in practice (allocator slack, page tables, GC timing). Success needs
# request >= peak * (1 + jitter).
MEMORY_JITTER = 0.05
# With an `rng`, the jitter is drawn from this band instead — domain
# randomization on the success threshold, so the policy cannot ride a
# deterministic sim edge (the named failure mode is Simulation Optimization
# Bias: an optimizer exploiting simulator quirks the real world lacks).
# Training passes an rng; tests and eval scoring stay deterministic.
JITTER_BAND = (0.02, 0.12)


@dataclass(frozen=True)
class EpisodeResult:
    """Outcome of one episode, real or simulated. The reward functions
    consume ONLY this shape, so sim and cluster episodes are interchangeable."""

    ok: bool  # task reached SUCCEEDED
    oom: bool  # failed specifically by running out of memory
    requested_cpu: float
    requested_memory_mib: int
    peak_memory_mib: float  # observed (cluster) or analytic (sim)
    peak_cpu: float
    throttled: bool  # cpu demand exceeded request
    duration_s: float
    simulated: bool
    run_name: str = ""  # cluster episodes only


def simulate_episode(
    proposal: Proposal,
    true_peak_memory_mib: float,
    true_cpu_cores: float,
    duration_s: int,
    rng: Random | None = None,
) -> EpisodeResult:
    jitter = rng.uniform(*JITTER_BAND) if rng is not None else MEMORY_JITTER
    fits = proposal.memory_mib >= true_peak_memory_mib * (1 + jitter)
    throttled = proposal.cpu < true_cpu_cores
    # Throttling stretches wall-clock proportionally to the shortfall.
    duration = duration_s * (max(true_cpu_cores / proposal.cpu, 1.0))
    return EpisodeResult(
        ok=fits,
        oom=not fits,
        requested_cpu=proposal.cpu,
        requested_memory_mib=proposal.memory_mib,
        peak_memory_mib=min(true_peak_memory_mib, proposal.memory_mib)
        if not fits
        else true_peak_memory_mib,
        peak_cpu=min(true_cpu_cores, proposal.cpu),
        throttled=throttled,
        duration_s=duration,
        simulated=True,
    )
