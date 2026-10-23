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

from ..policy.actions import GPU_VRAM_MIB, Proposal

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
    oom: bool  # failed specifically by running out of memory (host OR VRAM)
    requested_cpu: float
    requested_memory_mib: int
    peak_memory_mib: float  # observed (cluster) or analytic (sim)
    peak_cpu: float
    throttled: bool  # cpu demand exceeded request
    duration_s: float
    simulated: bool
    run_name: str = ""  # cluster episodes only
    # GPU axis (defaults keep CPU-only episodes unchanged)
    requested_gpu: int = 0
    requested_gpu_type: str | None = None
    true_gpu_mem_mib: float = 0.0  # 0 = the task does not need a GPU
    gpu_missing: bool = False  # needed a GPU, none was requested


def simulate_episode(
    proposal: Proposal,
    true_peak_memory_mib: float,
    true_cpu_cores: float,
    duration_s: int,
    rng: Random | None = None,
    true_gpu_mem_mib: float = 0.0,
) -> EpisodeResult:
    jitter = rng.uniform(*JITTER_BAND) if rng is not None else MEMORY_JITTER
    fits = proposal.memory_mib >= true_peak_memory_mib * (1 + jitter)
    throttled = proposal.cpu < true_cpu_cores
    # GPU axis: memory-like semantics. A task that needs VRAM fails
    # outright without a GPU (gpu_missing — ImportError/no-CUDA in real
    # life) and OOMs on an undersized one; requesting a GPU a task doesn't
    # need succeeds and is scored as pure waste by the reward.
    gpu_missing, gpu_oom = False, False
    if true_gpu_mem_mib > 0:
        if proposal.gpu < 1:
            gpu_missing = True
        else:
            vram = GPU_VRAM_MIB.get(proposal.gpu_type or "T4", 0) * proposal.gpu
            gpu_oom = vram < true_gpu_mem_mib * (1 + jitter)
    ok = fits and not gpu_missing and not gpu_oom
    # Throttling stretches wall-clock proportionally to the shortfall.
    duration = duration_s * (max(true_cpu_cores / proposal.cpu, 1.0))
    return EpisodeResult(
        ok=ok,
        oom=(not fits) or gpu_oom,
        requested_cpu=proposal.cpu,
        requested_memory_mib=proposal.memory_mib,
        peak_memory_mib=min(true_peak_memory_mib, proposal.memory_mib)
        if not fits
        else true_peak_memory_mib,
        peak_cpu=min(true_cpu_cores, proposal.cpu),
        throttled=throttled,
        duration_s=duration,
        simulated=True,
        requested_gpu=proposal.gpu,
        requested_gpu_type=proposal.gpu_type,
        true_gpu_mem_mib=true_gpu_mem_mib,
        gpu_missing=gpu_missing,
    )
