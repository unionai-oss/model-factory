"""On-cluster episodes: run a corpus task under a proposal, observe truth.

The expensive half of the environment. A driver task (CPU env) fans these
out with asyncio.gather; each episode is a child action of the driver's run
whose resources are the policy's proposal, verbatim. Underprovisioned
memory → the pod OOMs → the child action fails → that IS the negative
signal, not an error to retry.
"""

from __future__ import annotations

import flyte
import flyte.errors

import os

from ..policy.actions import CPU_GRID, MEMORY_GRID_MIB, Proposal, parse_memory_to_mib
from .harness import run_generated
from .simulator import EpisodeResult

# Execution ceiling — the PRD's quota clamp. The action grid intentionally
# extends beyond what this tenant schedules (the policy must be free to
# over-ask and get punished for it), but the POD REQUEST is clamped to what
# a CPU node can actually hold; otherwise an over-ask queues forever
# instead of costing waste. The clamp itself is recorded on the episode.
EPISODE_MAX_CPU = float(os.environ.get("RT_EPISODE_MAX_CPU", "6"))
EPISODE_MAX_MEMORY_MIB = parse_memory_to_mib(os.environ.get("RT_EPISODE_MAX_MEMORY", "12Gi"))


def clamp_for_execution(proposal: Proposal) -> tuple[Proposal, bool]:
    """(clamped proposal, was_clamped). Clamps FLOOR to the action grid —
    rounding up (bucket_*) would bounce back above the ceiling."""
    if proposal.cpu <= EPISODE_MAX_CPU and proposal.memory_mib <= EPISODE_MAX_MEMORY_MIB:
        return proposal, False
    cpu = max(c for c in CPU_GRID if c <= min(proposal.cpu, EPISODE_MAX_CPU))
    mem = max(m for m in MEMORY_GRID_MIB if m <= min(proposal.memory_mib, EPISODE_MAX_MEMORY_MIB))
    return Proposal(cpu=cpu, memory_mib=mem, gpu=proposal.gpu), True


def _looks_like_oom(err: Exception) -> bool:
    if isinstance(err, getattr(flyte.errors, "OOMError", ())):
        return True
    text = str(err).lower()
    return "oom" in text or "out of memory" in text or "137" in text


async def run_cluster_episode(record: dict, proposal: Proposal) -> EpisodeResult:
    """One real episode. `record` is a corpus row (see contracts).

    The episode's reward-facing request stays the POLICY's proposal (an
    over-ask must cost waste); only the pod request is clamped so it can
    schedule at all.
    """
    executed, _clamped = clamp_for_execution(proposal)
    overridden = run_generated.override(resources=flyte.Resources(**executed.to_kwargs()))
    try:
        out = await overridden(
            harness_code=record["harness_code"], task_id=record["task_id"]
        )
    except Exception as e:  # noqa: BLE001 — failed child action
        oom = _looks_like_oom(e)
        return EpisodeResult(
            ok=False,
            oom=oom,
            requested_cpu=proposal.cpu,
            requested_memory_mib=proposal.memory_mib,
            peak_memory_mib=float(proposal.memory_mib) if oom else 0.0,
            peak_cpu=0.0,
            throttled=False,
            duration_s=0.0,
            simulated=False,
        )
    return EpisodeResult(
        ok=bool(out["ok"]),
        oom=False,
        requested_cpu=proposal.cpu,
        requested_memory_mib=proposal.memory_mib,
        peak_memory_mib=float(out["peak_rss_mib"]),
        peak_cpu=min(float(record["true_cpu_cores"]), proposal.cpu),
        throttled=proposal.cpu < float(record["true_cpu_cores"]),
        duration_s=float(out["duration_s"]),
        simulated=False,
    )
