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

from ..policy.actions import Proposal
from .harness import run_generated
from .simulator import EpisodeResult


def _looks_like_oom(err: Exception) -> bool:
    if isinstance(err, getattr(flyte.errors, "OOMError", ())):
        return True
    text = str(err).lower()
    return "oom" in text or "out of memory" in text or "137" in text


async def run_cluster_episode(record: dict, proposal: Proposal) -> EpisodeResult:
    """One real episode. `record` is a corpus row (see contracts)."""
    overridden = run_generated.override(resources=flyte.Resources(**proposal.to_kwargs()))
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
