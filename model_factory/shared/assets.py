"""Asset resolution: find the latest version of each factory asset.

Primary path: the Flyte artifact registry (`flyte.remote.Artifact`). The
demo control plane doesn't serve the artifact CRUD API yet (trigger
*definitions* on artifacts register fine, but `Artifact.get/listall` return
Not Found), so this module falls back to scanning recent runs for the
station task that produces each asset and reading its output path. The
fallback disappears naturally once the backend ships the artifact service.
"""

from __future__ import annotations

from dataclasses import dataclass

import flyte.remote as remote

from ..contracts import (
    ARTIFACT_CHECKPOINT,
    ARTIFACT_EVAL_REPORT,
    ARTIFACT_INFERENCE_ENDPOINT,
    ARTIFACT_PROMOTED,
    ARTIFACT_RL_DATASET,
    ARTIFACT_SYNTHETIC,
)

# artifact name -> station task(s) that produce it (task_name suffix match)
# Old mf-* names kept so assets from pre-decomposition runs still resolve.
PRODUCERS: dict[str, tuple[str, ...]] = {
    ARTIFACT_RL_DATASET: ("de-cpu.publish_dataset", "mf-cpu.publish_dataset"),
    ARTIFACT_SYNTHETIC: ("de-gpu.generate_synthetic_tasks", "mf-gpu.generate_synthetic_tasks"),
    ARTIFACT_CHECKPOINT: ("trainer.train_grpo", "mf-gpu.train_grpo"),
    ARTIFACT_EVAL_REPORT: ("eval-gpu.evaluate_checkpoint", "mf-gpu.evaluate_checkpoint"),
    ARTIFACT_PROMOTED: ("eval-cpu.promote_checkpoint", "mf-cpu.promote_checkpoint"),
    ARTIFACT_INFERENCE_ENDPOINT: ("inference-ops.refresh_inference_service",),
}

_SCAN_RUNS = 15


@dataclass(frozen=True)
class AssetVersion:
    artifact_name: str
    path: str  # object-store path of the produced File/Dir
    run_name: str
    action_name: str
    via: str  # "artifact-api" | "run-scan"


async def _first_output_path(details) -> str | None:
    outs = details.outputs() if callable(getattr(details, "outputs", None)) else details.outputs
    import asyncio

    if asyncio.iscoroutine(outs):
        outs = await outs
    if outs is None:
        return None
    first = getattr(outs, "o0", None) or (outs[0] if len(outs) else None)
    return getattr(first, "path", None)


async def list_versions(
    artifact_name: str,
    project: str | None = None,
    domain: str | None = None,
    limit: int = 10,
) -> list[AssetVersion]:
    """All known versions of an asset, newest first."""
    # Primary: artifact registry.
    try:
        found = []
        async for a in remote.Artifact.listall.aio(name=artifact_name, limit=limit):
            found.append(
                AssetVersion(
                    artifact_name=artifact_name,
                    path=a.url,
                    run_name=str(getattr(a, "source", "") or ""),
                    action_name="",
                    via="artifact-api",
                )
            )
        if found:
            return found
    except Exception:
        pass

    # Fallback: scan recent runs for producing station actions.
    producers = PRODUCERS.get(artifact_name, ())
    versions: list[AssetVersion] = []
    try:
        count = 0
        async for run in remote.Run.listall.aio(
            project=project,
            domain=domain,
            limit=_SCAN_RUNS * 3,
            sort_by=("created_at", "desc"),  # "latest" must mean newest
        ):
            count += 1
            if count > _SCAN_RUNS or len(versions) >= limit:
                break
            try:
                async for action in remote.Action.listall.aio(
                    for_run_name=run.name, in_phase=("succeeded",)
                ):
                    tn = action.task_name or ""
                    if any(tn.endswith(p) or p.endswith(tn) for p in producers if tn):
                        details = await action.details()
                        path = await _first_output_path(details)
                        if path:
                            versions.append(
                                AssetVersion(
                                    artifact_name=artifact_name,
                                    path=path,
                                    run_name=run.name,
                                    action_name=action.name,
                                    via="run-scan",
                                )
                            )
            except Exception:
                continue
    except Exception:
        pass
    return versions[:limit]


async def latest(
    artifact_name: str, project: str | None = None, domain: str | None = None
) -> AssetVersion:
    versions = await list_versions(artifact_name, project=project, domain=domain, limit=1)
    if not versions:
        raise FileNotFoundError(
            f"no version of asset {artifact_name!r} found via artifact API or run scan"
        )
    return versions[0]
