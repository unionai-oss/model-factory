"""Asset resolution: look up versions of a factory artifact in the registry.

This used to carry a fallback that scanned recent runs for the station task
producing each artifact, because the demo control plane did not serve the
artifact CRUD API. It does now (`Artifact.listall` verified 2026-10-02), and
the factory publishes every artifact with its partition values, so the
registry is the only source this needs.

Consumers are the lineage dashboard and the serving app's fallback path — the
factory itself never comes through here: it resolves versions in its own
driver and hands them to each build.
"""

from __future__ import annotations

from dataclasses import dataclass

import flyte.remote as remote

from ..contracts import (
    ARTIFACT_CHECKPOINT,
    ARTIFACT_EVAL_REPORT,
    ARTIFACT_PROMOTED,
    ARTIFACT_RL_DATASET,
    ARTIFACT_SEED_TASKS,
    ARTIFACT_SYNTHETIC,
)

#: artifact name -> station task(s) that build it, as `<env>.<task>`. The
#: lineage dashboard uses this to label a run with the station it belongs to.
#: Old `mf-*` names are kept so pre-decomposition runs still map.
PRODUCERS: dict[str, tuple[str, ...]] = {
    ARTIFACT_SEED_TASKS: ("de-cpu.ingest_and_curate", "mf-cpu.ingest_and_curate"),
    ARTIFACT_SYNTHETIC: ("de-gpu.generate_synthetic_tasks", "mf-gpu.generate_synthetic_tasks"),
    ARTIFACT_RL_DATASET: ("de-cpu.assemble_dataset", "de-cpu.publish_dataset", "mf-cpu.publish_dataset"),
    ARTIFACT_CHECKPOINT: ("trainer.train_grpo", "mf-gpu.train_grpo"),
    ARTIFACT_EVAL_REPORT: ("eval-gpu.evaluate_checkpoint", "mf-gpu.evaluate_checkpoint"),
    ARTIFACT_PROMOTED: ("eval-cpu.promote_checkpoint", "mf-cpu.promote_checkpoint"),
}


@dataclass(frozen=True)
class AssetVersion:
    artifact_name: str
    path: str  # object-store URI of the produced File/Dir
    run_name: str
    action_name: str
    version: str = ""
    partitions: dict[str, str] | None = None


def _blob_uri(a) -> str | None:
    """Object-store URI of an artifact version's value (spec.value.scalar.blob.uri).

    ``Artifact.url`` is the *console* URL (https://...), which flyte's storage
    layer routes through fsspec's HTTP filesystem — that breaks inside task
    pods (aiohttp RuntimeError). Always resolve the underlying s3:// URI.
    """
    try:
        spec = a.to_dict().get("spec", {})
        return spec["value"]["scalar"]["blob"]["uri"]
    except (AttributeError, KeyError, TypeError):
        return None


def _source_ids(a) -> tuple[str, str]:
    """``(run_name, action_name)`` of the action that produced this version.

    ``Artifact.source`` is a pre-rendered DISPLAY string — literally
    ``"run <run>/<action> (attempt 1)"`` — not an identifier. Using it as a
    run name builds console URLs like ``/runs/run%20u.../...%20(attempt%201)``
    (404) and silently breaks any lookup keyed by run name. The identifiers
    live in the structured spec, so read them from there.
    """
    try:
        action = a.to_dict()["spec"]["source"]["taskAction"]["action"]
        return str(action["run"]["name"] or ""), str(action.get("name", "") or "")
    except (AttributeError, KeyError, TypeError):
        return "", ""


def _partitions(a) -> dict[str, str]:
    """Partition values this version was published with, as plain strings.

    A version's time partition and its string partitions live in two separate
    places on the spec — `spec.timePartition` (one, with a granularity) and
    `spec.partitions` (any number) — so both are read here. Verified against a
    factory-published version on 2026-10-02:

        spec.timePartition = {"key": "date", "granularity": "DAY",
                              "value": {"timeValue": "2026-10-02T00:00:00Z"}}
    """
    out: dict[str, str] = {}
    try:
        spec = a.to_dict().get("spec", {})
    except (AttributeError, TypeError):
        return out

    tp = spec.get("timePartition") or {}
    key = tp.get("key")
    if key:
        out[key] = str((tp.get("value") or {}).get("timeValue") or "")[:10]

    raw = (spec.get("partitions") or {}).get("value") or {}
    for name, val in raw.items():
        if isinstance(val, dict):
            out[name] = str(val.get("staticValue") or val.get("timeValue") or "")
        else:
            out[name] = str(val)
    return out


async def list_versions(
    artifact_name: str,
    project: str | None = None,
    domain: str | None = None,
    limit: int = 10,
) -> list[AssetVersion]:
    """All known versions of an artifact, newest first."""
    found: list[AssetVersion] = []
    try:
        async for a in remote.Artifact.listall.aio(name=artifact_name, limit=limit):
            uri = _blob_uri(a)
            if not uri:
                # Deliberately skip rather than falling back to ``a.url``: the
                # console URL is an https:// address that sends File/Dir
                # downloads through fsspec's HTTP filesystem, which blows up
                # inside a task pod. A version we cannot resolve to object
                # storage is worse than no version at all.
                continue
            run_name, action_name = _source_ids(a)
            found.append(
                AssetVersion(
                    artifact_name=artifact_name,
                    path=uri,
                    run_name=run_name,
                    action_name=action_name,
                    version=str(getattr(a, "version", "") or ""),
                    partitions=_partitions(a),
                )
            )
    except Exception:
        # An artifact nobody has built yet is a normal state, not an error:
        # the lineage dashboard renders it as an empty station.
        return []
    return found[:limit]


async def latest(
    artifact_name: str, project: str | None = None, domain: str | None = None
) -> AssetVersion:
    versions = await list_versions(artifact_name, project=project, domain=domain, limit=1)
    if not versions:
        raise FileNotFoundError(f"no version of artifact {artifact_name!r} found in the registry")
    return versions[0]
