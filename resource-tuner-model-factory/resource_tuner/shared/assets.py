"""Artifact-version helpers (ported from basic-model-factory's assets).

Two hard-won rules baked in:
- resolve the s3 blob URI from the structured spec, NEVER `a.url` (the
  console URL 404s fsspec inside pods);
- `Artifact.source` is a display string; run/action identifiers live in
  the spec.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AssetVersion:
    artifact_name: str
    path: str  # object-store URI of the payload
    run_name: str
    action_name: str
    created_at: str = ""


def _blob_uri(a) -> str | None:
    try:
        spec = a.to_dict()["spec"]
        return spec["value"]["scalar"]["blob"]["uri"]
    except (AttributeError, KeyError, TypeError):
        return None


def _source_ids(a) -> tuple[str, str]:
    try:
        action = a.to_dict()["spec"]["source"]["taskAction"]["action"]
        return str(action["run"]["name"] or ""), str(action.get("name", "") or "")
    except (AttributeError, KeyError, TypeError):
        return "", ""


def _created_at(a) -> str:
    try:
        return str(a.to_dict().get("metadata", {}).get("createdAt", "") or "")
    except (AttributeError, TypeError):
        return ""


async def list_versions(artifact_name: str, limit: int = 50) -> list[AssetVersion]:
    """Versions of an artifact, newest-first, blob-resolvable only."""
    import flyte.remote as remote

    found: list[AssetVersion] = []
    async for a in remote.Artifact.listall.aio(name=artifact_name, limit=limit):
        uri = _blob_uri(a)
        if not uri:
            continue  # unresolvable beats a console URL that breaks downloads
        run_name, action_name = _source_ids(a)
        found.append(
            AssetVersion(
                artifact_name=artifact_name,
                path=uri,
                run_name=run_name,
                action_name=action_name,
                created_at=_created_at(a),
            )
        )
    found.sort(key=lambda v: v.created_at, reverse=True)
    return found


async def latest_version(artifact_name: str) -> AssetVersion | None:
    versions = await list_versions(artifact_name, limit=10)
    return versions[0] if versions else None
