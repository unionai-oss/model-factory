"""HTTP client for the inference team's serving app (stdlib only)."""

from __future__ import annotations

import json
import urllib.error
import urllib.request

_CHUNK = 8  # chats per request — keeps each request under ingress timeouts
_TIMEOUT_S = 280


class InferenceServiceError(RuntimeError):
    pass


def _post(url: str, payload: dict, timeout: float = _TIMEOUT_S) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:500]
        raise InferenceServiceError(f"{url} -> HTTP {e.code}: {body}") from e
    except Exception as e:
        raise InferenceServiceError(f"{url} -> {e}") from e


def resolve_endpoint(app_name: str = "mf-inference") -> str:
    """Base URL of the serving app.

    Task pods must use the internal service DNS — the apps gateway returns
    403 for pod-originated requests to the public URL (verified empirically).
    Outside the cluster, resolve the public endpoint from the control plane.
    """
    import os

    if os.environ.get("KUBERNETES_SERVICE_HOST"):
        project = os.environ.get("MF_PROJECT", "model-factory")
        domain = os.environ.get("MF_DOMAIN", "development")
        try:
            import flyte

            ctx = flyte.ctx()
            if ctx is not None:
                project, domain = ctx.action.project, ctx.action.domain
        except Exception:
            pass
        return f"http://{app_name}.{project}-{domain}.svc.cluster.local"
    import flyte.remote

    return str(flyte.remote.App.get(app_name).endpoint)


def health(base_url: str) -> dict:
    with urllib.request.urlopen(f"{base_url}/health", timeout=60) as resp:
        return json.loads(resp.read())


def wait_until_ready(base_url: str, deadline_s: float = 600, poll_s: float = 10) -> dict:
    """Poll /health until the app answers, and return that first health dict.

    The serving app scales to zero, so the first request after an idle period
    has to wait for a pod to be scheduled and started. Requests sent during
    that window are dropped by the gateway rather than queued, so callers must
    wake the app *before* asking it to do work.
    """
    import time

    started = time.monotonic()
    last: Exception | None = None
    while time.monotonic() - started < deadline_s:
        try:
            return health(base_url)
        except Exception as e:  # not up yet (cold start, activator timeout)
            last = e
            time.sleep(poll_s)
    raise InferenceServiceError(
        f"{base_url} did not become reachable within {deadline_s:.0f}s "
        f"(last error: {last}). The app may be unable to schedule — check its "
        f"accelerator against the tenant's available node pools."
    )


def reload_checkpoint(
    base_url: str,
    checkpoint_path: str | None = None,
    deadline_s: float = 1500,
    poll_s: float = 10,
    ready_s: float = 600,
) -> dict:
    """Point the service at ``checkpoint_path`` and wait until it serves it.

    ``/reload`` is fire-and-forget server-side (it returns immediately and
    loads in the background), so we poll ``/health`` until the requested
    checkpoint is live.

    Two failure modes this has to survive, both seen in practice:

    * The app is scaled to zero, so the POST dies at the gateway and never
      reaches the server — nothing starts loading. We wake the app first, and
      re-issue the POST whenever /health says the service is neither loading
      nor already serving the target. Assuming a timed-out POST means "loading
      started" is what made this poll a healthy-but-idle app until its
      deadline and then fail with a misleading message.
    * A load is genuinely in flight and the gateway cuts the POST off. Then
      /health reports ``loading`` and we simply wait.
    """
    import time

    started = time.monotonic()

    # Wake the app before asking it for work; requests sent while it is
    # scaling from zero are dropped, not queued.
    wait_until_ready(base_url, deadline_s=min(ready_s, deadline_s), poll_s=poll_s)

    def kick() -> dict:
        try:
            out = _post(f"{base_url}/reload", {"checkpoint_path": checkpoint_path}, timeout=120)
        except InferenceServiceError as e:
            # Gateway cut us off. Whether the server got it is unknown —
            # /health is the source of truth, and we re-kick if it did not.
            if "504" in str(e) or "timed out" in str(e).lower():
                return {"ok": True, "loading": True}
            raise
        if not out.get("ok"):
            raise InferenceServiceError(f"reload failed: {out}")
        return out

    out = kick()
    if not out.get("loading"):
        return out  # already serving the requested checkpoint

    unreachable_s = 0.0
    while time.monotonic() - started < deadline_s:
        time.sleep(poll_s)
        try:
            h = health(base_url)
        except Exception:
            # Briefly unreachable mid-reload is normal; permanently is not.
            unreachable_s += poll_s
            if unreachable_s > ready_s:
                raise InferenceServiceError(
                    f"{base_url} stopped responding for {unreachable_s:.0f}s during reload"
                )
            continue
        unreachable_s = 0.0
        if h.get("reload_error"):
            raise InferenceServiceError(f"reload failed server-side:\n{h['reload_error']}")
        if h.get("loaded") and (
            checkpoint_path is None or h.get("checkpoint_path") == checkpoint_path
        ):
            return {
                "ok": True,
                "base_model": h.get("base_model"),
                "checkpoint_path": h.get("checkpoint_path"),
            }
        if not h.get("loading"):
            # Idle and not serving what we asked for: our POST never landed.
            kick()
    raise InferenceServiceError(
        f"service not serving {checkpoint_path or 'latest checkpoint'} after {deadline_s:.0f}s"
    )


def generate(
    base_url: str,
    chats: list[list[dict]],
    *,
    use_adapter: bool = True,
    max_new_tokens: int = 512,
    checkpoint_path: str | None = None,
    do_sample: bool = False,
    temperature: float = 1.0,
) -> list[str]:
    """Generate completions for chat prompts, chunked across requests."""
    outs: list[str] = []
    for i in range(0, len(chats), _CHUNK):
        out = _post(
            f"{base_url}/generate",
            {
                "chats": chats[i : i + _CHUNK],
                "use_adapter": use_adapter,
                "max_new_tokens": max_new_tokens,
                "checkpoint_path": checkpoint_path,
                "do_sample": do_sample,
                "temperature": temperature,
            },
        )
        if "completions" not in out:
            raise InferenceServiceError(f"generate failed: {out}")
        outs.extend(out["completions"])
    return outs
