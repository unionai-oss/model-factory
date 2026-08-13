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
    """Public URL of the serving app, from the control plane."""
    import flyte.remote

    return str(flyte.remote.App.get(app_name).endpoint)


def health(base_url: str) -> dict:
    with urllib.request.urlopen(f"{base_url}/health", timeout=60) as resp:
        return json.loads(resp.read())


def reload_checkpoint(base_url: str, checkpoint_path: str | None = None) -> dict:
    out = _post(f"{base_url}/reload", {"checkpoint_path": checkpoint_path}, timeout=600)
    if not out.get("ok"):
        raise InferenceServiceError(f"reload failed: {out}")
    return out


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
