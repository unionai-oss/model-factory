"""Client for the Union-hosted teacher LLMs (llama.cpp, OpenAI-compatible).

The llm-service apps (project `llm-service`, demo.hosted) expose `/v1` per
llama-server. Two access paths:

- IN-CLUSTER (the synthetic-data task): internal service DNS —
  `http://<app>.llm-service-development.svc.cluster.local` — because the
  public `*.apps.demo.hosted...` URL sits behind the OIDC gateway and
  answers task pods with a login redirect, not JSON.
- Locally: the public URL only works with a browser session; use
  RT_TEACHER_URL to point at a port-forward or any OpenAI-compatible server.

The apps scale to zero and a 27B llama.cpp server takes minutes to come up,
so `wait_until_ready` polls /v1/models before the first real request
(requests sent mid-scale-up are dropped at the gateway, not queued).
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

TEACHERS: dict[str, str] = {
    "qwen38-27b": "http://qwen38-27b.llm-service-development.svc.cluster.local",
    "glm-5-2": "http://glm-5-2.llm-service-development.svc.cluster.local",
}
DEFAULT_TEACHER = "qwen38-27b"


class TeacherError(RuntimeError):
    pass


def resolve_teacher(name_or_url: str | None = None) -> str:
    """Teacher name or URL → base URL. RT_TEACHER_URL overrides everything."""
    override = os.environ.get("RT_TEACHER_URL")
    if override:
        return override.rstrip("/")
    key = name_or_url or DEFAULT_TEACHER
    if key.startswith("http"):
        return key.rstrip("/")
    try:
        return TEACHERS[key]
    except KeyError:
        raise TeacherError(f"unknown teacher {key!r}; choose from {sorted(TEACHERS)}")


def _get(url: str, timeout: float = 30) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read())


def wait_until_ready(base_url: str, deadline_s: float = 900, poll_s: float = 15) -> None:
    """Poll /v1/models until the server answers; scale-from-zero for the
    27B takes a few minutes (weights load into VRAM)."""
    started = time.monotonic()
    last: Exception | None = None
    while time.monotonic() - started < deadline_s:
        try:
            _get(f"{base_url}/v1/models", timeout=20)
            return
        except Exception as e:  # noqa: BLE001 — cold start
            last = e
            time.sleep(poll_s)
    raise TeacherError(
        f"teacher {base_url} not reachable within {deadline_s:.0f}s (last: {last}). "
        "If calling from outside the cluster, the public app URL requires "
        "browser auth — set RT_TEACHER_URL or run in-cluster."
    )


def chat(
    base_url: str,
    messages: list[dict],
    max_tokens: int = 4096,
    temperature: float = 0.7,
    timeout: float = 600,
) -> str:
    """One chat completion; returns the assistant text (content only —
    llama.cpp puts hybrid-thinking traces in reasoning_content, which we
    deliberately drop)."""
    body = json.dumps(
        {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            # llama.cpp accepts and ignores model for single-model servers
            "model": "default",
            # The hosted presets default to reasoning_effort=medium; a
            # thinking model spends the whole token budget in
            # reasoning_content and returns EMPTY content (observed: 6/6
            # teacher responses with "no JSON object"). Data generation
            # wants the answer, not the chain of thought.
            "chat_template_kwargs": {"reasoning_effort": "none", "enable_thinking": False},
        }
    ).encode()
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            out = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        raise TeacherError(f"teacher HTTP {e.code}: {e.read()[:300]!r}")
    except Exception as e:  # noqa: BLE001
        raise TeacherError(f"teacher request failed: {e}")
    try:
        return out["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError):
        raise TeacherError(f"malformed completion response: {str(out)[:300]}")
