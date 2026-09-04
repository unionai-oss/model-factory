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

# Public app endpoints, usable from anywhere WITH the LLM_SERVICE_API_KEY
# bearer token (the gateway is OIDC-gated otherwise). Preferred: the
# gateway wakes scale-to-zero apps reliably, and the svc DNS name
# disappears entirely when the platform unassigns an idle app (observed
# 2026-09-03: "Service marked for deletion" → NXDOMAIN).
TEACHERS_PUBLIC: dict[str, str] = {
    "qwen38-27b": "https://qwen38-27b-llm-service-development.apps.demo.hosted.unionai.cloud",
    "glm-5-2": "https://glm-5-2-llm-service-development.apps.demo.hosted.unionai.cloud",
}
# In-cluster service DNS — the keyless fallback.
TEACHERS_SVC: dict[str, str] = {
    "qwen38-27b": "http://qwen38-27b.llm-service-development.svc.cluster.local",
    "glm-5-2": "http://glm-5-2.llm-service-development.svc.cluster.local",
}
DEFAULT_TEACHER = "qwen38-27b"


class TeacherError(RuntimeError):
    pass


def _api_key() -> str:
    return os.environ.get("LLM_SERVICE_API_KEY", "")


def resolve_teacher(name_or_url: str | None = None) -> str:
    """Teacher name or URL → base URL.

    Precedence: RT_TEACHER_URL override → public endpoint when
    LLM_SERVICE_API_KEY is present → in-cluster svc DNS.
    """
    override = os.environ.get("RT_TEACHER_URL")
    if override:
        return override.rstrip("/")
    key = name_or_url or DEFAULT_TEACHER
    if key.startswith("http"):
        return key.rstrip("/")
    table = TEACHERS_PUBLIC if _api_key() else TEACHERS_SVC
    try:
        return table[key]
    except KeyError:
        raise TeacherError(f"unknown teacher {key!r}; choose from {sorted(table)}")


def _headers() -> dict:
    headers = {"Content-Type": "application/json"}
    if _api_key():
        headers["Authorization"] = f"Bearer {_api_key()}"
    return headers


def _get(url: str, timeout: float = 30) -> dict:
    req = urllib.request.Request(url, headers=_headers())
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read()
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        # The auth gateway answers non-JSON (a signin redirect page) when
        # the bearer token is missing/invalid — say so instead of looping.
        raise TeacherError(
            f"{url} answered non-JSON (auth redirect? bad LLM_SERVICE_API_KEY?): "
            f"{body[:120]!r}"
        )


def wait_until_ready(base_url: str, deadline_s: float = 1800, poll_s: float = 15) -> None:
    """Poll /v1/models until the server answers. Scale-from-zero can take
    15+ minutes when the wake also provisions a fresh GPU node (observed:
    image pull + L40S node scale-up + 18GB weight load blew a 900s
    deadline), so the deadline is generous."""
    started = time.monotonic()
    last: Exception | None = None
    while time.monotonic() - started < deadline_s:
        try:
            _get(f"{base_url}/v1/models", timeout=20)
            return
        except TeacherError:
            raise  # auth-shaped failure: polling will never fix it
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
        f"{base_url}/v1/chat/completions", data=body, headers=_headers(), method="POST"
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
