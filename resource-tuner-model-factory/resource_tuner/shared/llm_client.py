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
    # Frontier-class teachers (deployed 2026-09-06): far stronger code
    # writers than the 27B, and — the point for synthetic diversity —
    # three DIFFERENT model families writing archetypes.
    "minimax-m3": "https://minimax-m3-llm-service-development.apps.demo.hosted.unionai.cloud",
    "qwen35-397b": "https://qwen35-397b-llm-service-development.apps.demo.hosted.unionai.cloud",
}
# In-cluster service DNS — the keyless fallback.
TEACHERS_SVC: dict[str, str] = {
    "qwen38-27b": "http://qwen38-27b.llm-service-development.svc.cluster.local",
    "glm-5-2": "http://glm-5-2.llm-service-development.svc.cluster.local",
    "minimax-m3": "http://minimax-m3.llm-service-development.svc.cluster.local",
    "qwen35-397b": "http://qwen35-397b.llm-service-development.svc.cluster.local",
}
DEFAULT_TEACHER = "qwen38-27b"


class TeacherError(RuntimeError):
    """Transport-level teacher failure.

    `status` carries the HTTP code when there was one (None for timeouts /
    resets), so callers can tell a persistently-down endpoint (504) from a
    request the endpoint rejected (400) — the difference between "drop this
    teacher from the pool" and "this prompt was bad".
    """

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class TeacherPool:
    """Round-robin over healthy teacher endpoints, with a circuit breaker.

    Round 13's 1M release (run upnz22h5) fed work to a fixed
    `idx % n_teachers` endpoint no matter how long that endpoint had been
    returning 504s: 277 archetypes died on a gateway that never recovered,
    each after burning its retry backoff. Retrying a DOWN service is not
    resilience — the fix is to stop sending to it.

    Health is per endpoint and counts only TRANSPORT failures. An endpoint
    that answers with unusable content is healthy but unproductive; that is
    a prompt problem, and dropping the endpoint would not fix it.
    """

    def __init__(self, names: list[str], trip_after: int = 6):
        self.names = list(names)
        self.trip_after = trip_after
        self.health = {
            n: {"fail": 0, "ok": 0, "errs": 0, "dead": False} for n in self.names
        }

    def live(self) -> list[int]:
        return [i for i, n in enumerate(self.names) if not self.health[n]["dead"]]

    def pick(self, idx: int) -> int | None:
        """Index into `names` for work item `idx`, or None if all are down."""
        live = self.live()
        return live[idx % len(live)] if live else None

    def note_ok(self, i: int) -> None:
        h = self.health[self.names[i]]
        h["fail"] = 0
        h["ok"] += 1

    def note_fail(self, i: int) -> bool:
        """Record a transport failure; True if this one tripped the breaker."""
        h = self.health[self.names[i]]
        h["fail"] += 1
        h["errs"] += 1
        # Never trip the LAST live endpoint: a degraded pool still beats a
        # pool with nowhere to send work.
        if h["fail"] >= self.trip_after and not h["dead"] and len(self.live()) > 1:
            h["dead"] = True
            return True
        return False

    def revive(self) -> list[str]:
        """Half-open probe: give tripped endpoints another chance (they do
        come back — these apps scale to zero), but pre-load the failure
        counter so a still-dead one re-trips after 2 strikes instead of
        absorbing another full `trip_after` of work. Returns revived names."""
        revived = []
        for name, h in self.health.items():
            if h["dead"]:
                h["dead"] = False
                h["fail"] = max(self.trip_after - 2, 0)
                revived.append(name)
        return revived


def _api_key() -> str:
    return os.environ.get("LLM_SERVICE_API_KEY", "")


def _in_cluster() -> bool:
    return bool(os.environ.get("KUBERNETES_SERVICE_HOST")) or os.path.exists(
        "/var/run/secrets/kubernetes.io"
    )


def resolve_teacher_candidates(name_or_url: str | None = None) -> list[str]:
    """Teacher name or URL → ordered candidate base URLs.

    In-cluster, svc DNS comes FIRST: task pods are rejected (403) at the
    public app gateway regardless of bearer token — observed on
    basic-model-factory and re-confirmed 2026-09-04 when a synthetic run
    sat "waking teacher" against the public URL while the app was Active.
    The public endpoint (+ API key) is the off-cluster path and the
    in-cluster fallback for when an unassigned app has no svc DNS.
    """
    override = os.environ.get("RT_TEACHER_URL")
    if override:
        return [override.rstrip("/")]
    key = name_or_url or DEFAULT_TEACHER
    if key.startswith("http"):
        return [key.rstrip("/")]
    if key not in TEACHERS_SVC:
        raise TeacherError(f"unknown teacher {key!r}; choose from {sorted(TEACHERS_SVC)}")
    candidates: list[str] = []
    if _in_cluster():
        candidates.append(TEACHERS_SVC[key])
        if _api_key():
            candidates.append(TEACHERS_PUBLIC[key])
    else:
        if _api_key():
            candidates.append(TEACHERS_PUBLIC[key])
        else:
            candidates.append(TEACHERS_SVC[key])  # honest failure off-cluster
    return candidates


def resolve_teacher(name_or_url: str | None = None) -> str:
    """First candidate (kept for callers that need a single URL)."""
    return resolve_teacher_candidates(name_or_url)[0]


def _headers() -> dict:
    # The User-Agent is load-bearing: the public app endpoints sit behind
    # Cloudflare, which bans python-urllib's default UA with 403 error 1010
    # BEFORE auth is evaluated (proven by probe runs uvpm5vx2bhf2lvgc4n8t /
    # umsxdmqzdtf9xbn8g7sc: same request 403s with default UA, 200s with a
    # normal one — key valid in both).
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "resource-tuner-llm-client/0.1",
    }
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


def wait_until_ready(
    candidates: str | list[str],
    deadline_s: float = 1800,
    poll_s: float = 15,
    on_status=None,
) -> str:
    """Poll candidates' /health until one answers 200; return that base URL.

    Mirrors llm-service's own smoke tests: llama-server answers 200 once
    the model is loaded and 503 while loading; the gateway answers 502/504
    while a scaled-to-zero pod comes up. Every poll's status is recorded
    per candidate (and streamed to `on_status`), because a swallowed
    status is exactly how a "stuck at waking teacher" report happens.
    Scale-from-zero can exceed 15 min when it also provisions a GPU node,
    so the deadline is generous.
    """
    urls = [candidates] if isinstance(candidates, str) else list(candidates)
    started = time.monotonic()
    last: dict[str, str] = {u: "not tried" for u in urls}
    while time.monotonic() - started < deadline_s:
        for url in urls:
            try:
                _get(f"{url}/health", timeout=20)
                return url
            except urllib.error.HTTPError as e:
                last[url] = f"HTTP {e.code}"
            except Exception as e:  # noqa: BLE001 — DNS/conn during cold start
                last[url] = f"{type(e).__name__}: {str(e)[:120]}"
        status = " | ".join(f"{u.split('/')[2]}: {s}" for u, s in last.items())
        print(f"[teacher] waiting — {status}")
        if on_status:
            try:
                on_status(status)
            except Exception:  # noqa: BLE001 — reporting must not break the wait
                pass
        time.sleep(poll_s)
    raise TeacherError(
        f"no teacher endpoint became healthy within {deadline_s:.0f}s — "
        + "; ".join(f"{u} → {s}" for u, s in last.items())
    )


# Gateway/activator hiccups are transient and MUST NOT burn a work item:
# a 504 "activator request timeout" cost 59 archetypes in the first
# round-13 release before this retry existed.
_RETRY_STATUS = {429, 500, 502, 503, 504}

# ...but retrying is only worth it for a HICCUP. Round 13 proved the other
# case: in run upnz22h5, ~290 archetypes each burned 5+10+20s of backoff and
# then failed anyway (retries 1/2/3 failed at essentially the same rate) —
# ~2.8 hours of pure sleeping against the concurrency limit for zero
# recovered work, because the endpoints were persistently unavailable, not
# hiccuping. So the per-call retry budget is now small and the CALLER is
# expected to stop sending to an endpoint that keeps failing (see the
# circuit breaker in training/stations.py).
def chat(
    base_url: str,
    messages: list[dict],
    max_tokens: int = 4096,
    temperature: float = 0.7,
    timeout: float = 600,
    retries: int = 2,
    backoff_s: float = 2.0,
) -> str:
    """One chat completion; returns the assistant text (content only —
    llama.cpp puts hybrid-thinking traces in reasoning_content, which we
    deliberately drop). Transient gateway statuses are retried with
    exponential backoff; anything else fails fast."""
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
    last: Exception | None = None
    for attempt in range(max(retries, 0) + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                out = json.loads(resp.read())
            break
        except urllib.error.HTTPError as e:
            last = TeacherError(f"teacher HTTP {e.code}: {e.read()[:300]!r}", status=e.code)
            if e.code not in _RETRY_STATUS or attempt == retries:
                raise last
        except Exception as e:  # noqa: BLE001 — timeouts/resets are transient too
            last = TeacherError(f"teacher request failed: {e}")
            if attempt == retries:
                raise last
        sleep_s = backoff_s * (2**attempt)
        print(f"[teacher] {last} — retry {attempt + 1}/{retries} in {sleep_s:.0f}s")
        time.sleep(sleep_s)
    else:  # pragma: no cover — loop always breaks or raises
        raise last or TeacherError("teacher request failed")
    try:
        return out["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError):
        raise TeacherError(f"malformed completion response: {str(out)[:300]}")
