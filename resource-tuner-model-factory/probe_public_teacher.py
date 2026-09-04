"""TEMPORARY diagnostic: call the qwen38-27b PUBLIC endpoint from a task pod
with the flyte API key, per the llm-service README pattern
(Authorization: Bearer $FLYTE_API_KEY). Tests whether in-cluster pods can
use the public app gateway at all, or get rejected regardless of bearer.

    uv run flyte --config .flyte/config.yaml run probe_public_teacher.py probe_public_endpoint
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

import flyte

from resource_tuner.config import LLM_SERVICE_SECRET
from resource_tuner.shared.images import driver_image

PUBLIC = "https://qwen38-27b-llm-service-development.apps.demo.hosted.unionai.cloud"
SVC = "http://qwen38-27b.llm-service-development.svc.cluster.local"

probe_env = flyte.TaskEnvironment(
    name="rt-probe",
    image=driver_image,
    resources=flyte.Resources(cpu=1, memory="1Gi"),
    secrets=[flyte.Secret(key=LLM_SERVICE_SECRET, as_env_var="LLM_SERVICE_API_KEY")],
)


def _call(base: str, path: str, bearer: str | None, body: dict | None = None,
          ua: str | None = None) -> dict:
    headers = {"Content-Type": "application/json"}
    if ua:
        headers["User-Agent"] = ua
    if bearer:
        headers["Authorization"] = f"Bearer {bearer}"
    req = urllib.request.Request(
        f"{base}{path}",
        data=json.dumps(body).encode() if body else None,
        headers=headers,
        method="POST" if body else "GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            raw = resp.read()
            return {"status": resp.status, "body": raw[:400].decode(errors="replace")}
    except urllib.error.HTTPError as e:
        return {"status": e.code, "body": e.read()[:400].decode(errors="replace")}
    except Exception as e:  # noqa: BLE001
        return {"status": None, "body": f"{type(e).__name__}: {e}"}


@probe_env.task(timeout=flyte.Timeout(max_runtime=600))
async def probe_public_endpoint() -> dict:
    import os

    key = os.environ.get("LLM_SERVICE_API_KEY", "")
    chat_body = {
        "model": "default",
        "max_tokens": 32,
        "messages": [{"role": "user", "content": "Say OK and nothing else."}],
    }
    out = {
        "api_key_present": bool(key),
        "api_key_len": len(key),
        "public_health_with_key": _call(PUBLIC, "/health", key),
        "public_models_with_key": _call(PUBLIC, "/v1/models", key),
        "public_chat_with_key": _call(PUBLIC, "/v1/chat/completions", key, chat_body),
        "public_health_key_curl_ua": _call(PUBLIC, "/health", key, ua="curl/8.7.1"),
        "public_chat_key_curl_ua": _call(PUBLIC, "/v1/chat/completions", key, chat_body, ua="curl/8.7.1"),
        "public_health_no_key": _call(PUBLIC, "/health", None),
        "svc_health_no_key": _call(SVC, "/health", None),
    }
    print(json.dumps(out, indent=2))
    return out
