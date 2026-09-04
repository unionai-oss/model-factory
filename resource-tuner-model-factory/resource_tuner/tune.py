"""`@tune.resources` — opt a task into model-proposed resources.

The PRD's one-line adoption surface:

    from resource_tuner import tune

    @tune.resources
    @env.task
    async def train_model(...): ...

    @env.task
    async def driver(...):
        await train_model(...)   # invocation transparently becomes
                                 # train_model.override(resources=<proposal>)(...)

At call time the wrapper assembles the estimation context (the decorated
task's own source code + the env's declared resources as the prior), asks
the tune service for a proposal, and invokes the underlying task with
`.override(resources=flyte.Resources(**proposal))`. On ANY failure —
service down, timeout, invalid proposal — the task runs unchanged on its
prior (the PRD's DEGRADED path): tuning can slow a call slightly, but it
can never break one.
"""

from __future__ import annotations

import inspect
import json
import os
import urllib.request

TUNE_TIMEOUT_S = float(os.environ.get("RT_TUNE_TIMEOUT", "120"))


def service_url() -> str:
    """In-cluster svc DNS by default; RT_TUNE_URL overrides (local runs,
    port-forwards, or the public endpoint + a curl-shaped UA)."""
    override = os.environ.get("RT_TUNE_URL")
    if override:
        return override.rstrip("/")
    project = os.environ.get("RT_PROJECT", "resource-tuner-model-factory")
    domain = os.environ.get("RT_DOMAIN", "development")
    return f"http://rt-tune.{project}-{domain}.svc.cluster.local"


def request_proposal(
    task_id: str, source_code: str, input_profile: str = "", prior: dict | None = None
) -> dict | None:
    """One blocking proposal request; None on any failure (caller falls
    back to the prior). The User-Agent matters: the public endpoint sits
    behind Cloudflare, which 403s python-urllib's default UA."""
    body = json.dumps(
        {
            "task_id": task_id,
            "source_code": source_code,
            "input_profile": input_profile,
            "prior": prior or {},
        }
    ).encode()
    req = urllib.request.Request(
        f"{service_url()}/v1/propose",
        data=body,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "resource-tuner-tune-client/0.1",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=TUNE_TIMEOUT_S) as resp:
            out = json.loads(resp.read())
    except Exception as e:  # noqa: BLE001 — DEGRADED, never broken
        print(f"[tune] service unavailable for {task_id}: {e} — running on the prior")
        return None
    if out.get("source") == "fallback_prior" or not out.get("proposal"):
        print(f"[tune] no proposal for {task_id} ({out.get('error', 'no detail')}) — prior")
        return None
    print(f"[tune] {task_id}: {out['proposal']} (source={out['source']})")
    return out["proposal"]


class TunedTask:
    """Callable wrapper: each invocation resolves a proposal, then calls
    the wrapped task with `.override(resources=...)`."""

    def __init__(self, task, enabled: bool = True):
        self._task = task
        self._enabled = enabled
        try:
            self._source = inspect.getsource(task.func)
        except Exception:  # noqa: BLE001 — no source, no context
            self._source = ""
        self.name = getattr(task, "name", getattr(task, "__name__", "tuned-task"))

    def _prior(self) -> dict:
        res = getattr(getattr(self._task, "parent_env", None), "resources", None)
        prior: dict = {}
        for attr in ("cpu", "memory", "gpu"):
            v = getattr(res, attr, None)
            if v is not None:
                prior[attr] = v if isinstance(v, (int, float, str)) else str(v)
        return prior

    def __call__(self, *args, **kwargs):
        import asyncio

        import flyte

        async def invoke():
            if not self._enabled or not self._source:
                return await self._task(*args, **kwargs)
            proposal = await asyncio.to_thread(
                request_proposal, self.name, self._source, "", self._prior()
            )
            if proposal is None:
                return await self._task(*args, **kwargs)  # DEGRADED: prior
            try:
                overridden = self._task.override(resources=flyte.Resources(**proposal))
            except Exception as e:  # noqa: BLE001 — bad kwargs never break the call
                print(f"[tune] proposal rejected by flyte.Resources: {e} — prior")
                return await self._task(*args, **kwargs)
            return await overridden(*args, **kwargs)

        return invoke()

    def override(self, *args, **kwargs):
        """Explicit user override outbids tuning (PRD: user stays in charge)."""
        return self._task.override(*args, **kwargs)


def resources(task=None, *, enabled: bool = True):
    """Decorator (bare or parameterized): `@tune.resources` /
    `@tune.resources(enabled=False)`."""
    if task is None:
        return lambda t: TunedTask(t, enabled=enabled)
    return TunedTask(task, enabled=enabled)
