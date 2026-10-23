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


def report_outcome(
    task_id: str,
    requested: dict | None,
    ok: bool = True,
    oom: bool = False,
    peak_rss_mib: float | None = None,
    duration_s: float | None = None,
) -> None:
    """Best-effort outcome post to the value ledger; never raises."""
    body = json.dumps(
        {
            "task_id": task_id,
            "ok": ok,
            "oom": oom,
            "peak_rss_mib": peak_rss_mib,
            "duration_s": duration_s,
            "requested": requested,
        }
    ).encode()
    req = urllib.request.Request(
        f"{service_url()}/v1/outcome",
        data=body,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "resource-tuner-tune-client/0.1",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            resp.read()
    except Exception as e:  # noqa: BLE001 — the ledger is best-effort
        print(f"[tune] outcome report failed for {task_id}: {e}")


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
        # Task-level resources win; parent_env is a WEAKREF on the v2
        # template — dereference it before reading the env's declaration.
        res = getattr(self._task, "resources", None)
        if res is None:
            env_ref = getattr(self._task, "parent_env", None)
            env = env_ref() if callable(env_ref) else env_ref
            res = getattr(env, "resources", None)
        prior: dict = {}
        for attr in ("cpu", "memory", "gpu"):
            v = getattr(res, attr, None)
            if v is not None:
                prior[attr] = v if isinstance(v, (int, float, str)) else str(v)
        return prior

    def __call__(self, *args, **kwargs):
        import asyncio
        import time

        import flyte

        async def invoke():
            if not self._enabled or not self._source:
                return await self._task(*args, **kwargs)
            # The invocation's actual inputs ARE the input profile — the
            # same task called with rows=10_000 vs rows=10_000_000 should
            # get different proposals.
            profile = f"invoked with args={args!r} kwargs={kwargs!r}"[:500]
            proposal = await asyncio.to_thread(
                request_proposal, self.name, self._source, profile, self._prior()
            )
            if proposal is None:
                return await self._task(*args, **kwargs)  # DEGRADED: prior
            try:
                overridden = self._task.override(resources=flyte.Resources(**proposal))
            except Exception as e:  # noqa: BLE001 — bad kwargs never break the call
                print(f"[tune] proposal rejected by flyte.Resources: {e} — prior")
                return await self._task(*args, **kwargs)
            started = time.monotonic()
            try:
                result = await overridden(*args, **kwargs)
            except Exception as e:
                # Best-effort ledger entry for the failure, then re-raise —
                # tuning never swallows a task error.
                is_oom = "oom" in type(e).__name__.lower() or "OOMKilled" in str(e)
                await asyncio.to_thread(
                    report_outcome, self.name, proposal,
                    ok=False, oom=is_oom, duration_s=time.monotonic() - started,
                )
                raise
            # Convention: tasks that return a dict with peak_rss_mib get
            # their outcome auto-reported to the value ledger.
            if isinstance(result, dict) and "peak_rss_mib" in result:
                await asyncio.to_thread(
                    report_outcome, self.name, proposal,
                    ok=True, oom=False,
                    peak_rss_mib=result["peak_rss_mib"],
                    duration_s=time.monotonic() - started,
                )
            return result

        return invoke()

    def __getattr__(self, name):
        # The in-pod runtime resolves this module attribute where the real
        # task used to live and expects the full TaskTemplate surface
        # (native_interface, execute, report, ...). Forward everything to
        # the wrapped task — the runtime path never touches __call__, so
        # the child pod runs the task body without re-tuning.
        if name == "_task":
            raise AttributeError(name)
        return getattr(self._task, name)

    def override(self, *args, **kwargs):
        """Explicit user override outbids tuning (PRD: user stays in charge)."""
        return self._task.override(*args, **kwargs)


def resources(task=None, *, enabled: bool = True):
    """Decorator (bare or parameterized): `@tune.resources` /
    `@tune.resources(enabled=False)`."""
    if task is None:
        return lambda t: TunedTask(t, enabled=enabled)
    return TunedTask(task, enabled=enabled)
