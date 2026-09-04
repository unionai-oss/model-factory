"""Append-only metrics store for the tune service.

State is a remote directory of immutable JSON shards — one file per
record, unique names, nothing ever rewritten — read back as a
`flyte.io.Dir`. That makes the store append-only by construction, safe
for a scale-to-zero app (replicas die; the Dir persists) and for
concurrent writers (no shared file to contend on).

Two record kinds:
- ``proposal``: every /v1/propose answer (task registry + requested
  savings vs the prior);
- ``outcome``: reported run results (fit/oom + measured peak), posted by
  callers (the @tune.resources decorator auto-reports when the wrapped
  task returns a dict containing ``peak_rss_mib``).

Aggregations are pure functions over record lists — unit-testable without
storage.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid

from . import pricing
from .policy.actions import parse_memory_to_mib

_DEFAULT_STORE = (
    "s3://union-oc-production-demo-raw/rt-tune-store/"
    f"{os.environ.get('RT_PROJECT', 'resource-tuner-model-factory')}/"
    f"{os.environ.get('RT_DOMAIN', 'development')}"
)


def store_uri() -> str:
    return os.environ.get("RT_TUNE_STORE", _DEFAULT_STORE).rstrip("/")


async def append_record(record: dict) -> str:
    """Write one immutable shard; returns its URI. Never overwrites —
    the name embeds nanotime + uuid."""
    import flyte.storage

    name = f"{record.get('kind', 'record')}-{time.time_ns()}-{uuid.uuid4().hex[:8]}.json"
    uri = f"{store_uri()}/{name}"
    await flyte.storage.put_stream(json.dumps(record, default=str).encode(), to_path=uri)
    return uri


class StoreReader:
    """Incremental reader: only fetches shards not seen before."""

    def __init__(self):
        self._seen: dict[str, dict] = {}

    async def load(self) -> list[dict]:
        import flyte.io
        import flyte.storage

        try:
            if not await flyte.storage.exists(store_uri()):
                return sorted(self._seen.values(), key=lambda r: r.get("ts", 0))
            d = flyte.io.Dir.from_existing_remote(store_uri())
            async for f in d.walk():
                path = getattr(f, "path", "")
                if not path or path in self._seen:
                    continue
                chunks = []
                async for chunk in flyte.storage.get_stream(path):
                    chunks.append(chunk)
                try:
                    self._seen[path] = json.loads(b"".join(chunks))
                except json.JSONDecodeError:
                    self._seen[path] = {"kind": "corrupt", "path": path}
        except Exception as e:  # noqa: BLE001 — a flaky list must not blank the page
            print(f"[tune-store] load failed (serving cached): {e}")
        return sorted(self._seen.values(), key=lambda r: r.get("ts", 0))


# ── pure aggregations ───────────────────────────────────────────────────


def _mem_mib(kwargs: dict | None) -> float | None:
    if not kwargs or "memory" not in kwargs:
        return None
    try:
        return parse_memory_to_mib(kwargs["memory"])
    except (ValueError, TypeError):
        return None


def _cpu(kwargs: dict | None) -> float | None:
    if not kwargs or "cpu" not in kwargs:
        return None
    v = kwargs["cpu"]
    try:
        return float(str(v).rstrip("m")) / (1000 if str(v).endswith("m") else 1)
    except ValueError:
        return None


def savings_of(record: dict) -> tuple[float | None, float | None]:
    """(mem_saved_mib, cpu_saved) vs the prior for one proposal record.
    Positive = tuned asked for less than the hard-coded prior; negative =
    tuning added safety headroom."""
    pm, tm = _mem_mib(record.get("prior")), _mem_mib(record.get("proposal"))
    pc, tc = _cpu(record.get("prior")), _cpu(record.get("proposal"))
    return (
        None if pm is None or tm is None else pm - tm,
        None if pc is None or tc is None else pc - tc,
    )


def dollars_saved_of(record: dict) -> float | None:
    """$/hr saved by one proposal vs its prior (pricing.py rates); None if
    either side is unpriceable."""
    prior = pricing.kwargs_dollars_per_hr(record.get("prior"))
    proposed = pricing.kwargs_dollars_per_hr(record.get("proposal"))
    if prior is None or proposed is None:
        return None
    return prior - proposed


def savings_series(records: list[dict]) -> dict:
    """Cumulative requested savings over time (proposal records only)."""
    ts, mem, cpu, usd = [], [], [], []
    m_total = c_total = d_total = 0.0
    for r in records:
        if r.get("kind") != "proposal":
            continue
        dm, dc = savings_of(r)
        m_total += dm or 0.0
        c_total += dc or 0.0
        d_total += dollars_saved_of(r) or 0.0
        ts.append(r.get("ts", 0))
        mem.append(m_total)
        cpu.append(c_total)
        usd.append(d_total)
    return {
        "ts": ts,
        "cum_mem_saved_mib": mem,
        "cum_cpu_saved": cpu,
        "cum_dollars_saved_per_hr": usd,
    }


def task_registry(records: list[dict]) -> list[dict]:
    """Per-task rollup: every task the service has ever been asked about."""
    reg: dict[str, dict] = {}
    for r in records:
        tid = r.get("task_id") or "?"
        row = reg.setdefault(
            tid,
            {
                "task_id": tid,
                "proposals": 0,
                "outcomes": 0,
                "fit": 0,
                "oom": 0,
                "last_prior": None,
                "last_proposal": None,
                "last_source": "",
                "last_peak_rss_mib": None,
                "last_ts": 0,
            },
        )
        row["last_ts"] = max(row["last_ts"], r.get("ts", 0))
        if r.get("kind") == "proposal":
            row["proposals"] += 1
            row["last_prior"] = r.get("prior")
            row["last_proposal"] = r.get("proposal")
            row["last_source"] = r.get("source", "")
        elif r.get("kind") == "outcome":
            row["outcomes"] += 1
            row["fit"] += 1 if r.get("ok") else 0
            row["oom"] += 1 if r.get("oom") else 0
            if r.get("peak_rss_mib") is not None:
                row["last_peak_rss_mib"] = r["peak_rss_mib"]
    return sorted(reg.values(), key=lambda x: -x["last_ts"])


def totals(records: list[dict]) -> dict:
    proposals = [r for r in records if r.get("kind") == "proposal"]
    outcomes = [r for r in records if r.get("kind") == "outcome"]
    series = savings_series(records)
    model_served = sum(1 for r in proposals if r.get("source") in ("model", "cache"))
    return {
        "proposals": len(proposals),
        "distinct_tasks": len({r.get("task_id") for r in proposals}),
        "model_or_cache_rate": (model_served / len(proposals)) if proposals else 0.0,
        "outcomes": len(outcomes),
        "outcome_fit_rate": (
            sum(1 for o in outcomes if o.get("ok")) / len(outcomes) if outcomes else None
        ),
        "outcome_oom_count": sum(1 for o in outcomes if o.get("oom")),
        "cum_mem_saved_mib": series["cum_mem_saved_mib"][-1] if series["ts"] else 0.0,
        "cum_cpu_saved": series["cum_cpu_saved"][-1] if series["ts"] else 0.0,
        # $/hr the fleet of tuned requests is cheaper than its priors —
        # multiply by task runtime to get real dollars.
        "cum_dollars_saved_per_hr": (
            series["cum_dollars_saved_per_hr"][-1] if series["ts"] else 0.0
        ),
    }


def fire_and_forget(coro, holder: set) -> None:
    """Schedule a store write without blocking the request path; the
    holder set keeps a strong ref so asyncio can't GC it mid-write."""
    task = asyncio.get_running_loop().create_task(coro)
    holder.add(task)
    task.add_done_callback(holder.discard)
