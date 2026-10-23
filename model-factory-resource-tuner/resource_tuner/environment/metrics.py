"""Pod-metrics cross-check via flyteplugins-union (branch niels/get-metrics).

`Metrics.get_for_action(run_name, action_name)` answers the dataplane's
Prometheus series for a task pod — the same numbers the Union UI charts and
the numbers the PRD's production reward would use. In this prototype they
CROSS-CHECK the harness's rusage self-report rather than drive the reward:

- pod "used memory" is the cgroup charge including page cache, so it
  overstates the working set (but is what OOM decisions act on);
- `used_cpu_avg` is an irate over 5m, smoothing bursts;
- attempts under ~30s answer OutOfRange — the corpus holds every workload
  ≥60s specifically so this path has data.

The plugin is an optional dependency (private repo); everything degrades to
rusage-only when it is absent.
"""

from __future__ import annotations

MEMORY_METRICS = ["request_memory_bytes", "used_memory_bytes_avg"]
CPU_METRICS = ["request_cpu", "used_cpu_avg"]


def metrics_available() -> bool:
    try:
        from flyteplugins.union.remote import Metrics  # noqa: F401

        return True
    except ImportError:
        return False


def _peak(series_list) -> float | None:
    values = [v for s in series_list for _, v in s.values]
    return max(values) if values else None


async def pod_peaks_for_action(run_name: str, action_name: str | None = None) -> dict:
    """{'peak_memory_mib': float|None, 'peak_cpu': float|None, 'error': str}.

    Post-hoc: one call returns the whole attempt window for a finished
    action. Callers treat None as "no data" (short attempt, scrape gap,
    plugin missing) and fall back to the harness self-report.
    """
    if not metrics_available():
        return {"peak_memory_mib": None, "peak_cpu": None, "error": "flyteplugins-union not installed"}

    from flyteplugins.union.remote import Metrics

    try:
        metrics = await Metrics.get_for_action.aio(
            run_name=run_name, action_name=action_name, metrics=MEMORY_METRICS + CPU_METRICS
        )
    except Exception as e:  # noqa: BLE001 — OutOfRange for short attempts etc.
        return {"peak_memory_mib": None, "peak_cpu": None, "error": f"{type(e).__name__}: {e}"}

    by_name = {r.name: r for r in metrics}
    out: dict = {"error": ""}
    mem = by_name.get("used_memory_bytes_avg")
    out["peak_memory_mib"] = (
        _peak(mem.series) / (1024 * 1024) if mem and not mem.error and _peak(mem.series) else None
    )
    cpu = by_name.get("used_cpu_avg")
    out["peak_cpu"] = _peak(cpu.series) if cpu and not cpu.error else None
    errors = "; ".join(f"{r.name}: {r.error}" for r in metrics if r.error)
    out["error"] = errors
    return out


async def harness_action_peaks(run_name: str) -> list[dict]:
    """Pod peaks for every harness action in a run.

    Child actions carry auto-generated names; find the harness's by task
    name via Action.listall (per-run listing is the only actions query the
    backend supports), then fetch metrics per action.
    """
    import flyte.remote

    peaks: list[dict] = []
    async for action in flyte.remote.Action.listall.aio(for_run_name=run_name):
        task_name = ""
        try:
            task_name = action.pb2.metadata.task.id.name
        except AttributeError:
            pass
        if not task_name.endswith("run_generated"):
            continue
        row = await pod_peaks_for_action(run_name, action.name)
        row["action_name"] = action.name
        peaks.append(row)
    return peaks
