"""The episode harness: one generic task that runs any generated workload.

The RL environment needs to execute arbitrary corpus tasks under arbitrary
resource requests. Rather than deploying a task per generated snippet, ONE
harness task is deployed and every episode overrides its resources with the
policy's proposal:

    harness = run_generated.override(resources=flyte.Resources(**proposal))

The harness exec()s the generated code (our own templates — this is the
environment executing its own corpus, not user code) and self-reports peak
RSS via getrusage. That self-report is the reward's fallback ground truth;
pod-level metrics (environment/metrics.py) are the cross-check — pod
"used memory" includes page cache, so RSS is the cleaner working-set number
while the pod number is what OOM decisions are actually made on.

retries=0 is load-bearing: an underprovisioned episode OOMs, and a retry
would just OOM again while distorting the episode's cost.
"""

from __future__ import annotations

import time

import flyte

from ..config import HARNESS_TIMEOUT_S, cluster_env_vars
from ..shared.images import harness_image

harness_env = flyte.TaskEnvironment(
    name="rt-harness",
    image=harness_image,
    # Env defaults only — every episode overrides these with the proposal.
    resources=flyte.Resources(cpu=1, memory="1Gi"),
    env_vars=cluster_env_vars(),
)


# max_queued_time is load-bearing: an unschedulable pod (a proposal no node
# satisfies) queues FOREVER otherwise — one such episode hung an eval for
# 75+ minutes. Queue timeout turns "can't schedule" into a failed episode.
@harness_env.task(
    retries=0,
    timeout=flyte.Timeout(max_runtime=HARNESS_TIMEOUT_S, max_queued_time=300),
    report=True,
)
async def run_generated(harness_code: str, task_id: str = "") -> dict:
    """Execute one generated workload and report what it actually used.

    Returns {ok, peak_rss_mib, duration_s, error}. An OOM never returns —
    the pod is killed and the caller sees the failed action instead (the
    report shows the code that was running when the pod died).
    """
    import resource

    from ..shared.reporting import Reporter, esc, ok_pill

    rep = Reporter("Episode harness", task_id or "<generated>")
    rep.p("Workload executing — an OOM kills this pod before a final report.")
    rep.raw(
        f'<pre style="background:#131316;padding:10px;border-radius:8px;font-size:11px;'
        f'color:#c9c9cf;overflow-x:auto">{esc(harness_code[:1200])}</pre>'
    )
    await rep.flush()

    namespace: dict = {}
    started = time.monotonic()
    ru0 = resource.getrusage(resource.RUSAGE_SELF)
    try:
        exec(compile(harness_code, task_id or "<generated>", "exec"), namespace)
        result = namespace["run"]()
        ok, error = True, ""
    except Exception as e:  # noqa: BLE001 — corpus bug, not an OOM
        result, ok, error = None, False, f"{type(e).__name__}: {e}"
    duration = time.monotonic() - started
    ru1 = resource.getrusage(resource.RUSAGE_SELF)
    peak_rss_mib = ru1.ru_maxrss / 1024  # linux: KiB
    # Sustained parallelism ≈ CPU-seconds / wall-seconds. This is the
    # oracle's cpu label for synthetic tasks (rusage has no CPU peak).
    cpu_s = (ru1.ru_utime - ru0.ru_utime) + (ru1.ru_stime - ru0.ru_stime)
    measured = {
        "ok": ok,
        "error": error,
        "peak_rss_mib": float(peak_rss_mib),
        "cpu_avg_cores": float(cpu_s / duration) if duration > 0 else 0.0,
        "duration_s": float(duration),
        "result": result or {},
    }
    rep.reset_body().raw(ok_pill(ok)).kv(
        {
            "peak RSS": f"{measured['peak_rss_mib']:.0f} MiB",
            "avg CPU": f"{measured['cpu_avg_cores']:.2f} cores",
            "duration": f"{measured['duration_s']:.0f}s",
            **({"error": error[:200]} if error else {}),
        }
    )
    await rep.flush()
    return measured
