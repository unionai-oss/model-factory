"""Value-ledger aggregations (pure functions over record lists)."""

from resource_tuner.tune_store import savings_of, savings_series, task_registry, totals


def _proposal(ts, task, prior_mem, prop_mem, prior_cpu=4, prop_cpu=2, source="model"):
    return {
        "kind": "proposal", "ts": ts, "task_id": task,
        "prior": {"cpu": prior_cpu, "memory": prior_mem},
        "proposal": {"cpu": prop_cpu, "memory": prop_mem},
        "source": source,
    }


def _outcome(ts, task, ok=True, oom=False, peak=300.0):
    return {"kind": "outcome", "ts": ts, "task_id": task, "ok": ok, "oom": oom,
            "peak_rss_mib": peak}


RECORDS = [
    _proposal(1, "wf.sessionize", "8Gi", "2Gi"),          # saves 6Gi + 2 cores
    _outcome(2, "wf.sessionize", peak=900.0),
    _proposal(3, "wf.churn", "8Gi", "4Gi"),               # saves 4Gi + 2 cores
    _outcome(4, "wf.churn", ok=False, oom=True, peak=None),
    _proposal(5, "wf.embed", "1Gi", "4Gi"),               # ADDS 3Gi (safety)
]


def test_savings_of_signs():
    mem, cpu = savings_of(RECORDS[0])
    assert mem == 6 * 1024 and cpu == 2
    mem, _ = savings_of(RECORDS[4])
    assert mem == -3 * 1024  # headroom added counts against savings, honestly


def test_savings_series_is_cumulative_and_proposal_only():
    s = savings_series(RECORDS)
    assert len(s["ts"]) == 3  # outcomes don't move the savings ledger
    assert s["cum_mem_saved_mib"] == [6144.0, 10240.0, 7168.0]
    assert s["cum_cpu_saved"][-1] == 2 + 2 + 2  # every proposal trims 2 cores


def test_registry_rolls_up_per_task():
    reg = {r["task_id"]: r for r in task_registry(RECORDS)}
    assert set(reg) == {"wf.sessionize", "wf.churn", "wf.embed"}
    assert reg["wf.sessionize"]["fit"] == 1
    assert reg["wf.sessionize"]["last_peak_rss_mib"] == 900.0
    assert reg["wf.churn"]["oom"] == 1
    assert reg["wf.embed"]["outcomes"] == 0
    # newest-first ordering
    assert task_registry(RECORDS)[0]["task_id"] == "wf.embed"


def test_totals():
    t = totals(RECORDS)
    assert t["proposals"] == 3 and t["distinct_tasks"] == 3
    assert t["outcomes"] == 2 and t["outcome_oom_count"] == 1
    assert t["outcome_fit_rate"] == 0.5
    assert t["cum_mem_saved_mib"] == 7168.0
    # $-ledger: net memory+cpu trimming must price out positive
    assert t["cum_dollars_saved_per_hr"] > 0
    assert totals([])["outcome_fit_rate"] is None
