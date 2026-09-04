"""Evaluation: policy vs rule-based baseline, sim + real cluster episodes.

Three questions, in order of importance:
1. Schema validity — do proposals parse and bounds-check?
2. Does the policy beat the baseline on (success rate, waste) in sim?
3. Does simulator truth track pod truth? A small batch of REAL episodes
   runs on the cluster under the policy's proposals; harness rusage (and
   pod metrics, when the plugin is installed) measure the sim-to-real gap.
"""

from __future__ import annotations

import json
import statistics
import tempfile

import flyte
import flyte.io

from ..config import get_profile
from ..contracts import (
    ARTIFACT_EVAL_REPORT,
    ARTIFACT_TASK_CORPUS,
    ARTIFACT_TUNER_CHECKPOINT,
    EVAL_REPORT_KEYS,
    publish,
)
from ..shared import assets
from ..shared.reporting import GOOD, MUTED, Reporter, esc, ok_pill, pill
from ..environment.episodes import run_cluster_episode
from ..environment.metrics import harness_action_peaks, metrics_available
from ..environment.simulator import simulate_episode
from ..policy.actions import Proposal
from ..policy.parsing import try_extract_proposal
from ..rewards.rewards import overprovision_fraction
from ..training.baseline import baseline_proposal, fit_family_baseline
from .envs import driver_env


def _summarize(pairs: list[tuple[dict, Proposal]]) -> dict:
    """Sim-score (record, proposal) pairs into success/waste aggregates."""
    successes, wastes = [], []
    for record, proposal in pairs:
        ep = simulate_episode(
            proposal,
            float(record["true_peak_memory_mib"]),
            float(record["true_cpu_cores"]),
            int(record["duration_s"]),
        )
        successes.append(ep.ok)
        if ep.ok:
            wastes.append(
                100
                * (
                    overprovision_fraction(ep.requested_memory_mib, ep.peak_memory_mib)
                    + overprovision_fraction(ep.requested_cpu, ep.peak_cpu)
                )
                / 2
            )
    return {
        "success_rate": sum(successes) / len(successes) if successes else 0.0,
        "median_overprovision_pct": statistics.median(wastes) if wastes else None,
    }


async def _generate_proposals(
    records: list[dict], checkpoint_path: str
) -> list[Proposal | None]:
    """Greedy proposals via the reusable generator env.

    One `generate_proposal` child call per context — warm replicas batch
    them dynamically (see training/generator.py), so the fan-out costs one
    model load per replica instead of per eval, and generation runs in
    batched `model.generate` calls. Per-record failures (including a
    generator OOM) degrade to an invalid proposal rather than failing the
    eval: a missing proposal is exactly what schema_validity measures.
    """
    import asyncio

    import flyte.errors

    from .generator import generate_proposal

    async def one(record: dict) -> Proposal | None:
        try:
            text = await generate_proposal(
                checkpoint_path=checkpoint_path,
                source_code=record["source_code"],
                input_profile=record["input_profile"],
            )
        except flyte.errors.OOMError as e:
            print(f"[eval] generator OOM for {record['task_id']}: {e}")
            return None
        except Exception as e:  # noqa: BLE001 — one bad context ≠ failed eval
            print(f"[eval] generation failed for {record['task_id']}: {e}")
            return None
        return try_extract_proposal(text)

    return list(await asyncio.gather(*(one(r) for r in records)))


# Dark-mode wiring: a new checkpoint version IS the request to evaluate.
# The trigger binds only the checkpoint; the eval resolves the latest
# corpus artifact itself (trigger-shaped path, like basic-model-factory's
# eval_and_promote with dataset=None).
_eval_trigger = flyte.Trigger(
    name="eval-on-new-checkpoint",
    automation=flyte.OnArtifact(name=ARTIFACT_TUNER_CHECKPOINT),
    inputs={"checkpoint": flyte.TriggeredArtifact},
    description="New tuner-checkpoint version -> eval vs baseline",
    auto_activate=False,
)


@driver_env.task(
    triggers=[_eval_trigger],
    timeout=flyte.Timeout(max_runtime=2 * 3600),
    produces_artifacts=True,
    report=True,
)
async def eval_tuner(
    checkpoint: flyte.io.Dir,
    corpus: flyte.io.File | None = None,
    profile_name: str = "smoke",
    run_cluster_episodes: bool = True,
) -> flyte.io.File:
    """Score the checkpoint on the held-out split; emit tuner-eval-report."""
    import asyncio

    import pandas as pd

    profile = get_profile(profile_name)
    rep = Reporter("Checkpoint eval", f"profile={profile.name}")
    rep.p("Resolving corpus + checkpoint…")
    await rep.flush()

    if corpus is None:
        latest = await assets.latest_version(ARTIFACT_TASK_CORPUS)
        if latest is None:
            raise RuntimeError(
                f"no {ARTIFACT_TASK_CORPUS} artifact to evaluate against — "
                "run build_task_corpus (or synthetic_data_release) first"
            )
        corpus = flyte.io.File.from_existing_remote(latest.path)
    df = pd.read_parquet(await corpus.download())
    heldout = df[df["split"] == "heldout"].to_dict("records")[: profile.eval_contexts]
    train_records = df[df["split"] == "train"].to_dict("records")

    ckpt_path = getattr(checkpoint, "path", "") or ""
    rep.reset_body().kv(
        {
            "heldout contexts": len(heldout),
            "train contexts (baseline fit)": len(train_records),
            "checkpoint": ckpt_path[-60:],
        }
    ).p("Generating proposals via the reusable batched generator…")
    await rep.flush()
    proposals = await _generate_proposals(heldout, ckpt_path)

    valid = [(r, p) for r, p in zip(heldout, proposals) if p is not None]
    policy_stats = _summarize(valid)
    schema_validity = len(valid) / len(heldout) if heldout else 0.0

    baselines = fit_family_baseline(train_records)
    baseline_stats = _summarize(
        [(r, baseline_proposal(baselines, r["family"])) for r in heldout]
    )

    rep.reset_body().kv({"schema validity": f"{schema_validity:.0%}"})
    rep.h("Simulated scoring (policy vs baseline)")
    rep.table(
        ["metric", "policy", "baseline"],
        [
            [
                esc("success rate"),
                esc(f"{policy_stats['success_rate']:.0%}"),
                esc(f"{baseline_stats['success_rate']:.0%}"),
            ],
            [
                esc("median overprovision"),
                esc(
                    "-"
                    if policy_stats["median_overprovision_pct"] is None
                    else f"{policy_stats['median_overprovision_pct']:.0f}%"
                ),
                esc(
                    "-"
                    if baseline_stats["median_overprovision_pct"] is None
                    else f"{baseline_stats['median_overprovision_pct']:.0f}%"
                ),
            ],
        ],
    )
    if run_cluster_episodes and valid:
        rep.p("Running real cluster episodes under the policy's proposals…")
    await rep.flush()

    # Real episodes: the sim-to-real check. Fan out a small batch of
    # actual pods sized by the policy's proposals.
    cluster: list[dict] = []
    if run_cluster_episodes and valid:
        subset = valid[: profile.eval_cluster_episodes]
        results = await asyncio.gather(
            *(run_cluster_episode(r, p) for r, p in subset), return_exceptions=True
        )
        for (record, proposal), ep in zip(subset, results):
            if isinstance(ep, BaseException):
                cluster.append({"task_id": record["task_id"], "error": str(ep)})
                continue
            cluster.append(
                {
                    "task_id": record["task_id"],
                    "ok": ep.ok,
                    "oom": ep.oom,
                    "requested_memory_mib": ep.requested_memory_mib,
                    "sim_peak_memory_mib": record["true_peak_memory_mib"],
                    "real_peak_rss_mib": ep.peak_memory_mib,
                    "duration_s": ep.duration_s,
                }
            )
        # Pod-level cross-check, when the private metrics plugin is present.
        if metrics_available():
            ctx = flyte.ctx()
            run_name = getattr(getattr(ctx, "action", None), "run_name", None) or getattr(
                ctx, "run_name", None
            )
            if run_name:
                try:
                    pod_peaks = await harness_action_peaks(run_name)
                except Exception as e:  # noqa: BLE001 — cross-check only
                    pod_peaks = [{"error": f"{type(e).__name__}: {e}"}]
                for row in cluster:
                    row["pod_metrics_available"] = True
                cluster.append({"pod_peaks": pod_peaks})

    gate = (
        schema_validity >= 0.95
        and policy_stats["success_rate"] >= baseline_stats["success_rate"]
        and (
            policy_stats["median_overprovision_pct"] is None
            or baseline_stats["median_overprovision_pct"] is None
            or policy_stats["median_overprovision_pct"]
            <= baseline_stats["median_overprovision_pct"]
        )
    )

    ckpt_dir = await checkpoint.download()
    with open(f"{ckpt_dir}/manifest.json") as f:
        base_model = json.load(f)["base_model"]
    report = {
        "base_model": base_model,
        # Links this report to the checkpoint version it scored — the
        # lineage app uses it to badge checkpoint cards with eval metrics.
        "checkpoint_path": getattr(checkpoint, "path", "") or "",
        "n_contexts": len(heldout),
        "schema_validity": schema_validity,
        "success_rate": policy_stats["success_rate"],
        "median_overprovision_pct": policy_stats["median_overprovision_pct"],
        "baseline_success_rate": baseline_stats["success_rate"],
        "baseline_median_overprovision_pct": baseline_stats["median_overprovision_pct"],
        "cluster_episodes": [c for c in cluster if "task_id" in c],
        "cluster_episode_details": cluster,
        "auto_gate_passed": gate,
    }
    assert all(k in report for k in EVAL_REPORT_KEYS)

    # Final report: verdict + real-episode table + pod-metric availability.
    rep.h("Verdict")
    rep.raw(ok_pill(gate, "GATE PASS", "gate fail"))
    real = [c for c in cluster if "task_id" in c and "error" not in c]
    if real:
        rep.h("Real episodes (sim-to-real)")
        rep.table(
            ["task", "requested MiB", "sim peak", "real RSS", "outcome"],
            [
                [
                    esc(c["task_id"]),
                    esc(c.get("requested_memory_mib", "-")),
                    esc(f"{c.get('sim_peak_memory_mib', 0):.0f}"),
                    esc(f"{c.get('real_peak_rss_mib', 0):.0f}"),
                    pill("oom", "#e69812") if c.get("oom") else ok_pill(bool(c.get("ok")), "fit"),
                ]
                for c in real
            ],
        )
    pod = next((c.get("pod_peaks") for c in cluster if "pod_peaks" in c), None)
    rep.h("Pod metrics cross-check")
    if pod:
        rep.table(
            ["action", "pod peak MiB", "pod peak CPU", "error"],
            [
                [
                    esc(p.get("action_name", "-")),
                    esc("-" if p.get("peak_memory_mib") is None else f"{p['peak_memory_mib']:.0f}"),
                    esc("-" if p.get("peak_cpu") is None else f"{p['peak_cpu']:.2f}"),
                    esc(p.get("error", "")[:80]),
                ]
                for p in pod
            ],
        )
    else:
        rep.p("pod metrics unavailable (plugin missing or no data)", color=MUTED)
    await rep.flush()

    path = tempfile.mktemp(suffix=".json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    out = await flyte.io.File.from_local(path)
    return publish(out, ARTIFACT_EVAL_REPORT, description=f"gate={'PASS' if gate else 'FAIL'}")
