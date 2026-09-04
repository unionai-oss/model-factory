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
from .. import pricing
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
    """Sim-score (record, proposal) pairs into the aggregates a human needs
    to judge a reward shape: success, per-axis waste, GPU decisions, and —
    the business metric — $/task-hour of what was requested."""
    successes, wastes, mem_wastes, cpu_wastes, costs = [], [], [], [], []
    gpu_needed = gpu_ok = gpu_missing = gpu_spurious = 0
    per_family: dict[str, dict] = {}
    for record, proposal in pairs:
        gpu_mem = float(record.get("true_gpu_mem_mib", 0.0) or 0.0)
        ep = simulate_episode(
            proposal,
            float(record["true_peak_memory_mib"]),
            float(record["true_cpu_cores"]),
            int(record["duration_s"]),
            true_gpu_mem_mib=gpu_mem,
        )
        successes.append(ep.ok)
        cost = pricing.dollars_per_hr(
            ep.requested_cpu, ep.requested_memory_mib, ep.requested_gpu_type, ep.requested_gpu
        )
        costs.append(cost)
        fam = per_family.setdefault(
            record.get("family", "?"),
            {"n": 0, "ok": 0, "cost": 0.0, "waste": []},
        )
        fam["n"] += 1
        fam["ok"] += 1 if ep.ok else 0
        fam["cost"] += cost
        if gpu_mem > 0:
            gpu_needed += 1
            gpu_missing += 1 if ep.gpu_missing else 0
            gpu_ok += 1 if ep.ok else 0
        elif ep.requested_gpu:
            gpu_spurious += 1
        if ep.ok:
            mem_w = 100 * overprovision_fraction(ep.requested_memory_mib, ep.peak_memory_mib)
            cpu_w = 100 * overprovision_fraction(ep.requested_cpu, ep.peak_cpu)
            mem_wastes.append(mem_w)
            cpu_wastes.append(cpu_w)
            wastes.append((mem_w + cpu_w) / 2)
            fam["waste"].append((mem_w + cpu_w) / 2)
    n = len(successes)
    return {
        "success_rate": sum(successes) / n if n else 0.0,
        "median_overprovision_pct": statistics.median(wastes) if wastes else None,
        "median_mem_overprovision_pct": statistics.median(mem_wastes) if mem_wastes else None,
        "median_cpu_overprovision_pct": statistics.median(cpu_wastes) if cpu_wastes else None,
        "cost_per_task_hr": (sum(costs) / n) if n else None,
        "gpu_contexts": gpu_needed,
        "gpu_success_rate": (gpu_ok / gpu_needed) if gpu_needed else None,
        "gpu_missing_count": gpu_missing,
        "gpu_spurious_count": gpu_spurious,
        "per_family": {
            f: {
                "n": v["n"],
                "success_rate": v["ok"] / v["n"] if v["n"] else 0.0,
                "cost_per_task_hr": v["cost"] / v["n"] if v["n"] else None,
                "median_overprovision_pct": statistics.median(v["waste"]) if v["waste"] else None,
            }
            for f, v in per_family.items()
        },
    }


async def _generate_proposals(
    records: list[dict], checkpoint_path: str, invalid_examples: list[str] | None = None
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
                prior_json=str(record.get("prior_json", "") or ""),
                history_json=str(record.get("history_json", "") or ""),
            )
        except flyte.errors.OOMError as e:
            print(f"[eval] generator OOM for {record['task_id']}: {e}")
            return None
        except Exception as e:  # noqa: BLE001 — one bad context ≠ failed eval
            print(f"[eval] generation failed for {record['task_id']}: {e}")
            return None
        proposal = try_extract_proposal(text)
        if proposal is None and invalid_examples is not None and len(invalid_examples) < 10:
            # Round-7 lesson: a validity drop with no visible completions
            # is undebuggable — keep the receipts.
            invalid_examples.append(f"{record['task_id']} [{record.get('family', '?')}]: {text[:220]}")
        return proposal

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
    invalid_examples: list[str] = []
    proposals = await _generate_proposals(heldout, ckpt_path, invalid_examples)

    valid = [(r, p) for r, p in zip(heldout, proposals) if p is not None]
    policy_stats = _summarize(valid)
    schema_validity = len(valid) / len(heldout) if heldout else 0.0

    baselines = fit_family_baseline(train_records)
    baseline_stats = _summarize(
        [(r, baseline_proposal(baselines, r["family"])) for r in heldout]
    )

    # Which reward produced this checkpoint — the manifest carries the full
    # shape config so a human reading the report knows the experiment arm.
    ckpt_dir = await checkpoint.download()
    with open(f"{ckpt_dir}/manifest.json") as f:
        manifest = json.load(f)
    base_model = manifest["base_model"]
    reward_stage = manifest.get("reward_stage", "?")
    reward_shape = manifest.get("reward_shape")
    # Native reward: each arm's own training curve. NOT comparable across
    # shapes in absolute terms (different formulas) — the comparable part
    # is the first→last improvement, shown alongside the shared $ metric.
    fm = manifest.get("final_metrics") or {}
    train_reward_first = fm.get("mean_reward_first")
    train_reward_last = fm.get("mean_reward_last")

    def _pct(v):
        return "-" if v is None else f"{v:.0f}%"

    def _usd(v):
        return "-" if v is None else f"${v:.4f}"

    rep.reset_body()
    rep.h("Reward configuration under evaluation")
    shape_kv = {"reward stage": reward_stage, "trained profile": manifest.get("profile", "?")}
    if reward_shape:
        shape_kv.update(
            {
                "waste form": reward_shape.get("waste_form"),
                "headroom band": str(reward_shape.get("headroom") or "off"),
                "cost-weighted": str(bool(reward_shape.get("cost_weighted"))),
                "baseline-relative": str(bool(reward_shape.get("baseline_relative"))),
                "waste weight": f"{reward_shape.get('w_waste')}"
                + (
                    f" → {reward_shape['w_waste_final']} (annealed)"
                    if reward_shape.get("w_waste_final") is not None
                    else ""
                ),
                "group tie-break": str(reward_shape.get("group_tiebreak") or "off"),
                "robustness samples": str(reward_shape.get("robustness_samples", 1)),
            }
        )
    if train_reward_first is not None and train_reward_last is not None:
        shape_kv["native reward (train, first → last)"] = (
            f"{train_reward_first:.3f} → {train_reward_last:.3f} "
            f"(Δ {train_reward_last - train_reward_first:+.3f})"
        )
    rep.kv(shape_kv)
    rep.kv({"schema validity": f"{schema_validity:.0%}"})

    # The business metric, first and biggest.
    saved = None
    if policy_stats["cost_per_task_hr"] is not None and baseline_stats["cost_per_task_hr"]:
        saved = baseline_stats["cost_per_task_hr"] - policy_stats["cost_per_task_hr"]
        rep.h("Dollars")
        rep.kv(
            {
                "policy cost / task-hour": _usd(policy_stats["cost_per_task_hr"]),
                "baseline cost / task-hour": _usd(baseline_stats["cost_per_task_hr"]),
                "saved / 1,000 task-hours": f"${saved * 1000:,.2f}",
                "savings vs baseline": f"{100 * saved / baseline_stats['cost_per_task_hr']:.1f}%",
            }
        )

    rep.h("Simulated scoring (policy vs baseline)")
    rep.table(
        ["metric", "policy", "baseline"],
        [
            [esc(m), esc(p), esc(b)]
            for m, p, b in [
                (
                    "success rate",
                    f"{policy_stats['success_rate']:.0%}",
                    f"{baseline_stats['success_rate']:.0%}",
                ),
                (
                    "$ / task-hour",
                    _usd(policy_stats["cost_per_task_hr"]),
                    _usd(baseline_stats["cost_per_task_hr"]),
                ),
                (
                    "median overprovision",
                    _pct(policy_stats["median_overprovision_pct"]),
                    _pct(baseline_stats["median_overprovision_pct"]),
                ),
                (
                    "median mem overprovision",
                    _pct(policy_stats["median_mem_overprovision_pct"]),
                    _pct(baseline_stats["median_mem_overprovision_pct"]),
                ),
                (
                    "median cpu overprovision",
                    _pct(policy_stats["median_cpu_overprovision_pct"]),
                    _pct(baseline_stats["median_cpu_overprovision_pct"]),
                ),
            ]
        ],
    )

    if policy_stats["gpu_contexts"]:
        rep.h("GPU estimation")
        rep.table(
            ["metric", "policy", "baseline"],
            [
                [esc(m), esc(str(p)), esc(str(b))]
                for m, p, b in [
                    ("GPU contexts in heldout", policy_stats["gpu_contexts"], baseline_stats["gpu_contexts"]),
                    (
                        "success on GPU tasks",
                        "-" if policy_stats["gpu_success_rate"] is None else f"{policy_stats['gpu_success_rate']:.0%}",
                        "-" if baseline_stats["gpu_success_rate"] is None else f"{baseline_stats['gpu_success_rate']:.0%}",
                    ),
                    ("GPU missing (needed, not proposed)", policy_stats["gpu_missing_count"], baseline_stats["gpu_missing_count"]),
                    ("GPU spurious (proposed, not needed)", policy_stats["gpu_spurious_count"], baseline_stats["gpu_spurious_count"]),
                ]
            ],
        )

    rep.h("Per-family breakdown (policy)")
    rep.table(
        ["family", "n", "success", "$ / task-hr", "median waste"],
        [
            [
                esc(f),
                esc(str(v["n"])),
                esc(f"{v['success_rate']:.0%}"),
                esc(_usd(v["cost_per_task_hr"])),
                esc(_pct(v["median_overprovision_pct"])),
            ]
            for f, v in sorted(policy_stats["per_family"].items())
        ],
    )
    if run_cluster_episodes and valid:
        rep.p("Running real cluster episodes under the policy's proposals…")
    await rep.flush()

    # Real episodes: the sim-to-real check. Fan out a small batch of
    # actual pods sized by the policy's proposals. GPU-family contexts are
    # sim-only (a GPU pod per episode would burn the training pool), so
    # the real subset is CPU tasks.
    cluster: list[dict] = []
    if run_cluster_episodes and valid:
        cpu_valid = [
            (r, p) for r, p in valid if not float(r.get("true_gpu_mem_mib", 0.0) or 0.0)
        ]
        subset = cpu_valid[: profile.eval_cluster_episodes]
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

    # The gate is the business question: at least as reliable as the
    # baseline AND cheaper (or equal) in dollars.
    gate = (
        schema_validity >= 0.95
        and policy_stats["success_rate"] >= baseline_stats["success_rate"]
        and (
            policy_stats["median_overprovision_pct"] is None
            or baseline_stats["median_overprovision_pct"] is None
            or policy_stats["median_overprovision_pct"]
            <= baseline_stats["median_overprovision_pct"]
        )
        and (
            policy_stats["cost_per_task_hr"] is None
            or baseline_stats["cost_per_task_hr"] is None
            or policy_stats["cost_per_task_hr"] <= baseline_stats["cost_per_task_hr"]
        )
    )

    report = {
        "base_model": base_model,
        # Which reward produced the checkpoint — the comparison key for
        # humans and the lineage dashboard.
        "reward_stage": reward_stage,
        "reward_shape": reward_shape,
        "train_reward_first": train_reward_first,
        "train_reward_last": train_reward_last,
        # Links this report to the checkpoint version it scored — the
        # lineage app uses it to badge checkpoint cards with eval metrics.
        "checkpoint_path": getattr(checkpoint, "path", "") or "",
        "n_contexts": len(heldout),
        "schema_validity": schema_validity,
        "success_rate": policy_stats["success_rate"],
        "median_overprovision_pct": policy_stats["median_overprovision_pct"],
        "median_mem_overprovision_pct": policy_stats["median_mem_overprovision_pct"],
        "median_cpu_overprovision_pct": policy_stats["median_cpu_overprovision_pct"],
        "baseline_success_rate": baseline_stats["success_rate"],
        "baseline_median_overprovision_pct": baseline_stats["median_overprovision_pct"],
        # ── the business metric ──
        "policy_cost_per_task_hr": policy_stats["cost_per_task_hr"],
        "baseline_cost_per_task_hr": baseline_stats["cost_per_task_hr"],
        "dollars_saved_per_1k_task_hrs": (saved * 1000 if saved is not None else None),
        # ── GPU estimation ──
        "gpu_contexts": policy_stats["gpu_contexts"],
        "gpu_success_rate": policy_stats["gpu_success_rate"],
        "gpu_missing_count": policy_stats["gpu_missing_count"],
        "gpu_spurious_count": policy_stats["gpu_spurious_count"],
        "per_family": policy_stats["per_family"],
        "invalid_completion_examples": invalid_examples,
        "cluster_episodes": [c for c in cluster if "task_id" in c],
        "cluster_episode_details": cluster,
        "auto_gate_passed": gate,
    }
    assert all(k in report for k in EVAL_REPORT_KEYS)

    # Final report: verdict + real-episode table + pod-metric availability.
    rep.h("Verdict")
    rep.raw(ok_pill(gate, "GATE PASS", "gate fail"))
    if saved is not None:
        rep.p(
            f"reward shape {reward_stage!r}: "
            f"{'saves' if saved >= 0 else 'COSTS AN EXTRA'} "
            f"${abs(saved) * 1000:,.2f} per 1,000 task-hours vs the rule baseline",
            color=GOOD if saved >= 0 else "#F43B3E",
        )
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
    if invalid_examples:
        rep.h("Invalid completions (first 10 — why validity is below 100%)")
        for ex in invalid_examples:
            rep.p(ex[:300], color=MUTED)  # Reporter.p escapes internally
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
