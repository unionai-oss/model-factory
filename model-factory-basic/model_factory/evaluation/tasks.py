"""Eval team tasks: score models on the fixed held-out split, decide the release,
and crown tomorrow's champion.

Inside the factory (each returns a plain value the factory publishes):
- `evaluate_baseline`   -> `baseline-report`  (base model; built once)
- `evaluate_checkpoint` -> `eval-report[D]`   (candidate vs base vs champion)
- `promote_checkpoint`  -> `promoted-model[D]` (champion/challenger + human gate)

Outside the factory (they publish `mf-champion` themselves, which is the point):
- `crown_champion` — OnArtifact trigger on `promoted-model`: publishes
  `champion[D+1] = promoted[D]`. This is the one edge that crosses days, and
  the reason the factory graph can stay acyclic.
- `seed_champion` — run once to start the loop: `champion[D0] = base model`.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
from datetime import datetime, timedelta

import flyte
import flyte.artifacts
import flyte.io
import flyte.report
import pandas as pd

from ..config import get_profile
from ..contracts import ARTIFACT_CHAMPION, ARTIFACT_PROMOTED, EVAL_FILE, PARTITION_DATE, heldout_fingerprint
from ..shared import reporting
from ..shared.gates import gate
from ..shared.rewards import build_prompt, score_completion
from .envs import eval_cpu_env, eval_gpu_env
from .metrics import paired_delta, pass_at_1, promotion_decision


def _read_manifest(path: str) -> dict:
    with open(os.path.join(path, "manifest.json")) as f:
        m = json.load(f)
    m.setdefault("adapter", True)
    m.setdefault("generation", 0)
    return m


@flyte.trace
async def _generate(
    base_model: str, adapter_dir: str | None, prompts: list[list[dict]], max_new_tokens: int, batch: int
) -> list[str]:
    """Greedy batched generation in-task; ``adapter_dir=None`` is the base model."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base_model, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
    )
    if adapter_dir:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()

    outs: list[str] = []
    for i in range(0, len(prompts), batch):
        chats = [
            tok.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
            for p in prompts[i : i + batch]
        ]
        inputs = tok(chats, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        outs.extend(tok.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return outs


async def _score_all(completions: list[str], tests: list[str]) -> list[dict]:
    sem = asyncio.Semaphore(8)

    async def one(c: str, t: str) -> dict:
        async with sem:
            b = await asyncio.to_thread(score_completion, c, t)
            return b.as_dict()

    return list(await asyncio.gather(*(one(c, t) for c, t in zip(completions, tests))))


async def _run_heldout(heldout: pd.DataFrame, base_model: str, adapter_dir: str | None, profile) -> tuple[list[str], list[dict]]:
    prompts = [build_prompt(r.question, r.function_declaration or None) for r in heldout.itertuples()]
    outs = await _generate(base_model, adapter_dir, prompts, profile.max_completion_length, profile.eval_batch)
    return outs, await _score_all(outs, list(heldout["tests"]))


@eval_gpu_env.task(report=True, cache="auto", timeout=flyte.Timeout(max_runtime=3600))
async def evaluate_baseline(heldout: flyte.io.File, profile_name: str = "daily") -> flyte.io.File:
    """The base model's pass@1 on the held-out split. Built once, read every day."""
    profile = get_profile(profile_name)
    df = pd.read_parquet(await heldout.download())
    _, scores = await _run_heldout(df, profile.base_model, None, profile)
    passed = [s["tests_passed"] for s in scores]
    result = {
        "base_model": profile.base_model,
        "n_heldout": len(df),
        "heldout_fingerprint": heldout_fingerprint(df["task_id"]),
        "pass_at_1": pass_at_1(passed),
        "per_task": {tid: bool(p) for tid, p in zip(df["task_id"], passed)},
    }
    out = "/tmp/baseline_report.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    await flyte.report.replace.aio(
        reporting.page(
            "Baseline: base model on the fixed held-out split",
            reporting.stats_row(
                {
                    "base model": profile.base_model.split("/")[-1],
                    "held-out tasks": len(df),
                    "pass@1": f"{result['pass_at_1']:.2%}",
                    "fingerprint": result["heldout_fingerprint"],
                }
            ),
        )
    )
    await flyte.report.flush.aio()
    return await flyte.io.File.from_local(out)


@eval_gpu_env.task(report=True, timeout=flyte.Timeout(max_runtime=3600))
async def evaluate_checkpoint(
    checkpoint: flyte.io.Dir,
    champion: flyte.io.Dir,
    heldout: flyte.io.File,
    baseline: flyte.io.File,
    profile_name: str = "daily",
) -> flyte.io.File:
    """Score the candidate on the fixed held-out split against base AND champion."""
    profile = get_profile(profile_name)
    df = pd.read_parquet(await heldout.download())
    base = json.loads(open(await baseline.download()).read())
    fingerprint = heldout_fingerprint(df["task_id"])
    if base["heldout_fingerprint"] != fingerprint:
        raise flyte.errors.NonRecoverableError(
            f"baseline was scored on held-out {base['heldout_fingerprint']}, this eval on {fingerprint}"
        )
    cand_dir = await checkpoint.download()
    champ_dir = await champion.download()
    cand_m, champ_m = _read_manifest(cand_dir), _read_manifest(champ_dir)

    cand_out, cand_scores = await _run_heldout(df, cand_m["base_model"], cand_dir, profile)
    cand = [s["tests_passed"] for s in cand_scores]
    base_passed = [base["per_task"][t] for t in df["task_id"]]
    if champ_m["adapter"]:
        _, champ_scores = await _run_heldout(df, champ_m["base_model"], champ_dir, profile)
        champ = [s["tests_passed"] for s in champ_scores]
    else:
        champ = base_passed  # the champion IS the base model; its score is the baseline

    vs_base = paired_delta(cand, base_passed)
    vs_champ = paired_delta(cand, champ)
    result = {
        "profile": profile.name,
        "base_model": cand_m["base_model"],
        "generation_backend": "local",
        "n_heldout": len(df),
        "heldout_fingerprint": fingerprint,
        "candidate_generation": cand_m["generation"],
        "champion_generation": champ_m["generation"],
        "champion_is_base": not champ_m["adapter"],
        "candidate_pass_at_1": pass_at_1(cand),
        "base_pass_at_1": pass_at_1(base_passed),
        "champion_pass_at_1": pass_at_1(champ),
        "delta_vs_base": vs_base["delta"],
        "delta_vs_base_ci95": vs_base["ci95"],
        "delta_vs_champion": vs_champ["delta"],
        "delta_vs_champion_ci95": vs_champ["ci95"],
        "wins_losses_vs_champion": [vs_champ["wins"], vs_champ["losses"]],
        "promotion_margin": profile.promotion_margin,
        "auto_gate_passed": vs_champ["delta"] >= profile.promotion_margin,
        "train_summary": {k: cand_m.get(k) for k in ("warm_start", "train_tasks", "hard_tasks", "max_steps")},
        "per_task": [
            {
                "task_id": t,
                "difficulty": d,
                "candidate": cand_scores[i],
                "base": bool(base_passed[i]),
                "champion": bool(champ[i]),
            }
            for i, (t, d) in enumerate(zip(df["task_id"], df["difficulty"]))
        ],
    }
    out = "/tmp/eval_report.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)

    body = reporting.stats_row(
        {
            "held-out tasks": len(df),
            "base pass@1": f"{result['base_pass_at_1']:.2%}",
            f"champion (gen {champ_m['generation']})": f"{result['champion_pass_at_1']:.2%}",
            f"candidate (gen {cand_m['generation']})": f"{result['candidate_pass_at_1']:.2%}",
            "Δ vs champion": f"{vs_champ['delta']:+.2%} [{vs_champ['ci95'][0]:+.1%}, {vs_champ['ci95'][1]:+.1%}]",
            "Δ vs base": f"{vs_base['delta']:+.2%}",
            "auto gate": "PASS" if result["auto_gate_passed"] else "FAIL",
        }
    )
    body += "<h3>Per-task results</h3>" + reporting.table(
        ["task", "difficulty", "candidate", "champion", "base", "candidate execution"],
        [
            [
                p["task_id"],
                p["difficulty"],
                reporting.pill(p["candidate"]["tests_passed"]),
                reporting.pill(p["champion"]),
                reporting.pill(p["base"]),
                p["candidate"]["execution_summary"],
            ]
            for p in result["per_task"]
        ],
    )
    body += "<h3>Sample completions (vibe check)</h3>"
    for i in range(min(3, len(df))):
        body += f"<h4>{reporting.esc(df.iloc[i]['task_id'])}</h4><pre>{reporting.esc(cand_out[i][:1200])}</pre>"
    await flyte.report.replace.aio(reporting.page("Evaluation: candidate vs champion vs base", body))
    await flyte.report.flush.aio()

    return await flyte.io.File.from_local(out)


@eval_cpu_env.task(report=True)
async def promote_checkpoint(
    checkpoint: flyte.io.Dir,
    champion: flyte.io.Dir,
    eval_report: flyte.io.File,
    auto_approve: bool = False,
) -> flyte.io.Dir:
    """Release the better of candidate and champion as `promoted-model[D]`.

    The candidate wins only if it beats the champion by the profile's margin
    and passes the human gate; otherwise the champion is carried forward. The
    returned directory carries the deciding eval report and a promotion record.
    """
    ev = json.loads(open(await eval_report.download()).read())
    approved = False
    if ev["auto_gate_passed"]:
        approved = await gate(
            "approve-promotion",
            "## Checkpoint promotion gate\n\n"
            f"Candidate pass@1 **{ev['candidate_pass_at_1']:.2%}** vs champion "
            f"**{ev['champion_pass_at_1']:.2%}** (Δ {ev['delta_vs_champion']:+.2%}, 95% CI "
            f"{ev['delta_vs_champion_ci95'][0]:+.1%}..{ev['delta_vs_champion_ci95'][1]:+.1%}) and base "
            f"**{ev['base_pass_at_1']:.2%}** on {ev['n_heldout']} fixed held-out tasks.\n\n"
            "Inspect the eval report for reward hacking or degenerate outputs.\n\n"
            "**Promote this checkpoint?**",
            auto_approve,
        )
    winner, reason = promotion_decision(ev["delta_vs_champion"], ev["promotion_margin"], approved)

    src = await (checkpoint if winner == "candidate" else champion).download()
    out_dir = "/tmp/promoted-model"
    shutil.copytree(src, out_dir, dirs_exist_ok=True)
    if winner == "candidate":
        shutil.copy(await eval_report.download(), os.path.join(out_dir, EVAL_FILE))
    with open(os.path.join(out_dir, "promotion.json"), "w") as f:
        json.dump(
            {
                "released": winner,
                "reason": reason,
                "candidate_pass_at_1": ev["candidate_pass_at_1"],
                "champion_pass_at_1": ev["champion_pass_at_1"],
                "base_pass_at_1": ev["base_pass_at_1"],
                "heldout_fingerprint": ev["heldout_fingerprint"],
            },
            f,
            indent=2,
        )

    await flyte.report.replace.aio(
        reporting.page(
            "Promotion decision",
            reporting.stats_row(
                {
                    "released": winner,
                    "candidate pass@1": f"{ev['candidate_pass_at_1']:.2%}",
                    "champion pass@1": f"{ev['champion_pass_at_1']:.2%}",
                    "Δ vs champion": f"{ev['delta_vs_champion']:+.2%}",
                    "margin": f"{ev['promotion_margin']:+.2%}",
                }
            )
            + f"<p>{reporting.esc(reason)}</p>",
        )
    )
    await flyte.report.flush.aio()
    print(f"released {winner}: {reason}")
    return await flyte.io.Dir.from_local(out_dir)


# ── the loop's one cross-day edge ───────────────────────────────────────


def champion_metadata(day: datetime, version: str, description: str) -> flyte.artifacts.Metadata:
    return flyte.artifacts.Metadata(
        name=ARTIFACT_CHAMPION,
        version=version,
        partitions={PARTITION_DATE: day.date()},
        kind="model",
        description=description,
    )


crown_on_promotion = flyte.Trigger(
    name="crown-champion",
    automation=flyte.OnArtifact(name=ARTIFACT_PROMOTED),
    inputs={"promoted": flyte.TriggeredArtifact, "date": flyte.TriggeredPartition(PARTITION_DATE)},
    # Live on deploy, like everything else in this factory: the loop closes
    # itself. Pause it by setting this False and redeploying, not in the console.
    auto_activate=True,
)


@eval_cpu_env.task(triggers=[crown_on_promotion], produces_artifacts=True)
async def crown_champion(promoted: flyte.io.Dir, date: datetime) -> flyte.io.Dir:
    """Make day D's release the model day D+1 starts from and must beat.

    Fired by every new `promoted-model` version (a cache-hit re-materialization
    publishes no new version, so it does not fire). The version is derived from
    the promoted model's location, so re-crowning the same release is
    idempotent and does not invalidate tomorrow's cache.
    """
    nxt = date + timedelta(days=1)
    version = hashlib.sha256(promoted.path.encode()).hexdigest()[:16]
    print(f"crowning {promoted.path} as {ARTIFACT_CHAMPION}[{nxt:%Y-%m-%d}] ({version})")
    return flyte.artifacts.new(
        promoted, champion_metadata(nxt, version, f"Released on {date:%Y-%m-%d}; the model {nxt:%Y-%m-%d} must beat")
    )


@eval_cpu_env.task(produces_artifacts=True)
async def seed_champion(date: datetime, profile_name: str = "daily") -> flyte.io.Dir:
    """Start (or restart) the loop: the base model becomes `champion[date]`.

        flyte run team_eval.py seed_champion --date 2026-10-13
    """
    profile = get_profile(profile_name)
    out_dir = "/tmp/base-champion"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump({"base_model": profile.base_model, "adapter": False, "generation": 0}, f, indent=2)
    d = await flyte.io.Dir.from_local(out_dir)
    version = f"base-{profile.name}-{date:%Y%m%d}"
    return flyte.artifacts.new(d, champion_metadata(date, version, f"Base model {profile.base_model}"))
