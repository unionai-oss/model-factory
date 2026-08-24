"""Eval team tasks: candidate-vs-base pass@1, gates, promotion.

Generation goes through the inference team's serving app (the same weights
service both variants: adapter on = candidate, adapter off = base), falling
back to local in-task generation if the service is unreachable — an eval
must never be blocked by a serving outage.
"""

from __future__ import annotations

import asyncio
import json

import flyte
import flyte.io
import flyte.report

from ..config import get_profile
from ..contracts import (
    ARTIFACT_CHECKPOINT,
    ARTIFACT_EVAL_REPORT,
    ARTIFACT_PROMOTED,
    ARTIFACT_RL_DATASET,
    publish,
)
from ..shared import assets, inference_client, reporting
from ..shared.gates import gate
from ..shared.rewards import build_prompt, score_completion
from .envs import eval_cpu_env, eval_gpu_env

_GEN_BATCH = 8


@flyte.trace
async def _generate_local(
    base_model: str, adapter_dir: str | None, prompts: list[list[dict]], max_new_tokens: int
) -> list[str]:
    """Fallback: greedy batched generation in-task; adapter_dir=None → base."""
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
    for i in range(0, len(prompts), _GEN_BATCH):
        chunk = prompts[i : i + _GEN_BATCH]
        chats = [
            tok.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
            for p in chunk
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


@flyte.trace
async def _generate_via_service(
    checkpoint_path: str, prompts: list[list[dict]], use_adapter: bool, max_new_tokens: int
) -> list[str]:
    url = inference_client.resolve_endpoint()
    return await asyncio.to_thread(
        inference_client.generate,
        url,
        prompts,
        use_adapter=use_adapter,
        max_new_tokens=max_new_tokens,
        checkpoint_path=checkpoint_path,
    )


async def _score_all(completions: list[str], tests: list[str]) -> list[dict]:
    sem = asyncio.Semaphore(8)

    async def one(c: str, t: str) -> dict:
        async with sem:
            b = await asyncio.to_thread(score_completion, c, t)
            return b.as_dict()

    return list(await asyncio.gather(*(one(c, t) for c, t in zip(completions, tests))))


@eval_gpu_env.task(report=True, timeout=flyte.Timeout(max_runtime=3600), produces_artifacts=True)
async def evaluate_checkpoint(
    checkpoint: flyte.io.Dir,
    dataset: flyte.io.File,
    profile_name: str = "smoke",
    use_service: bool = True,
) -> flyte.io.File:
    """Compare candidate vs base on the held-out split; emit `eval-report`."""
    import pandas as pd

    profile = get_profile(profile_name)
    df = pd.read_parquet(await dataset.download())
    heldout = df[df["split"] == "heldout"].reset_index(drop=True)
    adapter_dir = await checkpoint.download()

    with open(f"{adapter_dir}/manifest.json") as f:
        manifest = json.load(f)
    base_model = manifest["base_model"]

    prompts = [
        build_prompt(r.question, r.function_declaration or None)
        for r in heldout.itertuples()
    ]
    tests = list(heldout["tests"])

    backend = "inference-service"
    if use_service:
        try:
            cand_out = await _generate_via_service(
                checkpoint.path, prompts, True, profile.max_completion_length
            )
            base_out = await _generate_via_service(
                checkpoint.path, prompts, False, profile.max_completion_length
            )
        except Exception as e:
            print(f"inference service unavailable ({e}); falling back to local generation")
            backend = f"local (service failed: {str(e)[:120]})"
            cand_out = await _generate_local(base_model, adapter_dir, prompts, profile.max_completion_length)
            base_out = await _generate_local(base_model, None, prompts, profile.max_completion_length)
    else:
        backend = "local (requested)"
        cand_out = await _generate_local(base_model, adapter_dir, prompts, profile.max_completion_length)
        base_out = await _generate_local(base_model, None, prompts, profile.max_completion_length)

    cand_scores = await _score_all(cand_out, tests)
    base_scores = await _score_all(base_out, tests)

    def pass_rate(scores: list[dict]) -> float:
        return sum(s["tests_passed"] for s in scores) / max(len(scores), 1)

    result = {
        "profile": profile.name,
        "base_model": base_model,
        "generation_backend": backend,
        "n_heldout": len(heldout),
        "candidate_pass_at_1": pass_rate(cand_scores),
        "base_pass_at_1": pass_rate(base_scores),
        "delta": pass_rate(cand_scores) - pass_rate(base_scores),
        "promotion_margin": profile.promotion_margin,
        "auto_gate_passed": pass_rate(cand_scores) - pass_rate(base_scores) >= profile.promotion_margin,
        "per_task": [
            {
                "task_id": r.task_id,
                "difficulty": r.difficulty,
                "candidate": cand_scores[i],
                "base": base_scores[i],
            }
            for i, r in enumerate(heldout.itertuples())
        ],
    }
    out = "/tmp/eval_report.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)

    # --- report: the promotion-decision window ---
    body = reporting.stats_row(
        {
            "held-out tasks": len(heldout),
            "candidate pass@1": f"{result['candidate_pass_at_1']:.2%}",
            "base pass@1": f"{result['base_pass_at_1']:.2%}",
            "delta": f"{result['delta']:+.2%}",
            "auto gate": "PASS" if result["auto_gate_passed"] else "FAIL",
            "generation": backend,
        }
    )
    rows = []
    for i, r in enumerate(heldout.itertuples()):
        rows.append(
            [
                r.task_id,
                r.difficulty,
                reporting.pill(cand_scores[i]["tests_passed"]),
                reporting.pill(base_scores[i]["tests_passed"]),
                cand_scores[i]["execution_summary"],
            ]
        )
    body += "<h3>Per-task results (candidate vs base)</h3>"
    body += reporting.table(["task", "difficulty", "candidate", "base", "candidate execution"], rows)
    body += "<h3>Sample completions (vibe check)</h3>"
    for i in range(min(3, len(heldout))):
        body += f"<h4>{reporting.esc(heldout.iloc[i]['task_id'])}</h4>"
        body += f"<pre>{reporting.esc(cand_out[i][:1200])}</pre>"
    await flyte.report.replace.aio(reporting.page("Evaluation: candidate vs base", body))
    await flyte.report.flush.aio()

    f = await flyte.io.File.from_local(out)
    return publish(
        f,
        ARTIFACT_EVAL_REPORT,
        description=(
            f"pass@1 candidate {result['candidate_pass_at_1']:.2%} vs base "
            f"{result['base_pass_at_1']:.2%} on {len(heldout)} held-out tasks ({backend})"
        ),
    )


@eval_cpu_env.task(produces_artifacts=True)
async def promote_checkpoint(checkpoint: flyte.io.Dir, eval_report: flyte.io.File) -> flyte.io.Dir:
    """Re-publish an approved checkpoint as the `promoted-model` artifact."""
    local = await checkpoint.download()
    with open(f"{local}/manifest.json") as f:
        manifest = json.load(f)
    ev = json.loads(open(await eval_report.download()).read())
    d = await flyte.io.Dir.from_local(local)
    return publish(
        d,
        ARTIFACT_PROMOTED,
        description=(
            f"Promoted {manifest['base_model']} adapter — candidate pass@1 "
            f"{ev['candidate_pass_at_1']:.2%} (Δ {ev['delta']:+.2%}), human-approved"
        ),
        kind="model",
    )


# ── dark-mode trigger: new checkpoint → eval + gates + maybe promote ────

_eval_trigger = flyte.Trigger(
    name="eval-on-new-checkpoint",
    automation=flyte.OnArtifact(name=ARTIFACT_CHECKPOINT),
    inputs={"checkpoint": flyte.TriggeredArtifact, "profile_name": "smoke"},
    description="New checkpoint version → evaluation + gates",
    auto_activate=False,
)


@eval_cpu_env.task(triggers=[_eval_trigger], report=True)
async def eval_and_promote(
    checkpoint: flyte.io.Dir,
    profile_name: str = "smoke",
    auto_approve: bool = False,
    dataset: flyte.io.File | None = None,
) -> str:
    """The eval team's public entrypoint (also the trigger target).

    ``dataset`` defaults to the latest published `rl-tasks-dataset` so the
    trigger needs nothing beyond the checkpoint artifact.
    """
    if dataset is None:
        latest_dataset = await assets.latest(ARTIFACT_RL_DATASET)
        dataset = flyte.io.File.from_existing_remote(latest_dataset.path)

    eval_report = await evaluate_checkpoint(
        checkpoint=checkpoint, dataset=dataset, profile_name=profile_name
    )
    ev = json.loads(open(await eval_report.download()).read())
    if not ev["auto_gate_passed"]:
        return f"not promoted: delta {ev['delta']:+.2%} below margin"

    approved = await gate(
        "approve-promotion",
        "## Checkpoint promotion gate\n\n"
        f"Candidate pass@1 **{ev['candidate_pass_at_1']:.2%}** vs base "
        f"**{ev['base_pass_at_1']:.2%}** (Δ {ev['delta']:+.2%}) on "
        f"{ev['n_heldout']} held-out tasks (generation: {ev['generation_backend']}).\n\n"
        "Inspect the eval report for reward hacking or degenerate outputs.\n\n"
        "**Promote this checkpoint?**",
        auto_approve,
    )
    if not approved:
        return "not promoted: human gate"
    await promote_checkpoint(checkpoint=checkpoint, eval_report=eval_report)
    return (
        f"promoted: candidate {ev['candidate_pass_at_1']:.2%} vs base "
        f"{ev['base_pass_at_1']:.2%} on {ev['n_heldout']} held-out tasks"
    )
