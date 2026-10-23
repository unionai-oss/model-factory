"""Eval station: score each fine-tune, then pick the champion.

Two tasks, two very different shapes in the factory:

`score_model` is per-instance. It runs once for each `model` partition value,
so it inherits `[date, model]` from its inputs and produces one scorecard per
fine-tune.

`select_champion` is the *collapse*. It takes `decision-model.all("model")`
and `decision-scorecard.all("model")` — lists, one entry per suite member —
and produces a single artifact partitioned by `[date]` only. That one `.all`
is the whole "compare the suite and promote the winner" step; without it this
would be a task that has to go find the other models' outputs by itself.

Both return plain values; the factory publishes them.
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime

import flyte
import flyte.io
import flyte.report

from ..config import get_candidate, get_profile
from ..harness.policy import build_chat
from ..harness.scoring import grade_batch, majority_baseline, summarize
from ..harness.tools import TOOL_NAMES
from ..shared import reporting
from .envs import scorer_env, selector_env

_GEN_BATCH = 16


def _observations(df) -> list[dict]:
    from ..training.tasks import _clean, _clean_int

    return [
        {
            "customer_message": r.obs_customer_message,
            "intent": r.obs_intent,
            "order_id": _clean(r.obs_order_id),
            "order_looked_up": bool(r.obs_order_looked_up),
            "order_status": _clean(r.obs_order_status),
            "order_total_cents": _clean_int(r.obs_order_total_cents),
            "days_since_delivery": _clean_int(r.obs_days_since_delivery),
            "tracking_number": _clean(r.obs_tracking_number),
        }
        for r in df.itertuples()
    ]


@flyte.trace
async def _generate(
    base_model: str, adapter_dir: str | None, chats: list[list[dict]], max_new_tokens: int
) -> list[str]:
    """Greedy batched generation. ``adapter_dir=None`` scores the base model."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(adapter_dir or base_model, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=torch.bfloat16,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
    )
    if adapter_dir:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()

    outs: list[str] = []
    for i in range(0, len(chats), _GEN_BATCH):
        chunk = chats[i : i + _GEN_BATCH]
        rendered = [
            tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True) for c in chunk
        ]
        inputs = tok(
            rendered, return_tensors="pt", padding=True, truncation=True, max_length=2048
        ).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        outs.extend(
            tok.batch_decode(out[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return outs


@scorer_env.task(report=True, timeout=flyte.Timeout(max_runtime=5400))
async def score_model(
    candidate: flyte.io.Dir,
    episodes: flyte.io.File,
    model: str,
    profile_name: str = "smoke",
) -> flyte.io.File:
    """Score one fine-tune on the eval split; return the scorecard JSON."""
    import pandas as pd

    profile = get_profile(profile_name)
    spec = get_candidate(model)

    df = pd.read_parquet(await episodes.download())
    held = df[df["split"] == "eval"].reset_index(drop=True)
    if len(held) == 0:
        raise flyte.errors.NonRecoverableError("no eval-split episodes in the dataset")

    adapter_dir = await candidate.download()
    with open(os.path.join(adapter_dir, "manifest.json")) as f:
        manifest = json.load(f)
    base_model = manifest["base_model"]

    chats = [build_chat(obs) for obs in _observations(held)]
    ids = list(held["episode_id"])
    labels = list(held["label_json"])
    hard = [bool(h) for h in held["hard"]]

    tuned_out = await _generate(base_model, adapter_dir, chats, profile.max_new_tokens)
    tuned = summarize(grade_batch(ids, labels, tuned_out, hard))

    # Score the untuned base too. Without it a scorecard cannot distinguish
    # "this model learned the policy" from "this model already knew it", and
    # the champion choice would reward size rather than trainability.
    base = None
    if profile.score_base_model:
        base_out = await _generate(base_model, None, chats, profile.max_new_tokens)
        base = summarize(grade_batch(ids, labels, base_out, hard))

    baseline = majority_baseline(list(held["label_tool"]))
    scorecard = {
        "model_slug": spec.slug,
        "base_model": base_model,
        "params_m": spec.params_m,
        "profile": profile.name,
        "majority_baseline": baseline,
        **{k: v for k, v in tuned.items()},
        "base_model_scores": base,
        "lift_over_base": (tuned["exact_accuracy"] - base["exact_accuracy"]) if base else None,
        "lift_over_majority": tuned["exact_accuracy"] - baseline["accuracy"],
    }
    out = "/tmp/scorecard.json"
    with open(out, "w") as f:
        json.dump(scorecard, f, indent=2)

    body = reporting.stats_row(
        {
            "model": spec.slug,
            "params": f"{spec.params_m}M",
            "eval episodes": tuned["n"],
            "exact accuracy": reporting.pct(tuned["exact_accuracy"]),
            "tool accuracy": reporting.pct(tuned["tool_accuracy"]),
            "json valid": reporting.pct(tuned["json_valid_rate"]),
            "near-boundary exact": reporting.pct(tuned["hard_exact_accuracy"]),
        }
    )
    body += reporting.stats_row(
        {
            "majority baseline": reporting.pct(baseline["accuracy"]),
            "base model exact": reporting.pct(base["exact_accuracy"]) if base else "not scored",
            "lift over base": f"{scorecard['lift_over_base']:+.1%}" if base else "n/a",
            "lift over majority": f"{scorecard['lift_over_majority']:+.1%}",
        }
    )

    body += "<h3>Per-tool accuracy</h3>"
    body += reporting.table(
        ["tool", "n", "tool acc", "exact acc", ""],
        [
            [
                tool,
                tuned["per_tool"][tool]["n"],
                reporting.pct(tuned["per_tool"][tool]["tool_accuracy"]),
                reporting.pct(tuned["per_tool"][tool]["exact_accuracy"]),
                reporting.bar(tuned["per_tool"][tool]["exact_accuracy"]),
            ]
            for tool in TOOL_NAMES
        ],
    )

    if tuned["confusion"]:
        body += "<h3>Most common mistakes</h3>"
        body += reporting.table(
            ["expected -> predicted", "count"],
            [[k, v] for k, v in tuned["confusion"].items()],
        )

    body += "<h3>Sample predictions</h3>"
    for i in range(min(5, len(held))):
        body += f"<h4>{reporting.esc(ids[i])} — expected {reporting.esc(labels[i])}</h4>"
        body += f"<pre>{reporting.esc(tuned_out[i][:400])}</pre>"

    await flyte.report.replace.aio(reporting.page(f"Scorecard: {spec.slug}", body))
    await flyte.report.flush.aio()

    return await flyte.io.File.from_local(out)


@selector_env.task(report=True)
async def select_champion(
    candidates: list[flyte.io.Dir],
    scorecards: list[flyte.io.File],
    profile_name: str = "smoke",
    date: datetime | None = None,
) -> flyte.io.Dir:
    """Pick the best fine-tune in the suite and return it as the champion.

    ``candidates`` and ``scorecards`` are the `.all("model")` collapses, so
    both lists cover the whole suite for this day. They arrive in matching
    order, but this does not rely on that: each scorecard names its own
    `model_slug` and the candidate Dirs are matched by the slug in their
    manifests, so a reordering cannot silently promote the wrong directory.
    """
    profile = get_profile(profile_name)
    if not scorecards:
        raise flyte.errors.NonRecoverableError("no scorecards: nothing to choose from")

    # slug -> (scorecard, local dir)
    cards: dict[str, dict] = {}
    for f in scorecards:
        with open(await f.download()) as fh:
            card = json.load(fh)
        cards[card["model_slug"]] = card

    dirs: dict[str, str] = {}
    for d in candidates:
        local = await d.download()
        with open(os.path.join(local, "manifest.json")) as fh:
            dirs[json.load(fh)["model_slug"]] = local

    missing = sorted(set(cards) - set(dirs))
    if missing:
        raise flyte.errors.NonRecoverableError(
            f"scored models with no model directory: {missing}"
        )

    ranked = sorted(cards.values(), key=lambda c: -c["exact_accuracy"])
    winner = ranked[0]
    if winner["exact_accuracy"] < profile.min_exact_accuracy:
        # Deploy nothing rather than the least-bad thing: the serving app keeps
        # whatever champion it already had.
        raise flyte.errors.NonRecoverableError(
            f"no model cleared the floor: best is {winner['model_slug']} at "
            f"{winner['exact_accuracy']:.1%}, floor is {profile.min_exact_accuracy:.1%}"
        )

    champion_dir = "/tmp/decision-champion"
    if os.path.isdir(champion_dir):
        shutil.rmtree(champion_dir)
    shutil.copytree(dirs[winner["model_slug"]], champion_dir)

    with open(os.path.join(champion_dir, "manifest.json")) as f:
        manifest = json.load(f)
    manifest.update(
        {
            "selected_from": [c["model_slug"] for c in ranked],
            "exact_accuracy": winner["exact_accuracy"],
            "tool_accuracy": winner["tool_accuracy"],
            "runner_up": ranked[1]["model_slug"] if len(ranked) > 1 else None,
            "margin_over_runner_up": (
                winner["exact_accuracy"] - ranked[1]["exact_accuracy"] if len(ranked) > 1 else None
            ),
            "selection_date": date.strftime("%Y-%m-%d") if date else "",
        }
    )
    with open(os.path.join(champion_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    body = reporting.stats_row(
        {
            "champion": winner["model_slug"],
            "exact accuracy": reporting.pct(winner["exact_accuracy"]),
            "suite size": len(ranked),
            "runner-up": manifest["runner_up"] or "—",
            "margin": (
                f"{manifest['margin_over_runner_up']:+.1%}"
                if manifest["margin_over_runner_up"] is not None
                else "n/a"
            ),
            "floor": reporting.pct(profile.min_exact_accuracy),
        }
    )
    body += "<h3>Suite leaderboard</h3>"
    body += reporting.table(
        ["", "model", "params", "exact", "tool", "json valid", "near-boundary", "lift vs base", ""],
        [
            [
                reporting.win_pill() if c["model_slug"] == winner["model_slug"] else "",
                c["model_slug"],
                f"{c.get('params_m', 0)}M",
                reporting.pct(c["exact_accuracy"]),
                reporting.pct(c["tool_accuracy"]),
                reporting.pct(c["json_valid_rate"]),
                reporting.pct(c.get("hard_exact_accuracy", 0.0)),
                f"{c['lift_over_base']:+.1%}" if c.get("lift_over_base") is not None else "n/a",
                reporting.bar(c["exact_accuracy"]),
            ]
            for c in ranked
        ],
    )
    await flyte.report.replace.aio(reporting.page("Champion selection", body))
    await flyte.report.flush.aio()

    print(
        f"champion: {winner['model_slug']} at {winner['exact_accuracy']:.1%} exact "
        f"(suite: {[c['model_slug'] for c in ranked]})"
    )
    return await flyte.io.Dir.from_local(champion_dir)
