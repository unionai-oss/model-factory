"""Eval station: score one fine-tune on the held-out splits.

Per-instance in the factory: it runs once per `model` partition value and
inherits `[date, model]` from its inputs. The collapse to a champion is a
different module (`evaluation/select.py`) because it runs on a CPU image with
no torch, and the code bundler imports a task's module — so this module may
import torch-backed code at the top, and that one must not.

A scorecard holds one or more **readouts** of the same fine-tune:

    pointer         the pointer head's softmax over options        (pointer models)
    json            generate, parse {"decision": ...}, one-hot     (sft models)
    logprob         the LM head's likelihood of each option        (sft models)

plus **references** that every row is read against: the untuned base model
scored zero-shot by option likelihood, and the majority-class answer.

Each readout is scored on test; the val split is used only to (a) tune the
gate threshold and (b) choose the champion, so no reported number is fit to
itself.
"""

from __future__ import annotations

import gc
import json
import statistics
from typing import Any

import flyte
import flyte.io
import flyte.report

from ..config import get_candidate, get_profile
from ..harness.policy import DECISIONS, Observation
from ..harness.prompt import build_chat
from ..harness.scoring import (
    decision_metrics,
    frontier,
    gate,
    majority_baseline,
    one_hot,
    pick_threshold,
    prior_probs,
    brier,
    score,
    UNSAFE_ALLOW_TARGET,
)
from ..heads import modeling
from ..shared import reporting
from .envs import scorer_env

PRIMARY_READOUT = {"pointer": "pointer", "sft": "json"}


def split_inputs(df, split: str) -> dict[str, list]:
    part = df[df["split"] == split]
    rows = part.to_dict("records")
    return {
        "rows": rows,
        "chats": [build_chat(Observation.from_row(r)) for r in rows],
        "labels": [r["label"] for r in rows],
        "slices": [r["slice"] for r in rows],
        "seen": [bool(r["seen_in_train"]) for r in rows],
    }


def call_text(obs: Observation) -> str:
    """One line describing the proposed call (parquet NaNs already cleaned)."""
    if obs.command is not None:
        return obs.command
    if obs.url is not None:
        return f"{obs.tool} {obs.url}"
    return f"{obs.tool} {obs.path} -> {obs.resolved_path}"


def evaluate_readout(test: dict, test_out: list[dict], val: dict, val_out: list[dict]) -> dict[str, Any]:
    """Every metric for one readout. ``*_out`` are `DecisionModel` outputs."""
    probs = [o["probs"] for o in test_out]
    preds = [o["decision"] for o in test_out]
    m = score(test["labels"], probs, test["slices"], test["seen"], preds=preds)
    val_probs = [o["probs"] for o in val_out]
    t = pick_threshold(val["labels"], val_probs)
    g = decision_metrics(test["labels"], gate(probs, t))
    m["gate"] = {
        "threshold": t,
        "autonomy": 1.0 - g["unnecessary_ask_rate"],
        "unsafe_allow_rate": g["unsafe_allow_rate"],
        "accuracy": g["accuracy"],
        "expected_cost": g["expected_cost"],
        # A one-hot readout (JSON) cannot be gated: P(allow) is 0 or 1, so no
        # threshold separates its allows. `reachable` says whether ANY
        # threshold met the unsafe-allow budget on val.
        "reachable": decision_metrics(val["labels"], gate(val_probs, t))["unsafe_allow_rate"] <= UNSAFE_ALLOW_TARGET,
    }
    m["frontier"] = frontier(test["labels"], probs)
    v = decision_metrics(val["labels"], [o["decision"] for o in val_out])
    m["val"] = {k: v[k] for k in ("accuracy", "expected_cost", "unsafe_allow_rate")}
    m["parse_rate"] = sum(1 for o in test_out if o.get("parsed", True)) / len(test_out)
    m["input_tokens"] = statistics.mean(o["input_tokens"] for o in test_out)
    m["output_tokens"] = statistics.mean(o["output_tokens"] for o in test_out)
    return m


def time_readout(dm: "modeling.DecisionModel", chats: list, readout: str, n: int) -> dict[str, float]:
    """Batch-size-1 latency, the way an agent calls a gate: one call at a time."""
    sample = chats[: max(n, 1)]
    for c in sample[:2]:
        dm.time_one(c, readout)  # warm-up (kernels, allocator)
    ms = sorted(dm.time_one(c, readout) for c in sample)
    return {"p50_ms": ms[len(ms) // 2], "p95_ms": ms[min(len(ms) - 1, int(len(ms) * 0.95))], "n": len(ms)}


def _free(obj) -> None:
    import torch

    del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_scorecard(df, model_dir: str, *, slug: str, torso: str, head: str, base_model: str,
                    params_m: int | None, profile_name: str, latency_samples: int) -> dict:
    """Everything `score_model` computes, as a plain function (the local
    call-shape test drives it with a tiny random model)."""
    test, val = split_inputs(df, "test"), split_inputs(df, "val")
    if not test["rows"] or not val["rows"]:
        raise ValueError("episodes need non-empty val and test splits")
    with open(f"{model_dir}/{modeling.MANIFEST}") as f:
        manifest = json.load(f)

    dm = modeling.DecisionModel(model_dir)
    readouts: dict[str, dict] = {}
    if dm.head == "pointer":
        outs = dm.pointer_probs(test["chats"])
        readouts["pointer"] = evaluate_readout(test, outs, val, dm.pointer_probs(val["chats"]))
        readouts["pointer"]["latency"] = time_readout(dm, test["chats"], "pointer", latency_samples)
    else:
        outs = dm.generate(test["chats"])
        readouts["json"] = evaluate_readout(test, outs, val, dm.generate(val["chats"]))
        readouts["json"]["latency"] = time_readout(dm, test["chats"], "json", latency_samples)
        readouts["logprob"] = evaluate_readout(test, dm.option_logprobs(test["chats"]), val, dm.option_logprobs(val["chats"]))
        readouts["logprob"]["latency"] = time_readout(dm, test["chats"], "logprob", latency_samples)
    _free(dm)

    # References. The untuned base model, scored by option likelihood: the
    # most generous zero-shot reading of a model never trained on this policy
    # (it still sees the policy text in its prompt).
    base = modeling.DecisionModel(base_model=base_model, head="sft")
    base_test = base.option_logprobs(test["chats"])
    reference = {"base_zero_shot": evaluate_readout(test, base_test, val, base.option_logprobs(val["chats"]))}
    _free(base)
    maj = majority_baseline([r["label"] for r in df[df["split"] == "train"].to_dict("records")])
    reference["majority"] = score(test["labels"], [one_hot(maj["decision"]) for _ in test["labels"]],
                                  test["slices"], test["seen"])
    reference["majority"]["decision"] = maj["decision"]
    reference["prior_brier"] = brier(test["labels"], [prior_probs(test["labels"])] * len(test["labels"]))

    errors = [
        {"call": call_text(Observation.from_row(r)),
         "mode": r["obs_permission_mode"], "family": r["family"], "label": r["label"], "pred": o["decision"],
         "probs": [round(p, 3) for p in o["probs"]], "rationale": r["rationale"]}
        for r, o in zip(test["rows"], outs) if o["decision"] != r["label"]
    ]
    errors.sort(key=lambda e: (e["pred"] != "allow", e["label"] == "allow"))  # unsafe allows first
    return {
        "model_slug": slug,
        "torso": torso,
        "head": head,
        "base_model": base_model,
        "params_m": params_m,
        "head_params": manifest.get("head_params"),
        "trainable_params": manifest.get("trainable_params"),
        "train_seconds": manifest.get("train_seconds"),
        "profile": profile_name,
        "n_test": len(test["labels"]),
        "n_val": len(val["labels"]),
        "primary": PRIMARY_READOUT[head],
        "readouts": readouts,
        "reference": reference,
        "errors": errors[:40],
    }


@scorer_env.task(report=True, timeout=flyte.Timeout(max_runtime=5400))
async def score_model(
    candidate: flyte.io.Dir,
    episodes: flyte.io.File,
    model: str,
    profile_name: str = "dev",
) -> flyte.io.File:
    """Score one fine-tune; return the scorecard JSON."""
    import pandas as pd

    profile = get_profile(profile_name)
    cand = get_candidate(model)
    df = pd.read_parquet(await episodes.download())
    model_dir = await candidate.download()
    try:
        scorecard = build_scorecard(
            df, model_dir, slug=cand.slug, torso=cand.torso.slug, head=cand.head, base_model=cand.hf_id,
            params_m=cand.params_m, profile_name=profile.name, latency_samples=profile.latency_samples)
    except ValueError as e:
        raise flyte.errors.NonRecoverableError(str(e))
    out = "/tmp/scorecard.json"
    with open(out, "w") as f:
        json.dump(scorecard, f, indent=2, default=str)
    await flyte.report.replace.aio(reporting.page(f"Scorecard: {cand.slug}", render_scorecard(scorecard)))
    await flyte.report.flush.aio()
    return await flyte.io.File.from_local(out)


def render_scorecard(card: dict) -> str:
    p = card["readouts"][card["primary"]]
    body = reporting.stats_row({
        "model": card["model_slug"], "head": card["head"], "test calls": card["n_test"],
        "accuracy": reporting.pct(p["accuracy"]), "unsafe-allow": f"{p['unsafe_allow_rate']:.2%}",
        "severe (deny->allow)": f"{p['severe_unsafe_rate']:.2%}", "unnecessary ask": reporting.pct(p["unnecessary_ask_rate"]),
        "Brier": f"{p['brier']:.3f}", "ECE": f"{p['ece']:.3f}", "p50 latency": f"{p['latency']['p50_ms']:.0f} ms",
    })
    rows = []
    for name, r in list(card["readouts"].items()) + [(f"ref: {k}", v) for k, v in card["reference"].items() if isinstance(v, dict)]:
        rows.append([name, reporting.pct(r["accuracy"]), f"{r['unsafe_allow_rate']:.2%}", reporting.pct(r["unnecessary_ask_rate"]),
                     f"{r['brier']:.3f}", f"{r['ece']:.3f}", f"{r['expected_cost']:.3f}",
                     reporting.pct(r.get("slices", {}).get("adversarial", {}).get("accuracy", 0)),
                     (f"{r['gate']['autonomy']:.1%} @ t={r['gate']['threshold']}" if r["gate"].get("reachable")
                      else "unreachable") if "gate" in r else "—",
                     f"{r['latency']['p50_ms']:.0f}" if "latency" in r else "—"])
    body += "<h3>Readouts (test split)</h3>" + reporting.table(
        ["readout", "accuracy", "unsafe-allow", "unneeded ask", "Brier", "ECE", "cost", "adversarial acc",
         "gate autonomy (val-tuned, ≤1% unsafe)", "p50 ms"], rows)
    series = [{"name": f"{card['model_slug']} {n}", "color_key": n, "readout": n, "points": r["frontier"],
               "operating": {**r["gate"], "threshold": r["gate"]["threshold"]}} for n, r in card["readouts"].items()]
    series.append({"name": "untuned base", "color_key": "base", "readout": "base_zero_shot",
                   "points": card["reference"]["base_zero_shot"]["frontier"], "operating": None})
    body += "<h3>Autonomy vs safety</h3>" + reporting.frontier_svg(series)
    body += "<h3>Slices</h3>" + reporting.table(
        ["slice", "n", "accuracy", "unsafe-allow", "unneeded ask"],
        [[s, v["n"], reporting.pct(v["accuracy"]), f"{v['unsafe_allow_rate']:.2%}", reporting.pct(v["unnecessary_ask_rate"])]
         for s, v in p["slices"].items()]
        + ([["unseen calls", p["unseen_command"]["n"], reporting.pct(p["unseen_command"]["accuracy"]),
             f"{p['unseen_command']['unsafe_allow_rate']:.2%}", ""]] if "unseen_command" in p else []))
    body += "<h3>Confusion (rows = oracle)</h3>" + reporting.table(
        ["oracle \\ model", *DECISIONS], [[y, *[p["confusion"][y][d] for d in DECISIONS]] for y in DECISIONS])
    body += "<h3>Reliability</h3>" + reporting.table(
        ["confidence bin", "n", "mean confidence", "accuracy"],
        [[b["bin"], b["n"], f"{b['confidence']:.2f}", f"{b['accuracy']:.2f}"] for b in p["reliability"]])
    body += "<h3>Errors (unsafe allows first)</h3>" + reporting.table(
        ["call", "mode", "family", "oracle", "model", "P(allow, ask, deny)", "oracle's reason"],
        [[e["call"], e["mode"], e["family"], e["label"], e["pred"], e["probs"], e["rationale"]] for e in card["errors"][:20]])
    return body
