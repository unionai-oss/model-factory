"""The collapse: compare the whole suite, promote one champion.

`select_champion` takes `permission-model.all("model")` and
`permission-scorecard.all("model")` — lists covering every torso x head for
the day — and returns a single Dir that the factory publishes as
`permission-champion[date]`. That one `.all` is the entire "compare and
promote" step.

The champion is the lowest **expected cost on the val split** (cost matrix in
`harness/scoring.py`: an unsafe allow of a deny-worthy call costs 50, an
unnecessary prompt costs 1). Not accuracy: a gate that is 1% more accurate by
auto-allowing a few more deletes is worse. And val, not test, so the numbers
reported for the winner are not the numbers it was picked on.

CPU-only, and deliberately imports nothing torch-backed: the code bundler
imports a task's module on its image, and the selector's image has no torch.
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime

import flyte
import flyte.io
import flyte.report

from ..config import get_profile
from ..shared import reporting
from .envs import selector_env

MANIFEST = "manifest.json"


def leaderboard(cards: list[dict]) -> list[dict]:
    """One row per (model, readout) plus per-torso references, best first."""
    rows = []
    for c in cards:
        for name, r in c["readouts"].items():
            rows.append({"model": c["model_slug"], "torso": c["torso"], "head": c["head"], "readout": name,
                         "primary": name == c["primary"], "params_m": c.get("params_m"),
                         "head_params": c.get("head_params") if name != "logprob" else None, **_summary(r)})
    seen = set()
    for c in cards:
        if c["torso"] in seen:
            continue
        seen.add(c["torso"])
        rows.append({"model": f"{c['torso']} (untuned)", "torso": c["torso"], "head": "-", "readout": "base_zero_shot",
                     "primary": False, "params_m": c.get("params_m"), **_summary(c["reference"]["base_zero_shot"])})
    maj = cards[0]["reference"]["majority"]
    rows.append({"model": f"majority ({maj['decision']})", "torso": "-", "head": "-", "readout": "majority",
                 "primary": False, **_summary(maj)})
    return rows


def _summary(r: dict) -> dict:
    return {
        "accuracy": r["accuracy"], "unsafe_allow_rate": r["unsafe_allow_rate"],
        "severe_unsafe_rate": r["severe_unsafe_rate"], "unnecessary_ask_rate": r["unnecessary_ask_rate"],
        "brier": r["brier"], "ece": r["ece"], "expected_cost": r["expected_cost"],
        "adversarial_accuracy": r.get("slices", {}).get("adversarial", {}).get("accuracy"),
        "adversarial_unsafe_allow": r.get("slices", {}).get("adversarial", {}).get("unsafe_allow_rate"),
        "unseen_accuracy": r.get("unseen_command", {}).get("accuracy"),
        "gate": r.get("gate"), "val": r.get("val"), "latency": r.get("latency"),
        "heldout": {k: v for k, v in (r.get("heldout") or {}).items() if k not in ("frontier", "confusion")} or None,
        "input_tokens": r.get("input_tokens"), "output_tokens": r.get("output_tokens"),
        "parse_rate": r.get("parse_rate"),
    }


#: How many standard errors of the best val cost count as a tie.
TIE_SE = 2.0


def rank(cards: list[dict]) -> list[dict]:
    """Suite members, champion first.

    1. Val expected cost of the primary readout (lower is better).
    2. But a val split of ~800 calls cannot separate 0.009 from 0.019 — that
       is one mistake. Every model within ``TIE_SE`` standard errors of the
       best is a statistical tie, and among ties the **cheapest to run** (p50
       latency) wins. Don't let noise pick a model that is 6x slower.
    """
    def cost(c):
        return c["readouts"][c["primary"]]["val"]["expected_cost"]

    def latency(c):
        return (c["readouts"][c["primary"]].get("latency") or {}).get("p50_ms", float("inf"))

    by_cost = sorted(cards, key=lambda c: (cost(c), latency(c)))
    best = by_cost[0]["readouts"][by_cost[0]["primary"]]["val"]
    band = cost(by_cost[0]) + TIE_SE * best.get("expected_cost_se", 0.0)
    tied = sorted([c for c in by_cost if cost(c) <= band], key=lambda c: (latency(c), cost(c)))
    return tied + [c for c in by_cost if cost(c) > band]


@selector_env.task(report=True)
async def select_champion(
    candidates: list[flyte.io.Dir],
    scorecards: list[flyte.io.File],
    profile_name: str = "dev",
    date: datetime | None = None,
) -> flyte.io.Dir:
    """Pick the best suite member and return it as the champion.

    Candidates and scorecards are matched by the slug inside each, not by list
    position, so a reordering cannot promote the wrong directory.
    """
    profile = get_profile(profile_name)
    if not scorecards:
        raise flyte.errors.NonRecoverableError("no scorecards: nothing to choose from")
    cards: dict[str, dict] = {}
    for f in scorecards:
        with open(await f.download()) as fh:
            card = json.load(fh)
        cards[card["model_slug"]] = card
    dirs: dict[str, str] = {}
    for d in candidates:
        local = await d.download()
        with open(os.path.join(local, MANIFEST)) as fh:
            dirs[json.load(fh)["model_slug"]] = local
    missing = sorted(set(cards) - set(dirs))
    if missing:
        raise flyte.errors.NonRecoverableError(f"scored models with no model directory: {missing}")

    out = "/tmp/permission-champion"
    try:
        manifest, ranked = choose(cards, dirs, profile.min_accuracy, out, date)
    except ValueError as e:
        # Deploy nothing rather than the least-bad thing: the app keeps
        # whatever champion it already had.
        raise flyte.errors.NonRecoverableError(str(e))
    winner = ranked[0]
    await flyte.report.replace.aio(
        reporting.page("Champion selection", render_selection(ranked, manifest["leaderboard"], winner)))
    await flyte.report.flush.aio()
    w = winner["readouts"][winner["primary"]]
    print(f"champion: {winner['model_slug']} (val cost {w['val']['expected_cost']:.3f}, test acc {w['accuracy']:.1%})")
    return await flyte.io.Dir.from_local(out)


def choose(cards: dict[str, dict], dirs: dict[str, str], min_accuracy: float, out: str,
           date: datetime | None = None) -> tuple[dict, list[dict]]:
    """Rank, check the floor, and write the champion Dir to ``out``.
    Returns (champion manifest, ranked scorecards)."""
    ranked = rank(list(cards.values()))
    winner = ranked[0]
    w = winner["readouts"][winner["primary"]]
    if w["accuracy"] < min_accuracy:
        raise ValueError(f"no model cleared the floor: best is {winner['model_slug']} at {w['accuracy']:.1%}, "
                         f"floor is {min_accuracy:.1%}")
    if os.path.isdir(out):
        shutil.rmtree(out)
    shutil.copytree(dirs[winner["model_slug"]], out)
    with open(os.path.join(out, MANIFEST)) as f:
        manifest = json.load(f)
    manifest.pop("history", None)
    manifest.update({
        "selected_from": [c["model_slug"] for c in ranked],
        "selection_metric": f"val expected_cost of the primary readout; ties within {TIE_SE} SE go to the lowest p50 latency",
        "primary_readout": winner["primary"],
        "gate_threshold": w["gate"]["threshold"],
        "test_metrics": _summary(w),
        "runner_up": ranked[1]["model_slug"] if len(ranked) > 1 else None,
        "leaderboard": leaderboard(ranked),
        "selection_date": date.strftime("%Y-%m-%d") if date else "",
    })
    with open(os.path.join(out, MANIFEST), "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    return manifest, ranked


def render_selection(ranked: list[dict], board: list[dict], winner: dict) -> str:
    w = winner["readouts"][winner["primary"]]
    body = reporting.stats_row({
        "champion": winner["model_slug"], "test accuracy": reporting.pct(w["accuracy"]),
        "unsafe-allow": f"{w['unsafe_allow_rate']:.2%}", "Brier": f"{w['brier']:.3f}", "ECE": f"{w['ece']:.3f}",
        "gate": f"t={w['gate']['threshold']} -> {w['gate']['autonomy']:.1%} autonomy at {w['gate']['unsafe_allow_rate']:.2%} unsafe",
        "suite size": len(ranked),
    })
    body += "<h3>Leaderboard (test split; primary readouts ranked by val expected cost)</h3>"
    body += reporting.table(
        ["", "model", "readout", "acc", "unsafe-allow", "severe", "unneeded ask", "Brier", "ECE", "cost",
         "adv. acc", "unseen acc", "gate autonomy @≤1%", "held-out acc", "held-out unsafe", "held-out ECE",
         "p50 ms", "out tok"],
        [[reporting.win_pill() if r["model"] == winner["model_slug"] and r["primary"] else "",
          r["model"], r["readout"], reporting.pct(r["accuracy"]), f"{r['unsafe_allow_rate']:.2%}",
          f"{r['severe_unsafe_rate']:.2%}", reporting.pct(r["unnecessary_ask_rate"]), f"{r['brier']:.3f}",
          f"{r['ece']:.3f}", f"{r['expected_cost']:.3f}",
          reporting.pct(r["adversarial_accuracy"]) if r["adversarial_accuracy"] is not None else "—",
          reporting.pct(r["unseen_accuracy"]) if r["unseen_accuracy"] is not None else "—",
          (f"{r['gate']['autonomy']:.1%} (t={r['gate']['threshold']}, {r['gate']['unsafe_allow_rate']:.2%})"
           if r.get("gate", {}).get("reachable") else "unreachable") if r.get("gate") else "—",
          reporting.pct(r["heldout"]["accuracy"]) if r.get("heldout") else "—",
          f"{r['heldout']['unsafe_allow_rate']:.2%}" if r.get("heldout") else "—",
          f"{r['heldout']['ece']:.3f}" if r.get("heldout") else "—",
          f"{r['latency']['p50_ms']:.0f}" if r.get("latency") else "—",
          f"{r['output_tokens']:.1f}" if r.get("output_tokens") is not None else "—"]
         for r in board])
    series = []
    for c in ranked:
        for name, r in c["readouts"].items():
            series.append({"name": f"{c['model_slug']} · {name}", "color_key": c["torso"], "readout": name,
                           "points": r["frontier"], "operating": r["gate"]})
    body += "<h3>The money chart: autonomy vs safety (in-distribution test)</h3>" + reporting.frontier_svg(series)
    ood = [{"name": f"{c['model_slug']} · {n}", "color_key": c["torso"], "readout": n,
            "points": r["heldout"]["frontier"], "operating": r["heldout"]["gate"]}
           for c in ranked for n, r in c["readouts"].items() if r.get("heldout")]
    if ood:
        body += ("<h3>The same chart on held-out families</h3><p>Same val-tuned thresholds (filled dots). "
                 "If a model's confidence does not drop on unfamiliar calls, its dot slides right.</p>"
                 + reporting.frontier_svg(ood))
    return body
