"""Factory stations for the decision-model arm.

Three tasks, all CPU, all returning plain values — the factory declares and
publishes the artifacts (see `resource_tuner/factory.py`):

    fit_decision_model     -> rt-decision-model      [arm]
    score_decision_model   -> rt-decision-scorecard  [arm]
    select_decision_tuner  -> rt-decision-champion    (collapses `arm`)

The LLM arm is untouched by all of this. The corpus these read is a
`factory.source` — an artifact the existing `build_task_corpus` station
publishes outside the factory — so adding this arm required no change to the
GRPO path, the trigger wiring, or the tune service.

Scoring reuses `environment.simulator.simulate_episode` and `pricing`, the
same instruments the LLM arm's eval uses, so the arms land on one comparable
set of numbers: fit rate, OOM rate, median over-provisioning, $/task-hr.
"""

from __future__ import annotations

import json
import statistics
import tempfile
from typing import Sequence

import flyte
import flyte.io
import flyte.report

from .. import pricing
from ..environment.simulator import simulate_episode
from ..policy.action_space import grid_headroom
from ..shared.reporting import Reporter, esc
from .baseline import baseline_proposal, fit_family_baseline
from .decision_model import DEFAULT_ARMS, fit, get_arm, load, save
from .envs import decision_env

#: Keys every rt-decision-scorecard carries. `fit_rate` and `cost_per_task_hr`
#: are what the champion is chosen on.
DECISION_SCORECARD_KEYS = [
    "arm",
    "action_space",
    "under_penalty",
    "n_heldout",
    "fit_rate",
    "oom_rate",
    "median_overprovision_pct",
    "cost_per_task_hr",
    "baseline_fit_rate",
    "baseline_cost_per_task_hr",
    "grid_headroom_pct",
]


#: Only the columns the features and labels actually read. `harness_code` is
#: deliberately absent: it is a second full copy of every workload's source and
#: nothing here looks at it.
_NEEDED_COLUMNS = [
    "task_id",
    "family",
    "source_code",
    "input_profile",
    "params_json",
    "generator",
    "prior_json",
    "history_json",
    "true_peak_memory_mib",
    "true_cpu_cores",
    "true_gpu_mem_mib",
    "duration_s",
    "split",
]


def read_corpus(
    path: str, max_train: int, max_eval: int
) -> tuple[list[dict], list[dict]]:
    """Stream a corpus parquet and return capped (train, heldout) record lists.

    Streamed in batches with an explicit column projection rather than
    `pd.read_parquet(path)`, because the corpus runs to ~1e6 rows carrying two
    source-code columns: materializing the whole frame OOMs the pod before any
    model is built (run rt-decision-r1, exit 137). Reading only what the
    features touch, and stopping once both caps are met, makes the station's
    memory a function of the caps instead of the corpus.

    The cap is a prefix, not a random sample. The corpus is written with
    templates and archetypes interleaved by the generator rather than grouped,
    so a prefix is already mixed across families; a reservoir sample would cost
    a full pass for no benefit here.
    """
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    available = {f.name for f in pf.schema_arrow}
    columns = [c for c in _NEEDED_COLUMNS if c in available]

    train: list[dict] = []
    heldout: list[dict] = []
    for batch in pf.iter_batches(batch_size=16_384, columns=columns):
        for row in batch.to_pylist():
            split = row.get("split")
            if split == "train":
                if len(train) < max_train:
                    train.append(row)
            elif len(heldout) < max_eval:
                heldout.append(row)
        if len(train) >= max_train and len(heldout) >= max_eval:
            break
    return train, heldout


def _simulate(records: Sequence[dict], proposals: Sequence) -> dict:
    """Run every (record, proposal) pair through the simulator and aggregate."""
    fits, ooms, wastes, costs = 0, 0, [], []
    for r, p in zip(records, proposals):
        ep = simulate_episode(
            p,
            float(r["true_peak_memory_mib"]),
            float(r["true_cpu_cores"]),
            int(r["duration_s"]),
            true_gpu_mem_mib=float(r.get("true_gpu_mem_mib", 0.0) or 0.0),
        )
        fits += bool(ep.ok)
        ooms += not bool(ep.ok)
        costs.append(
            pricing.dollars_per_hr(
                ep.requested_cpu,
                ep.requested_memory_mib,
                ep.requested_gpu_type,
                ep.requested_gpu,
            )
        )
        if ep.ok:
            wastes.append(
                100 * (ep.requested_memory_mib - ep.peak_memory_mib) / ep.requested_memory_mib
            )
    n = max(len(records), 1)
    return {
        "fit_rate": fits / n,
        "oom_rate": ooms / n,
        "median_overprovision_pct": statistics.median(wastes) if wastes else None,
        "cost_per_task_hr": sum(costs) / n if costs else None,
    }


@decision_env.task(report=True, timeout=flyte.Timeout(max_runtime=3600))
async def fit_decision_model(corpus: flyte.io.File, arm: str = "oom-averse") -> flyte.io.Dir:
    """Train one decision-model arm on the corpus train split.

    ``arm`` is the configuration name and also the `arm` partition value — the
    factory passes it with ``factory.partition("arm")``.
    """
    spec = get_arm(arm)
    train, heldout = read_corpus(
        await corpus.download(), spec.max_train_rows, spec.max_eval_rows
    )
    if not train:
        raise flyte.errors.NonRecoverableError("corpus has no train split")

    model = fit(train, spec, val_records=heldout or None)

    out_dir = tempfile.mkdtemp(prefix="rt-decision-")
    save(
        model,
        out_dir,
        extra_manifest={
            "n_train": len(train),
            "n_heldout": len(heldout),
            "corpus_path": corpus.path,
        },
    )

    final = model.history[-1] if model.history else {}
    rep = Reporter(f"Decision model — arm {spec.name}")
    rep.kv(
        {
            "arm": spec.name,
            "action space": spec.action_space,
            "under-penalty": spec.under_penalty,
            "train rows": f"{len(train)} (cap {spec.max_train_rows})",
            "heldout rows": f"{len(heldout)} (cap {spec.max_eval_rows})",
            "epochs": spec.epochs,
            "final loss": f"{final.get('train_loss', float('nan')):.4f}",
        }
    )
    rep.kv(
        {
            "memory exact": _pct(final.get("val_memory_exact")),
            "memory under": _pct(final.get("val_memory_under")),
            "cpu exact": _pct(final.get("val_cpu_exact")),
            "gpu exact": _pct(final.get("val_gpu_exact")),
            "all-heads fit": _pct(final.get("val_all_fit")),
        }
    )
    rep.h("Training")
    rep.table(
        ["epoch", "train loss", "mem exact", "mem under", "all fit"],
        [
            [
                h.get("epoch"),
                f"{h.get('train_loss', float('nan')):.4f}",
                _pct(h.get("val_memory_exact")),
                _pct(h.get("val_memory_under")),
                _pct(h.get("val_all_fit")),
            ]
            # Every epoch would be a 40-row wall; the shape is what matters.
            for h in model.history[:: max(1, len(model.history) // 20)]
        ],
    )
    await rep.flush()

    return await flyte.io.Dir.from_local(out_dir)


@decision_env.task(report=True, timeout=flyte.Timeout(max_runtime=3600))
async def score_decision_model(
    model_dir: flyte.io.Dir, corpus: flyte.io.File, arm: str = "oom-averse"
) -> flyte.io.File:
    """Simulate the arm's proposals on the heldout split; return the scorecard.

    Scored against the SAME rule-based baseline the LLM arm's eval uses, on
    the same split, through the same simulator — so the three arms' numbers
    can be put in one table without an apples-to-oranges caveat.
    """
    spec = get_arm(arm)
    # Train rows ARE needed here, even though the model is already trained:
    # `baseline_proposal` is a lookup into a per-family baseline that has to be
    # FITTED from the train split first (`fit_family_baseline`), and the
    # comparison is only honest if the baseline never saw heldout either.
    train, heldout = read_corpus(
        await corpus.download(), spec.max_train_rows, spec.max_eval_rows
    )
    if not heldout:
        raise flyte.errors.NonRecoverableError("corpus has no heldout split")
    if not train:
        raise flyte.errors.NonRecoverableError(
            "corpus has no train split, so the rule baseline cannot be fitted"
        )

    model = load(await model_dir.download())
    proposals = model.predict(heldout)
    tuned = _simulate(heldout, proposals)

    baselines = fit_family_baseline(train)
    base_proposals = [baseline_proposal(baselines, r["family"]) for r in heldout]
    base = _simulate(heldout, base_proposals)

    headroom = grid_headroom(
        model.space, [float(r["true_peak_memory_mib"]) for r in heldout]
    )

    scorecard = {
        "arm": model.arm.name,
        "action_space": model.arm.action_space,
        "under_penalty": model.arm.under_penalty,
        "n_heldout": len(heldout),
        **tuned,
        "baseline_fit_rate": base["fit_rate"],
        "baseline_oom_rate": base["oom_rate"],
        "baseline_median_overprovision_pct": base["median_overprovision_pct"],
        "baseline_cost_per_task_hr": base["cost_per_task_hr"],
        # The waste floor the grid itself imposes: a 20% number reads very
        # differently against a 15% floor than against zero.
        "grid_headroom_pct": 100 * headroom,
        "dollars_saved_per_1k_task_hrs": (
            1000 * (base["cost_per_task_hr"] - tuned["cost_per_task_hr"])
            if base["cost_per_task_hr"] is not None and tuned["cost_per_task_hr"] is not None
            else None
        ),
    }

    out = tempfile.mktemp(suffix=".json", prefix="rt-decision-scorecard-")
    with open(out, "w") as f:
        json.dump(scorecard, f, indent=2)

    rep = Reporter(f"Decision scorecard — arm {model.arm.name}")
    rep.kv(
        {
            "arm": model.arm.name,
            "under-penalty": model.arm.under_penalty,
            "heldout": len(heldout),
            "baseline fitted on": len(train),
            "fit rate": _pct(tuned["fit_rate"]),
            "OOM rate": _pct(tuned["oom_rate"]),
            "median waste": _num(tuned["median_overprovision_pct"], "%"),
            "grid floor": f"{100 * headroom:.1f}%",
        }
    )
    rep.kv(
        {
            "baseline fit": _pct(base["fit_rate"]),
            "baseline waste": _num(base["median_overprovision_pct"], "%"),
            "$/task-hr": _num(tuned["cost_per_task_hr"], "", 4),
            "baseline $/task-hr": _num(base["cost_per_task_hr"], "", 4),
            "$ saved / 1k task-hr": _num(scorecard["dollars_saved_per_1k_task_hrs"], "", 2),
        }
    )
    rep.h("Arm vs rule baseline")
    rep.table(
        ["", "fit", "OOM", "median waste", "$/task-hr"],
        [
            [
                f"decision ({model.arm.name})",
                _pct(tuned["fit_rate"]),
                _pct(tuned["oom_rate"]),
                _num(tuned["median_overprovision_pct"], "%"),
                _num(tuned["cost_per_task_hr"], "", 4),
            ],
            [
                "rule baseline",
                _pct(base["fit_rate"]),
                _pct(base["oom_rate"]),
                _num(base["median_overprovision_pct"], "%"),
                _num(base["cost_per_task_hr"], "", 4),
            ],
        ],
    )
    rep.h("Sample proposals")
    rep.table(
        ["task", "family", "true peak", "proposed", "cpu", "gpu"],
        [
            [
                esc(r["task_id"]),
                esc(r["family"]),
                f"{float(r['true_peak_memory_mib']):.0f}Mi",
                f"{p.memory_mib}Mi",
                p.cpu,
                f"{p.gpu_type}:{p.gpu}" if p.gpu else "—",
            ]
            for r, p in list(zip(heldout, proposals))[:15]
        ],
    )
    await rep.flush()

    return await flyte.io.File.from_local(out)


@decision_env.task(report=True)
async def select_decision_tuner(
    models: list[flyte.io.Dir], scorecards: list[flyte.io.File]
) -> flyte.io.Dir:
    """Pick the best arm and return its model directory as the champion.

    Selection is **lexicographic, not a weighted score**: highest fit rate
    first, cheapest $/task-hr as the tie-break. That ordering is the policy
    decision this factory encodes — a cheaper arm that OOMs more is not a
    better arm, because an OOM costs the whole run plus a retry, and no
    plausible weighting of a blended objective expresses that cliff honestly.

    Both arguments are `.all("arm")` collapses, so each covers the whole
    suite. Arms are matched by the name in their own manifest rather than by
    list position, so a reordering cannot promote the wrong directory.
    """
    import os
    import shutil

    if not scorecards:
        raise flyte.errors.NonRecoverableError("no scorecards: nothing to choose from")

    cards: dict[str, dict] = {}
    for f in scorecards:
        with open(await f.download()) as fh:
            card = json.load(fh)
        cards[card["arm"]] = card

    dirs: dict[str, str] = {}
    for d in models:
        local = await d.download()
        with open(os.path.join(local, "manifest.json")) as fh:
            dirs[json.load(fh)["arm"]["name"]] = local

    missing = sorted(set(cards) - set(dirs))
    if missing:
        raise flyte.errors.NonRecoverableError(f"scored arms with no model dir: {missing}")

    ranked = sorted(
        cards.values(),
        key=lambda c: (-c["fit_rate"], c.get("cost_per_task_hr") or float("inf")),
    )
    winner = ranked[0]

    champion_dir = tempfile.mkdtemp(prefix="rt-decision-champion-")
    shutil.rmtree(champion_dir)
    shutil.copytree(dirs[winner["arm"]], champion_dir)
    manifest_path = os.path.join(champion_dir, "manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)
    manifest.update(
        {
            "selected_arm": winner["arm"],
            "selected_from": [c["arm"] for c in ranked],
            "fit_rate": winner["fit_rate"],
            "cost_per_task_hr": winner.get("cost_per_task_hr"),
            "runner_up": ranked[1]["arm"] if len(ranked) > 1 else None,
        }
    )
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    rep = Reporter("Decision-arm champion")
    rep.kv(
        {
            "champion": winner["arm"],
            "under-penalty": winner["under_penalty"],
            "fit rate": _pct(winner["fit_rate"]),
            "OOM rate": _pct(winner["oom_rate"]),
            "median waste": _num(winner["median_overprovision_pct"], "%"),
            "$/task-hr": _num(winner.get("cost_per_task_hr"), "", 4),
            "arms compared": len(ranked),
        }
    )
    rep.h("Frontier (fit rate first, then cost)")
    rep.table(
        ["arm", "space", "under-penalty", "fit", "OOM", "median waste", "$/task-hr"],
        [
            [
                ("* " if c["arm"] == winner["arm"] else "") + c["arm"],
                c["action_space"],
                c["under_penalty"],
                _pct(c["fit_rate"]),
                _pct(c["oom_rate"]),
                _num(c["median_overprovision_pct"], "%"),
                _num(c.get("cost_per_task_hr"), "", 4),
            ]
            for c in ranked
        ],
    )
    await rep.flush()

    print(
        f"decision champion: {winner['arm']} fit={winner['fit_rate']:.1%} "
        f"oom={winner['oom_rate']:.1%} (arms: {[c['arm'] for c in ranked]})"
    )
    return await flyte.io.Dir.from_local(champion_dir)


def _pct(value) -> str:
    return "n/a" if value is None else f"{float(value):.1%}"


def _num(value, suffix: str = "", places: int = 1) -> str:
    return "n/a" if value is None else f"{float(value):.{places}f}{suffix}"


#: The arms a full suite run trains. Exposed so the factory and the CLI agree.
SUITE = DEFAULT_ARMS
