"""RightSizeCPU stations: the Flyte tasks the factory wires together.

    rt-cpubench-runs        generate_cpu_runs     code + inputs, unmeasured
    rt-cpubench-corpus      measure_cpu_bench     every run executed, CPU sampled per second
    rt-cpubench-arm         fit_cpu_arm           one arm: fit on train, predict val/test/ood
    rt-cpubench-scorecard   score_cpu_arm         accuracy within margin, throttling cost
    rt-cpubench-leaderboard cpu_leaderboard       all arms side by side
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time

import flyte
import flyte.io
import pandas as pd

from ..bench import charts
from ..bench import hybrid
from ..bench.common import (
    collect,
    hinted_corpus,
    pooled_arm,
    hosted_subsample,
    pred_rows,
    preds_by_variant,
    profile_for_release,
    read_arm,
    read_rows,
    shard,
    tasks_for_release,
    usable_by_split,
    write_arm_dir,
    write_parquet,
)
from ..bench.gpu_stations import train_llm_arm
from ..bench.llm_arms import zeroshot_predict
from ..bench.versioning import cpu_arms_cache, cpu_corpus_cache
from ..shared.reporting import Reporter
from . import arms as cpu_arms
from . import dataset, grid, scoring
from .envs import cpu_measure_env, cpu_measure_pair_env, cpubench_env
from .measure import measure_run, slot_cpus
from .spec import CPU

TITLE = CPU.title


#: Tasks per family by release prefix. A full release is ~4x small: the
#: learned arms were data-starved at small (GPU jev-pointer fit a constant).
#: big is RightSizeMem v2's order of magnitude: 11 x 450 x 2 = 9,900 runs.
RELEASE_TASKS = {"small": 10, "full": 40, "big": 450}


@cpubench_env.task(cache=cpu_corpus_cache(), report=True)
async def generate_cpu_runs(release: str, tasks_per_family: int = 0, seed: int = 0,
                            runs_per_task: int = 2) -> flyte.io.File:
    tasks_per_family = tasks_per_family or tasks_for_release(release, RELEASE_TASKS)
    profile = profile_for_release(release)
    rows = dataset.generate_runs(tasks_per_family, seed=seed, runs_per_task=runs_per_task, profile=profile)
    rep = Reporter(f"{TITLE}: workloads", f"release {release}, profile {profile}")
    by = pd.DataFrame(rows).groupby(["family", "split"]).size().unstack(fill_value=0)
    rep.kv({"tasks": len({r["task_id"] for r in rows}), "runs": len(rows)})
    rep.table(["family", *by.columns], [[f, *map(str, v)] for f, v in zip(by.index, by.values.tolist())])
    await rep.flush()
    return await flyte.io.File.from_local(write_parquet(rows, "runs.parquet"))


def _measure_one(r: dict, shared: bool = False) -> dict:
    try:
        m = measure_run(r["harness_code"], json.loads(r["inputs_json"]), seed=r["seed"], shared=shared)
    except Exception as e:  # noqa: BLE001 -- a broken prep is a result, not a crash
        m = {"ok": False, "error": f"prep: {type(e).__name__}: {e}"[:500]}
    print(f"{r['run_id']}: ok={m.get('ok')} peak={m.get('peak_cores', 0):.2f} mean={m.get('mean_cores', 0):.2f} "
          f"wall={m.get('wall_s', 0):.0f}s src={m.get('cpu_source')} {str(m.get('error', ''))[:120]}", flush=True)
    return {"run_id": r["run_id"], **{k: v for k, v in m.items() if k != "result"}}


# One workload at a time: CPU is what is measured. retries=3 / max_runtime:
# interruptible pods (a spot reclaim must become a retry, not a stuck
# RUNNING action -- rsb-v2-corpus-1). A shard of 3 runs takes ~6 min.
@cpu_measure_env.task(retries=3, cache=cpu_corpus_cache(), timeout=flyte.Timeout(max_runtime=1200))
async def measure_cpu_shard(rows_json: str) -> str:
    return json.dumps([_measure_one(r) for r in json.loads(rows_json)])


def _slot(cpus: set[int], rows: list[dict]) -> list[dict]:
    # Linux affinity is per thread and inherited by children: input prep and
    # the workload both run on this slot's cores only.
    os.sched_setaffinity(0, cpus)
    return [_measure_one(r, shared=True) for r in rows]


# Two pinned slots per pod (big releases), each working through half the
# shard. 8 runs of ~45 s + prep ~= 4 min of work.
@cpu_measure_pair_env.task(retries=3, cache=cpu_corpus_cache(), timeout=flyte.Timeout(max_runtime=1200))
async def measure_cpu_pair_shard(rows_json: str) -> str:
    rows = json.loads(rows_json)
    slots = slot_cpus(2)
    halves = await asyncio.gather(*(asyncio.to_thread(_slot, cpus, rows[i::2]) for i, cpus in enumerate(slots)))
    return json.dumps([m for h in halves for m in h])


@cpubench_env.task(cache=cpu_corpus_cache(), report=True)
async def measure_cpu_bench(runs: flyte.io.File, shard_size: int = 3, pairing: str = "auto") -> flyte.io.File:
    """`pairing`: "auto" pairs big releases; "pair" / "single" force it (the pilot
    measures one set of runs both ways to check the pinned slots against the cgroup)."""
    rows = await read_rows(runs)
    paired = pairing == "pair" or (pairing == "auto" and bool(rows) and rows[0].get("profile") == "big")
    shard_size = 8 if paired else shard_size
    shards = shard(rows, ("run_id", "harness_code", "inputs_json", "seed"), shard_size)
    how = "two pinned slots of 7 cores per pod" if paired else "one at a time per pod"
    rep = Reporter(f"{TITLE}: measurement", f"{len(rows)} runs in {len(shards)} shards, {how}")
    await rep.flush()
    measured: dict[str, dict] = {}
    run_shard = measure_cpu_pair_shard if paired else measure_cpu_shard
    lost = collect(await asyncio.gather(*(run_shard(s) for s in shards), return_exceptions=True), measured)
    if lost > 0.1 * len(shards):
        raise RuntimeError(f"{lost}/{len(shards)} measurement shards failed; first error above")
    table = [dataset.attach_measurement(r, measured.get(r["run_id"], {"ok": False, "error": "shard lost"})) for r in rows]
    df = pd.DataFrame(dataset.attach_history(table))
    ok = df[df["ok"]]
    rep.reset_body()
    rep.kv({"runs": len(df), "measured ok": len(ok), "failed": int((~df["ok"]).sum()), "shards lost": lost,
            "cpu source": ", ".join(sorted(set(ok["cpu_source"]))) if len(ok) else "-",
            "median wall": f"{ok['wall_s'].median():.0f} s" if len(ok) else "-",
            "sampled / kernel CPU s (median)": f"{(ok['cpu_s'] / ok['ru_cpu_s'].clip(lower=1e-6)).median():.3f}"
            if len(ok) and "ru_cpu_s" in ok else "-"})
    fam = ok.groupby("family").agg(runs=("run_id", "size"), peak50=("peak_cores", "median"),
                                   peak_max=("peak_cores", "max"), mean50=("mean_cores", "median"))
    rep.h("Per family (cores)")
    rep.table(["family", "runs", "median peak (15 s)", "max peak", "median mean"],
              [[f, str(int(r.runs)), f"{r.peak50:.2f}", f"{r.peak_max:.2f}", f"{r.mean50:.2f}"] for f, r in fam.iterrows()])
    errs = df[~df["ok"]]["error"].str.slice(0, 120).value_counts().head(10)
    if len(errs):
        rep.h("Failures")
        rep.table(["error", "n"], [[e, str(n)] for e, n in errs.items()])
    await rep.flush()
    return await flyte.io.File.from_local(write_parquet(df.to_dict("records"), "corpus.parquet"))


CPU_ARMS = tuple(cpu_arms.ARMS)
GPU_ARMS = ("sft", "jev", "jev-pointer", "jev-pointer-4b")
#: The stacked hybrids (``bench.hybrid``): each LLM arm with the trees' prediction in its prompt.
STACK_ARMS = tuple(f"{hybrid.STACK_PREFIX}{a}" for a in ("sft", "jev-pointer", "jev-pointer-4b"))
HOSTED_ARMS = {"zeroshot-27b": "qwen38-27b-vllm"}
ALL_ARMS = CPU_ARMS + tuple(HOSTED_ARMS) + GPU_ARMS + STACK_ARMS
_EVAL_SPLITS = ("val", "test", "ood")
HOSTED_RUNS_PER_SPLIT = 300


@cpubench_env.task(cache=cpu_arms_cache(), report=True)
async def fit_cpu_arm(corpus: flyte.io.File, arm: str, llm_model: str = "Qwen/Qwen3-1.7B", llm_epochs: int = 3,
                      llm_max_train_rows: int = 0) -> flyte.io.Dir:
    rows = await read_rows(corpus)
    splits = usable_by_split(rows)
    rep = Reporter(f"{TITLE}: arm {arm}", ", ".join(f"{k}={len(v)}" for k, v in splits.items()))
    await rep.flush()
    if arm in CPU_ARMS:
        fit, predict = cpu_arms.ARMS[arm]
        state = fit(splits["train"], splits["val"])
        eval_rows = [r for s in _EVAL_SPLITS for r in splits[s]]
        t0 = time.perf_counter()
        preds = predict(state, eval_rows)
        meta = {"knobs": {k: v for k, v in state.items() if isinstance(v, (int, float, str))},
                "ms_per_decision": (time.perf_counter() - t0) * 1000 / max(1, len(eval_rows))}
    elif arm in HOSTED_ARMS:
        eval_rows = hosted_subsample(splits, HOSTED_RUNS_PER_SPLIT)
        preds, stats = await asyncio.to_thread(zeroshot_predict, eval_rows, HOSTED_ARMS[arm], spec=CPU)
        meta = {"stats": stats}
    elif arm in GPU_ARMS:
        return await train_llm_arm(corpus, arm, llm_model, llm_epochs, llm_max_train_rows, bench="cpu")
    elif arm in STACK_ARMS:
        fit, predict = cpu_arms.ARMS[hybrid.ANCHOR]
        hinted, hmeta = await hinted_corpus(rows, fit, predict, CPU)
        rep.kv({k: str(v)[:300] for k, v in hmeta.items()})
        await rep.flush()
        # the hint lengthens every prompt; 14 h as the GPU bench's sft needed
        return await train_llm_arm.override(timeout=flyte.Timeout(max_runtime=14 * 3600))(
            hinted, arm, llm_model, llm_epochs, llm_max_train_rows, bench="cpu")
    else:
        raise ValueError(f"unknown arm {arm!r}; choose from {ALL_ARMS}")
    meta["arm"] = arm
    rep.kv({k: str(v)[:200] for k, v in meta.items()})
    await rep.flush()
    return await write_arm_dir(pred_rows(arm, preds, eval_rows, CPU), meta)


@cpubench_env.task(cache=cpu_arms_cache(), report=True)
async def score_cpu_arm(corpus: flyte.io.File, arm_dir: flyte.io.Dir, arm: str) -> flyte.io.File:
    rows = [r for r in await read_rows(corpus) if dataset.usable(r)]
    preds, meta = await read_arm(arm_dir)
    card = {"arm": arm, "meta": meta, "variants": {}, "frontier": {}}
    rep = Reporter(f"{TITLE}: scorecard {arm}")
    for variant, p in preds_by_variant(preds, CPU).items():
        card["variants"][variant] = scoring.score_slices(rows, p)
        test_rows = [r for r in rows if r["split"] == "test" and r["run_id"] in p]
        card["frontier"][variant] = scoring.frontier(test_rows, p, point_estimate=variant.endswith("-argmax"))
        t = card["variants"][variant].get("split=test", {})
        if t.get("n"):
            rep.kv({f"{variant} test": f"within margin {t['within_margin']:.1%}, under {t['under_rate']:.1%}, "
                                       f"cpu regret {t['cpu_regret']:.1%}"})
    await rep.flush()
    path = os.path.join(tempfile.mkdtemp(prefix="rsc-card-"), "scorecard.json")
    with open(path, "w") as f:
        json.dump(card, f, indent=2, default=float)
    return await flyte.io.File.from_local(path)


@cpubench_env.task(cache=cpu_arms_cache(), report=True)
async def pool_cpu_arms(corpus: flyte.io.File, arm_dirs: list[flyte.io.Dir]) -> flyte.io.Dir:
    """The pooled hybrids: gbt-dist x every distributional LLM variant (``bench.hybrid``)."""
    rows = [r for r in await read_rows(corpus) if dataset.usable(r)]
    out = await pooled_arm(rows, arm_dirs, CPU)
    rep = Reporter(f"{TITLE}: pooled hybrids", f"{hybrid.ANCHOR} x each LLM variant, weight tuned on val")
    _preds, meta = await read_arm(out)
    rep.table(["pool", "w (trees)", "val headline (lower is better)", "secondary from"],
              [[v, str(k["w_trees"]), f"{k['val_headline']:.3f}", k["secondary_from"]] for v, k in meta["knobs"].items()])
    await rep.flush()
    return out


HEADLINE = ("static", "vpa", "code-rule", "gbt", "gbt-dist", "zeroshot-27b", "sft-greedy", "sft-sampled",
            "jev-argmax", "jev", "jev-pointer-argmax", "jev-pointer",
            "jev-pointer-4b-argmax", "jev-pointer-4b",
            "stack-sft-greedy", "stack-sft-sampled", "stack-jev-pointer-argmax", "stack-jev-pointer",
            "stack-jev-pointer-4b-argmax", "stack-jev-pointer-4b")


@cpubench_env.task(cache=cpu_arms_cache(), report=True)
async def cpu_leaderboard(scorecards: list[flyte.io.File], pool_scorecard: flyte.io.File | None = None) -> flyte.io.File:
    cards = []
    for f in [*scorecards, *([pool_scorecard] if pool_scorecard is not None else [])]:
        with open(await f.download()) as fh:
            cards.append(json.load(fh))
    board, fronts = {}, {}
    for c in cards:
        board.update(c["variants"])
        fronts.update(c.get("frontier", {}))
    order = [v for v in HEADLINE if v in board] + sorted(v for v in board if v not in HEADLINE)
    rep = Reporter(f"{TITLE}: leaderboard", f"{len(order)} decision rules")
    for split in ("test", "ood"):
        key = f"split={split}"
        rows = [(v, board[v][key]) for v in order if board[v].get(key, {}).get("n")]
        if not rows:
            continue
        rep.h(f"{split}: peak CPU (busiest 15 s window)")
        rep.table(["arm", "within ±25%", "under (throttled)", "MAE cores", "exact cell", "cpu regret",
                   "mean within ±25%", "Brier", "ECE", "n"],
                  [[v, f"{s['within_margin']:.1%}", f"{s['under_rate']:.1%}", f"{s['mae_cores']:.2f}",
                    f"{s['exact']:.1%}", f"{s['cpu_regret']:.1%}", f"{s['mean_within_margin']:.1%}",
                    f"{s['brier']:.3f}" if "brier" in s else "-", f"{s['ece_under']:.3f}" if "ece_under" in s else "-",
                    str(s["n"])] for v, s in rows])
        rep.raw(charts.hbars([(v, s["within_margin"], charts.arm_color(v)) for v, s in rows],
                             title=f"{split}: peak predicted within ±25% (higher is better)"))
    rep.h("test: by driver (peak within ±25%)")
    keys = sorted({k for v in order for k in board[v] if k.startswith("test/driver=")})
    rep.table(["slice", *order], [[k[12:], *[f"{board[v].get(k, {}).get('within_margin', float('nan')):.0%}" for v in order]]
                                   for k in keys])
    await rep.flush()
    path = os.path.join(tempfile.mkdtemp(prefix="rsc-board-"), "leaderboard.json")
    with open(path, "w") as f:
        json.dump({"order": order, "board": board, "frontier": fronts}, f, indent=2, default=float)
    return await flyte.io.File.from_local(path)


_PEEK_DROP = ("task_code", "harness_code", "inputs_desc", "history_json", "inputs_json")


@cpubench_env.task
async def cpu_peek(corpus: flyte.io.File | None = None, artifact_file: flyte.io.File | None = None) -> str:
    """Corpus numbers (code dropped) or a JSON artifact, inline, readable from a laptop."""
    if corpus is not None:
        df = pd.read_parquet(await corpus.download())
        return df.drop(columns=[c for c in _PEEK_DROP if c in df.columns]).to_json(orient="records")
    with open(await artifact_file.download()) as f:
        return f.read()


@cpubench_env.task
async def cpu_pilot(release: str = "pilot", tasks_per_family: int = 1, shard_size: int = 4,
                    pairing: str = "auto") -> flyte.io.File:
    """generate -> measure, for shaking out the pipeline before a factory run."""
    runs = await generate_cpu_runs(release, tasks_per_family)
    return await measure_cpu_bench(runs, shard_size, pairing)


__all__ = ["grid"]
