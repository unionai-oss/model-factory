"""RightSizeGPU stations: the Flyte tasks the factory wires together.

    rt-gpubench-runs        generate_gpu_runs     code + inputs, unmeasured
    rt-gpubench-corpus      measure_gpu_bench     every run on a T4; CUDA OOMs again on an L40S
    rt-gpubench-arm         fit_gpu_arm           one arm: fit on train, predict val/test/ood
    rt-gpubench-scorecard   score_gpu_arm         device cost, memory and utilization accuracy
    rt-gpubench-leaderboard gpu_leaderboard       all arms side by side
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
from ..bench.common import (
    collect,
    hosted_subsample,
    pred_rows,
    preds_by_variant,
    read_arm,
    read_rows,
    shard,
    usable_by_split,
    write_arm_dir,
    write_parquet,
)
from ..bench.gpu_stations import train_llm_arm
from ..bench.llm_arms import zeroshot_predict
from ..bench.versioning import gpu_arms_cache, gpu_corpus_cache
from ..shared.reporting import Reporter
from . import arms as gpu_arms
from . import dataset, grid, scoring
from .envs import GPU_BIG, gpu_measure_big_env, gpu_measure_env, gpubench_env
from .measure import measure_run
from .spec import GPU

TITLE = GPU.title


@gpubench_env.task(cache=gpu_corpus_cache(), report=True)
async def generate_gpu_runs(release: str, tasks_per_family: int = 8, seed: int = 0,
                            runs_per_task: int = 2) -> flyte.io.File:
    rows = dataset.generate_runs(tasks_per_family, seed=seed, runs_per_task=runs_per_task)
    rep = Reporter(f"{TITLE}: workloads", f"release {release}")
    by = pd.DataFrame(rows).groupby(["family", "split"]).size().unstack(fill_value=0)
    rep.kv({"tasks": len({r["task_id"] for r in rows}), "runs": len(rows)})
    rep.table(["family", *by.columns], [[f, *map(str, v)] for f, v in zip(by.index, by.values.tolist())])
    await rep.flush()
    return await flyte.io.File.from_local(write_parquet(rows, "runs.parquet"))


def _measure_one(r: dict) -> dict:
    try:
        m = measure_run(r["harness_code"], json.loads(r["inputs_json"]), seed=r["seed"])
    except Exception as e:  # noqa: BLE001 -- a broken prep is a result, not a crash
        m = {"ok": False, "error": f"prep: {type(e).__name__}: {e}"[:500]}
    print(f"{r['run_id']}: ok={m.get('ok')} oom={m.get('censored')} peak={m.get('peak_gpu_mib', 0):.0f}MiB "
          f"util={m.get('mean_util', 0):.0f}% wall={m.get('wall_s', 0):.0f}s on {m.get('device_name')} "
          f"{str(m.get('error', ''))[:120]}", flush=True)
    return {"run_id": r["run_id"], **{k: v for k, v in m.items() if k != "result"}}


# One workload per GPU. max_runtime: a shard of 5 runs takes ~15-20 min; the
# timeout turns a node lost mid-shard into a retry.
@gpu_measure_env.task(retries=2, cache=gpu_corpus_cache(), timeout=flyte.Timeout(max_runtime=3600))
async def measure_gpu_shard(rows_json: str) -> str:
    return json.dumps([_measure_one(r) for r in json.loads(rows_json)])


@gpu_measure_big_env.task(retries=1, cache=gpu_corpus_cache(),
                          timeout=flyte.Timeout(max_runtime=3600, max_queued_time=2 * 3600))
async def measure_gpu_shard_big(rows_json: str) -> str:
    """Pass 2: runs that did not fit a T4, on a 48 GB L40S."""
    return json.dumps([_measure_one(r) for r in json.loads(rows_json)])


@gpubench_env.task(cache=gpu_corpus_cache(), report=True)
async def measure_gpu_bench(runs: flyte.io.File, shard_size: int = 5) -> flyte.io.File:
    rows = await read_rows(runs)
    keep = ("run_id", "harness_code", "inputs_json", "seed")
    shards = shard(rows, keep, shard_size)
    rep = Reporter(f"{TITLE}: measurement", f"{len(rows)} runs in {len(shards)} T4 shards")
    await rep.flush()
    measured: dict[str, dict] = {}
    lost = collect(await asyncio.gather(*(measure_gpu_shard(s) for s in shards), return_exceptions=True), measured)
    if lost > 0.1 * len(shards):
        raise RuntimeError(f"{lost}/{len(shards)} measurement shards failed; first error above")
    big = [r for r in rows if measured.get(r["run_id"], {}).get("censored")]
    lost_big = 0
    if big:
        rep.p(f"pass 2: {len(big)} runs hit a CUDA OOM on the T4; re-measuring on {GPU_BIG}")
        await rep.flush()
        before = dict(measured)
        lost_big = collect(await asyncio.gather(*(measure_gpu_shard_big(s) for s in shard(big, keep, 4)),
                                                return_exceptions=True), measured)
        for r in big:  # pass 2 lost: keep the T4 OOM (censored, excluded) rather than nothing
            measured.setdefault(r["run_id"], before[r["run_id"]])
    table = [dataset.attach_measurement(r, measured.get(r["run_id"], {"ok": False, "error": "shard lost"})) for r in rows]
    df = pd.DataFrame(dataset.attach_history(table))
    ok = df[df["ok"] & ~df["censored"]]
    rep.reset_body()
    rep.kv({"runs": len(df), "measured ok": len(ok), "CUDA OOM on every device": int(df["censored"].sum()),
            "failed": int((~df["ok"] & ~df["censored"]).sum()), "shards lost": f"{lost} + {lost_big}",
            "devices": ", ".join(sorted(set(ok["measured_on"]))) if len(ok) else "-",
            "median wall": f"{ok['wall_s'].median():.0f} s" if len(ok) else "-",
            "median DCGM samples / run": f"{(ok['wall_s'] / 30).median():.1f}" if len(ok) else "-"})
    fam = ok.groupby("family").agg(runs=("run_id", "size"), p50=("peak_gpu_gib", "median"),
                                   pmax=("peak_gpu_gib", "max"), util=("mean_util", "median"))
    rep.h("Per family")
    rep.table(["family", "runs", "median peak GiB", "max peak GiB", "median util %"],
              [[f, str(int(r.runs)), f"{r.p50:.2f}", f"{r.pmax:.2f}", f"{r.util:.0f}"] for f, r in fam.iterrows()])
    errs = df[~df["ok"]]["error"].str.slice(0, 120).value_counts().head(10)
    if len(errs):
        rep.h("Failures")
        rep.table(["error", "n"], [[e, str(n)] for e, n in errs.items()])
    await rep.flush()
    return await flyte.io.File.from_local(write_parquet(df.to_dict("records"), "corpus.parquet"))


CPU_ARMS = tuple(gpu_arms.ARMS)
GPU_ARMS = ("sft", "jev", "jev-pointer")  # LLM arms, trained on a T4
HOSTED_ARMS = {"zeroshot-27b": "qwen38-27b-vllm"}
ALL_ARMS = CPU_ARMS + tuple(HOSTED_ARMS) + GPU_ARMS
_EVAL_SPLITS = ("val", "test", "ood")
HOSTED_RUNS_PER_SPLIT = 300


@gpubench_env.task(cache=gpu_arms_cache(), report=True)
async def fit_gpu_arm(corpus: flyte.io.File, arm: str, llm_model: str = "Qwen/Qwen3-1.7B", llm_epochs: int = 3,
                      llm_max_train_rows: int = 0) -> flyte.io.Dir:
    rows = await read_rows(corpus)
    splits = usable_by_split(rows)
    rep = Reporter(f"{TITLE}: arm {arm}", ", ".join(f"{k}={len(v)}" for k, v in splits.items()))
    await rep.flush()
    if arm in CPU_ARMS:
        fit, predict = gpu_arms.ARMS[arm]
        state = fit(splits["train"], splits["val"])
        eval_rows = [r for s in _EVAL_SPLITS for r in splits[s]]
        t0 = time.perf_counter()
        preds = predict(state, eval_rows)
        meta = {"knobs": {k: v for k, v in state.items() if isinstance(v, (int, float, str))},
                "ms_per_decision": (time.perf_counter() - t0) * 1000 / max(1, len(eval_rows))}
    elif arm in HOSTED_ARMS:
        eval_rows = hosted_subsample(splits, HOSTED_RUNS_PER_SPLIT)
        preds, stats = await asyncio.to_thread(zeroshot_predict, eval_rows, HOSTED_ARMS[arm], spec=GPU)
        meta = {"stats": stats}
    elif arm in GPU_ARMS:
        return await train_llm_arm(corpus, arm, llm_model, llm_epochs, llm_max_train_rows, bench="gpu")
    else:
        raise ValueError(f"unknown arm {arm!r}; choose from {ALL_ARMS}")
    meta["arm"] = arm
    rep.kv({k: str(v)[:200] for k, v in meta.items()})
    await rep.flush()
    return await write_arm_dir(pred_rows(arm, preds, eval_rows, GPU), meta)


@gpubench_env.task(cache=gpu_arms_cache(), report=True)
async def score_gpu_arm(corpus: flyte.io.File, arm_dir: flyte.io.Dir, arm: str) -> flyte.io.File:
    rows = [r for r in await read_rows(corpus) if dataset.usable(r)]
    preds, meta = await read_arm(arm_dir)
    card = {"arm": arm, "meta": meta, "variants": {}, "frontier": {}}
    rep = Reporter(f"{TITLE}: scorecard {arm}")
    for variant, p in preds_by_variant(preds, GPU).items():
        card["variants"][variant] = scoring.score_slices(rows, p)
        test_rows = [r for r in rows if r["split"] == "test" and r["run_id"] in p]
        card["frontier"][variant] = scoring.frontier(test_rows, p, point_estimate=variant.endswith("-argmax"))
        t = card["variants"][variant].get("split=test", {})
        if t.get("n"):
            rep.kv({f"{variant} test": f"gpu regret {t['gpu_regret']:.1%}, OOM {t['oom_rate']:.1%}, "
                                       f"util MAE {t['util_mae']:.1f} pts"})
    await rep.flush()
    path = os.path.join(tempfile.mkdtemp(prefix="rsg-card-"), "scorecard.json")
    with open(path, "w") as f:
        json.dump(card, f, indent=2, default=float)
    return await flyte.io.File.from_local(path)


HEADLINE = ("static", "vpa", "size-rule", "gbt", "gbt-dist", "zeroshot-27b", "sft-greedy", "sft-sampled",
            "jev-argmax", "jev", "jev-pointer-argmax", "jev-pointer")


@gpubench_env.task(cache=gpu_arms_cache(), report=True)
async def gpu_leaderboard(scorecards: list[flyte.io.File]) -> flyte.io.File:
    cards = []
    for f in scorecards:
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
        rep.h(f"{split}: GPU memory -> device, and mean GPU utilization")
        rep.table(["arm", "memory within ±20%", "memory within 1 cell", "MAE GiB", "gpu $ regret", "OOM (1st try)",
                   "device = oracle", "util MAE (pts)", "util within ±10", "Brier", "ECE", "n"],
                  [[v, f"{s['mem_within_margin']:.1%}", f"{s['mem_within_one_cell']:.1%}", f"{s['mem_mae_gib']:.2f}",
                    f"{s['gpu_regret']:.1%}", f"{s['oom_rate']:.1%}", f"{s['device_exact']:.1%}", f"{s['util_mae']:.1f}",
                    f"{s['util_within_10']:.1%}", f"{s['brier']:.3f}" if "brier" in s else "-",
                    f"{s['ece_oom']:.3f}" if "ece_oom" in s else "-", str(s["n"])] for v, s in rows])
        rep.raw(charts.hbars([(v, s["mem_within_margin"], charts.arm_color(v)) for v, s in rows],
                             title=f"{split}: peak GPU memory predicted within ±20% (higher is better)"))
    rep.h("test: by driver (GPU memory within ±20%)")
    keys = sorted({k for v in order for k in board[v] if k.startswith("driver=")})
    rep.table(["slice", *order], [[k[7:], *[f"{board[v].get(k, {}).get('mem_within_margin', float('nan')):.0%}" for v in order]]
                                   for k in keys])
    await rep.flush()
    path = os.path.join(tempfile.mkdtemp(prefix="rsg-board-"), "leaderboard.json")
    with open(path, "w") as f:
        json.dump({"order": order, "board": board, "frontier": fronts}, f, indent=2, default=float)
    return await flyte.io.File.from_local(path)


_PEEK_DROP = ("task_code", "harness_code", "inputs_desc", "history_json", "inputs_json")


@gpubench_env.task
async def gpu_peek(corpus: flyte.io.File | None = None, artifact_file: flyte.io.File | None = None) -> str:
    """Corpus numbers (code dropped) or a JSON artifact, inline, readable from a laptop."""
    if corpus is not None:
        df = pd.read_parquet(await corpus.download())
        return df.drop(columns=[c for c in _PEEK_DROP if c in df.columns]).to_json(orient="records")
    with open(await artifact_file.download()) as f:
        return f.read()


@gpubench_env.task
async def gpu_pilot(release: str = "pilot", tasks_per_family: int = 1, shard_size: int = 4) -> flyte.io.File:
    """generate -> measure, for shaking out the pipeline before a factory run."""
    runs = await generate_gpu_runs(release, tasks_per_family)
    return await measure_gpu_bench(runs, shard_size)


__all__ = ["grid"]
