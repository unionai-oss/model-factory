"""RightSizeGPU stations: the Flyte tasks the factory wires together.

    rt-gpubench-runs        generate_gpu_runs     code + inputs, unmeasured
    rt-gpubench-corpus      measure_gpu_bench     every run on a T4; CUDA OOMs climb the machine ladder
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
from ..bench.versioning import gpu_arms_cache, gpu_corpus_cache
from ..shared.reporting import Reporter
from . import arms as gpu_arms
from . import dataset, grid, scoring
from .envs import (
    gpu_measure_big_env,
    gpu_measure_env,
    gpu_measure_l4x4_env,
    gpu_measure_l40sx4_env,
    gpu_measure_t4x8_env,
    gpubench_env,
)
from .measure import measure_many, measure_run
from .spec import GPU

TITLE = GPU.title


#: Tasks per family by release prefix. A full release is ~4x small: the
#: learned arms were data-starved at small (GPU jev-pointer fit a constant).
#: big is RightSizeMem v2's order of magnitude: 10 families x 500 x 2 =
#: 10,000 runs (big-1 had 8 families x 625).
RELEASE_TASKS = {"small": 8, "full": 40, "big": 500}


@gpubench_env.task(cache=gpu_corpus_cache(), report=True)
async def generate_gpu_runs(release: str, tasks_per_family: int = 0, seed: int = 0,
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


def _measure_one(r: dict, gpus: list[int] | None = None) -> dict:
    try:
        m = measure_run(r["harness_code"], json.loads(r["inputs_json"]), seed=r["seed"], gpus=gpus)
    except Exception as e:  # noqa: BLE001 -- a broken prep is a result, not a crash
        m = {"ok": False, "error": f"prep: {type(e).__name__}: {e}"[:500]}
    print(f"{r['run_id']}: ok={m.get('ok')} oom={m.get('censored')} peak={m.get('peak_gpu_mib', 0):.0f}MiB "
          f"on {m.get('gpu_n_used', 0)} GPU(s) util={m.get('mean_util', 0):.0f}% wall={m.get('wall_s', 0):.0f}s "
          f"on {m.get('device_name')} {str(m.get('error', ''))[:120]}", flush=True)
    return {"run_id": r["run_id"], **{k: v for k, v in m.items() if k != "result"}}


def _shard(rows_json: str, mode: str) -> str:
    return json.dumps(measure_many(json.loads(rows_json), _measure_one, mode))


# One workload per GPU. max_runtime: a shard of 3 runs takes ~10-12 min (6
# runs of a big release's shorter budgets, about the same); the timeout turns
# a node lost mid-shard into a retry.
@gpu_measure_env.task(retries=2, cache=gpu_corpus_cache(), timeout=flyte.Timeout(max_runtime=1800))
async def measure_gpu_shard(rows_json: str) -> str:
    return _shard(rows_json, "per_gpu")


@gpu_measure_big_env.task(retries=1, cache=gpu_corpus_cache(),
                          timeout=flyte.Timeout(max_runtime=3600, max_queued_time=8 * 3600))
async def measure_gpu_shard_big(rows_json: str) -> str:
    """Runs that did not fit a smaller GPU, on a 48 GB L40S."""
    return _shard(rows_json, "per_gpu")


@gpu_measure_l4x4_env.task(retries=2, cache=gpu_corpus_cache(),
                           timeout=flyte.Timeout(max_runtime=3600, max_queued_time=2 * 3600))
async def measure_gpu_shard_l4x4(rows_json: str) -> str:
    """Single-GPU runs that did not fit a T4: four at a time, one per L4."""
    return _shard(rows_json, "per_gpu")


@gpu_measure_t4x8_env.task(retries=2, cache=gpu_corpus_cache(),
                           timeout=flyte.Timeout(max_runtime=3600, max_queued_time=2 * 3600))
async def measure_gpu_shard_t4x8(rows_json: str) -> str:
    """Sharded runs that did not fit one T4: each run across all eight."""
    return _shard(rows_json, "whole")


# The g6e.12xlarge pool did not admit a pod in an hour (rt-gpuprobe-1): single-GPU
# runs wait an hour for it, then fall back to single L40S nodes (`_climb`).
@gpu_measure_l40sx4_env.task(retries=1, cache=gpu_corpus_cache(),
                             timeout=flyte.Timeout(max_runtime=3600, max_queued_time=3600))
async def measure_gpu_shard_l40sx4(rows_json: str) -> str:
    """Single-GPU runs that did not fit an L4, four at a time, one per L40S."""
    return _shard(rows_json, "per_gpu")


@gpu_measure_l40sx4_env.task(retries=1, cache=gpu_corpus_cache(),
                             timeout=flyte.Timeout(max_runtime=3600, max_queued_time=2 * 3600))
async def measure_gpu_shard_l40sx4_whole(rows_json: str) -> str:
    """The top of the ladder: sharded runs that did not fit 8 T4s, each across all four L40S."""
    return _shard(rows_json, "whole")


#: Rows per shard on each pass: pass 1 is 6 runs on one T4; a 4-GPU node takes
#: 4 at once; a whole-node shard runs its runs one after another.
_PASS_SHARD = {"t4": 6, "l4x4": 16, "l40sx4": 12, "l40s": 4, "t4x8": 4, "l40sx4_whole": 3}


async def _run_shards(name: str, shards: list[str], call, measured: dict[str, dict]) -> list[str]:
    """Measure `shards` with `call`; returns the shards that were lost (queue timeout, node loss)."""
    res = await asyncio.gather(*(call(s) for s in shards), return_exceptions=True)
    lost = []
    for s_, r in zip(shards, res):
        if isinstance(r, BaseException):
            print(f"{name} shard lost: {type(r).__name__}: {str(r)[:200]}", flush=True)
            lost.append(s_)
        else:
            for m in json.loads(r):
                measured[m["run_id"]] = m
    return lost


async def _climb(rows: list[dict], measured: dict[str, dict], keep: tuple, rep) -> dict[str, int]:
    """Re-measure each run that CUDA-OOMed, one machine up its ladder (envs.py).

    The L40S pools admit slowly and either may be full (rt-gpuprobe-1: one
    g6e.2xlarge in 37 min, the g6e.12xlarge pool not at all), so a 4xL40S
    shard lost in the queue is re-sent as single-L40S shards. A run whose
    shards are all lost keeps the OOM below (censored, excluded)."""
    lost: dict[str, int] = {}
    sharded = {r["run_id"]: bool(r.get("sharded")) for r in rows}

    def oomed(r):
        return bool(measured.get(r["run_id"], {}).get("censored"))

    async def one_pass(name: str, todo: list[dict], call, fallback=None) -> None:
        if not todo:
            return
        rep.p(f"{name}: {len(todo)} runs that CUDA-OOMed the machine below")
        await rep.flush()
        missed = await _run_shards(name, shard(todo, keep, _PASS_SHARD[name]), call, measured)
        if missed and fallback is not None:
            fb_name, fb_call = fallback
            again = [r for s_ in missed for r in json.loads(s_)]
            rep.p(f"{name}: {len(missed)} shards lost; {len(again)} runs re-sent to {fb_name}")
            await rep.flush()
            missed = await _run_shards(fb_name, shard(again, keep, _PASS_SHARD[fb_name]), fb_call, measured)
        lost[name] = len(missed)

    await asyncio.gather(
        one_pass("l4x4", [r for r in rows if oomed(r) and not sharded[r["run_id"]]], measure_gpu_shard_l4x4),
        one_pass("t4x8", [r for r in rows if oomed(r) and sharded[r["run_id"]]], measure_gpu_shard_t4x8))
    await asyncio.gather(
        one_pass("l40sx4", [r for r in rows if oomed(r) and not sharded[r["run_id"]]],
                 measure_gpu_shard_l40sx4, fallback=("l40s", measure_gpu_shard_big)),
        one_pass("l40sx4_whole", [r for r in rows if oomed(r) and sharded[r["run_id"]]],
                 measure_gpu_shard_l40sx4_whole))
    return lost


@gpubench_env.task(cache=gpu_corpus_cache(), report=True)
async def measure_gpu_bench(runs: flyte.io.File, shard_size: int = 3) -> flyte.io.File:
    rows = await read_rows(runs)
    keep = ("run_id", "harness_code", "inputs_json", "seed")
    if rows and rows[0].get("profile") in ("big", "big2"):
        shard_size = _PASS_SHARD["t4"]  # short budgets; ~1,700 shards instead of ~3,300
    shards = shard(rows, keep, shard_size)
    rep = Reporter(f"{TITLE}: measurement", f"{len(rows)} runs in {len(shards)} T4 shards")
    await rep.flush()
    measured: dict[str, dict] = {}
    lost = collect(await asyncio.gather(*(measure_gpu_shard(s) for s in shards), return_exceptions=True), measured)
    if lost > 0.1 * len(shards):
        raise RuntimeError(f"{lost}/{len(shards)} measurement shards failed; first error above")
    lost_up = await _climb(rows, measured, keep, rep)
    table = [dataset.attach_measurement(r, measured.get(r["run_id"], {"ok": False, "error": "shard lost"})) for r in rows]
    df = pd.DataFrame(dataset.attach_history(table))
    ok = df[df["ok"] & ~df["censored"]]
    rep.reset_body()
    rep.kv({"runs": len(df), "measured ok": len(ok), "CUDA OOM on every machine": int(df["censored"].sum()),
            "failed": int((~df["ok"] & ~df["censored"]).sum()),
            "shards lost": f"t4 {lost}, " + ", ".join(f"{k} {v}" for k, v in lost_up.items()),
            "machines": ", ".join(sorted(set(ok["measured_on"]))) if len(ok) else "-",
            "median wall": f"{ok['wall_s'].median():.0f} s" if len(ok) else "-",
            "median DCGM samples / run": f"{(ok['wall_s'] / 30).median():.1f}" if len(ok) else "-"})
    if len(ok):
        shapes = ok.assign(shape=[grid.DEVICES[grid.oracle_device(p, s, i) or 0].name for p, s, i in
                                  zip(ok["peak_gpu_gib"], ok["sharded"], ok["gpu_imbalance"])]).groupby("shape").size()
        rep.h("Cheapest machine that fits (usable runs)")
        rep.table(["machine", "runs"], [[k, str(int(v))] for k, v in shapes.items()])
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
GPU_ARMS = ("sft", "jev", "jev-pointer", "jev-pointer-4b")  # LLM arms, trained on a T4
#: The stacked hybrids (``bench.hybrid``): each LLM arm with the trees' prediction in its prompt.
STACK_ARMS = tuple(f"{hybrid.STACK_PREFIX}{a}" for a in ("sft", "jev-pointer", "jev-pointer-4b"))
HOSTED_ARMS = {"zeroshot-27b": "qwen38-27b-vllm"}
ALL_ARMS = CPU_ARMS + tuple(HOSTED_ARMS) + GPU_ARMS + STACK_ARMS
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
        # 14 h, not train_llm_arm's 8: big-1's sft arm (5,129 train runs, ~4,800
        # to decode greedy + 8 samples) ran past 8 h. Overridden here, not in
        # gpu_stations.py, which every LLM arm's cache version hashes.
        return await train_llm_arm.override(timeout=flyte.Timeout(max_runtime=14 * 3600))(
            corpus, arm, llm_model, llm_epochs, llm_max_train_rows, bench="gpu")
    elif arm in STACK_ARMS:
        fit, predict = gpu_arms.ARMS[hybrid.ANCHOR]
        hinted, hmeta = await hinted_corpus(rows, fit, predict, GPU)
        rep.kv({k: str(v)[:300] for k, v in hmeta.items()})
        await rep.flush()
        return await train_llm_arm.override(timeout=flyte.Timeout(max_runtime=14 * 3600))(
            hinted, arm, llm_model, llm_epochs, llm_max_train_rows, bench="gpu")
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


@gpubench_env.task(cache=gpu_arms_cache(), report=True)
async def pool_gpu_arms(corpus: flyte.io.File, arm_dirs: list[flyte.io.Dir]) -> flyte.io.Dir:
    """The pooled hybrids: gbt-dist x every distributional LLM variant (``bench.hybrid``)."""
    rows = [r for r in await read_rows(corpus) if dataset.usable(r)]
    out = await pooled_arm(rows, arm_dirs, GPU)
    rep = Reporter(f"{TITLE}: pooled hybrids", f"{hybrid.ANCHOR} x each LLM variant, weight tuned on val")
    _preds, meta = await read_arm(out)
    rep.table(["pool", "w (trees)", "val headline (lower is better)", "secondary from"],
              [[v, str(k["w_trees"]), f"{k['val_headline']:.3f}", k["secondary_from"]] for v, k in meta["knobs"].items()])
    await rep.flush()
    return out


HEADLINE = ("static", "vpa", "size-rule", "gbt", "gbt-dist", "zeroshot-27b", "sft-greedy", "sft-sampled",
            "jev-argmax", "jev", "jev-pointer-argmax", "jev-pointer",
            "jev-pointer-4b-argmax", "jev-pointer-4b",
            "stack-sft-greedy", "stack-sft-sampled", "stack-jev-pointer-argmax", "stack-jev-pointer",
            "stack-jev-pointer-4b-argmax", "stack-jev-pointer-4b")


@gpubench_env.task(cache=gpu_arms_cache(), report=True)
async def gpu_leaderboard(scorecards: list[flyte.io.File], pool_scorecard: flyte.io.File | None = None) -> flyte.io.File:
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
