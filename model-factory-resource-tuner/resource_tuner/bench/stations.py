"""RightSizeMem Bench stations: the Flyte tasks the factory wires together.

    rt-bench-runs        generate_bench_runs   code + inputs, unmeasured
    rt-bench-corpus      measure_bench         every run executed and measured
    rt-bench-arm         fit_bench_arm         one arm: fit on train, predict val/test/ood
    rt-bench-scorecard   score_bench_arm       cost, OOMs, regret vs oracle, calibration
    rt-bench-leaderboard bench_leaderboard     all arms side by side
    rt-bench-live        live_validate         chosen requests, run for real

Station tasks return plain File/Dir; the factory publishes the artifacts.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import tempfile
import time
from pathlib import Path
from random import Random

import flyte
import flyte.io
import numpy as np
import pandas as pd

from ..shared.reporting import Reporter
from . import dataset
from . import arms as cpu_arms
from . import charts, grid, scoring
from .envs import (
    BIG_GUARD_MIB,
    MEASURE_CONCURRENCY,
    MEASURE_GUARD_MIB,
    bench_env,
    live_env,
    measure_big_env,
    measure_env,
)
from .gpu_stations import train_llm_arm
from .llm_arms import zeroshot_predict
from .measure import measure_run
from .versioning import arms_cache, corpus_cache


def _write_parquet(rows: list[dict], name: str) -> str:
    path = os.path.join(tempfile.mkdtemp(prefix="rsb-"), name)
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


async def _read_rows(f: flyte.io.File) -> list[dict]:
    local = await f.download()
    return pd.read_parquet(local).to_dict("records")


# ── 1. generate ─────────────────────────────────────────────────────────


@bench_env.task(cache=corpus_cache(), report=True)
async def generate_bench_runs(release: str, tasks_per_family: int = 200, seed: int = 0) -> flyte.io.File:
    """Sample tasks from every family and expand each into its runs."""
    rows = dataset.generate_runs(tasks_per_family, seed=seed)
    rep = Reporter("RightSizeMem Bench: workloads", f"release {release}")
    by = pd.DataFrame(rows).groupby(["family", "split"]).size().unstack(fill_value=0)
    rep.kv({"tasks": len({r["task_id"] for r in rows}), "runs": len(rows)})
    rep.table(["family", *by.columns], [[f, *map(str, v)] for f, v in zip(by.index, by.values.tolist())])
    await rep.flush()
    return await flyte.io.File.from_local(_write_parquet(rows, "runs.parquet"))


# ── 2. measure ──────────────────────────────────────────────────────────


def _measure_one(r: dict, guard_mib: float) -> dict:
    try:
        m = measure_run(r["harness_code"], json.loads(r["inputs_json"]), seed=r["seed"],
                        kill_above_mib=guard_mib, timeout_s=900)
    except Exception as e:  # noqa: BLE001 -- a broken prep is a result, not a crash
        m = {"ok": False, "error": f"prep: {type(e).__name__}: {e}"[:500]}
    print(f"{r['run_id']}: ok={m.get('ok')} peak={m.get('peak_rss_mib', 0):.0f}MiB "
          f"wall={m.get('wall_s', 0):.1f}s {m.get('error', '')[:120]}", flush=True)
    return {"run_id": r["run_id"], **{k: v for k, v in m.items() if k != "result"}}


def _measure_rows(rows: list[dict], guard_mib: float, concurrency: int = 1) -> list[dict]:
    """Execute and measure runs, each in its own interpreter.

    `ru_maxrss` is per process, so running `concurrency` workloads side by
    side does not blur anyone's peak -- what it costs is the pod's memory,
    which is why each process gets its own guard (pod limit / concurrency,
    roughly) and a workload heading past it is killed and recorded as
    censored rather than taking the pod, and every peak in it, down. The
    project caps concurrent actions (~30 here), so runs per pod is the
    throughput lever, not pods.
    """
    from concurrent.futures import ThreadPoolExecutor

    if concurrency <= 1:
        return [_measure_one(r, guard_mib) for r in rows]
    with ThreadPoolExecutor(concurrency) as pool:
        return list(pool.map(lambda r: _measure_one(r, guard_mib), rows))


# retries=3: pass 1 runs on interruptible capacity, so a spot reclaim is
# an expected event, not a failure. max_runtime=1h (a shard normally takes
# 20-30 min) is what turns a reclaim into a retry: in rsb-v2-corpus-1, ~25
# shards whose pods were "terminated in response to imminent node shutdown"
# stayed RUNNING for 40+ min instead of failing over.
@measure_env.task(retries=3, cache=corpus_cache(), timeout=flyte.Timeout(max_runtime=3600))
async def measure_shard(rows_json: str, guard_mib: float = MEASURE_GUARD_MIB,
                        concurrency: int = MEASURE_CONCURRENCY) -> str:
    """Pass 1: measure a shard of runs, `concurrency` at a time."""
    return json.dumps(_measure_rows(json.loads(rows_json), guard_mib, concurrency))


@measure_big_env.task(retries=1, cache=corpus_cache(), timeout=flyte.Timeout(max_runtime=4 * 3600))
async def measure_shard_big(rows_json: str, guard_mib: float = BIG_GUARD_MIB) -> str:
    """Pass 2: re-measure the runs pass 1's guard censored, on a 24Gi pod."""
    return json.dumps(_measure_rows(json.loads(rows_json), guard_mib))


def _collect(results, measured: dict[str, dict]) -> int:
    lost = 0
    for res in results:
        if isinstance(res, BaseException):
            lost += 1
            print(f"shard failed: {res}")
            continue
        for m in json.loads(res):
            measured[m["run_id"]] = m
    return lost


@bench_env.task(cache=corpus_cache(), report=True)
async def measure_bench(runs: flyte.io.File, shard_size: int = 60) -> flyte.io.File:
    """Measure every run, in parallel shards; return the labelled corpus."""
    rows = await _read_rows(runs)
    order = list(range(len(rows)))
    Random(0).shuffle(order)  # spread the heavy families across shards
    keep = ("run_id", "harness_code", "inputs_json", "seed")
    shards = [
        json.dumps([{k: rows[i][k] for k in keep} for i in order[s : s + shard_size]])
        for s in range(0, len(order), shard_size)
    ]
    rep = Reporter("RightSizeMem Bench: measurement", f"{len(rows)} runs in {len(shards)} shards")
    await rep.flush()
    measured: dict[str, dict] = {}
    lost = _collect(await asyncio.gather(*(measure_shard(s) for s in shards), return_exceptions=True), measured)
    if lost > 0.1 * len(shards):
        # A corpus with holes is not a release. rsb-pilot-2 "succeeded" with
        # every shard lost to one bad input type; never again.
        raise RuntimeError(f"{lost}/{len(shards)} measurement shards failed; first error above")
    big = [rows[i] for i in order if measured.get(rows[i]["run_id"], {}).get("censored")]
    if big:
        big_shards = [json.dumps([{k: r[k] for k in keep} for r in big[s : s + 8]]) for s in range(0, len(big), 8)]
        rep.p(f"pass 2: re-measuring {len(big)} censored runs on 24Gi pods")
        await rep.flush()
        lost += _collect(await asyncio.gather(*(measure_shard_big(s) for s in big_shards), return_exceptions=True),
                         measured)
    table = [
        dataset.attach_measurement(r, measured.get(r["run_id"], {"ok": False, "error": "shard lost"}))
        for r in rows
    ]
    table = dataset.attach_history(table)
    df = pd.DataFrame(table)
    ok = df[df["ok"] & ~df["censored"]]
    rep.reset_body()
    rep.kv({
        "runs": len(df), "measured ok": len(ok), "censored (guard)": int(df["censored"].sum()),
        "failed": int((~df["ok"] & ~df["censored"]).sum()), "shards lost": lost,
        "median peak": f"{ok['peak_rss_mib'].median():.0f} MiB",
        "p95 peak": f"{ok['peak_rss_mib'].quantile(.95):.0f} MiB",
    })
    fam = ok.groupby("family").agg(runs=("run_id", "size"), p50=("peak_rss_mib", "median"),
                                   p95=("peak_rss_mib", lambda s: s.quantile(.95)),
                                   gib_in=("total_input_bytes", lambda s: s.median() / 2**30))
    rep.h("Per family (MiB peak; median GiB of input)")
    rep.table(["family", "runs", "p50", "p95", "input"],
              [[f, str(int(r.runs)), f"{r.p50:.0f}", f"{r.p95:.0f}", f"{r.gib_in:.2f}"] for f, r in fam.iterrows()])
    errs = df[~df["ok"] & ~df["censored"]]["error"].str.slice(0, 120).value_counts().head(10)
    if len(errs):
        rep.h("Failures")
        rep.table(["error", "n"], [[e, str(n)] for e, n in errs.items()])
    await rep.flush()
    return await flyte.io.File.from_local(_write_parquet(df.to_dict("records"), "corpus.parquet"))


@bench_env.task
async def bench_pilot(release: str = "pilot", tasks_per_family: int = 4, shard_size: int = 12) -> flyte.io.File:
    """generate -> measure in one run, for shaking out the pipeline before
    a factory materialization commits to a full release."""
    runs = await generate_bench_runs(release, tasks_per_family)
    return await measure_bench(runs, shard_size)


# ── 3. arms ─────────────────────────────────────────────────────────────

#: arm (partition value) -> where it runs. Every arm writes the same
#: predictions table, possibly for several decoding *variants*.
CPU_ARMS = ("static", "vpa", "size-rule", "gbt", "gbt-dist")
GPU_ARMS = ("sft", "jev", "jev-pointer", "jev-num", "jev-pointer-4b")
#: hosted zero-shot arms -> the llm-service app each one asks
# (the llama.cpp qwen38-27b could not get an L40S on 2026-10-16; its vLLM
# build was up -- same weights, so the arm points there)
HOSTED_ARMS = {"zeroshot": "qwen35-397b", "zeroshot-27b": "qwen38-27b-vllm"}
ALL_ARMS = CPU_ARMS + tuple(HOSTED_ARMS) + GPU_ARMS

_EVAL_SPLITS = ("val", "test", "ood")


#: Runs per split a hosted (zero-shot) arm is asked about.
HOSTED_RUNS_PER_SPLIT = 300


def hosted_subsample(splits: dict[str, list[dict]], n: int = HOSTED_RUNS_PER_SPLIT) -> list[dict]:
    """Deterministic, family-stratified test+ood subsample."""
    out = []
    for split in ("test", "ood"):
        by_fam: dict[str, list[dict]] = {}
        for r in sorted(splits[split], key=lambda r: r["run_id"]):
            by_fam.setdefault(r["family"], []).append(r)
        rng = Random(17)
        per = max(1, n // max(1, len(by_fam)))
        for fam in sorted(by_fam):
            out += rng.sample(by_fam[fam], min(per, len(by_fam[fam])))
    return out


def _usable_by_split(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {s: [] for s in dataset.SPLITS}
    for r in rows:
        if dataset.usable(r):
            out[r["split"]].append(r)
    return out


def _pred_rows(variant: str, preds: dict, rows: list[dict]) -> list[dict]:
    split = {r["run_id"]: r["split"] for r in rows}
    return [
        {"variant": variant, "run_id": rid, "split": split.get(rid, ""), "mem_idx": int(p["mem_idx"]),
         "cpu_idx": int(p["cpu_idx"]),
         "mem_probs": json.dumps([round(float(x), 6) for x in p["mem_probs"]]) if p.get("mem_probs") is not None else "",
         "raw": str(p.get("raw", ""))[:200]}
        for rid, p in preds.items()
    ]


async def _write_arm_dir(pred_rows: list[dict], meta: dict) -> flyte.io.Dir:
    d = tempfile.mkdtemp(prefix="rsb-arm-")
    pd.DataFrame(pred_rows).to_parquet(os.path.join(d, "predictions.parquet"), index=False)
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)
    return await flyte.io.Dir.from_local(d)


def _fit_cpu_arm(arm: str, splits: dict[str, list[dict]]) -> tuple[list[dict], dict]:
    fit, predict = cpu_arms.ARMS[arm]
    state = fit(splits["train"], splits["val"])
    rows = [r for s in _EVAL_SPLITS for r in splits[s]]
    t0 = time.perf_counter()
    preds = predict(state, rows)
    ms = (time.perf_counter() - t0) * 1000 / max(1, len(rows))
    knobs = {k: v for k, v in state.items() if isinstance(v, (int, float, str))}
    return _pred_rows(arm, preds, rows), {"knobs": knobs, "ms_per_decision": ms}


@bench_env.task(cache=arms_cache(), report=True)
async def fit_bench_arm(corpus: flyte.io.File, arm: str, llm_model: str = "Qwen/Qwen3-1.7B", llm_epochs: int = 3,
                        llm_max_train_rows: int = 0) -> flyte.io.Dir:
    """Fit one arm on train (knob tuned on val); predict val/test/ood.

    CPU arms run here; LLM arms train on the GPU env (this task only
    awaits them -- a GPU parent awaiting a GPU child deadlocks a small pool).
    The llm_* knobs are typed (not a JSON string) so a materialization can
    set them with `--param rt-bench-arm.llm_epochs=1`.
    """
    rows = await _read_rows(corpus)
    splits = _usable_by_split(rows)
    rep = Reporter(f"RightSizeMem Bench: arm {arm}", ", ".join(f"{k}={len(v)}" for k, v in splits.items()))
    await rep.flush()
    if arm in CPU_ARMS:
        pred_rows, meta = _fit_cpu_arm(arm, splits)
    elif arm in HOSTED_ARMS:
        # A stratified subsample: at ~1k reasoning tokens per decision, a
        # frontier call on every run is hours (v1's 2,318 took >4 h). The
        # leaderboard scores every arm on this same subset as well.
        eval_rows = hosted_subsample(splits)
        preds, stats = await asyncio.to_thread(zeroshot_predict, eval_rows, HOSTED_ARMS[arm])
        pred_rows, meta = _pred_rows(arm, preds, eval_rows), {"stats": stats}
    elif arm in GPU_ARMS:
        return await train_llm_arm(corpus, arm, llm_model, llm_epochs, llm_max_train_rows)
    else:
        raise ValueError(f"unknown arm {arm!r}; choose from {ALL_ARMS}")
    meta["arm"] = arm
    rep.kv({k: str(v)[:200] for k, v in meta.items()})
    await rep.flush()
    return await _write_arm_dir(pred_rows, meta)


# ── 4. score ────────────────────────────────────────────────────────────


async def _read_arm(d: flyte.io.Dir) -> tuple[pd.DataFrame, dict]:
    local = await d.download()
    preds = pd.read_parquet(os.path.join(local, "predictions.parquet"))
    with open(os.path.join(local, "meta.json")) as f:
        meta = json.load(f)
    return preds, meta


def _preds_by_variant(preds: pd.DataFrame) -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = {}
    for r in preds.to_dict("records"):
        out.setdefault(r["variant"], {})[r["run_id"]] = {
            "mem_idx": r["mem_idx"], "cpu_idx": r["cpu_idx"],
            "mem_probs": json.loads(r["mem_probs"]) if r.get("mem_probs") else None,
        }
    return out


@bench_env.task(cache=arms_cache(), report=True)
async def score_bench_arm(corpus: flyte.io.File, arm_dir: flyte.io.Dir, arm: str) -> flyte.io.File:
    """Score every decoding variant of one arm on every split and slice."""
    rows = [r for r in await _read_rows(corpus) if dataset.usable(r)]
    preds, meta = await _read_arm(arm_dir)
    card = {"arm": arm, "meta": meta, "variants": {}}
    subset_ids = {r["run_id"] for r in hosted_subsample(_usable_by_split(rows))}
    rep = Reporter(f"RightSizeMem Bench: scorecard {arm}")
    for variant, p in _preds_by_variant(preds).items():
        card["variants"][variant] = scoring.score_slices(rows, p)
        card["variants"][variant]["subset=hosted"] = scoring.score([r for r in rows if r["run_id"] in subset_ids], p)
        test_rows = [r for r in rows if r["split"] == "test" and r["run_id"] in p]
        card.setdefault("frontier", {})[variant] = scoring.frontier(test_rows, p, point_estimate=variant.endswith("-argmax"))
        test = card["variants"][variant].get("split=test", {})
        rep.kv({f"{variant} test": f"memory regret {test.get('mem_regret', float('nan')):.1%}, "
                                   f"OOM {test.get('oom_rate', float('nan')):.1%}, $ regret {test.get('regret', float('nan')):.1%}"})
    await rep.flush()
    path = os.path.join(tempfile.mkdtemp(prefix="rsb-card-"), "scorecard.json")
    with open(path, "w") as f:
        json.dump(card, f, indent=2, default=float)
    return await flyte.io.File.from_local(path)


# ── 5. leaderboard ──────────────────────────────────────────────────────

#: Variants on the headline board, in order. Anything else an arm emits is
#: still in the JSON, just not on the chart.
HEADLINE = ("static", "vpa", "size-rule", "gbt", "gbt-dist", "zeroshot-27b", "zeroshot",
            "sft-greedy", "sft-sampled", "jev-argmax", "jev", "jev-pointer-argmax", "jev-pointer",
            "jev-num-argmax", "jev-num", "jev-pointer-4b-argmax", "jev-pointer-4b", "ensemble")


@bench_env.task(cache=arms_cache(), report=True)
async def bench_leaderboard(scorecards: list[flyte.io.File],
                            ensemble_scorecard: flyte.io.File | None = None) -> flyte.io.File:
    """Every arm side by side: the board, the frontier, and the slices."""
    cards = []
    for f in [*scorecards, *([ensemble_scorecard] if ensemble_scorecard is not None else [])]:
        with open(await f.download()) as fh:
            cards.append(json.load(fh))
    board: dict[str, dict] = {}
    meta: dict[str, dict] = {}
    fronts: dict[str, list[dict]] = {}
    for c in cards:
        for variant, slices in c["variants"].items():
            board[variant] = slices
            meta[variant] = c["meta"]
        fronts.update(c.get("frontier", {}))
    order = [v for v in HEADLINE if v in board] + sorted(v for v in board if v not in HEADLINE)
    rep = Reporter("RightSizeMem Bench: leaderboard", f"{len(order)} decision rules, {len(cards)} arms")
    for split in ("test", "ood"):
        key = f"split={split}"
        rows = [(v, board[v].get(key, {})) for v in order if board[v].get(key, {}).get("n")]
        if not rows:
            continue
        static = board.get("static", {}).get(key, {}).get("gib_hours_per_run")
        rep.h(f"{split}: memory reserved to get every run to SUCCEEDED (CPU held at oracle)")
        rep.table(
            ["arm", "memory regret", "vs static", "OOM (1st try)", "attempts", "median over-prov.",
             "$ regret (packed)", "$ regret (fargate)", "CPU exact", "Brier", "ECE(P[OOM])"],
            [[v, f"{s['mem_regret']:.1%}",
              f"{s['gib_hours_per_run'] / static - 1:+.1%}" if static else "-",
              f"{s['oom_rate']:.1%}", f"{s['mean_attempts']:.2f}", f"{s['median_over_provision']:.0%}",
              f"{s['regret']:.1%}", f"{s['fargate_regret']:.1%}", f"{s['cpu_exact']:.0%}",
              f"{s['brier']:.3f}" if "brier" in s else "-",
              f"{s['ece_oom']:.3f}" if "ece_oom" in s else "-"] for v, s in rows],
        )
        rep.raw(charts.hbars([(v, s["mem_regret"], charts.arm_color(v)) for v, s in rows],
                             title=f"{split}: memory regret vs oracle (lower is better)"))
        rep.raw(charts.scatter([(v, s["oom_rate"], s["mem_regret"], charts.arm_color(v)) for v, s in rows],
                               "first-attempt OOM rate", "memory regret vs oracle"))
    sub = {v: board[v].get("subset=hosted", {}) for v in order}
    if any(x.get("n") for x in sub.values()):
        rep.h("Same runs for everyone: the hosted arms' stratified subsample (test+ood)")
        rep.table(["arm", "memory regret", "OOM (1st try)", "n"],
                  [[v, f"{x['mem_regret']:.1%}", f"{x['oom_rate']:.1%}", str(x["n"])] for v, x in sub.items() if x.get("n")])
    if fronts:
        rep.h("test: risk frontier -- OOM rate vs cost as each arm's knob moves")
        rep.p("Distributional arms re-decode the same predictions under OOM penalties 0-16x (no retraining); "
              "point-estimate arms can only add or remove margin.")
        series = [(v, charts.PALETTE[i % len(charts.PALETTE)], [(p["oom_rate"], p["mem_regret"]) for p in fronts[v]])
                  for i, v in enumerate(o for o in order if o in fronts)]
        rep.raw(charts.frontier_chart(series, "first-attempt OOM rate", "memory regret vs oracle"))
    rep.h("Where the wins come from (test memory regret by slice)")
    slice_keys = sorted({k for v in order for k in board[v] if k.startswith(("driver=", "test/history="))})
    rep.table(["slice", *order],
              [[k, *[f"{board[v].get(k, {}).get('mem_regret', float('nan')):.1%}" for v in order]] for k in slice_keys])
    fam_keys = sorted({k for v in order for k in board[v] if k.startswith("family=")})
    rep.h("By family (memory regret, val+test+ood)")
    rep.table(["family", *order],
              [[k[7:], *[f"{board[v].get(k, {}).get('mem_regret', float('nan')):.1%}" for v in order]] for k in fam_keys])
    await rep.flush()
    path = os.path.join(tempfile.mkdtemp(prefix="rsb-board-"), "leaderboard.json")
    with open(path, "w") as f:
        json.dump({"order": order, "board": board, "meta": meta, "frontier": fronts}, f, indent=2, default=float)
    return await flyte.io.File.from_local(path)


# ── 6. live validation ──────────────────────────────────────────────────

#: Decision rules re-run for real. One per arm family, the strongest
#: decoding of each, so the live table reads like the leaderboard.
LIVE_VARIANTS = ("static", "size-rule", "gbt-dist", "sft-greedy", "jev-pointer")
#: Largest request a CPU node here can schedule (c5.4xlarge allocatable).
LIVE_MAX_MEMORY_MIB = 24 * 1024


# max_queued_time: long enough to wait out the project's action cap (rsb-v1-live-1 lost
# 161 of 600 pods to a 15 min queue limit while v2 measurement held the slots),
# short enough that a request no node can hold still fails rather than hangs.
@live_env.task(retries=0, timeout=flyte.Timeout(max_runtime=1800, max_queued_time=3 * 3600))
async def live_run(harness_code: str, inputs_json: str, seed: int) -> str:
    """One workload, in a pod sized by an arm's decision. An OOM never
    returns: the kernel kills the pod and the caller sees a failed action."""
    import resource

    os.environ["RSB_PREP_CHUNK_CELLS"] = "500000"  # keep input prep far below any cell
    m = measure_run(harness_code, json.loads(inputs_json), seed=seed, kill_above_mib=None, timeout_s=1500)
    m.pop("result", None)
    m["task_process_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    return json.dumps(m)


def _is_oom(e: BaseException) -> bool:
    import flyte.errors

    if isinstance(e, getattr(flyte.errors, "OOMError", ())):
        return True
    text = str(e).lower()
    return "oom" in text or "out of memory" in text or "exit code 137" in text


async def _live_one(row: dict, variant: str, mem_idx: int, cpu_idx: int, sem: asyncio.Semaphore) -> dict:
    mem = min(grid.MEMORY_GRID_MIB[mem_idx], LIVE_MAX_MEMORY_MIB)
    cpu = grid.CPU_GRID[cpu_idx]
    task = live_run.override(resources=flyte.Resources(cpu=cpu if cpu % 1 else int(cpu), memory=f"{mem}Mi"))
    predicted_fit = grid.MEMORY_GRID_MIB[mem_idx] >= row["required_mib"]
    out = {"run_id": row["run_id"], "variant": variant, "family": row["family"], "mem_mib": mem,
           "cpu": cpu, "required_mib": row["required_mib"], "sim_fit": predicted_fit}
    async with sem:
        try:
            m = json.loads(await task(harness_code=row["harness_code"], inputs_json=row["inputs_json"],
                                      seed=int(row["seed"])))
            out.update({"real": "ok" if m.get("ok") else "error", "peak_rss_mib": m.get("peak_rss_mib"),
                        "task_process_rss_mib": m.get("task_process_rss_mib"), "error": m.get("error", "")[:200]})
        except Exception as e:  # noqa: BLE001 -- the OOM is the measurement
            out.update({"real": "oom" if _is_oom(e) else "failed", "error": str(e)[:300]})
    return out


@bench_env.task(cache=arms_cache(), report=True)
async def live_validate(corpus: flyte.io.File, arm_dirs: list[flyte.io.Dir], runs_per_family: int = 10,
                        concurrency: int = 150) -> flyte.io.File:
    """Re-run a stratified sample of test/ood runs for real, under each
    arm's chosen request, and compare with what the simulator said."""
    rows = {r["run_id"]: r for r in await _read_rows(corpus) if dataset.usable(r)}
    decisions: dict[str, dict[str, dict]] = {}
    for d in arm_dirs:
        preds, _ = await _read_arm(d)
        for variant, p in _preds_by_variant(preds).items():
            if variant in LIVE_VARIANTS:
                decisions[variant] = p
    rng = Random(7)
    sample = []
    by_family: dict[str, list[dict]] = {}
    for r in rows.values():
        if r["split"] in ("test", "ood"):
            by_family.setdefault(r["family"], []).append(r)
    for fam in sorted(by_family):
        pool = sorted(by_family[fam], key=lambda r: r["run_id"])
        sample += rng.sample(pool, min(runs_per_family, len(pool)))
    sem = asyncio.Semaphore(concurrency)
    jobs = [_live_one(r, v, p[r["run_id"]]["mem_idx"], p[r["run_id"]]["cpu_idx"], sem)
            for v, p in sorted(decisions.items()) for r in sample if r["run_id"] in p]
    rep = Reporter("RightSizeMem Bench: live validation",
                   f"{len(sample)} runs x {len(decisions)} decision rules = {len(jobs)} real pods")
    await rep.flush()
    results = await asyncio.gather(*jobs)
    df = pd.DataFrame(results)
    rep.reset_body()
    agree = df[df["real"].isin(["ok", "oom"])]
    agree = agree.assign(real_fit=agree["real"] == "ok")
    rep.kv({"pods": len(df), "ok": int((df["real"] == "ok").sum()), "oom": int((df["real"] == "oom").sum()),
            "other failures": int(df["real"].isin(["failed", "error"]).sum()),
            "simulator agreement": f"{(agree['real_fit'] == agree['sim_fit']).mean():.1%}" if len(agree) else "-"})
    by_v = df.groupby("variant").agg(pods=("run_id", "size"), oom=("real", lambda s: (s == "oom").mean()),
                                     sim_oom=("sim_fit", lambda s: 1 - s.mean()),
                                     gib_hr=("mem_mib", lambda s: s.sum() / 1024))
    rep.h("Per decision rule: real vs simulated first-attempt OOM rate")
    rep.table(["variant", "pods", "real OOM", "sim OOM", "requested GiB (sum)"],
              [[v, str(int(r.pods)), f"{r.oom:.1%}", f"{r.sim_oom:.1%}", f"{r.gib_hr:.0f}"] for v, r in by_v.iterrows()])
    if len(agree):
        cm = pd.crosstab(agree["sim_fit"], agree["real_fit"])
        rep.h("Simulator vs reality (rows: simulator says fits; cols: really fit)")
        rep.table(["sim \\ real", *map(str, cm.columns)], [[str(i), *map(str, row)] for i, row in zip(cm.index, cm.values.tolist())])
    await rep.flush()
    return await flyte.io.File.from_local(_write_parquet(results, "live.parquet"))


# ── 7. peek (analysis from a laptop without data-plane credentials) ─────

_PEEK_DROP = ("task_code", "harness_code", "inputs_desc", "history_json", "inputs_json")


@bench_env.task
async def bench_peek(corpus: flyte.io.File | None = None, artifact_dir: flyte.io.Dir | None = None,
                     artifact_file: flyte.io.File | None = None) -> str:
    """The numeric part of a corpus (code columns dropped), or an arm's
    predictions + meta, or a JSON artifact -- inline, as a string, so it
    can be read with `Run.outputs()` from anywhere."""
    if corpus is not None:
        df = pd.read_parquet(await corpus.download())
        return df.drop(columns=[c for c in _PEEK_DROP if c in df.columns]).to_json(orient="records")
    if artifact_dir is not None:
        preds, meta = await _read_arm(artifact_dir)
        return json.dumps({"meta": meta, "predictions": preds.to_dict("records")}, default=float)
    with open(await artifact_file.download()) as f:
        return f.read()


# ── 8. ensemble ─────────────────────────────────────────────────────────

#: The classical member of the ensemble: tabular features, strong in-distribution.
ENSEMBLE_ANCHOR = "gbt-dist"
_POOL_WEIGHTS = tuple(round(w, 2) for w in np.linspace(0, 1, 11))


def log_pool(p_anchor, p_llm, w: float):
    """Logarithmic opinion pool: p ∝ p_anchor^w · p_llm^(1-w). Both members are
    calibrated distributions over the same cells, so pooling is meaningful."""
    p = np.exp(w * np.log(np.asarray(p_anchor) + 1e-9) + (1 - w) * np.log(np.asarray(p_llm) + 1e-9))
    return p / p.sum()


def build_ensemble(rows: list[dict], variants: dict[str, dict[str, dict]]) -> tuple[dict, dict]:
    """Pool the anchor with the LLM variant that pools best on val, weight tuned on val."""
    val = [r for r in rows if r["split"] == "val"]
    anchor = variants[ENSEMBLE_ANCHOR]
    candidates = [v for v, p in variants.items() if v != ENSEMBLE_ANCHOR and not v.endswith("-argmax")
                  and all(x.get("mem_probs") is not None for x in p.values())]

    def pooled(v, w):
        out = {}
        for rid, a in anchor.items():
            b = variants[v].get(rid)
            if b is None:
                continue
            p = log_pool(a["mem_probs"], b["mem_probs"], w)
            out[rid] = {"mem_idx": scoring.bayes_mem_index(p), "cpu_idx": b["cpu_idx"], "mem_probs": p}
        return out

    if not candidates:  # every LLM arm failed or is absent: the anchor alone
        return dict(anchor), {"anchor": ENSEMBLE_ANCHOR, "llm_member": None, "w_anchor": 1.0,
                              "val_mem_regret": scoring.score(val, anchor).get("mem_regret"), "candidates": []}
    grid_ = [(v, w) for v in candidates for w in _POOL_WEIGHTS]
    scored = [(scoring.score(val, pooled(v, w))["mem_regret"], v, w) for v, w in grid_]
    best_val, v, w = min(scored)
    return pooled(v, w), {"anchor": ENSEMBLE_ANCHOR, "llm_member": v, "w_anchor": w,
                          "val_mem_regret": best_val, "candidates": candidates}


@bench_env.task(cache=arms_cache(), report=True)
async def fit_bench_ensemble(corpus: flyte.io.File, arm_dirs: list[flyte.io.Dir]) -> flyte.io.Dir:
    """The classical anchor pooled with the best calibrated LLM arm.

    The arms fail differently -- the GBT on code semantics it cannot read, the
    LLM on numbers it reads as tokens -- and both emit calibrated
    distributions over the same grid, so a logarithmic pool is one line of
    math and needs no retraining.
    """
    rows = [r for r in await _read_rows(corpus) if dataset.usable(r)]
    variants: dict[str, dict[str, dict]] = {}
    for d in arm_dirs:
        preds, _ = await _read_arm(d)
        variants.update(_preds_by_variant(preds))
    preds, meta = build_ensemble(rows, variants)
    rep = Reporter("RightSizeMem Bench: ensemble", f"{meta['anchor']} x {meta['llm_member']}, w={meta['w_anchor']}")
    rep.kv({k: str(v) for k, v in meta.items()})
    await rep.flush()
    meta["arm"] = "ensemble"
    return await _write_arm_dir(_pred_rows("ensemble", preds, rows), meta)
