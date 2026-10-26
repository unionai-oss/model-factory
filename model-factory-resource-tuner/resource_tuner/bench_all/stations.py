"""The combined bench's stations: the Flyte tasks its factory wires together.

    rt-allbench-corpus      join_corpora        three measured corpora -> one, labelled on every axis
    rt-allbench-arm         fit_all_arm         one arm: the whole request (memory, CPU, GPU)
    rt-allbench-rl          rl_all_policy       the supervised decision model, RL-tuned on the joint cost
    rt-allbench-scorecard   score_all_arm       joint regret, first-attempt success, per-axis accuracy
    rt-allbench-leaderboard all_leaderboard     every arm side by side
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import flyte
import flyte.io
import numpy as np
import pandas as pd

from ..bench import charts
from ..bench.common import hosted_subsample, read_rows, write_parquet
from ..bench.envs import bench_gpu_env
from ..bench.llm_arms import TrainConfig, _amp_dtype, _device, config_dict, measure_latency
from ..bench.scoring import bayes_mem_index
from ..bench.versioning import all_arms_cache, all_corpus_cache
from ..bench_cpu.scoring import decode_index as cpu_decode
from ..bench_gpu.scoring import decode_index as gpu_decode
from ..shared import llm_client
from ..shared.reporting import Reporter
from . import arms as all_arms
from . import dataset, grid, policy, prompting, scoring
from .envs import allbench_env

TITLE = "RightSize Bench"
_EVAL = ("val", "test", "ood")

CPU_ARMS = tuple(all_arms.ARMS)
HOSTED_ARMS = {"zeroshot-27b": "qwen38-27b-vllm"}
GPU_ARMS = ("jev-pointer",)
ALL_ARMS = CPU_ARMS + tuple(HOSTED_ARMS) + GPU_ARMS
HOSTED_RUNS_PER_SPLIT = 300
#: The decision model's training budget: rows per source (the three corpora
#: differ 1.5x in size; balanced, none drowns the others) and epochs.
TRAIN_PER_SOURCE = 3500
EPOCHS = 2


@allbench_env.task(cache=all_corpus_cache(), report=True)
async def join_corpora(mem: flyte.io.File, cpu: flyte.io.File, gpu: flyte.io.File) -> flyte.io.File:
    parts = await asyncio.gather(read_rows(mem), read_rows(cpu), read_rows(gpu))
    rows = dataset.unify(*parts)
    df = pd.DataFrame(rows)
    rep = Reporter(f"{TITLE}: corpus", "RightSizeMem + RightSizeCPU + RightSizeGPU, one request per run")
    ok = df[df["ok"]]
    rep.kv({"runs": len(df), "usable": len(ok), **{f"{s} usable": int((ok["source"] == s).sum()) for s in dataset.SOURCES}})
    by = ok.groupby(["source", "split"]).size().unstack(fill_value=0)
    rep.table(["source", *by.columns], [[s, *map(str, v)] for s, v in zip(by.index, by.values.tolist())])
    await rep.flush()
    return await flyte.io.File.from_local(write_parquet(rows, "corpus.parquet"))


def _pred_table(variant: str, preds: dict[str, dict], split_of: dict[str, str]) -> list[dict]:
    return [{"variant": variant, "run_id": rid, "split": split_of.get(rid, ""), "mem_idx": int(p["mem_idx"]),
             "cpu_idx": int(p["cpu_idx"]), "gpu_idx": int(p["gpu_idx"]),
             "probs": json.dumps([[round(float(x), 5) for x in ax] for ax in p["probs"]]) if p.get("probs") else "",
             "raw": str(p.get("raw", ""))[:200]} for rid, p in preds.items()]


async def _write_arm(table: list[dict], meta: dict, extra: str | None = None) -> flyte.io.Dir:
    out = tempfile.mkdtemp(prefix="rsa-arm-")
    pd.DataFrame(table).to_parquet(os.path.join(out, "predictions.parquet"), index=False)
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=float)
    if extra:
        os.rename(extra, os.path.join(out, "model"))
    return await flyte.io.Dir.from_local(out)


def zeroshot_predict(rows: list[dict], teacher: str, concurrency: int = 16, max_tokens: int = 1024) -> tuple[dict, dict]:
    """The hosted model, prompted with the combined bench's prompt. Unparseable
    answers fall back to 4Gi / 1 core / no GPU, and are counted."""
    url = llm_client.wait_until_ready(llm_client.resolve_teacher_candidates(teacher), deadline_s=1800)
    model = llm_client.served_model(url)
    stats = {"teacher": teacher, "model": model, "parse_failures": 0, "errors": 0}
    lat, toks = [], []
    fallback = (grid.mem_cell(4096), grid.cpu_cell(1), grid.NO_GPU)

    def one(r):
        t0, usage = time.perf_counter(), {}
        try:
            text = llm_client.chat(url, prompting.messages(r), max_tokens=max_tokens, temperature=0.0,
                                   timeout=600, model=model, usage=usage)
        except Exception as e:  # noqa: BLE001
            return r["run_id"], None, f"{type(e).__name__}: {e}", 0.0, usage
        return r["run_id"], text, "", (time.perf_counter() - t0) * 1000, usage

    preds = {}
    with ThreadPoolExecutor(concurrency) as pool:
        for i, (rid, text, err, ms, usage) in enumerate(pool.map(one, rows)):
            ans = prompting.parse_answer(text) if text else None
            if err:
                stats["errors"] += 1
            elif ans is None:
                stats["parse_failures"] += 1
            else:
                lat.append(ms)
                toks.append(usage.get("completion_tokens", 0))
            m, c, g = ans or fallback
            preds[rid] = {"mem_idx": m, "cpu_idx": c, "gpu_idx": g, "raw": (text or err)[-200:]}
            if i % 50 == 0:
                print(f"[zeroshot] {i}/{len(rows)} parse_failures={stats['parse_failures']} errors={stats['errors']}",
                      flush=True)
    stats["p50_ms"] = float(np.median(lat)) if lat else float("nan")
    stats["mean_completion_tokens"] = float(np.mean(toks)) if toks else float("nan")
    return preds, stats


def _decode(prob: dict[str, list[np.ndarray]], threshold: float) -> dict[str, dict]:
    out = {}
    for rid, (pm, pc, pg) in prob.items():
        m, c, g = scoring.decode((pm, pc, pg), threshold, cpu_decode, bayes_mem_index, gpu_decode)
        out[rid] = {"mem_idx": m, "cpu_idx": c, "gpu_idx": g, "probs": [pm.tolist(), pc.tolist(), pg.tolist()]}
    return out


def _argmax(prob: dict[str, list[np.ndarray]]) -> dict[str, dict]:
    return {rid: {"mem_idx": int(pm.argmax()), "cpu_idx": int(pc.argmax()), "gpu_idx": int(pg.argmax())}
            for rid, (pm, pc, pg) in prob.items()}


def _policy_predictions(tok, model, head, eval_rows: list[dict], cfg: TrainConfig, name: str, meta: dict) -> list[dict]:
    """Calibrate on val, tune the no-GPU threshold on val, decode every split two ways."""
    lgs = policy.logits(tok, model, head, eval_rows, cfg)
    val_idx = [i for i, r in enumerate(eval_rows) if r["split"] == "val"]
    temps = policy.calibrate([z[val_idx] for z in lgs], [eval_rows[i] for i in val_idx])
    P = policy.probs(lgs, temps)
    prob = {r["run_id"]: [P[0][i], P[1][i], P[2][i]] for i, r in enumerate(eval_rows)}
    val_rows = [eval_rows[i] for i in val_idx]
    val_prob = {r["run_id"]: prob[r["run_id"]] for r in val_rows}
    threshold = min(all_arms.GPU_THRESHOLDS, key=lambda t: scoring.score(val_rows, _decode(val_prob, t))["regret"])
    meta.update(temperatures=temps, gpu_threshold=threshold)
    split_of = {r["run_id"]: r["split"] for r in eval_rows}
    return _pred_table(name, _decode(prob, threshold), split_of) + _pred_table(f"{name}-argmax", _argmax(prob), split_of)


def knobs(state: dict) -> dict:
    """The JSON-serializable part of an arm's fitted state (fitted models dropped)."""
    out = {}
    for k, v in state.items():
        try:
            json.dumps(v)
        except TypeError:
            continue
        out[k] = v
    return out


def _save(tok, model, head) -> str:
    import torch

    d = tempfile.mkdtemp(prefix="rsa-model-")
    model.save_pretrained(d)
    torch.save(head.state_dict(), os.path.join(d, "decision_head.pt"))
    return d


@bench_gpu_env.task(retries=1, cache=all_arms_cache(), report=True, timeout=flyte.Timeout(max_runtime=10 * 3600))
async def train_all_policy(corpus: flyte.io.File, arm: str = "jev-pointer", model_name: str = "Qwen/Qwen3-1.7B",
                           epochs: int = EPOCHS, per_source: int = TRAIN_PER_SOURCE) -> flyte.io.Dir:
    """The supervised three-head decision model (`policy.train_supervised`)."""
    rows = await read_rows(corpus)
    splits = dataset.by_split(rows)
    train = dataset.balanced(splits["train"], per_source)
    eval_rows = [r for s in _EVAL for r in splits[s]]
    cfg = TrainConfig(model_name=model_name, epochs=epochs, head="pointer", bench="all")
    rep = Reporter(f"{TITLE}: {arm} on {model_name}", f"train {len(train)} runs ({per_source}/source), eval {len(eval_rows)}")
    await rep.flush()
    t0 = time.time()
    tok, model, head, hist = policy.train_supervised(train, cfg)
    meta = {"arm": arm, "config": config_dict(cfg), "n_train": len(train), "train_s": time.time() - t0, "loss": hist}
    table = _policy_predictions(tok, model, head, eval_rows, cfg, arm, meta)
    meta["latency"] = measure_latency(lambda rs: policy.logits(tok, model, head, rs, cfg, batch_size=1), eval_rows)
    rep.kv({k: json.dumps(v, default=float)[:200] for k, v in meta.items()})
    await rep.flush()
    return await _write_arm(table, meta, _save(tok, model, head))


@allbench_env.task(cache=all_arms_cache(), report=True)
async def fit_all_arm(corpus: flyte.io.File, arm: str) -> flyte.io.Dir:
    rows = await read_rows(corpus)
    splits = dataset.by_split(rows)
    rep = Reporter(f"{TITLE}: arm {arm}", ", ".join(f"{k}={len(v)}" for k, v in splits.items()))
    await rep.flush()
    if arm in GPU_ARMS:
        return await train_all_policy(corpus, arm)
    eval_rows = [r for s in _EVAL for r in splits[s]]
    if arm in CPU_ARMS:
        fit, predict = all_arms.ARMS[arm]
        state = fit(splits["train"], splits["val"])
        t0 = time.perf_counter()
        preds = predict(state, eval_rows)
        meta = {"knobs": knobs(state),
                "ms_per_decision": (time.perf_counter() - t0) * 1000 / max(1, len(eval_rows))}
    elif arm in HOSTED_ARMS:
        eval_rows = hosted_subsample(splits, HOSTED_RUNS_PER_SPLIT)
        preds, meta = await asyncio.to_thread(zeroshot_predict, eval_rows, HOSTED_ARMS[arm])
    else:
        raise ValueError(f"unknown arm {arm!r}; choose from {ALL_ARMS}")
    meta["arm"] = arm
    rep.kv({k: str(v)[:200] for k, v in meta.items()})
    await rep.flush()
    return await _write_arm(_pred_table(arm, preds, {r["run_id"]: r["split"] for r in eval_rows}), meta)


async def _read_arm(d: flyte.io.Dir) -> tuple[str, pd.DataFrame, dict]:
    local = await d.download()
    with open(os.path.join(local, "meta.json")) as f:
        meta = json.load(f)
    return local, pd.read_parquet(os.path.join(local, "predictions.parquet")), meta


@bench_gpu_env.task(retries=1, cache=all_arms_cache(), report=True, timeout=flyte.Timeout(max_runtime=8 * 3600))
async def rl_all_policy(corpus: flyte.io.File, arm_dirs: list[flyte.io.Dir], base_arm: str = "jev-pointer",
                        prompts: int = 6000, k_samples: int = 16, kl_beta: float = 0.05) -> flyte.io.Dir:
    """`base_arm`'s decision model, RL-tuned against the joint cost (`policy.train_rl`)."""
    import torch
    from peft import PeftModel
    from transformers import AutoModel, AutoTokenizer

    base_dir = None
    for d in arm_dirs:
        local, _preds, meta = await _read_arm(d)
        if meta.get("arm") == base_arm:
            base_dir, base_meta = local, meta
    if base_dir is None:
        raise ValueError(f"rl_all_policy: no {base_arm!r} among the arms built for this release")
    cfg = TrainConfig(**{k: v for k, v in base_meta["config"].items() if k in TrainConfig.__dataclass_fields__})
    rows = await read_rows(corpus)
    splits = dataset.by_split(rows)
    # RL prompts: training rows the supervised model did NOT see, then the rest
    seen = {r["run_id"] for r in dataset.balanced(splits["train"], TRAIN_PER_SOURCE)}
    fresh = [r for r in splits["train"] if r["run_id"] not in seen]
    pool = dataset.balanced(fresh, prompts // 3) + dataset.balanced([r for r in splits["train"] if r["run_id"] in seen],
                                                                     prompts // 3)
    rl_cfg = policy.RLConfig(prompts=prompts, k_samples=k_samples, kl_beta=kl_beta)
    eval_rows = [r for s in _EVAL for r in splits[s]]
    rep = Reporter(f"{TITLE}: rl-pointer from {base_arm}", f"{min(prompts, len(pool))} RL prompts x {k_samples} samples")
    await rep.flush()

    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    base = AutoModel.from_pretrained(cfg.model_name, dtype=_amp_dtype(), attn_implementation="sdpa").to(_device())
    base.config.use_cache = False
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    base.enable_input_require_grads()
    model = PeftModel.from_pretrained(base, os.path.join(base_dir, "model"), is_trainable=True)
    head = policy.head_module(model.config.hidden_size, policy.option_offsets(tok)).to(_device())
    head.load_state_dict(torch.load(os.path.join(base_dir, "model", "decision_head.pt"), map_location=_device()))

    t0 = time.time()
    curve = policy.train_rl(tok, model, head, pool, cfg, rl_cfg)
    meta = {"arm": "rl-pointer", "base_arm": base_arm, "config": config_dict(cfg), "rl": rl_cfg.__dict__,
            "rl_s": time.time() - t0, "curve": curve}
    table = _policy_predictions(tok, model, head, eval_rows, cfg, "rl-pointer", meta)
    rep.kv({k: json.dumps(v, default=float)[:300] for k, v in meta.items()})
    if curve:
        rep.raw(charts.hbars([(f"step {c['step']}", -c["mean_reward"] - 1, "#7aa2f7") for c in curve[:: max(1, len(curve) // 12)]],
                             title="mean sampled regret during RL (lower is better)"))
    await rep.flush()
    return await _write_arm(table, meta, _save(tok, model, head))


@allbench_env.task(cache=all_arms_cache(), report=True)
async def score_all_arm(corpus: flyte.io.File, arm_dir: flyte.io.Dir, arm: str) -> flyte.io.File:
    rows = [r for r in await read_rows(corpus) if dataset.usable(r)]
    _local, preds, meta = await _read_arm(arm_dir)
    card = {"arm": arm, "meta": meta, "variants": {}}
    rep = Reporter(f"{TITLE}: scorecard {arm}")
    for variant, sub in preds.groupby("variant"):
        p = {r["run_id"]: r for r in sub.to_dict("records")}
        card["variants"][variant] = scoring.score_slices(rows, p)
        t = card["variants"][variant].get("split=test", {})
        if t.get("n"):
            rep.kv({f"{variant} test": f"regret {t['regret']:.1%}, all right {t['all_right']:.1%}, "
                                       f"first attempt {t['first_attempt_ok']:.1%}"})
    await rep.flush()
    path = os.path.join(tempfile.mkdtemp(prefix="rsa-card-"), "scorecard.json")
    with open(path, "w") as f:
        json.dump(card, f, indent=2, default=float)
    return await flyte.io.File.from_local(path)


HEADLINE = ("static", "rules", "gbt-dist", "zeroshot-27b", "jev-pointer-argmax", "jev-pointer",
            "rl-pointer-argmax", "rl-pointer")


@allbench_env.task(cache=all_arms_cache(), report=True)
async def all_leaderboard(scorecards: list[flyte.io.File], rl_scorecard: flyte.io.File | None = None) -> flyte.io.File:
    cards = []
    for f in [*scorecards, *([rl_scorecard] if rl_scorecard is not None else [])]:
        with open(await f.download()) as fh:
            cards.append(json.load(fh))
    board = {}
    for c in cards:
        board.update(c["variants"])
    order = [v for v in HEADLINE if v in board] + sorted(v for v in board if v not in HEADLINE)
    rep = Reporter(f"{TITLE}: leaderboard", f"{len(order)} decision rules, one cost model")
    for split in ("test", "ood"):
        key = f"split={split}"
        rows = [(v, board[v][key]) for v in order if board[v].get(key, {}).get("n")]
        if not rows:
            continue
        rep.h(f"{split}: the whole request")
        rep.table(["arm", "regret", "geo regret", "$ regret", "first attempt ok", "all axes right", "memory right",
                   "cpu right", "gpu right", "GPU for a CPU task", "n"],
                  [[v, f"{s['regret']:.1%}", f"{s['geo_regret']:.1%}", f"{s['dollar_regret']:.1%}",
                    f"{s['first_attempt_ok']:.1%}", f"{s['all_right']:.1%}", f"{s['mem_right']:.1%}",
                    f"{s['cpu_right']:.1%}", f"{s['gpu_right']:.1%}", f"{s['gpu_for_cpu_task']:.1%}", str(s["n"])]
                   for v, s in rows])
        rep.raw(charts.hbars([(v, s["regret"], charts.arm_color(v)) for v, s in rows],
                             title=f"{split}: mean regret vs the oracle (lower is better)"))
        rep.h(f"{split}: regret by source bench")
        rep.table(["arm", "mem", "cpu", "gpu"],
                  [[v, *[f"{board[v].get(f'{split}/source={s}', {}).get('regret', float('nan')):.1%}"
                         for s in dataset.SOURCES]] for v, _ in rows])
    await rep.flush()
    path = os.path.join(tempfile.mkdtemp(prefix="rsa-board-"), "leaderboard.json")
    with open(path, "w") as f:
        json.dump({"order": order, "board": board,
                   "meta": {c["arm"]: c.get("meta", {}) for c in cards}}, f, indent=2, default=float)
    return await flyte.io.File.from_local(path)


_PEEK_DROP = ("task_code", "inputs_desc", "history_json", "inputs_json", "cpu_series_json")


@allbench_env.task
async def all_peek(corpus: flyte.io.File | None = None, artifact_file: flyte.io.File | None = None) -> str:
    """Corpus numbers (code dropped) or a JSON artifact, inline, readable from a laptop."""
    if corpus is not None:
        df = pd.read_parquet(await corpus.download())
        return df.drop(columns=[c for c in _PEEK_DROP if c in df.columns]).to_json(orient="records")
    with open(await artifact_file.download()) as f:
        return f.read()
