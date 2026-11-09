"""The combined bench's stations: the Flyte tasks its factory wires together.

    rt-allbench-corpus      join_corpora        three measured corpora -> one, labelled on every axis
    rt-allbench-arm         fit_all_arm         one arm: the whole request (memory, CPU, GPU)
    rt-allbench-rl          rl_all_policy       the supervised decision model, RL-tuned on the joint cost
    rt-allbench-rl-stack    rl_all_policy       the same, from the stacked decision model (trees in its prompt)
    rt-allbench-pool        pool_all_arms       gbt-dist x every decision model, pooled per axis
    rt-allbench-scorecard   score_all_arm       joint regret, first-attempt success, per-axis accuracy
    rt-allbench-leaderboard all_leaderboard     every arm side by side
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import flyte
import flyte.io
import numpy as np
import pandas as pd

from ..bench import charts, hybrid
from ..bench.common import hosted_subsample, read_rows, write_parquet
from ..bench.llm_arms import (
    TrainConfig,
    _amp_dtype,
    _device,
    config_dict,
    measure_latency,
)
from ..bench.scoring import bayes_mem_index
from ..bench.versioning import all_arms_cache, all_corpus_cache, all_train_cache
from ..bench_cpu.scoring import decode_index as cpu_decode
from ..bench_gpu.scoring import decode_index as gpu_decode
from ..shared import llm_client
from ..shared.reporting import Reporter
from . import arms as all_arms
from . import counterfactual, dataset, features, gating, grid, k8s, policy, prompting, scoring
from .envs import allbench_env, allbench_train_env

TITLE = "RightSize Bench"
_EVAL = ("val", "test", "ood")

CPU_ARMS = tuple(all_arms.ARMS)
#: The hosted model, two ways. It reasons in plain text before it answers even
#: with thinking off, and at 1,024 tokens 444 of 560 k8s-1 answers stopped
#: mid-reasoning (rsk-k8s-1-gate): ``zeroshot-27b`` gets the room to finish;
#: ``-json`` decodes straight into the answer's schema (no reasoning at all).
HOSTED_ARMS = {"zeroshot-27b": "qwen38-27b-vllm", "zeroshot-27b-json": "qwen38-27b-vllm"}
HOSTED_KNOBS = {"zeroshot-27b": {"max_tokens": 8192}, "zeroshot-27b-json": {"max_tokens": 256, "structured": True}}
#: Single-platform ablations of the decision model (round 22): trained on the
#: Flyte rows only (``-flyte``: a history that is all Flyte, sizing raw Jobs)
#: or the raw-Job rows only (``-k8s``: does the Flyte interface's data help?).
#: Every platform is evaluated.
GPU_ARMS = ("jev-pointer", "jev-pointer-flyte", "jev-pointer-k8s")
FLYTE_ONLY_SUFFIX = "-flyte"
PLATFORM_SUFFIXES = {"-flyte": "flyte", "-k8s": "k8s"}
#: The trees fit on one platform's rows: the single-platform decision models' pool partners.
TREE_PLATFORM_ARMS = tuple(f"{hybrid.ANCHOR}{sfx}" for sfx in PLATFORM_SUFFIXES)


def seed_of(release: str, default: int) -> int:
    """A replicate's seed: ``<release>-s<k>`` trains at seed k (k8s-2-s2, k8s-2-s3), anything else at `default`."""
    m = re.search(r"-s(\d+)$", release or "")
    return int(m.group(1)) if m else default


def platform_suffix(variant: str) -> str:
    """'-flyte' for jev-pointer-flyte(-argmax, -adapt...), '' for an arm trained on both."""
    v = variant.removesuffix("-argmax")
    return next((sfx for sfx in PLATFORM_SUFFIXES if v.endswith(sfx)), "")
#: The stacked hybrid (``bench.hybrid``): the decision model with the trees' estimate in its prompt.
STACK_ARMS = (f"{hybrid.STACK_PREFIX}jev-pointer",)
ALL_ARMS = CPU_ARMS + TREE_PLATFORM_ARMS + tuple(HOSTED_ARMS) + GPU_ARMS + STACK_ARMS
HINTS_FILE = "hints.parquet"
HOSTED_RUNS_PER_SPLIT = 300
#: The decision model's training budget: rows per source (the three corpora
#: differ 1.5x in size; balanced, none drowns the others) and epochs.
TRAIN_PER_SOURCE = 3500
EPOCHS = 2
#: Prompt budget: a K8s twin carries its manifest too (median +390 tokens;
#: 25% of GPU twins pass 2,048 -- round 22), and truncation drops the code's top.
MAX_LEN = 2048


@allbench_env.task(cache=all_corpus_cache(), report=True)
async def join_corpora(mem: flyte.io.File, cpu: flyte.io.File, gpu: flyte.io.File, platforms: str = "flyte") -> flyte.io.File:
    """The three corpora as one; ``platforms="flyte,k8s"`` adds every run's Kubernetes twin (``k8s``)."""
    parts = await asyncio.gather(read_rows(mem), read_rows(cpu), read_rows(gpu))
    rows = dataset.unify(*parts)
    if "k8s" in platforms.split(","):
        rows = dataset.with_k8s(rows)
    df = pd.DataFrame(rows)
    rep = Reporter(f"{TITLE}: corpus", "RightSizeMem + RightSizeCPU + RightSizeGPU, one request per run")
    ok = df[df["ok"]]
    rep.kv({"runs": len(df), "usable": len(ok), **{f"{s} usable": int((ok["source"] == s).sum()) for s in dataset.SOURCES}})
    by = ok.groupby(["platform", "source", "split"]).size().unstack(fill_value=0)
    rep.table(["platform", "source", *by.columns], [[p, s, *map(str, v)] for (p, s), v in zip(by.index, by.values.tolist())])
    if "k8s" in platforms.split(","):
        tw = ok[ok["platform"] == "k8s"]
        flyte_need = ok[ok["platform"] == "flyte"].set_index("run_id")["label_mem"]
        moved = tw[tw["source"] != "gpu"].assign(d=lambda x: x["label_mem"] - x["twin_of"].map(flyte_need))
        rep.h("K8s twins: memory cell vs the Flyte run's, by staging")
        rep.table(["staging", "twins", "lower", "same", "higher", "mean shift (cells)"],
                  [[st, str(len(g)), f"{(g['d'] < 0).mean():.1%}", f"{(g['d'] == 0).mean():.1%}",
                    f"{(g['d'] > 0).mean():.1%}", f"{g['d'].mean():+.2f}"] for st, g in moved.groupby("staging")])
    await rep.flush()
    return await flyte.io.File.from_local(write_parquet(rows, "corpus.parquet"))


def paired_subsample(splits: dict[str, list[dict]], n: int) -> list[dict]:
    """The hosted arm's test+ood subsample: `n` per split, half Flyte runs
    (``hosted_subsample``, family-stratified) and half their K8s twins, so
    its platform gap is measured on the same workloads."""
    flyte_only = {k: [r for r in v if dataset.platform_of(r) == "flyte"] for k, v in splits.items()}
    twins = {r.get("twin_of"): r for v in splits.values() for r in v if r.get("twin_of")}
    if not twins:
        return hosted_subsample(splits, n)
    picked = hosted_subsample(flyte_only, n // 2)
    return picked + [twins[r["run_id"]] for r in picked if r["run_id"] in twins]


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


def answer_schema() -> dict:
    """The answer's JSON schema: one option string per axis, as the prompt lists them."""
    props = {k: {"type": "string", "enum": list(o)} for k, o in zip(prompting.KEYS, prompting.OPTIONS)}
    return {"type": "object", "properties": props, "required": list(prompting.KEYS), "additionalProperties": False}


def zeroshot_predict(rows: list[dict], teacher: str, concurrency: int = 16, max_tokens: int = 1024,
                     structured: bool = False) -> tuple[dict, dict]:
    """The hosted model, prompted with the combined bench's prompt. ``structured``
    constrains decoding to ``answer_schema`` (vLLM structured output).
    Unparseable answers fall back to 4Gi / 1 core / no GPU, and are counted,
    as are answers that hit the token limit."""
    url = llm_client.wait_until_ready(llm_client.resolve_teacher_candidates(teacher), deadline_s=1800)
    model = llm_client.served_model(url)
    stats = {"teacher": teacher, "model": model, "max_tokens": max_tokens, "structured": structured, "parse_failures": 0,
             "errors": 0, "at_token_limit": 0}
    lat, toks = [], []
    fallback = (grid.mem_cell(4096), grid.cpu_cell(1), grid.NO_GPU)
    extra = ({"response_format": {"type": "json_schema", "json_schema": {"name": "request", "schema": answer_schema()}}}
             if structured else None)

    def one(r):
        t0, usage = time.perf_counter(), {}
        try:
            text = llm_client.chat(url, prompting.messages(r), max_tokens=max_tokens, temperature=0.0,
                                   timeout=1800, model=model, usage=usage, extra=extra)
        except Exception as e:  # noqa: BLE001
            return r["run_id"], None, f"{type(e).__name__}: {e}", 0.0, usage
        return r["run_id"], text, "", (time.perf_counter() - t0) * 1000, usage

    preds = {}
    with ThreadPoolExecutor(concurrency) as pool:
        for i, (rid, text, err, ms, usage) in enumerate(pool.map(one, rows)):
            ans = prompting.parse_answer(text) if text else None
            if usage.get("completion_tokens", 0) >= max_tokens:
                stats["at_token_limit"] += 1
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
    stats["p95_completion_tokens"] = float(np.percentile(toks, 95)) if toks else float("nan")
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


def _policy_predictions(tok, model, head, eval_rows: list[dict], cfg: TrainConfig, name: str, meta: dict,
                        tune_platforms: str = "") -> list[dict]:
    """Calibrate on val, tune the no-GPU threshold on val, decode every split two
    ways. ``tune_platforms`` keeps the tuning to the val rows of the platforms the
    model trained on: a single-platform ablation must not see the other
    platform's labels anywhere (through k8s-1 it was calibrated on both -- review)."""
    lgs = policy.logits(tok, model, head, eval_rows, cfg)
    plats = tune_platforms.split(",") if tune_platforms else None
    val_idx = [i for i, r in enumerate(eval_rows)
               if r["split"] == "val" and (plats is None or dataset.platform_of(r) in plats)]
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


@allbench_train_env.task(retries=1, cache=all_train_cache(), report=True, timeout=flyte.Timeout(max_runtime=12 * 3600))
async def train_all_policy(corpus: flyte.io.File, arm: str = "jev-pointer", model_name: str = "Qwen/Qwen3-1.7B",
                           epochs: int = EPOCHS, per_source: int = TRAIN_PER_SOURCE, platforms: str = "",
                           max_len: int = MAX_LEN, seed: int = 0) -> flyte.io.Dir:
    """The supervised three-head decision model (`policy.train_supervised`).
    ``platforms`` restricts the *training* rows (``"flyte"``: the transfer
    ablations); every platform is evaluated. ``seed`` seeds the model (init,
    batch order); the training rows are the same draw for every seed."""
    rows = await read_rows(corpus)
    splits = dataset.by_split(rows)
    pool = [r for r in splits["train"] if not platforms or dataset.platform_of(r) in platforms.split(",")]
    train = dataset.balanced(pool, per_source)
    eval_rows = [r for s in _EVAL for r in splits[s]]
    cfg = TrainConfig(model_name=model_name, epochs=epochs, head="pointer", bench="all", max_len=max_len, seed=seed)
    rep = Reporter(f"{TITLE}: {arm} on {model_name}", f"train {len(train)} runs ({per_source}/source), eval {len(eval_rows)}")
    await rep.flush()
    t0 = time.time()
    tok, model, head, hist = policy.train_supervised(train, cfg)
    meta = {"arm": arm, "config": config_dict(cfg), "n_train": len(train), "train_s": time.time() - t0, "loss": hist,
            "train_platforms": platforms or "all", "per_source": per_source}
    table = _policy_predictions(tok, model, head, eval_rows, cfg, arm, meta, tune_platforms=platforms)
    meta["latency"] = measure_latency(lambda rs: policy.logits(tok, model, head, rs, cfg, batch_size=1), eval_rows)
    meta["stacked"] = hybrid.is_stacked(arm)
    rep.kv({k: json.dumps(v, default=float)[:200] for k, v in meta.items()})
    await rep.flush()
    saved = _save(tok, model, head)
    if meta["stacked"]:  # the RL phase prompts with the same hints
        pd.DataFrame([{"run_id": r["run_id"], "hint": r.get("hint") or ""} for r in rows]).to_parquet(
            os.path.join(saved, HINTS_FILE), index=False)
    return await _write_arm(table, meta, saved)


def all_hints(preds: dict[str, dict]) -> dict[str, str]:
    return {rid: prompting.hint_text(p["probs"], (p["mem_idx"], p["cpu_idx"], p["gpu_idx"])) for rid, p in preds.items()}


async def hinted_all_corpus(splits: dict[str, list[dict]], eval_rows: list[dict]) -> tuple[flyte.io.File, dict]:
    """Every usable row with gbt-dist's joint estimate in a ``hint`` column,
    out-of-fold on train (``hybrid.crossfit``)."""
    t0 = time.perf_counter()
    fit, predict = all_arms.ARMS[hybrid.ANCHOR]
    hints = all_hints(hybrid.crossfit(splits["train"], splits["val"], eval_rows, fit, predict))
    rows = [{**r, "hint": hints.get(r["run_id"], "")} for s in dataset.SPLITS for r in splits[s]]
    meta = {"hint_anchor": hybrid.ANCHOR, "hinted_rows": len(hints), "crossfit_s": time.perf_counter() - t0,
            "hint_example": next(iter(hints.values()), "")}
    return await flyte.io.File.from_local(write_parquet(rows, "hinted.parquet")), meta


@allbench_env.task(cache=all_train_cache(), report=True)
async def fit_all_arm(corpus: flyte.io.File, arm: str, per_source: int = TRAIN_PER_SOURCE, epochs: int = EPOCHS,
                      max_len: int = MAX_LEN, seed: int = 0, release: str = "") -> flyte.io.Dir:
    seed = seed_of(release, seed)
    rows = await read_rows(corpus)
    splits = dataset.by_split(rows)
    rep = Reporter(f"{TITLE}: arm {arm}", ", ".join(f"{k}={len(v)}" for k, v in splits.items()))
    await rep.flush()
    if arm in GPU_ARMS:
        for suffix, platform in PLATFORM_SUFFIXES.items():
            if arm.endswith(suffix):  # half the rows: twice the epochs, the same number of steps
                return await train_all_policy(corpus, arm, epochs=2 * epochs, per_source=per_source,
                                              platforms=platform, max_len=max_len, seed=seed)
        return await train_all_policy(corpus, arm, epochs=epochs, per_source=per_source, max_len=max_len, seed=seed)
    eval_rows = [r for s in _EVAL for r in splits[s]]
    if arm in STACK_ARMS:
        hinted, hmeta = await hinted_all_corpus(splits, eval_rows)
        rep.kv({k: str(v)[:300] for k, v in hmeta.items()})
        await rep.flush()
        return await train_all_policy(hinted, arm, epochs=epochs, per_source=per_source, max_len=max_len)
    if arm in CPU_ARMS or arm in TREE_PLATFORM_ARMS:
        sfx = platform_suffix(arm)
        fit, predict = all_arms.ARMS[arm.removesuffix(sfx)]
        mine = {s: [r for r in splits[s] if not sfx or dataset.platform_of(r) == PLATFORM_SUFFIXES[sfx]]
                for s in ("train", "val")}
        state = fit(mine["train"], mine["val"])
        t0 = time.perf_counter()
        preds = predict(state, eval_rows)
        meta = {"knobs": knobs(state),
                "ms_per_decision": (time.perf_counter() - t0) * 1000 / max(1, len(eval_rows))}
    elif arm in HOSTED_ARMS:
        eval_rows = paired_subsample(splits, HOSTED_RUNS_PER_SPLIT)
        preds, meta = await asyncio.to_thread(zeroshot_predict, eval_rows, HOSTED_ARMS[arm], **HOSTED_KNOBS[arm])
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


@allbench_train_env.task(retries=1, cache=all_train_cache(), report=True, timeout=flyte.Timeout(max_runtime=10 * 3600))
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
    name = f"{hybrid.STACK_PREFIX}rl-pointer" if hybrid.is_stacked(base_arm) else "rl-pointer"
    hints_path = os.path.join(base_dir, "model", HINTS_FILE)
    if os.path.exists(hints_path):  # a stacked base: its prompts carried the trees' estimate
        hints = dict(pd.read_parquet(hints_path)[["run_id", "hint"]].itertuples(index=False))
        rows = [{**r, "hint": hints.get(r["run_id"], "")} for r in rows]
    splits = dataset.by_split(rows)
    # RL prompts: training rows the supervised model did NOT see, then the rest
    # the base's own training draw (its budget and platforms, as train_all_policy drew it)
    plats = base_meta.get("train_platforms", "all")
    base_pool = [r for r in splits["train"] if plats == "all" or dataset.platform_of(r) in plats.split(",")]
    seen = {r["run_id"] for r in dataset.balanced(base_pool, int(base_meta.get("per_source", TRAIN_PER_SOURCE)))}
    fresh = [r for r in splits["train"] if r["run_id"] not in seen]
    pool = dataset.balanced(fresh, prompts // 3) + dataset.balanced([r for r in splits["train"] if r["run_id"] in seen],
                                                                     prompts // 3)
    rl_cfg = policy.RLConfig(prompts=prompts, k_samples=k_samples, kl_beta=kl_beta, seed=cfg.seed)
    eval_rows = [r for s in _EVAL for r in splits[s]]
    rep = Reporter(f"{TITLE}: {name} from {base_arm}", f"{min(prompts, len(pool))} RL prompts x {k_samples} samples")
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
    meta = {"arm": name, "base_arm": base_arm, "config": config_dict(cfg), "rl": rl_cfg.__dict__,
            "rl_s": time.time() - t0, "curve": curve}
    table = _policy_predictions(tok, model, head, eval_rows, cfg, name, meta)
    rep.kv({k: json.dumps(v, default=float)[:300] for k, v in meta.items()})
    if curve:
        rep.raw(charts.hbars([(f"step {c['step']}", -c["mean_reward"] - 1, "#7aa2f7") for c in curve[:: max(1, len(curve) // 12)]],
                             title="mean sampled regret during RL (lower is better)"))
    await rep.flush()
    return await _write_arm(table, meta, _save(tok, model, head))


def _probs_of(preds: pd.DataFrame) -> dict[str, dict[str, list[np.ndarray]]]:
    """variant -> run_id -> [P(mem), P(cpu), P(gpu)], for the variants that carry distributions."""
    out: dict[str, dict[str, list[np.ndarray]]] = {}
    for r in preds.to_dict("records"):
        if r.get("probs"):
            out.setdefault(r["variant"], {})[r["run_id"]] = [np.asarray(x, dtype=float) for x in json.loads(r["probs"])]
    return out


def pool_all(rows: list[dict], variants: dict[str, dict[str, list[np.ndarray]]]) -> tuple[list[dict], dict]:
    """gbt-dist x each decision-model variant: a log pool on every axis with one
    weight, the weight and the no-GPU threshold tuned on val for regret."""
    split_of = {r["run_id"]: r["split"] for r in rows}
    table, knobs = [], {}
    for v in sorted(variants):
        if (v == hybrid.ANCHOR or v in TREE_PLATFORM_ARMS or v.endswith("-argmax")
                or v.startswith(hybrid.POOL_PREFIX)):
            continue
        member = variants[v]
        # a single-platform model pools with trees fit on that platform, tuned on its val rows
        sfx = platform_suffix(v)
        anchor_name = f"{hybrid.ANCHOR}{sfx}" if f"{hybrid.ANCHOR}{sfx}" in variants else hybrid.ANCHOR
        anchor = variants[anchor_name]
        val = [r for r in rows if r["split"] == "val" and (not sfx or dataset.platform_of(r) == PLATFORM_SUFFIXES[sfx])]

        def pooled(w):
            return {rid: [hybrid.log_pool(a, b, w) for a, b in zip(anchor[rid], member[rid])]
                    for rid in anchor if rid in member}

        vrows = [r for r in val if r["run_id"] in anchor and r["run_id"] in member]
        if not vrows:
            continue
        best = None
        for w in hybrid.POOL_WEIGHTS:
            pv = pooled(w)
            vp = {r["run_id"]: pv[r["run_id"]] for r in vrows}
            for t in all_arms.GPU_THRESHOLDS:
                reg = scoring.score(vrows, _decode(vp, t))["regret"]
                if best is None or (reg, -w) < (best[0], -best[1]):
                    best = (reg, w, t)
        reg, w, t = best
        prob = pooled(w)
        name = f"{hybrid.POOL_PREFIX}{v}"
        table += _pred_table(name, _decode(prob, t), split_of) + _pred_table(f"{name}-argmax", _argmax(prob), split_of)
        knobs[name] = {"w_trees": w, "gpu_threshold": t, "val_regret": reg, "anchor": anchor_name,
                       "tuned_on": PLATFORM_SUFFIXES.get(sfx, "all")}
    return table, knobs


@allbench_env.task(cache=all_arms_cache(), report=True)
async def pool_all_arms(corpus: flyte.io.File, arm_dirs: list[flyte.io.Dir], rl_dir: flyte.io.Dir | None = None,
                        rl_stack_dir: flyte.io.Dir | None = None) -> flyte.io.Dir:
    """The pooled hybrids: the trees' joint distribution x every decision model's
    (supervised, stacked, RL from each), per axis (``pool_all``)."""
    rows = [r for r in await read_rows(corpus) if dataset.usable(r)]
    variants: dict[str, dict[str, list[np.ndarray]]] = {}
    for d in [*arm_dirs, *[x for x in (rl_dir, rl_stack_dir) if x is not None]]:
        _local, preds, _meta = await _read_arm(d)
        variants.update(_probs_of(preds))
    if hybrid.ANCHOR not in variants:
        raise ValueError(f"pool_all_arms: no {hybrid.ANCHOR!r} among the arms built for this release")
    table, knobs = await asyncio.to_thread(pool_all, rows, variants)
    rep = Reporter(f"{TITLE}: pooled hybrids", f"{hybrid.ANCHOR} x {len(knobs)} decision-model variants")
    rep.table(["pool", "w (trees)", "no-GPU threshold", "val regret"],
              [[k, str(x["w_trees"]), str(x["gpu_threshold"]), f"{x['val_regret']:.1%}"] for k, x in knobs.items()])
    await rep.flush()
    return await _write_arm(table, {"arm": "pool", "anchor": hybrid.ANCHOR, "knobs": knobs})


def _decoded_of(preds: pd.DataFrame) -> dict[str, dict[str, dict]]:
    """variant -> run_id -> the request it made."""
    out: dict[str, dict[str, dict]] = {}
    for r in preds.to_dict("records"):
        out.setdefault(r["variant"], {})[r["run_id"]] = {k: int(r[k]) for k in ("mem_idx", "cpu_idx", "gpu_idx")}
    return out


def gate_signals_of(rows: list[dict], probs: dict, member: str) -> dict[str, dict[str, float]]:
    """``gating.GATE_SIGNALS``, run_id -> value, for every evaluated run."""
    feats = {r["run_id"]: features.row_features(r) for r in rows}
    return {"knn": gating.novelty_knn(rows, feats), "knn-numeric": gating.novelty_knn(rows, feats, numeric_only=True),
            "knn-raw": gating.novelty_knn(rows, feats, legacy_scale=True),
            "js": {rid: gating.disagreement(p, probs[member][rid]) for rid, p in probs[hybrid.ANCHOR].items()
                   if rid in probs[member]}}


def gate_all(rows: list[dict], probs: dict, decoded: dict, metas: dict, inner: str, outer: str,
             member: str) -> tuple[list[dict], dict]:
    """The gates (``gating.gate``: inner on familiar runs, outer on novel ones,
    for each signal and val quantile) and every Flyte-only arm, adapted."""
    split_of = {r["run_id"]: r["split"] for r in rows}
    signals = gate_signals_of(rows, probs, member)
    knobs = {"inner": inner, "outer": outer, "member": member, "headline": f"{gating.GATE_SIGNAL}@{gating.GATE_Q}",
             "signals": {}}
    table = []
    for sig, signal in signals.items():
        by_split = {s: [signal[r["run_id"]] for r in rows if r["split"] == s and r["run_id"] in signal] for s in _EVAL}
        kn = knobs["signals"][sig] = {
            "auroc_val_vs_ood": gating.auroc(by_split["val"], by_split["ood"]),
            "auroc_test_vs_ood": gating.auroc(by_split["test"], by_split["ood"]),
            "median": {s: float(np.median(v)) for s, v in by_split.items() if v}, "q": {}}
        for q in gating.GATE_QS:
            g, tau = gating.gate(rows, signal, decoded[inner], decoded[outer], q)
            name = f"{gating.GATE_PREFIX}{sig}-q{round(q * 100)}"
            table += _pred_table(name, g, split_of)
            kn["q"][name] = {"tau": tau, "gated_out": {
                s: float(np.mean([g[r["run_id"]]["gated_out"] for r in rows if r["split"] == s and r["run_id"] in g]))
                for s in _EVAL}}
            if sig == gating.GATE_SIGNAL and q == gating.GATE_Q:
                table += _pred_table(f"{gating.GATE_PREFIX}{member}", g, split_of)
    for v in sorted(probs):
        base = v.removesuffix(FLYTE_ONLY_SUFFIX)
        if not v.endswith(FLYTE_ONLY_SUFFIX) or base not in metas or v not in metas:
            continue
        t = metas[v].get("gpu_threshold", 0.5)
        for suffix, memory_from in gating.ADAPT_SOURCES.items():
            prob = gating.adapted(rows, probs[v], memory_from)
            name = f"{v}{gating.ADAPT_SUFFIX}{suffix}"
            table += _pred_table(name, _decode(prob, t), split_of) + _pred_table(f"{name}-argmax", _argmax(prob), split_of)
            knobs[name] = {"from": v, "memory_from": memory_from, "gpu_threshold": t}
    return table, knobs


@allbench_env.task(cache=all_arms_cache(), report=True)
async def gate_all_arms(corpus: flyte.io.File, arm_dirs: list[flyte.io.Dir], rl_dir: flyte.io.Dir,
                        pool_dir: flyte.io.Dir, inner: str = "pool-rl-pointer-argmax", outer: str = "rl-pointer",
                        member: str = "rl-pointer") -> flyte.io.Dir:
    """Post-hoc arms over the trained ones (``gating``): the novelty gate and the
    Flyte -> K8s adapter. Nothing is retrained."""
    rows = [r for r in await read_rows(corpus) if dataset.usable(r)]
    probs, decoded, metas = {}, {}, {}
    for d in [*arm_dirs, rl_dir, pool_dir]:
        _local, preds, meta = await _read_arm(d)
        probs.update(_probs_of(preds))
        decoded.update(_decoded_of(preds))
        for v in preds["variant"].unique():
            metas[v] = meta
    table, knobs = await asyncio.to_thread(gate_all, rows, probs, decoded, metas, inner, outer, member)
    rep = Reporter(f"{TITLE}: gates and adapters", f"{inner} on familiar runs, {outer} on novel ones")
    rep.table(["signal", "AUROC (val vs unseen)", "AUROC (test vs unseen)"],
              [[k, f"{x['auroc_val_vs_ood']:.3f}", f"{x['auroc_test_vs_ood']:.3f}"] for k, x in knobs["signals"].items()])
    rep.table(["gate", "threshold", *[f"to {outer} ({s})" for s in _EVAL]],
              [[k, f"{x['tau']:.3g}", *[f"{x['gated_out'][s]:.1%}" for s in _EVAL]]
               for sk in knobs["signals"].values() for k, x in sk["q"].items()])
    await rep.flush()
    return await _write_arm(table, {"arm": "gate", **knobs})


def _load_policy(local: str, meta: dict):
    """A trained decision model (``_save``'s layout) for inference: tokenizer, model, head, config."""
    import torch
    from peft import PeftModel
    from transformers import AutoModel, AutoTokenizer

    cfg = TrainConfig(**{k: v for k, v in meta["config"].items() if k in TrainConfig.__dataclass_fields__})
    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    base = AutoModel.from_pretrained(cfg.model_name, dtype=_amp_dtype(), attn_implementation="sdpa").to(_device())
    model = PeftModel.from_pretrained(base, os.path.join(local, "model")).eval()
    head = policy.head_module(model.config.hidden_size, policy.option_offsets(tok)).to(_device())
    head.load_state_dict(torch.load(os.path.join(local, "model", "decision_head.pt"), map_location=_device()))
    head.eval()
    return tok, model, head, cfg


@allbench_train_env.task(cache=all_arms_cache(), report=True, timeout=flyte.Timeout(max_runtime=6 * 3600))
async def staging_counterfactual(corpus: flyte.io.File, arm_dirs: list[flyte.io.Dir], rl_dir: flyte.io.Dir,
                                 pool_dir: flyte.io.Dir, n_per_split: int = 600) -> flyte.io.File:
    """``counterfactual``: sampled CPU-node runs rendered as Jobs under all four
    stagings, sized by every approach; does the request move with the manifest?
    The trees and rules are refit (their arms keep no fitted state); the decision
    models are the trained ones, read with their own val temperatures and thresholds."""
    import gc

    import torch

    rows = await read_rows(corpus)
    splits = dataset.by_split(rows)
    picked = counterfactual.sample_runs(rows, n_per_split)
    rends = counterfactual.renditions(rows, picked)
    # the name test: the picked runs and their real twins, as named and with a neutral name
    by_id = {r["run_id"]: r for r in rows}
    named = [r for p in picked for r in (p, by_id.get(p["run_id"] + k8s.SUFFIX)) if r is not None and dataset.usable(r)]
    rends = rends + named + [counterfactual.neutral(r) for r in named]
    rep = Reporter(f"{TITLE}: staging counterfactual", f"{len(picked)} runs x 4 stagings = {len(rends)} renditions")
    await rep.flush()
    picks: dict[str, dict[str, dict]] = {}
    probs: dict[str, dict[str, list[np.ndarray]]] = {}
    for arm in ("static", "rules", "gbt-dist"):
        fit, predict = all_arms.ARMS[arm]
        state = await asyncio.to_thread(fit, splits["train"], splits["val"])
        picks[arm] = predict(state, rends)
        if arm == hybrid.ANCHOR:
            probs[arm] = {rid: [np.asarray(x, dtype=float) for x in p["probs"]] for rid, p in picks[arm].items()}
    for d in [*arm_dirs, rl_dir]:
        local, _preds, meta = await _read_arm(d)
        if not os.path.exists(os.path.join(local, "model", "decision_head.pt")) or "config" not in meta:
            continue
        name = meta["arm"]
        tok, model, head, cfg = _load_policy(local, meta)
        with torch.no_grad():
            P = policy.probs(policy.logits(tok, model, head, rends, cfg), meta["temperatures"])
        probs[name] = {r["run_id"]: [P[0][i], P[1][i], P[2][i]] for i, r in enumerate(rends)}
        picks[name] = _decode(probs[name], meta["gpu_threshold"])
        picks[f"{name}-argmax"] = _argmax(probs[name])
        del model, head
        gc.collect()
        torch.cuda.empty_cache()
    _local, _preds, pool_meta = await _read_arm(pool_dir)
    for pname, kn in pool_meta.get("knobs", {}).items():
        member = pname.removeprefix(hybrid.POOL_PREFIX)
        anchor = kn.get("anchor", hybrid.ANCHOR)
        if member not in probs or anchor not in probs:
            continue
        pooled = {rid: [hybrid.log_pool(a, b, kn["w_trees"]) for a, b in zip(probs[anchor][rid], m)]
                  for rid, m in probs[member].items()}
        picks[pname] = _decode(pooled, kn["gpu_threshold"])
        picks[f"{pname}-argmax"] = _argmax(pooled)
    stag = [r for r in rends if counterfactual.SEP in r["run_id"]]
    rep_json = counterfactual.report(stag, picks)
    names_json = counterfactual.name_report(named, picks)
    rep.h("medium contrast (emptyDir {} -> medium: Memory), test: direction right where the label moved")
    rep.table(["approach", "label moved", "direction right (moved)", "kept (unmoved)", "exact (moved)"],
              [[v, f"{x['split']['test']['medium']['label_moved']:.1%}",
                f"{x['split']['test']['medium']['moved_direction']:.1%}",
                f"{x['split']['test']['medium']['unmoved_kept']:.1%}",
                f"{x['split']['test']['medium']['moved_exact']:.1%}"] for v, x in rep_json.items()])
    await rep.flush()
    rep.h("the task's name: regret with the family's name and with a neutral one (test)")
    rep.table(["approach", "named", "neutral", "difference"],
              [[v, *(f"{x.get('test', {}).get(k, float('nan')):.1%}" for k in ("named", "neutral", "diff"))]
               for v, x in names_json.items()])
    out = {"n_runs": len(picked), "n_renditions": len(stag), "report": rep_json, "names": names_json,
           "picks": {v: {rid: [int(p[k]) for k in ("mem_idx", "cpu_idx", "gpu_idx")] for rid, p in pv.items()}
                     for v, pv in picks.items()},
           "rows": [{k: r.get(k) for k in ("run_id", "split", "family", "staging", "label_mem", "need_mib")}
                    for r in rends]}
    path = os.path.join(tempfile.mkdtemp(prefix="rsa-cf-"), "counterfactual.json")
    with open(path, "w") as f:
        json.dump(out, f, default=float)
    return await flyte.io.File.from_local(path)


def _entropy(p: np.ndarray) -> float:
    p = np.clip(np.asarray(p, dtype=float), 1e-12, None)
    return float(-(p * np.log(p)).sum())


@allbench_env.task(report=True)
async def gate_signals(corpus: flyte.io.File, arm_dirs: list[flyte.io.Dir], rl_dir: flyte.io.Dir,
                       pool_dir: flyte.io.Dir, member: str = "rl-pointer") -> str:
    """Analysis, not a station: per evaluated run, candidate novelty signals and
    every arm's cost ratio, gzipped JSON (base64) for choosing a gate offline."""
    import base64
    import gzip

    all_rows = await read_rows(corpus)
    rows = [r for r in all_rows if dataset.usable(r)]
    probs, decoded = {}, {}
    for d in [*arm_dirs, rl_dir, pool_dir]:
        _local, preds, _meta = await _read_arm(d)
        probs.update(_probs_of(preds))
        decoded.update(_decoded_of(preds))
    sigs = await asyncio.to_thread(gate_signals_of, rows, probs, member)
    a, m = probs[hybrid.ANCHOR], probs[member]
    out = []
    for r in rows:
        rid = r["run_id"]
        if r["split"] not in _EVAL or rid not in a or rid not in m:
            continue
        x = {"id": rid, "split": r["split"], "platform": dataset.platform_of(r), "source": r["source"],
             "family": r["family"], "knn": round(sigs["knn"].get(rid, float("nan")), 4),
             "knn_numeric": round(sigs["knn-numeric"].get(rid, float("nan")), 4),
             "js": [round(gating.js(p, q), 4) for p, q in zip(a[rid], m[rid])],
             "h_member": [round(_entropy(p), 4) for p in m[rid]], "h_anchor": [round(_entropy(p), 4) for p in a[rid]],
             "ratio": {v: round(scoring.ratio(r, p[rid]["mem_idx"], p[rid]["cpu_idx"], p[rid]["gpu_idx"]), 4)
                       for v, p in decoded.items() if rid in p}}
        out.append(x)
    return base64.b64encode(gzip.compress(json.dumps(out).encode())).decode()


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


HEADLINE = ("static", "rules", "rules-sized", "gbt-dist", "gbt-dist-flyte", "gbt-dist-k8s", "zeroshot-27b",
            "zeroshot-27b-json", "jev-pointer-k8s-argmax", "jev-pointer-k8s", "pool-jev-pointer-k8s", "jev-pointer-argmax", "jev-pointer",
            "rl-pointer-argmax", "rl-pointer", "stack-jev-pointer-argmax", "stack-jev-pointer",
            "stack-rl-pointer-argmax", "stack-rl-pointer", "pool-jev-pointer", "pool-rl-pointer",
            "pool-stack-jev-pointer", "pool-stack-rl-pointer", "jev-pointer-flyte-argmax", "jev-pointer-flyte",
            "pool-jev-pointer-flyte", "pool-rl-pointer-argmax", "jev-pointer-flyte-adapt", "gate-rl-pointer")


@allbench_env.task(cache=all_arms_cache(), report=True)
async def all_leaderboard(scorecards: list[flyte.io.File], rl_scorecard: flyte.io.File | None = None,
                          rl_stack_scorecard: flyte.io.File | None = None,
                          pool_scorecard: flyte.io.File | None = None,
                          gate_scorecard: flyte.io.File | None = None,
                          counterfactual: flyte.io.File | None = None) -> flyte.io.File:
    cards = []
    extra = (rl_scorecard, rl_stack_scorecard, pool_scorecard, gate_scorecard)
    for f in [*scorecards, *[c for c in extra if c is not None]]:
        with open(await f.download()) as fh:
            cards.append(json.load(fh))
    board = {}
    for c in cards:
        board.update(c["variants"])
    order = [v for v in HEADLINE if v in board] + sorted(v for v in board if v not in HEADLINE)
    rep = Reporter(f"{TITLE}: leaderboard", f"{len(order)} decision rules, one cost model")
    both = [(v, board[v]["split=test"]["regret"], board[v]["split=ood"]["regret"]) for v in order
            if board[v].get("split=test", {}).get("n") and board[v].get("split=ood", {}).get("n")]
    if both:
        rep.h("the worse of test and unseen families (lower is better)")
        rep.table(["arm", "test regret", "unseen regret", "worse of the two"],
                  [[v, f"{t:.1%}", f"{o:.1%}", f"{max(t, o):.1%}"] for v, t, o in sorted(both, key=lambda x: max(x[1:]))])
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
        if any(f"{split}/platform=k8s" in board[v] for v, _ in rows):
            def cell(v, k, m="regret"):
                return f"{board[v].get(k, {}).get(m, float('nan')):.1%}"

            rep.h(f"{split}: Flyte tasks vs their Kubernetes twins")
            rep.table(["arm", "Flyte regret", "K8s regret", "Flyte $ regret", "K8s $ regret", "twin regret gap",
                       "memory shift right (direction)", "both memory right"],
                      [[v, cell(v, f"{split}/platform=flyte"), cell(v, f"{split}/platform=k8s"),
                        cell(v, f"{split}/platform=flyte", "dollar_regret"), cell(v, f"{split}/platform=k8s", "dollar_regret"),
                        f"{board[v].get(f'{split}/pairs', {}).get('regret_gap', float('nan')):+.1%}",
                        cell(v, f"{split}/pairs", "mem_shift_direction"), cell(v, f"{split}/pairs", "mem_both_right")]
                       for v, _ in rows])
            stagings = sorted({k.split("=", 1)[1] for v, _ in rows for k in board[v] if k.startswith(f"{split}/staging=")})
            rep.h(f"{split}: K8s twins (CPU-node runs) by staging")
            rep.table(["arm", *stagings], [[v, *[cell(v, f"{split}/staging={st}") for st in stagings]] for v, _ in rows])
    cf = None
    if counterfactual is not None:
        with open(await counterfactual.download()) as fh:
            cf = json.load(fh)
        cf.pop("picks", None), cf.pop("rows", None)
        rep.h("does the request move with the manifest? (the same runs under all four stagings, test)")
        rep.table(["approach", "medium: label moved", "direction right (moved)", "kept (unmoved)",
                   "fetcher: direction right (moved)", "named -> neutral name, regret"],
                  [[v, f"{x['split']['test']['medium']['label_moved']:.1%}",
                    f"{x['split']['test']['medium']['moved_direction']:.1%}",
                    f"{x['split']['test']['medium']['unmoved_kept']:.1%}",
                    f"{x['split']['test']['fetcher']['moved_direction']:.1%}",
                    f"{cf['names'].get(v, {}).get('test', {}).get('diff', float('nan')):+.1%}"]
                   for v, x in cf["report"].items()])
    await rep.flush()
    path = os.path.join(tempfile.mkdtemp(prefix="rsa-board-"), "leaderboard.json")
    with open(path, "w") as f:
        json.dump({"order": order, "board": board, "counterfactual": cf,
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
