"""The GPU half of the LLM arms: train on train, decode val/test/ood.

One task per arm (``sft`` or ``jev``), same torso and budget for both
(``llm_arms.TrainConfig``). It writes the same predictions table every arm
writes, with one row per (decoding variant, run):

- sft: ``sft-greedy`` (+k-cell margin tuned on val), ``sft-sampled`` (N
  samples -> empirical distribution -> expected-cost decode)
- jev / jev-pointer: ``<arm>`` (temperature-scaled -> expected-cost decode)
  and ``<arm>-argmax``; the two arms differ only in the head shape
"""

from __future__ import annotations

import json
import os
import tempfile
import time

import flyte
import flyte.io
import numpy as np
import pandas as pd

from ..shared.reporting import Reporter
from . import dataset, grid, prompting, scoring
from .envs import bench_gpu_env
from .llm_arms import (
    TrainConfig,
    config_dict,
    fit_temperature,
    jev_logits,
    measure_latency,
    samples_to_probs,
    sft_generate,
    shift,
    softmax,
    train_jev,
    train_sft,
)

_EVAL = ("val", "test", "ood")


def _rows_by_split(df: pd.DataFrame) -> dict[str, list[dict]]:
    out = {s: [] for s in dataset.SPLITS}
    for r in df.to_dict("records"):
        if dataset.usable(r):
            out[r["split"]].append(r)
    return out


def _pred_table(variant: str, preds: dict, split_of: dict) -> list[dict]:
    return [{"variant": variant, "run_id": rid, "split": split_of[rid], "mem_idx": int(p["mem_idx"]),
             "cpu_idx": int(p["cpu_idx"]),
             "mem_probs": json.dumps([round(float(x), 6) for x in p["mem_probs"]]) if p.get("mem_probs") is not None else "",
             "raw": str(p.get("raw", ""))[:200]} for rid, p in preds.items()]


@bench_gpu_env.task(retries=1, timeout=flyte.Timeout(max_runtime=6 * 3600))
async def train_llm_arm(corpus: flyte.io.File, arm: str, llm_config: str = "{}") -> flyte.io.Dir:
    cfg = TrainConfig(**json.loads(llm_config or "{}"))
    if arm == "jev-pointer":
        cfg.head = "pointer"
    df = pd.read_parquet(await corpus.download())
    splits = _rows_by_split(df)
    train = splits["train"]
    if cfg.max_train_rows:
        train = train[: cfg.max_train_rows]
    eval_rows = [r for s in _EVAL for r in splits[s]]
    split_of = {r["run_id"]: r["split"] for r in eval_rows}
    rep = Reporter(f"RightSize Bench: {arm} on {cfg.model_name}",
                   f"train {len(train)} runs, eval {len(eval_rows)}")
    logs: list[str] = []

    def log(msg: str) -> None:
        print(msg, flush=True)
        logs.append(msg)

    await rep.flush()
    t0 = time.time()
    table: list[dict] = []
    meta: dict = {"arm": arm, "config": config_dict(cfg), "n_train": len(train)}
    if arm in ("jev", "jev-pointer"):
        tok, model, head, hist = train_jev(train, splits["val"], cfg, log=log)
        meta["train_s"], meta["loss"] = time.time() - t0, hist
        mem_l, cpu_l = jev_logits(tok, model, head, eval_rows, cfg)
        val_idx = [i for i, r in enumerate(eval_rows) if r["split"] == "val"]
        T = fit_temperature(mem_l[val_idx], np.array([eval_rows[i]["label_mem_idx"] for i in val_idx]))
        meta["temperature"] = T
        probs = softmax(mem_l / T)
        cpu_idx = cpu_l.argmax(1)
        bayes, argmax = {}, {}
        for i, r in enumerate(eval_rows):
            p = probs[i]
            bayes[r["run_id"]] = {"mem_idx": scoring.bayes_mem_index(p), "cpu_idx": int(cpu_idx[i]), "mem_probs": p}
            argmax[r["run_id"]] = {"mem_idx": int(p.argmax()), "cpu_idx": int(cpu_idx[i]), "mem_probs": p}
        table += _pred_table(arm, bayes, split_of) + _pred_table(f"{arm}-argmax", argmax, split_of)
        meta["latency"] = measure_latency(lambda rs: jev_logits(tok, model, head, rs, cfg, batch_size=1), eval_rows)
        meta["tokens_generated_per_decision"] = 0
        adapter_dir = tempfile.mkdtemp(prefix="rsb-jev-")
        model.save_pretrained(adapter_dir)
        import torch

        torch.save(head.state_dict(), os.path.join(adapter_dir, "decision_head.pt"))
    elif arm == "sft":
        tok, model, hist = train_sft(train, cfg, log=log)
        meta["train_s"], meta["loss"] = time.time() - t0, hist
        greedy_txt = sft_generate(tok, model, eval_rows, cfg, n=1)
        greedy = {}
        parse_fail = 0
        for r, txt in zip(eval_rows, greedy_txt):
            a = prompting.parse_answer(txt[0])
            if a is None:
                parse_fail += 1
                a = (grid.mem_index_covering(4096), grid.cpu_index_covering(1))
            greedy[r["run_id"]] = {"mem_idx": a[0], "cpu_idx": a[1], "raw": txt[0]}
        meta["greedy_parse_failures"] = parse_fail
        val_ids = {r["run_id"] for r in splits["val"]}
        val_rows = splits["val"]
        best_k = min(range(0, 4), key=lambda k: scoring.score(
            val_rows, {rid: p for rid, p in shift(greedy, k).items() if rid in val_ids})["cost_per_1k_task_hr"])
        meta["greedy_margin_cells"] = best_k
        table += _pred_table("sft-greedy", shift(greedy, best_k), split_of)
        sampled_txt = sft_generate(tok, model, eval_rows, cfg, n=cfg.n_samples, batch_size=8)
        sampled = {}
        for r, txts in zip(eval_rows, sampled_txt):
            p, c = samples_to_probs(txts)
            sampled[r["run_id"]] = {"mem_idx": scoring.bayes_mem_index(p), "cpu_idx": c, "mem_probs": p}
        table += _pred_table("sft-sampled", sampled, split_of)
        meta["latency_greedy"] = measure_latency(lambda rs: sft_generate(tok, model, rs, cfg, n=1, batch_size=1), eval_rows)
        meta["latency_sampled"] = measure_latency(
            lambda rs: sft_generate(tok, model, rs, cfg, n=cfg.n_samples, batch_size=1), eval_rows, n=16)
        meta["tokens_generated_per_decision"] = float(np.mean([len(tok(t[0])["input_ids"]) for t in greedy_txt]))
        adapter_dir = tempfile.mkdtemp(prefix="rsb-sft-")
        model.save_pretrained(adapter_dir)
    else:
        raise ValueError(f"train_llm_arm: unknown arm {arm!r}")
    meta["total_s"] = time.time() - t0
    out = tempfile.mkdtemp(prefix="rsb-arm-")
    pd.DataFrame(table).to_parquet(os.path.join(out, "predictions.parquet"), index=False)
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=float)
    with open(os.path.join(out, "train.log"), "w") as f:
        f.write("\n".join(logs))
    os.rename(adapter_dir, os.path.join(out, "adapter"))
    rep.kv({k: json.dumps(v, default=float)[:200] for k, v in meta.items()})
    await rep.flush()
    return await flyte.io.Dir.from_local(out)
