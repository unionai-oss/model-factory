"""Training station: fine-tune one torso x head.

One task call = one suite member. The factory instantiates this build once
per value of the `model` partition (`qwen2.5-0.5b-pointer`,
`qwen2.5-0.5b-sft`, ...), so the suite is a fan-out over a partition
dimension rather than a loop inside a task: each fine-tune is separately
cached, retried and visible.

Both heads go through the SAME loop (`heads.modeling.train`): same LoRA, same
prompt, same data, same optimizer, same number of steps. The only branch is
the head and its loss — cross-entropy over the options for ``pointer``,
cross-entropy over the completion tokens for ``sft``.

Why supervised and not RL: the oracle gives an exact label for every call.
"""

from __future__ import annotations

import json
import os
import random
from datetime import datetime

import flyte
import flyte.io
import flyte.report

from ..config import POINTER_RANK, get_candidate, get_profile
from ..harness.policy import Observation
from ..harness.prompt import build_chat, shuffled_options
from ..heads import modeling
from ..shared import reporting
from .envs import trainer_env


def training_inputs(rows: list[dict], seed: int = 0) -> tuple[list, list, list]:
    """Episodes -> (chats, option orders, labels).

    Option order is shuffled per example for BOTH heads, so the two arms see
    the identical prompt distribution; for the pointer head it is what stops
    it learning "the answer is usually the first slot".
    """
    rng = random.Random(seed)
    chats, options, labels = [], [], []
    for r in rows:
        opts = shuffled_options(rng)
        chats.append(build_chat(Observation.from_row(r), opts))
        options.append(opts)
        labels.append(r["label"])
    return chats, options, labels


def write_manifest(out_dir: str, info: dict, *, slug: str, torso: str, head: str, base_model: str,
                   params_m: int | None, profile, train_n: int, date: datetime | None = None) -> dict:
    """The model Dir's manifest: what the scorer, selector and app read to
    know how to load it (`base_model`, `head`, `pointer_rank`)."""
    manifest = {
        "model_slug": slug,
        "torso": torso,
        "head": head,
        "base_model": base_model,
        "params_m": params_m,
        "profile": profile.name,
        "train_episodes": train_n,
        "epochs": profile.epochs,
        "lora_r": profile.lora_r,
        "date": date.strftime("%Y-%m-%d") if date else "",
        "final_loss": info["history"][-1]["loss"] if info["history"] else None,
        **info,
    }
    with open(os.path.join(out_dir, modeling.MANIFEST), "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    return manifest


@trainer_env.task(report=True, timeout=flyte.Timeout(max_runtime=7200))
async def finetune_decision_model(
    episodes: flyte.io.File,
    model: str,
    profile_name: str = "dev",
    date: datetime | None = None,
) -> flyte.io.Dir:
    """``model`` is the suite slug and the `model` partition value — the
    factory passes it with ``factory.partition("model")``."""
    import pandas as pd

    profile = get_profile(profile_name)
    cand = get_candidate(model)
    df = pd.read_parquet(await episodes.download())
    rows = df[df["split"] == "train"].to_dict("records")
    if not rows:
        raise flyte.errors.NonRecoverableError("no train-split episodes in the dataset")
    chats, options, labels = training_inputs(rows)

    history: list[dict] = []

    def on_log(rec: dict) -> None:
        history.append(rec)
        try:
            body = reporting.stats_row({"model": cand.slug, "head": cand.head, "step": rec["step"],
                                        "loss": f"{rec['loss']:.4f}", "epoch": f"{rec['epoch']:.2f}"})
            body += reporting.table(["step", "loss", "lr", "elapsed"],
                                    [[h["step"], f"{h['loss']:.4f}", f"{h['lr']:.2e}", f"{h['elapsed_s']:.0f}s"]
                                     for h in history[-25:]])
            flyte.report.replace(reporting.page(f"Training {cand.slug}", body))
            flyte.report.flush()
        except Exception:
            pass  # reporting must never kill training

    out_dir = "/tmp/permission-model"
    info = modeling.train(
        cand.head, cand.hf_id, chats, options, labels, out_dir,
        epochs=profile.epochs, batch_size=profile.batch_size, grad_accum=profile.grad_accum,
        lr=profile.learning_rate, head_lr=profile.head_learning_rate, lora_r=profile.lora_r,
        pointer_rank=POINTER_RANK, max_len=profile.max_seq_length, on_log=on_log,
    )

    manifest = write_manifest(
        out_dir, info, slug=cand.slug, torso=cand.torso.slug, head=cand.head, base_model=cand.hf_id,
        params_m=cand.params_m, profile=profile, train_n=len(rows), date=date)

    body = reporting.stats_row({
        "model": cand.slug, "base": cand.hf_id, "head": cand.head,
        "head params": f"{info['head_params'] / 1e6:.2f}M", "trainable": f"{info['trainable_params'] / 1e6:.2f}M",
        "train examples": info["train_examples"], "steps": info["steps"],
        "train time": f"{info['train_seconds'] / 60:.1f} min",
        "final loss": f"{manifest['final_loss']:.4f}" if manifest["final_loss"] is not None else "n/a",
    })
    body += "<h3>Loss</h3>" + reporting.table(
        ["step", "loss", "epoch"], [[h["step"], f"{h['loss']:.4f}", f"{h['epoch']:.2f}"] for h in info["history"]])
    await flyte.report.replace.aio(reporting.page(f"Fine-tuned {cand.slug}", body))
    await flyte.report.flush.aio()
    return await flyte.io.Dir.from_local(out_dir)
