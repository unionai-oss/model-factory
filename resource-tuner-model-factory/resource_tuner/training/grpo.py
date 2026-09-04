"""GRPO training of the resource-tuner policy.

TRL `GRPOTrainer` + LoRA, reward computed in-process against the SIMULATOR
(the cheap loop): each completion is parsed into a Proposal and scored
against the context's analytic footprint. No cluster episodes inside the
training step — sim rewards keep a step at milliseconds; the on-cluster
truth enters through evaluation instead.

Recipe notes (hard-won upstream, see docs/DESIGN.md):
- attn_implementation="eager": SDPA emits nan logits during left-padded
  GRPO generation on some head_dim-128 models.
- dtype follows the device: bf16 on Ampere+ (A10G), fp16 on Turing (T4).
- plain-LoRA + gradient checkpointing needs enable_input_require_grads().
- QLoRA (nf4) for ladder rungs that don't fit the GPU in 16-bit.
"""

from __future__ import annotations

import json
import os
import tempfile

import flyte
import flyte.io

from ..config import TunerProfile, WANDB_PROJECT, get_profile
from ..contracts import ARTIFACT_TASK_CORPUS, ARTIFACT_TUNER_CHECKPOINT, publish
from ..environment.simulator import simulate_episode
from ..policy.parsing import try_extract_proposal
from ..policy.prompts import render_messages
from ..rewards.rewards import invalid_proposal_reward, score_episode
from ..shared.reporting import GOOD, Reporter, esc, line_chart, pill
from .envs import trainer_env


def make_reward_fn(stage: str, jitter_rng=None):
    """completions + per-sample truth columns → list of scalar rewards.

    TRL forwards non-reserved dataset columns to the reward function as
    kwargs of per-sample lists. `jitter_rng` enables the simulator's
    threshold randomization during training (see simulator.JITTER_BAND);
    tests omit it for determinism.
    """

    def resource_reward(
        completions, true_peak_memory_mib, true_cpu_cores, duration_s, **kwargs
    ) -> list[float]:
        rewards: list[float] = []
        for completion, peak, cpu, dur in zip(
            completions, true_peak_memory_mib, true_cpu_cores, duration_s
        ):
            text = completion[0]["content"] if isinstance(completion, list) else completion
            proposal = try_extract_proposal(text)
            if proposal is None:
                rewards.append(invalid_proposal_reward().total)
                continue
            episode = simulate_episode(
                proposal, float(peak), float(cpu), int(dur), rng=jitter_rng
            )
            rewards.append(score_episode(stage, episode).total)
        return rewards

    return resource_reward


def _records_to_dataset(records: list[dict]):
    from datasets import Dataset

    rows = [
        {
            "prompt": render_messages(r["source_code"], r["input_profile"]),
            "true_peak_memory_mib": r["true_peak_memory_mib"],
            "true_cpu_cores": r["true_cpu_cores"],
            "duration_s": r["duration_s"],
        }
        for r in records
    ]
    return Dataset.from_list(rows)


@flyte.trace
def _load_model(profile: TunerProfile):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float16

    quant_config = None
    if profile.use_qlora:
        from transformers import BitsAndBytesConfig

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )

    tok = AutoTokenizer.from_pretrained(profile.base_model, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        profile.base_model,
        dtype=dtype,
        device_map="auto",
        quantization_config=quant_config,
        attn_implementation="eager",
        trust_remote_code=True,
    )
    if profile.use_qlora:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.enable_input_require_grads()
    return model, tok, bf16


def _report_html(profile: TunerProfile, rows: list[dict], meta: dict | None = None) -> str:
    """Live training report, re-rendered every logged step.

    Reads as a page: config header (knobs / device / dataset / W&B link),
    the go/no-go reward curve, group-health curves (entropy +
    frac_reward_zero_std — the degenerate-group early warning), training
    dynamics (grad_norm, completion length), and the last step's stats.
    """
    meta = meta or {}
    rewards = [r["reward"] for r in rows if "reward" in r]
    knobs = (
        f"lr {profile.learning_rate} · group {profile.num_generations} · "
        f"batch {profile.per_device_batch} · comp_len {profile.max_completion_length} · "
        f"lora r{profile.lora_r}{' · qlora' if profile.use_qlora else ''}"
    )
    rep = Reporter(
        f"GRPO — {profile.name} / {profile.base_model}",
        f"reward stage: {profile.reward_stage} · {knobs}",
    )
    config_kv = {
        "device": meta.get("device", "?"),
        "dtype": meta.get("dtype", "?"),
        "train contexts": meta.get("n_contexts", "?"),
        "corpus": str(meta.get("corpus", ""))[-60:] or "?",
    }
    if meta.get("wandb_url"):
        rep.raw(
            f'<p style="margin:6px 0"><a style="color:#8b9bff" href="{esc(meta["wandb_url"])}">'
            "Weights &amp; Biases run ↗</a></p>"
        )
    rep.kv(config_kv)
    if not rewards:
        rep.p("model loading / waiting for the first logged step…")
        return rep.html()

    def col(key):
        return [r.get(key) for r in rows if "reward" in r]

    rep.progress(len(rewards), profile.max_steps, "steps")
    rep.p(
        f"step {len(rewards)}/{profile.max_steps} · mean reward {rewards[-1]:.3f} "
        f"(start {rewards[0]:.3f})"
    )
    rep.h("Reward")
    rep.raw(line_chart([("mean reward", GOOD, rewards)], y_fmt="{:.2f}"))
    rep.h("Group health (all-pass/all-fail groups yield no gradient)")
    rep.raw(
        line_chart(
            [
                ("frac_reward_zero_std", "#e69812", col("frac_reward_zero_std")),
                ("entropy", "#4d65ff", col("entropy")),
            ],
            y_max=1.0,
            y_fmt="{:.1f}",
        )
    )
    rep.h("Dynamics")
    rep.raw(
        line_chart(
            [
                ("grad_norm", "#F43B3E", col("grad_norm")),
                ("completion len", "#9a9aa4", col("completions/mean_length")),
            ],
            y_fmt="{:.0f}",
        )
    )
    last = rows[-1]
    rep.h("Last step")
    rep.table(
        ["metric", "value"],
        [
            [esc(k), esc(f"{v:.4g}")]
            for k, v in last.items()
            if isinstance(v, (int, float))
        ],
    )
    return rep.html()


@flyte.trace
async def _save_checkpoint(trainer, tok, out_dir: str, profile: TunerProfile,
                           history: list[dict], mean_rewards: list) -> "flyte.io.Dir":
    """Write adapter + tokenizer + manifest and upload as a Dir (traced —
    checkpoint writes are exactly where a crashed run wants a span)."""
    trainer.model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(
            {
                "base_model": profile.base_model,
                "profile": profile.name,
                "reward_stage": profile.reward_stage,
                "max_steps": profile.max_steps,
                "final_metrics": {
                    "mean_reward_first": mean_rewards[0] if mean_rewards else None,
                    "mean_reward_last": mean_rewards[-1] if mean_rewards else None,
                    "log_history": history,
                },
            },
            f,
            indent=2,
        )
    return await flyte.io.Dir.from_local(out_dir)


# Dark-mode wiring: a new corpus version IS the request to train.
_train_trigger = flyte.Trigger(
    name="train-on-new-corpus",
    automation=flyte.OnArtifact(name=ARTIFACT_TASK_CORPUS),
    inputs={"corpus": flyte.TriggeredArtifact, "profile_name": "smoke"},
    description="New tuning-task-corpus version -> GRPO training",
    auto_activate=False,
)


@trainer_env.task(
    triggers=[_train_trigger],
    timeout=flyte.Timeout(max_runtime=6 * 3600),
    produces_artifacts=True,
    report=True,
)
async def train_tuner(corpus: flyte.io.File, profile_name: str = "smoke") -> flyte.io.Dir:
    """Train the policy on the corpus's train split; emit tuner-checkpoint."""
    import asyncio

    import pandas as pd
    from peft import LoraConfig
    from transformers import TrainerCallback
    from trl import GRPOConfig, GRPOTrainer

    profile = get_profile(profile_name)
    meta: dict = {"corpus": getattr(corpus, "path", "")}
    # Page up immediately: the model download/load takes minutes and the
    # run page should say so rather than sit blank.
    await flyte.report.replace.aio(_report_html(profile, [], meta), do_flush=True)

    df = pd.read_parquet(await corpus.download())
    records = df[df["split"] == "train"].to_dict("records")[: profile.train_contexts]
    dataset = _records_to_dataset(records)
    meta["n_contexts"] = len(records)

    model, tok, bf16 = _load_model(profile)
    import torch

    meta["device"] = (
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    )
    meta["dtype"] = "bf16" if bf16 else "fp16"
    await flyte.report.replace.aio(_report_html(profile, [], meta), do_flush=True)
    lora = LoraConfig(
        r=profile.lora_r,
        lora_alpha=profile.lora_r * 2,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    out_dir = tempfile.mkdtemp(prefix="tuner-ckpt-")
    wandb_on = bool(os.environ.get("WANDB_API_KEY"))
    if wandb_on:
        os.environ.setdefault("WANDB_PROJECT", WANDB_PROJECT)
        run_name = ""
        try:
            run_name = flyte.ctx().action.run_name  # type: ignore[union-attr]
        except Exception:
            pass
        os.environ.setdefault("WANDB_NAME", f"{profile.name}-{run_name or 'local'}")
        os.environ.setdefault("WANDB_TAGS", f"{profile.reward_stage},{profile.base_model}")
    import dataclasses
    import random

    # Disable Qwen3-style thinking during rollouts where the installed TRL
    # supports it (the /no_think prompt switch covers older versions).
    extra_cfg = {}
    if any(f.name == "chat_template_kwargs" for f in dataclasses.fields(GRPOConfig)):
        extra_cfg["chat_template_kwargs"] = {"enable_thinking": False}

    config = GRPOConfig(
        output_dir=out_dir,
        max_steps=profile.max_steps,
        per_device_train_batch_size=profile.per_device_batch,
        num_generations=profile.num_generations,
        max_completion_length=profile.max_completion_length,
        learning_rate=profile.learning_rate,
        temperature=1.0,
        # 2026 small-model GRPO consensus (DAPO / Dr.GRPO line): no KL
        # anchor, token-level loss without length bias, batch-scaled
        # advantages. Watch frac_reward_zero_std in the logs — all-pass /
        # all-fail groups contribute no gradient.
        beta=0.0,
        loss_type="dapo",
        scale_rewards="batch",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=bf16,
        fp16=not bf16,
        logging_steps=1,
        save_strategy="no",
        report_to="wandb" if wandb_on else "none",
        **extra_cfg,
    )
    # Live flyte report: the trainer runs in a worker thread, so the
    # callback ships HTML back through the captured event loop.
    loop = asyncio.get_running_loop()

    class _ReportCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if wandb_on and not meta.get("wandb_url"):
                try:
                    import wandb

                    if wandb.run is not None:
                        meta["wandb_url"] = wandb.run.url
                except Exception:
                    pass
            rows = [r for r in state.log_history if isinstance(r, dict)]
            html = _report_html(profile, rows, meta)
            asyncio.run_coroutine_threadsafe(
                flyte.report.replace.aio(html, do_flush=True), loop
            )

    trainer = GRPOTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        reward_funcs=make_reward_fn(profile.reward_stage, jitter_rng=random.Random(0)),
        peft_config=lora,
        processing_class=tok,
        callbacks=[_ReportCallback()],
    )
    # The trainer is blocking for the whole run; keep the event loop alive.
    try:
        await asyncio.to_thread(trainer.train)
    except torch.cuda.OutOfMemoryError as e:
        # Explicit OOM handling: fail with a pointed message instead of a
        # bare CUDA traceback (the fix is a knob, not a retry).
        raise RuntimeError(
            f"GPU OOM during GRPO training on {meta.get('device')}: {e}. "
            f"Reduce per_device_batch/num_generations/max_completion_length "
            f"or set use_qlora for {profile.base_model}."
        ) from e

    history = [
        {k: v for k, v in row.items() if isinstance(v, (int, float))}
        for row in trainer.state.log_history
    ]
    mean_rewards = [h["reward"] for h in history if "reward" in h]
    ckpt = await _save_checkpoint(trainer, tok, out_dir, profile, history, mean_rewards)
    return publish(
        ckpt,
        ARTIFACT_TUNER_CHECKPOINT,
        description=f"{profile.base_model} LoRA, stage={profile.reward_stage}, "
        f"steps={profile.max_steps}",
        kind="model",
    )
