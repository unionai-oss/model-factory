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
from ..contracts import ARTIFACT_TUNER_CHECKPOINT, publish
from ..environment.simulator import simulate_episode
from ..policy.parsing import try_extract_proposal
from ..policy.prompts import render_messages
from ..rewards.rewards import invalid_proposal_reward, score_episode
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


@trainer_env.task(timeout=flyte.Timeout(max_runtime=6 * 3600), produces_artifacts=True)
async def train_tuner(corpus: flyte.io.File, profile_name: str = "smoke") -> flyte.io.Dir:
    """Train the policy on the corpus's train split; emit tuner-checkpoint."""
    import asyncio

    import pandas as pd
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    profile = get_profile(profile_name)
    df = pd.read_parquet(await corpus.download())
    records = df[df["split"] == "train"].to_dict("records")[: profile.train_contexts]
    dataset = _records_to_dataset(records)

    model, tok, bf16 = _load_model(profile)
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
    import random

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
    )
    trainer = GRPOTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        reward_funcs=make_reward_fn(profile.reward_stage, jitter_rng=random.Random(0)),
        peft_config=lora,
        processing_class=tok,
    )
    # The trainer is blocking for the whole run; keep the event loop alive.
    await asyncio.to_thread(trainer.train)

    history = [
        {k: v for k, v in row.items() if isinstance(v, (int, float))}
        for row in trainer.state.log_history
    ]
    mean_rewards = [h["reward"] for h in history if "reward" in h]
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

    ckpt = await flyte.io.Dir.from_local(out_dir)
    return publish(
        ckpt,
        ARTIFACT_TUNER_CHECKPOINT,
        description=f"{profile.base_model} LoRA, stage={profile.reward_stage}, "
        f"steps={profile.max_steps}",
        kind="model",
    )
