"""Model training team: GRPO with verifiable code-execution rewards.

Consumes: `rl-tasks-dataset` (OnArtifact trigger — a new approved dataset
version starts a training run automatically where the backend supports
artifact events). Publishes: `policy-checkpoint`. The team never calls data
engineering or eval code; artifacts are the only interface.

TRL GRPOTrainer + LoRA on a single 24GB GPU (A10G/L4). Rollout completions
are scored by the reward stack in rewards.py — sandboxed test execution is
the reward source, exactly the RLCEF pattern at POC scale.

Observability (bottleneck 2 of the abstract):
- live flyte report: reward/loss curves stream to the Union console
- W&B run when NIELS_WANDB_API_KEY is provisioned (else disabled mode)
- per-component reward means logged every step
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor

import flyte
import flyte.io
import flyte.report

from ..config import WANDB_PROJECT, get_profile
from ..contracts import ARTIFACT_CHECKPOINT, ARTIFACT_RL_DATASET, publish
from ..shared import reporting
from ..shared.rewards import MAX_REWARD, build_prompt, score_completion
from .envs import trainer_env

# Dark-mode wiring: a new approved dataset version IS the request to train.
_retrain_trigger = flyte.Trigger(
    name="train-on-new-dataset",
    automation=flyte.OnArtifact(name=ARTIFACT_RL_DATASET),
    inputs={"dataset": flyte.TriggeredArtifact, "profile_name": "smoke"},
    description="New approved dataset version -> GRPO training",
    auto_activate=False,
)

_SCORE_WORKERS = 8


def _completion_text(completion) -> str:
    """TRL passes chat completions as [{'role','content'}] lists."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        return completion[-1].get("content", "")
    return ""


def make_reward_fn(metrics_sink: list[dict]):
    """Build the TRL reward function; extra dataset columns arrive as kwargs."""

    def reward_fn(prompts, completions, tests, **kwargs):
        texts = [_completion_text(c) for c in completions]
        with ThreadPoolExecutor(max_workers=_SCORE_WORKERS) as pool:
            breakdowns = list(pool.map(score_completion, texts, tests))
        step_stats = {
            "reward/mean_total": sum(b.total for b in breakdowns) / len(breakdowns),
            "reward/format_rate": sum(b.format_ok for b in breakdowns) / len(breakdowns),
            "reward/compile_rate": sum(b.compiles for b in breakdowns) / len(breakdowns),
            "reward/pass_rate": sum(b.tests_passed for b in breakdowns) / len(breakdowns),
            "reward/guard_violations": sum(1 for b in breakdowns if b.guard_violation),
        }
        metrics_sink.append(step_stats)
        return [b.total for b in breakdowns]

    reward_fn.__name__ = "code_execution_reward"
    return reward_fn


@trainer_env.task(report=True, timeout=flyte.Timeout(max_runtime=7200), triggers=[_retrain_trigger])
async def train_grpo(dataset: flyte.io.File, profile_name: str = "smoke") -> flyte.io.Dir:
    """Run GRPO; emit the LoRA adapter as a `policy-checkpoint` artifact."""
    import pandas as pd
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import TrainerCallback
    from trl import GRPOConfig, GRPOTrainer

    profile = get_profile(profile_name)
    df = pd.read_parquet(await dataset.download())
    train_df = df[df["split"] == "train"].reset_index(drop=True)

    train_ds = Dataset.from_list(
        [
            {
                "prompt": build_prompt(r.question, r.function_declaration or None),
                "tests": r.tests,
            }
            for r in train_df.itertuples()
        ]
    )

    wandb_enabled = bool(os.environ.get("WANDB_API_KEY"))
    if not wandb_enabled:
        os.environ["WANDB_MODE"] = "disabled"
    run_name = f"grpo-{profile.name}-{flyte.ctx().action.run_name if flyte.ctx() else 'local'}"

    reward_metrics: list[dict] = []
    reward_fn = make_reward_fn(reward_metrics)

    args = GRPOConfig(
        output_dir="/tmp/grpo-out",
        run_name=run_name,
        max_steps=profile.max_steps,
        per_device_train_batch_size=profile.num_generations,
        gradient_accumulation_steps=2,
        num_generations=profile.num_generations,
        max_completion_length=profile.max_completion_length,
        learning_rate=profile.learning_rate,
        beta=0.0,                    # no KL / reference model (memory + R1-Zero school)
        loss_type="dapo",            # token-level norm, long-CoT friendly
        scale_rewards="batch",
        temperature=1.0,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=1,
        save_strategy="no",
        model_init_kwargs={"torch_dtype": "bfloat16"},
        report_to=["wandb"] if wandb_enabled else [],
        use_vllm=profile.use_vllm,
        **(
            dict(vllm_mode="colocate", vllm_gpu_memory_utilization=0.25, vllm_enable_sleep_mode=True)
            if profile.use_vllm
            else {}
        ),
    )

    history: list[dict] = []

    class LiveReportCallback(TrainerCallback):
        """Streams loss/reward curves to the Union console every step."""

        def on_log(self, args_, state, control, logs=None, **kw):
            if not logs:
                return
            entry = {"step": state.global_step, **logs}
            if reward_metrics:
                entry.update(reward_metrics[-1])
            history.append(entry)
            try:
                body = reporting.stats_row(
                    {
                        "step": state.global_step,
                        "mean reward": f"{entry.get('reward/mean_total', float('nan')):.3f}",
                        "pass rate": f"{entry.get('reward/pass_rate', float('nan')):.2%}",
                        "loss": f"{entry.get('loss', float('nan')):.4f}",
                    }
                )
                body += reporting.line_chart(
                    {
                        "mean reward": [h.get("reward/mean_total", float("nan")) for h in history],
                        "pass rate": [h.get("reward/pass_rate", float("nan")) for h in history],
                    },
                    title=f"reward (max {MAX_REWARD})",
                )
                body += reporting.line_chart(
                    {"loss": [h.get("loss", float("nan")) for h in history]},
                    title="loss",
                )
                flyte.report.replace(reporting.page(f"GRPO training — {run_name}", body))
                flyte.report.flush()
            except Exception:
                pass  # reporting must never kill training

    trainer = GRPOTrainer(
        model=profile.base_model,
        reward_funcs=reward_fn,
        args=args,
        train_dataset=train_ds,
        peft_config=LoraConfig(
            r=profile.lora_r,
            lora_alpha=profile.lora_r * 2,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        ),
        callbacks=[LiveReportCallback()],
    )
    trainer.train()

    out_dir = "/tmp/policy-checkpoint"
    trainer.save_model(out_dir)  # LoRA adapter + config
    trainer.processing_class.save_pretrained(out_dir)
    manifest = {
        "base_model": profile.base_model,
        "profile": profile.name,
        "max_steps": profile.max_steps,
        "final_metrics": history[-1] if history else {},
        "reward_history": history,
        "wandb_enabled": wandb_enabled,
        "wandb_project": WANDB_PROJECT if wandb_enabled else None,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    d = await flyte.io.Dir.from_local(out_dir)
    return publish(
        d,
        ARTIFACT_CHECKPOINT,
        description=(
            f"LoRA adapter for {profile.base_model}, {profile.max_steps} GRPO steps, "
            f"final mean reward {history[-1].get('reward/mean_total', 'n/a') if history else 'n/a'}"
        ),
        kind="model",
    )
