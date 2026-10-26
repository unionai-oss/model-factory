"""Model training team: GRPO with verifiable code-execution rewards.

Consumes `rl-tasks-dataset[D]` and `champion[D]`, produces
`policy-checkpoint[D]` — but neither the wiring nor the publishing lives here:
the factory declares this task as the `policy-checkpoint` build and publishes
its return value (see model_factory/factory.py).

Closed loop: training WARM-STARTS from the champion. The champion's weights
are loaded (merged, if it is a round-2 adapter), a FRESH LoRA is trained on top,
and the merge is saved as full weights, so generation N is one model directory,
never a stack of N adapters, and each day's optimizer starts clean. The checkpoint carries
`hard_tasks.parquet` — the tasks its rollouts kept failing — which, once the
checkpoint is crowned, seeds the next day's synthetic curriculum.

TRL GRPOTrainer + LoRA on a single 24GB GPU (A10G). Rollout completions are
scored by the reward stack in rewards.py — sandboxed test execution is the
reward source.

Observability:
- live flyte report: reward/loss curves stream to the Union console
- W&B run when WANDB_API_KEY is provisioned (else disabled mode)
- per-component reward means logged every step
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor

import flyte
import flyte.io
import flyte.report
import pandas as pd

from ..config import WANDB_PROJECT, get_profile
from ..contracts import HARD_TASKS_FILE
from ..shared import reporting
from ..shared.checkpoints import load_causal_lm, read_manifest
from ..shared.rewards import MAX_REWARD, build_prompt, score_completion
from .envs import trainer_env

_SCORE_WORKERS = 8


def _completion_text(completion) -> str:
    """TRL passes chat completions as [{'role','content'}] lists."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        return completion[-1].get("content", "")
    return ""


def make_reward_fn(metrics_sink: list[dict], task_sink: dict[str, list[int]] | None = None):
    """Build the TRL reward function; extra dataset columns arrive as kwargs.

    ``task_sink`` accumulates ``task_id -> [passes, rollouts]``: the record of
    what the policy could and could not solve, which becomes the next day's
    curriculum.
    """

    def reward_fn(prompts, completions, tests, task_id=None, **kwargs):
        texts = [_completion_text(c) for c in completions]
        with ThreadPoolExecutor(max_workers=_SCORE_WORKERS) as pool:
            breakdowns = list(pool.map(score_completion, texts, tests))
        if task_sink is not None and task_id is not None:
            for tid, b in zip(task_id, breakdowns):
                rec = task_sink.setdefault(tid, [0, 0])
                rec[0] += int(b.tests_passed)
                rec[1] += 1
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


def hard_tasks(train_df: pd.DataFrame, task_stats: dict[str, list[int]], threshold: float = 0.5) -> pd.DataFrame:
    """Train tasks whose rollout pass rate stayed at or below ``threshold``, hardest first."""
    rate = {t: p / n for t, (p, n) in task_stats.items() if n}
    df = train_df[train_df["task_id"].map(lambda t: t in rate and rate[t] <= threshold)].copy()
    df["rollout_pass_rate"] = df["task_id"].map(rate)
    return df.sort_values("rollout_pass_rate", kind="stable").reset_index(drop=True)


@trainer_env.task(report=True, timeout=flyte.Timeout(max_runtime=7200))
async def train_grpo(
    dataset: flyte.io.File, champion: flyte.io.Dir, profile_name: str = "daily"
) -> flyte.io.Dir:
    """GRPO from the champion's weights; return the merged full-weight model directory.

    Returns a plain Dir; the factory build publishes it as `policy-checkpoint`.
    """
    from datasets import Dataset
    from peft import LoraConfig, PeftModel
    from transformers import TrainerCallback
    from trl import GRPOConfig, GRPOTrainer

    profile = get_profile(profile_name)
    df = pd.read_parquet(await dataset.download())
    train_df = df[df["split"] == "train"].reset_index(drop=True)

    champ_dir = await champion.download()
    champ = read_manifest(champ_dir)
    if champ["base_model"] != profile.base_model:
        raise flyte.errors.NonRecoverableError(
            f"champion is a {champ['base_model']} model but profile {profile.name} trains {profile.base_model}"
        )

    train_ds = Dataset.from_list(
        [
            {
                "prompt": build_prompt(r.question, r.function_declaration or None),
                "tests": r.tests,
                "task_id": r.task_id,
            }
            for r in train_df.itertuples()
        ]
    )

    wandb_enabled = bool(os.environ.get("WANDB_API_KEY"))
    if not wandb_enabled:
        os.environ["WANDB_MODE"] = "disabled"
    os.environ.setdefault("WANDB_PROJECT", WANDB_PROJECT)
    run_name = f"grpo-{profile.name}-{flyte.ctx().action.run_name if flyte.ctx() else 'local'}"

    reward_metrics: list[dict] = []
    task_stats: dict[str, list[int]] = {}
    reward_fn = make_reward_fn(reward_metrics, task_stats)

    args = GRPOConfig(
        output_dir="/tmp/grpo-out",
        run_name=run_name,
        max_steps=profile.max_steps,
        per_device_train_batch_size=profile.per_device_batch,
        gradient_accumulation_steps=profile.grad_accum,
        num_generations=profile.num_generations,
        max_completion_length=profile.max_completion_length,
        learning_rate=profile.learning_rate,
        beta=0.0,                    # no KL / reference model (memory + R1-Zero school)
        loss_type="dapo",            # token-level norm, long-CoT friendly
        scale_rewards="batch",
        temperature=1.0,
        bf16=True,
        gradient_checkpointing=True,
        # Non-reentrant checkpointing also works on a pre-wrapped PeftModel.
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1,
        save_strategy="no",
        report_to=["wandb"] if wandb_enabled else [],
        # Rollouts on a colocated vLLM engine sharing the training GPU: the
        # rollout phase is most of a GRPO step, and HF generate is the slow part.
        use_vllm=profile.use_vllm,
        **(
            dict(vllm_mode="colocate", vllm_gpu_memory_utilization=profile.vllm_gpu_memory_utilization)
            if profile.use_vllm
            else {}
        ),
    )

    # Warm start: the champion's weights, with a fresh adapter on top.
    model = load_causal_lm(champ_dir, champ)
    if isinstance(model, PeftModel):  # a round-2 adapter champion: fold it in first
        model = model.merge_and_unload()
    print(
        f"warm start from the generation-{champ['generation']} champion ({champ['weights']} weights); fresh LoRA"
        if champ["generation"]
        else "cold start: the champion is the base model; fresh LoRA"
    )
    peft_config = LoraConfig(
        r=profile.lora_r,
        lora_alpha=profile.lora_r * 2,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
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
                        "starts from": f"gen {champ['generation']}",
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
        model=model,
        reward_funcs=reward_fn,
        args=args,
        train_dataset=train_ds,
        peft_config=peft_config,
        callbacks=[LiveReportCallback()],
    )
    trainer.train()

    out_dir = "/tmp/policy-checkpoint"
    trained = trainer.model
    if isinstance(trained, PeftModel):
        trained = trained.merge_and_unload()
    trained.save_pretrained(out_dir, safe_serialization=True)  # full merged weights
    trainer.processing_class.save_pretrained(out_dir)
    hard = hard_tasks(train_df, task_stats)
    hard.to_parquet(os.path.join(out_dir, HARD_TASKS_FILE), index=False)
    manifest = {
        "base_model": profile.base_model,
        "weights": "full",
        "generation": champ["generation"] + 1,
        "warm_start": champ["generation"] > 0,
        "use_vllm": profile.use_vllm,
        "learning_rate": profile.learning_rate,
        "profile": profile.name,
        "max_steps": profile.max_steps,
        "train_tasks": len(train_df),
        "tasks_seen": len(task_stats),
        "tasks_always_solved": sum(1 for p, n in task_stats.values() if n and p == n),
        "hard_tasks": len(hard),
        "final_metrics": history[-1] if history else {},
        "reward_history": history,
        "wandb_enabled": wandb_enabled,
        "wandb_project": WANDB_PROJECT if wandb_enabled else None,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    return await flyte.io.Dir.from_local(out_dir)
