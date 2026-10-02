"""Training station: supervised fine-tuning of one suite member.

One task call = one model. The factory instantiates this build once per value
of the `model` partition, so the suite is trained by fanning out over a
partition dimension rather than by a loop inside a task — which means each
fine-tune is separately cached, separately retried, and separately visible.

The task returns a plain Dir; the factory declares and publishes it as
`decision-model[date, model]`.

Why SFT and not RL: the decision problem has an exact oracle label for every
observation, so there is a supervised target. GRPO (what the sibling
basic-model-factory uses) earns its complexity when the reward is only
computable by *running* something — sandboxed tests there. Here, policy
agreement is the label, and SFT on labels is both cheaper and a tighter fit.

Training is completion-only on purpose. The prompt (system + observation) is
long and the target (`{"tool": ..., "args": {...}}`) is short, so training on
the concatenation would spend most of the loss on tokens the model is never
asked to produce. TRL masks the prompt automatically for prompt-completion
datasets, which is why the dataset is built in that shape rather than as flat
text.
"""

from __future__ import annotations

import json
import os
from datetime import datetime

import flyte
import flyte.io
import flyte.report

from ..config import get_candidate, get_profile
from ..harness.policy import build_chat
from ..shared import reporting
from .envs import trainer_env


def build_sft_rows(df) -> list[dict]:
    """Episodes -> conversational prompt/completion rows for TRL.

    The prompt is built by the SAME `build_chat` the eval and the serving app
    use, so what the model is trained on and what it is asked at inference
    cannot drift.
    """
    rows = []
    for r in df.itertuples():
        obs = {
            "customer_message": r.obs_customer_message,
            "intent": r.obs_intent,
            "order_id": _clean(r.obs_order_id),
            "order_looked_up": bool(r.obs_order_looked_up),
            "order_status": _clean(r.obs_order_status),
            "order_total_cents": _clean_int(r.obs_order_total_cents),
            "days_since_delivery": _clean_int(r.obs_days_since_delivery),
            "tracking_number": _clean(r.obs_tracking_number),
        }
        rows.append(
            {
                "prompt": build_chat(obs),
                "completion": [{"role": "assistant", "content": r.label_json}],
            }
        )
    return rows


def _clean(value):
    """Parquet turns missing strings into NaN; the prompt must show None."""
    if value is None:
        return None
    if isinstance(value, float) and value != value:  # NaN
        return None
    return str(value) if str(value) else None


def _clean_int(value):
    if value is None:
        return None
    if isinstance(value, float) and value != value:
        return None
    return int(value)


@trainer_env.task(report=True, timeout=flyte.Timeout(max_runtime=5400))
async def finetune_decision_model(
    episodes: flyte.io.File,
    model: str,
    profile_name: str = "smoke",
    date: datetime | None = None,
) -> flyte.io.Dir:
    """Fine-tune one suite member on the day's episodes.

    ``model`` is the suite slug and also the `model` partition value — the
    factory passes it with ``factory.partition("model")``.
    """
    import pandas as pd
    from datasets import Dataset
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    profile = get_profile(profile_name)
    candidate = get_candidate(model)

    df = pd.read_parquet(await episodes.download())
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    if len(train_df) == 0:
        raise flyte.errors.NonRecoverableError("no train-split episodes in the dataset")

    train_ds = Dataset.from_list(build_sft_rows(train_df))

    out_dir = "/tmp/decision-model"
    args = SFTConfig(
        output_dir="/tmp/sft-out",
        num_train_epochs=profile.epochs,
        per_device_train_batch_size=profile.per_device_batch,
        gradient_accumulation_steps=profile.grad_accum,
        learning_rate=profile.learning_rate,
        max_length=profile.max_seq_length,
        logging_steps=5,
        save_strategy="no",
        bf16=True,
        report_to=[],
        # Prompt tokens are masked; see the module docstring.
        completion_only_loss=True,
    )

    history: list[dict] = []
    trainer = SFTTrainer(
        model=candidate.hf_id,
        args=args,
        train_dataset=train_ds,
        peft_config=LoraConfig(
            r=profile.lora_r,
            lora_alpha=profile.lora_r * 2,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        ),
    )

    # Stream the loss curve to the console. A suite run has one of these per
    # model, so the reports are how you see which fine-tune actually converged.
    from transformers import TrainerCallback

    class LiveReport(TrainerCallback):
        def on_log(self, args_, state, control, logs=None, **kw):
            if not logs:
                return
            history.append({"step": state.global_step, **logs})
            try:
                body = reporting.stats_row(
                    {
                        "model": candidate.slug,
                        "step": state.global_step,
                        "loss": f"{logs.get('loss', float('nan')):.4f}",
                        "epoch": f"{logs.get('epoch', 0):.2f}",
                    }
                )
                body += reporting.table(
                    ["step", "loss", "epoch", "lr"],
                    [
                        [
                            h.get("step"),
                            f"{h.get('loss', float('nan')):.4f}",
                            f"{h.get('epoch', 0):.2f}",
                            f"{h.get('learning_rate', 0):.2e}",
                        ]
                        for h in history[-25:]
                    ],
                )
                flyte.report.replace(reporting.page(f"SFT — {candidate.slug}", body))
                flyte.report.flush()
            except Exception:
                pass  # reporting must never kill training

    trainer.add_callback(LiveReport())
    trainer.train()

    trainer.save_model(out_dir)
    trainer.processing_class.save_pretrained(out_dir)

    manifest = {
        "model_slug": candidate.slug,
        "base_model": candidate.hf_id,
        "params_m": candidate.params_m,
        "profile": profile.name,
        "train_episodes": len(train_df),
        "epochs": profile.epochs,
        "lora_r": profile.lora_r,
        "date": date.strftime("%Y-%m-%d") if date else "",
        "final_loss": history[-1].get("loss") if history else None,
        "loss_history": history,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    body = reporting.stats_row(
        {
            "model": candidate.slug,
            "base": candidate.hf_id,
            "params": f"{candidate.params_m}M",
            "train episodes": len(train_df),
            "epochs": profile.epochs,
            "final loss": f"{manifest['final_loss']:.4f}" if manifest["final_loss"] else "n/a",
        }
    )
    body += "<h3>Loss</h3>" + reporting.table(
        ["step", "loss", "epoch"],
        [[h.get("step"), f"{h.get('loss', float('nan')):.4f}", f"{h.get('epoch', 0):.2f}"] for h in history],
    )
    await flyte.report.replace.aio(reporting.page(f"Fine-tuned {candidate.slug}", body))
    await flyte.report.flush.aio()

    return await flyte.io.Dir.from_local(out_dir)
