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

from .. import pricing
from ..config import TunerProfile, WANDB_PROJECT, get_profile
from ..contracts import (
    ARTIFACT_TASK_CORPUS,
    ARTIFACT_TUNER_CHECKPOINT,
    ARTIFACT_TUNER_CHECKPOINT_INTERMEDIATE,
    publish,
)
from ..shared import assets
from ..environment.simulator import simulate_episode
from ..policy.parsing import try_extract_proposal
from ..policy.prompts import parse_context_fields, render_messages
from ..rewards import shaping
from ..rewards.rewards import invalid_proposal_reward, score_episode
from ..shared.reporting import GOOD, Reporter, esc, line_chart, pill
from .baseline import baseline_proposal, fit_family_baseline
from .envs import ckpt_publisher_env, trainer_env


def make_reward_fn(stage: str, jitter_rng=None, num_generations: int = 0, max_steps: int = 0):
    """completions + per-sample truth columns → list of scalar rewards.

    TRL forwards non-reserved dataset columns to the reward function as
    kwargs of per-sample lists. `jitter_rng` enables the simulator's
    threshold randomization during training (see simulator.JITTER_BAND);
    tests omit it for determinism.

    Stage A/B ("success"/"composite") keep their historical behavior.
    Shaped stages (shaping.SHAPES) add: GPU truth, robustness-averaged
    episodes, annealed waste weight (call count ≈ optimizer step since
    generation_batch == one group batch per step), and the in-group
    cheapest-survivor tie-break.
    """
    shaped = shaping.is_shaped_stage(stage)
    shape = shaping.get_shape(stage) if shaped else None
    calls = {"n": 0}

    def resource_reward(
        completions, true_peak_memory_mib, true_cpu_cores, duration_s, **kwargs
    ) -> list[float]:
        n = len(completions)
        gpu_col = kwargs.get("true_gpu_mem_mib") or [0.0] * n
        base_cost_col = kwargs.get("baseline_cost_per_hr") or [None] * n
        calls["n"] += 1
        step_frac = calls["n"] / max_steps if max_steps else 1.0

        rewards: list[float] = []
        oks: list[bool] = []
        costs: list[float | None] = []
        for completion, peak, cpu, dur, gpu_mem, base_cost in zip(
            completions, true_peak_memory_mib, true_cpu_cores, duration_s,
            gpu_col, base_cost_col,
        ):
            text = completion[0]["content"] if isinstance(completion, list) else completion
            proposal = try_extract_proposal(text)
            if proposal is None:
                rewards.append(invalid_proposal_reward().total)
                oks.append(False)
                costs.append(None)
                continue
            if not shaped:
                episode = simulate_episode(
                    proposal, float(peak), float(cpu), int(dur), rng=jitter_rng
                )
                rewards.append(score_episode(stage, episode).total)
                oks.append(episode.ok)
                costs.append(None)
                continue
            episodes = [
                simulate_episode(
                    proposal, float(peak), float(cpu), int(dur),
                    rng=jitter_rng, true_gpu_mem_mib=float(gpu_mem or 0.0),
                )
                for _ in range(max(shape.robustness_samples, 1))
            ]
            bd = shaping.score_shaped(
                shape, episodes, step_frac=step_frac,
                baseline_cost_per_hr=base_cost,
            )
            rewards.append(bd.total)
            oks.append(all(e.ok for e in episodes))
            costs.append(shaping.episode_dollars_per_hr(episodes[0]))
        if shaped:
            rewards = shaping.apply_group_tiebreak(
                shape, rewards, oks, costs, num_generations
            )
        return rewards

    return resource_reward


def _record_to_row(r: dict, baselines: dict | None = None) -> dict:
    base_cost = None
    if baselines:
        bp = baseline_proposal(baselines, r["family"])
        base_cost = pricing.dollars_per_hr(bp.cpu, bp.memory_mib, bp.gpu_type, bp.gpu)
    prior, history = parse_context_fields(r.get("prior_json"), r.get("history_json"))
    return {
        "prompt": render_messages(
            r["source_code"], r["input_profile"], prior=prior, history=history
        ),
        "true_peak_memory_mib": r["true_peak_memory_mib"],
        "true_cpu_cores": r["true_cpu_cores"],
        "duration_s": r["duration_s"],
        # Old corpora predate the GPU column — default to CPU task.
        "true_gpu_mem_mib": float(r.get("true_gpu_mem_mib", 0.0) or 0.0),
        "baseline_cost_per_hr": base_cost,
    }


def _records_to_dataset(records: list[dict], baselines: dict | None = None):
    from datasets import Dataset

    return Dataset.from_list([_record_to_row(r, baselines) for r in records])


# Deliberately NOT @flyte.trace'd: traced functions serialize inputs AND
# outputs as literals, and this returns the model object — the span costs
# a multi-GB pickle upload. Load timing is visible in the report instead.
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
async def _upload_checkpoint(out_dir: str, manifest: dict) -> "flyte.io.Dir":
    """Write the manifest and upload the checkpoint dir (traced — the span
    marks the write/upload). Traced functions serialize their INPUTS as
    literals, so only plain data crosses this boundary: passing the
    trainer object here pickled the accelerate-wrapped model and died with
    PicklingError (hit for real on run ullwh6kd4s727k5jvm59).
    """
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return await flyte.io.Dir.from_local(out_dir)


def find_trl_checkpoint(root: str) -> str | None:
    """Locate a resumable TRL checkpoint under a restored tree.

    flyte.Checkpoint tarballs a saved directory; depending on layout the
    restore lands either as the checkpoint dir's CONTENTS (trainer_state
    at root) or as `checkpoint-<N>/` subdirs. Handles both; picks the
    highest step. Pure function — unit-tested."""
    import pathlib

    p = pathlib.Path(root)
    if (p / "trainer_state.json").exists():
        return str(p)
    cands = []
    for d in p.glob("**/checkpoint-*"):
        if d.is_dir() and (d / "trainer_state.json").exists():
            try:
                cands.append((int(d.name.rsplit("-", 1)[-1]), d))
            except ValueError:
                continue
    return str(max(cands)[1]) if cands else None


@ckpt_publisher_env.task(produces_artifacts=True)
async def publish_intermediate_checkpoint(
    ckpt: flyte.io.Dir, step: int, profile_name: str, reward_stage: str
) -> flyte.io.Dir:
    """Version a mid-training adapter as a Union artifact.

    A separate task on purpose: publish() versions an artifact only when
    the wrapped value is RETURNED from a task, and a distinct artifact
    name (tuner-checkpoint-intermediate) keeps the eval-on-new-checkpoint
    trigger quiet until the FINAL checkpoint lands."""
    return publish(
        ckpt,
        ARTIFACT_TUNER_CHECKPOINT_INTERMEDIATE,
        description=f"step {step} — {profile_name}/{reward_stage} (intermediate)",
        kind="model",
    )


async def _resolve_resume(
    resume_from: "flyte.io.Dir | None", resume_from_artifact: str
) -> tuple["flyte.io.Dir | None", str]:
    """(Dir to warm-start from, human-readable source). The imperative
    path resolves the newest version of a named artifact via
    flyte.remote.Artifact (assets: always the s3 blob URI, never the
    console URL)."""
    if resume_from is not None:
        return resume_from, f"dir:{getattr(resume_from, 'path', '')}"
    if resume_from_artifact:
        ver = await assets.latest_version(resume_from_artifact)
        if ver is None:
            raise RuntimeError(
                f"resume_from_artifact={resume_from_artifact!r} has no "
                "blob-resolvable versions — train once with "
                "artifact_checkpoint_every > 0 first"
            )
        return flyte.io.Dir.from_existing_remote(ver.path), (
            f"artifact:{resume_from_artifact}@{ver.path[-32:]}"
        )
    return None, ""


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
    # Long-run posture: 3-day ceiling, and retries>0 so a lost pod
    # RESUMES from the intra-task checkpoint instead of restarting.
    timeout=flyte.Timeout(max_runtime=72 * 3600, max_queued_time=6 * 3600),
    retries=3,
    produces_artifacts=True,
    report=True,
)
async def train_tuner(
    corpus: flyte.io.File,
    profile_name: str = "smoke",
    resume_from: flyte.io.Dir | None = None,
    resume_from_artifact: str = "",
    fail_at_step: int = 0,
) -> flyte.io.Dir:
    """Train the policy on the corpus's train split; emit tuner-checkpoint.

    Checkpointing layers (all optional, driven by the profile):
    - intra-task (profile.save_steps): TRL writes full trainer state every
      N steps and flyte's Checkpoint uploads it; a retried attempt resumes
      at the last saved step (optimizer/scheduler/global_step intact).
    - artifact checkpoints (profile.artifact_checkpoint_every): the
      adapter is published mid-run as tuner-checkpoint-intermediate via a
      child task — Union-native lineage + a warm-start input for later runs.
    - warm start (resume_from / resume_from_artifact): initialize the LoRA
      adapter from a previous (intermediate or final) checkpoint Dir; the
      artifact form resolves the newest version imperatively.

    `fail_at_step` is a chaos hook for testing the intra-task path: the
    FIRST attempt raises after that step; retries then prove the resume.
    """
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
    train_df = df[df["split"] == "train"]
    if len(train_df) > profile.train_contexts:
        # Random (fixed-seed) subset, not a head slice: merged corpora are
        # ordered template-families-then-archetypes, and a head slice
        # would silently drop whole sources (e.g. every archetype row).
        train_df = train_df.sample(n=profile.train_contexts, random_state=0)
    records = train_df.to_dict("records")
    # Family baselines priced per record: the baseline_relative shapes score
    # "cheaper than the rule baseline" directly in the reward.
    baselines = fit_family_baseline(records) if records else {}
    dataset = _records_to_dataset(records, baselines=baselines)
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

    # Warm start: initialize the adapter from a previous (intermediate or
    # final) checkpoint instead of fresh LoRA init.
    peft_config = lora
    resume_dir, resume_source = await _resolve_resume(resume_from, resume_from_artifact)
    if resume_dir is not None:
        from peft import PeftModel

        local_resume = await resume_dir.download()
        model = PeftModel.from_pretrained(model, local_resume, is_trainable=True)
        peft_config = None  # already wrapped — GRPOTrainer must not re-wrap
        meta["resume"] = resume_source
        await flyte.report.replace.aio(_report_html(profile, [], meta), do_flush=True)

    # Intra-task checkpoint: if a previous ATTEMPT of this action saved
    # trainer state, restore it and resume mid-run (optimizer/scheduler/
    # global_step intact) instead of restarting from step 0.
    try:
        cp = flyte.ctx().checkpoint
    except Exception:  # noqa: BLE001 — local runs have no checkpoint prefix
        cp = None
    intra_resume = None
    if profile.save_steps and cp is not None and cp.prev_exists():
        restored = await cp.load()
        if restored is not None:
            intra_resume = find_trl_checkpoint(str(restored))
            if intra_resume:
                meta["intra_task_resume"] = intra_resume.rsplit("/", 1)[-1]
                print(f"[ckpt] resuming attempt from intra-task checkpoint {intra_resume}")

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
    if profile.save_steps:
        # Full trainer state on disk every N steps; the checkpoint callback
        # ships the newest one to the flyte Checkpoint prefix.
        extra_cfg.update(save_steps=profile.save_steps, save_total_limit=2)

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
        save_strategy="steps" if profile.save_steps else "no",
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

    pending_publishes: list = []

    class _CheckpointCallback(TrainerCallback):
        """Fires on every TRL save (profile.save_steps cadence): uploads
        the trainer-state dir to the flyte Checkpoint (intra-task resume),
        and every artifact_checkpoint_every steps snapshots the adapter and
        schedules publish_intermediate_checkpoint (a child task) on the
        main event loop."""

        def on_save(self, args, state, control, **cb_kwargs):
            step = state.global_step
            latest = find_trl_checkpoint(out_dir)
            if cp is not None and latest:
                try:
                    # Blocks the trainer thread for the upload — adapters
                    # are tens of MB, seconds at most.
                    asyncio.run_coroutine_threadsafe(cp.save(latest), loop).result(timeout=900)
                    print(f"[ckpt] intra-task checkpoint uploaded at step {step}")
                except Exception as e:  # noqa: BLE001 — a failed save must not kill training
                    print(f"[ckpt] intra-task save failed at step {step}: {e}")
            # Chaos hook: first attempt dies after this step; the retry
            # must resume from the checkpoint just uploaded.
            if (
                fail_at_step
                and profile.save_steps
                and step >= fail_at_step
                and not (cp is not None and cp.prev_exists())
            ):
                raise RuntimeError(
                    f"[chaos] injected failure at step {step} "
                    f"(fail_at_step={fail_at_step}) — the retried attempt "
                    "should resume from the intra-task checkpoint"
                )

        # Artifact cadence lives on on_step_end, NOT on_save: on_save only
        # fires every save_steps, so gating publishes there silently reduces
        # the cadence to the LCM of the two knobs (found by the round-9
        # chaos smoke: save 4 / publish 5 → zero intermediates published).
        def on_step_end(self, args, state, control, **cb_kwargs):
            step = state.global_step
            every = profile.artifact_checkpoint_every
            if every and step and step % every == 0 and step < profile.max_steps:
                snap = tempfile.mkdtemp(prefix=f"tuner-inter-{step}-")
                mdl = cb_kwargs.get("model") or trainer.model
                mdl.save_pretrained(snap)
                tok.save_pretrained(snap)
                with open(os.path.join(snap, "manifest.json"), "w") as f:
                    json.dump(
                        {
                            "base_model": profile.base_model,
                            "profile": profile.name,
                            "reward_stage": profile.reward_stage,
                            "step": step,
                            "intermediate": True,
                        },
                        f,
                    )

                async def _publish(snap=snap, step=step):
                    d = await flyte.io.Dir.from_local(snap)
                    return await publish_intermediate_checkpoint(
                        ckpt=d,
                        step=step,
                        profile_name=profile.name,
                        reward_stage=profile.reward_stage,
                    )

                pending_publishes.append(asyncio.run_coroutine_threadsafe(_publish(), loop))
                print(f"[ckpt] intermediate artifact publish scheduled at step {step}")

    trainer = GRPOTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        reward_funcs=make_reward_fn(
            profile.reward_stage,
            jitter_rng=random.Random(0),
            num_generations=profile.num_generations,
            max_steps=profile.max_steps,
        ),
        peft_config=peft_config,
        processing_class=tok,
        callbacks=[_ReportCallback(), _CheckpointCallback()],
    )
    # The trainer is blocking for the whole run; keep the event loop alive.
    try:
        await asyncio.to_thread(trainer.train, resume_from_checkpoint=intra_resume)
    except torch.cuda.OutOfMemoryError as e:
        # Explicit OOM handling: fail with a pointed message instead of a
        # bare CUDA traceback (the fix is a knob, not a retry).
        raise RuntimeError(
            f"GPU OOM during GRPO training on {meta.get('device')}: {e}. "
            f"Reduce per_device_batch/num_generations/max_completion_length "
            f"or set use_qlora for {profile.base_model}."
        ) from e

    # Drain intermediate publishes before the final upload — every
    # scheduled artifact either lands or is reported, never silently lost.
    for fut in pending_publishes:
        try:
            await asyncio.wrap_future(fut)
        except Exception as e:  # noqa: BLE001 — an intermediate is best-effort
            print(f"[ckpt] intermediate publish failed: {e}")

    history = [
        {k: v for k, v in row.items() if isinstance(v, (int, float))}
        for row in trainer.state.log_history
    ]
    mean_rewards = [h["reward"] for h in history if "reward" in h]
    # save_pretrained holds unpicklable objects — keep it OUTSIDE the
    # traced boundary; only paths/dicts cross into the traced upload.
    trainer.model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    manifest = {
        "base_model": profile.base_model,
        "profile": profile.name,
        "reward_stage": profile.reward_stage,
        # Full shape config for shaped stages — eval and the dashboard show
        # WHICH reward produced this checkpoint, not just a stage name.
        "reward_shape": (
            dataclasses.asdict(shaping.get_shape(profile.reward_stage))
            if shaping.is_shaped_stage(profile.reward_stage)
            else None
        ),
        "max_steps": profile.max_steps,
        "resume": meta.get("resume", ""),
        "intra_task_resume": meta.get("intra_task_resume", ""),
        "save_steps": profile.save_steps,
        "artifact_checkpoint_every": profile.artifact_checkpoint_every,
        "final_metrics": {
            "mean_reward_first": mean_rewards[0] if mean_rewards else None,
            "mean_reward_last": mean_rewards[-1] if mean_rewards else None,
            "log_history": history,
        },
    }
    ckpt = await _upload_checkpoint(out_dir, manifest)
    return publish(
        ckpt,
        ARTIFACT_TUNER_CHECKPOINT,
        description=f"{profile.base_model} LoRA, stage={profile.reward_stage}, "
        f"steps={profile.max_steps}",
        kind="model",
    )
