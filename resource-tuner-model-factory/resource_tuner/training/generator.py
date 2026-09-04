"""Reusable proposal generator: warm weights + dynamic batching.

Eval used to load the base model + adapter inside every eval task and
generate sequentially — a cold multi-minute load per eval and one prompt
at a time. This env fixes both:

- **ReusePolicy**: replicas stay warm between calls, so the checkpoint's
  weights load once per replica (cached by checkpoint path) instead of
  once per eval.
- **DynamicBatcher**: many concurrent `generate_proposal` calls (the eval
  fans out one per held-out context) are batched into single
  left-padded `model.generate` calls, maximizing GPU utilization
  (https://www.union.ai/docs/v2/union/user-guide/run-scaling/batch-inference/).

Gotchas honored: `submit()` is double-await (it returns a Future);
blocking generate runs via `asyncio.to_thread` so the replica's heartbeat
stays alive; reusable envs need `unionai-reuse` in the image and cannot
set a pod_template.
"""

from __future__ import annotations

import asyncio
import json

import flyte
import flyte.errors
import flyte.io

from ..config import cluster_env_vars, train_resources
from ..policy.prompts import render_messages
from ..shared.images import gpu_image, secrets

MAX_NEW_TOKENS = 128

generator_env = flyte.TaskEnvironment(
    name="rt-generator",
    image=gpu_image.with_pip_packages("unionai-reuse>=0.1.3"),
    resources=train_resources(),
    # Warm replicas between calls; scale back down after 5 idle minutes so
    # the T4 isn't parked forever.
    reusable=flyte.ReusePolicy(replicas=(1, 2), concurrency=8, idle_ttl=300, scaledown_ttl=120),
    secrets=secrets(),
    env_vars={**cluster_env_vars(), "TOKENIZERS_PARALLELISM": "false"},
)

# Replica-local state: engine + batcher per checkpoint path. A replica
# normally serves one checkpoint per eval; the dict handles transitions.
_engines: dict[str, tuple] = {}
_batchers: dict[str, object] = {}
_load_lock = asyncio.Lock()


@flyte.trace
async def _load_engine(checkpoint_path: str):
    """Download the adapter, load base + adapter once per replica."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    local = await flyte.io.Dir.from_existing_remote(checkpoint_path).download()
    with open(f"{local}/manifest.json") as f:
        base_model = json.load(f)["base_model"]
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    def load():
        tok = AutoTokenizer.from_pretrained(local, padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            dtype=torch.bfloat16 if bf16 else torch.float16,
            device_map="auto",
            attn_implementation="eager",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(model, local)
        model.eval()
        return model, tok

    return await asyncio.to_thread(load)


def _batched_generate(model, tok, prompts: list[str]) -> list[str]:
    """One left-padded batched generate; returns decoded completions."""
    import torch

    inputs = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=4096).to(
        model.device
    )
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
    return tok.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


async def _get_batcher(checkpoint_path: str):
    from flyte.extras import DynamicBatcher

    async with _load_lock:
        if checkpoint_path not in _engines:
            _engines[checkpoint_path] = await _load_engine(checkpoint_path)
        if checkpoint_path not in _batchers:
            model, tok = _engines[checkpoint_path]

            async def process(batch: list[str]) -> list[str]:
                import torch

                try:
                    return await asyncio.to_thread(_batched_generate, model, tok, batch)
                except torch.cuda.OutOfMemoryError:
                    # Explicit OOM handling: free the cache and retry the
                    # batch one prompt at a time rather than killing every
                    # future in it.
                    torch.cuda.empty_cache()
                    out = []
                    for prompt in batch:
                        try:
                            out.extend(
                                await asyncio.to_thread(_batched_generate, model, tok, [prompt])
                            )
                        except torch.cuda.OutOfMemoryError:
                            torch.cuda.empty_cache()
                            out.append("")  # caller treats empty as invalid
                    return out

            _batchers[checkpoint_path] = DynamicBatcher(
                process,
                cost_estimator=lambda p: max(len(p) // 4, 1),
                target_batch_cost=16_000,  # ~16 prompts × ~1k tokens
                max_batch_size=16,
                batch_timeout_s=0.25,
            )
    return _batchers[checkpoint_path]


@generator_env.task
async def generate_proposal(
    checkpoint_path: str, source_code: str, input_profile: str
) -> str:
    """One greedy proposal completion for one estimation context.

    Concurrent calls to warm replicas are transparently batched. Returns
    the raw completion text (parsing/validation stays with the caller so
    format failures remain visible to eval metrics).
    """
    batcher = await _get_batcher(checkpoint_path)
    _model, tok = _engines[checkpoint_path]
    messages = render_messages(source_code, input_profile)
    try:
        prompt = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:  # template without the thinking switch
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    fut = await batcher.submit(prompt)  # double-await: submit returns a Future
    return await fut
