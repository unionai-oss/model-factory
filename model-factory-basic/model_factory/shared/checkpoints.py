"""What a model directory holds, and how to load it.

Three shapes share one ``manifest.json`` convention (``contracts.CHECKPOINT_MANIFEST_KEYS``):

- ``weights: "base"``  — no weights at all; the model IS ``base_model`` (champion zero).
- ``weights: "lora"``  — a PEFT adapter on top of ``base_model`` (round-2 checkpoints).
- ``weights: "full"``  — merged full weights saved in the directory itself.

Since round 3 every day trains a FRESH LoRA on the champion's merged weights and
saves the merge, so a champion is always ``base`` or ``full`` and generation N
is never a stack of N adapters.

No flyte imports here: unit-testable pure Python.
"""

from __future__ import annotations

import json
import os


def read_manifest(path: str) -> dict:
    with open(os.path.join(path, "manifest.json")) as f:
        m = json.load(f)
    if "weights" not in m:  # round-2 manifests carried only `adapter`
        m["weights"] = "lora" if m.get("adapter", True) else "base"
    m.setdefault("generation", 0)
    return m


def weights_source(path: str, manifest: dict) -> tuple[str, str | None]:
    """``(what to from_pretrained, adapter dir or None)`` for a model directory."""
    kind = manifest["weights"]
    if kind == "base":
        return manifest["base_model"], None
    if kind == "lora":
        return manifest["base_model"], path
    if kind == "full":
        return path, None
    raise ValueError(f"unknown checkpoint weights kind {kind!r}")


def load_causal_lm(path: str, manifest: dict, device_map=None, trainable_adapter: bool = False):
    """Load the model a directory describes (bf16), applying an adapter if it has one."""
    import torch
    from transformers import AutoModelForCausalLM

    src, adapter = weights_source(path, manifest)
    model = AutoModelForCausalLM.from_pretrained(src, dtype=torch.bfloat16, device_map=device_map)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter, is_trainable=trainable_adapter)
    return model
