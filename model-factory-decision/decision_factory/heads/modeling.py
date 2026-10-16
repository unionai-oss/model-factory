"""The two heads, one training loop, one inference wrapper.

    torso (LoRA r=16) ──► hidden states ──┬─► pointer head ──► P(option) for every option, 1 pass
                                          └─► LM head ───────► generate {"decision": ...} token by token

Both arms share everything except the head: the torso, the LoRA config, the
prompt, the data, the optimizer and the number of steps. So a difference on
the scorecard is a difference the head made.

The pointer head (Strands Decider's "pointer" design, simplified):

    q = W_q · LN(h[answer])            h: last-layer hidden states
    k_i = W_k · LN(h[end of option i])
    score_i = q · k_i / sqrt(rank)     softmax over options, cross-entropy

It scores an option by its *content at its position*, so the candidate set is
not baked into the weights: the same head can rank 3 options or 30, in any
order (training shuffles the order so it cannot learn slots).

This module imports torch at the top; only GPU tasks and the serving app
import it. The CPU-only selector lives in `evaluation/select.py` and never
does.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from typing import Callable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..harness.policy import DECISIONS
from ..harness.prompt import parse_decision
from ..harness.scoring import one_hot
from .encoding import (
    SFTExample,
    PointerExample,
    completion_ids,
    encode_pointer,
    encode_sft,
    prompt_ids,
)

HEAD_FILE = "pointer_head.pt"
MANIFEST = "manifest.json"


class PointerHead(nn.Module):
    def __init__(self, hidden: int, rank: int = 512):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.q = nn.Linear(hidden, rank)
        self.k = nn.Linear(hidden, rank)
        self.rank = rank

    def forward(self, h_answer: torch.Tensor, h_options: torch.Tensor) -> torch.Tensor:
        """[B, H], [B, K, H] -> scores [B, K]."""
        q = self.q(self.norm(h_answer.float()))
        k = self.k(self.norm(h_options.float()))
        return torch.einsum("br,bkr->bk", q, k) / math.sqrt(self.rank)


def n_params(module: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad or not trainable_only)


def device_and_dtype() -> tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    # bf16 matmuls are slow or unsupported on many CPUs; serving a sub-1B
    # model in fp32 is a couple of GB of RAM and faster.
    return "cpu", torch.float32


def load_tokenizer(name_or_dir: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name_or_dir)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def load_torso(hf_id: str, head: str, dtype=None, device=None):
    """The base model for a head: no LM head for ``pointer``."""
    from transformers import AutoModel, AutoModelForCausalLM

    device = device or device_and_dtype()[0]
    dtype = dtype or device_and_dtype()[1]
    cls = AutoModel if head == "pointer" else AutoModelForCausalLM
    return cls.from_pretrained(hf_id, dtype=dtype).to(device)


def hidden_size(model) -> int:
    cfg = model.config
    return getattr(cfg, "hidden_size", None) or cfg.get_text_config().hidden_size


# ── batching ────────────────────────────────────────────────────────────


def _pad(seqs: list[list[int]], pad: int) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), width), pad, dtype=torch.long)
    mask = torch.zeros((len(seqs), width), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        mask[i, : len(s)] = 1
    return ids, mask


def _last_hidden(model, ids, mask):
    out = model(input_ids=ids, attention_mask=mask, output_hidden_states=False)
    return out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]


def pointer_scores(model, head: PointerHead, batch: Sequence[PointerExample], pad: int, device) -> torch.Tensor:
    ids, mask = _pad([b.input_ids for b in batch], pad)
    ids, mask = ids.to(device), mask.to(device)
    h = _last_hidden(model, ids, mask)  # [B, T, H]
    rows = torch.arange(len(batch), device=device)
    h_ans = h[rows, torch.tensor([b.answer_pos for b in batch], device=device)]
    opt_pos = torch.tensor([b.option_pos for b in batch], device=device)  # [B, K]
    h_opt = h[rows.unsqueeze(1), opt_pos]  # [B, K, H]
    return head(h_ans, h_opt)


def _left_pad(seqs: list[list[int]], pad: int):
    """Left-padded ids, mask and explicit position ids, so every sequence ENDS
    at the same index. That lets the LM head run on the last few positions
    only (`logits_to_keep`) instead of materialising a [batch, seq, vocab]
    tensor for a 150k-token vocabulary to score a 6-token answer."""
    width = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), width), pad, dtype=torch.long)
    mask = torch.zeros((len(seqs), width), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, width - len(s):] = torch.tensor(s, dtype=torch.long)
        mask[i, width - len(s):] = 1
    pos = (mask.cumsum(-1) - 1).clamp(min=0)
    return ids, mask, pos


def sft_loss(model, batch: Sequence[SFTExample], pad: int, device) -> torch.Tensor:
    ids, mask, pos = _left_pad([b.input_ids for b in batch], pad)
    labels, _, _ = _left_pad([b.labels for b in batch], -100)
    k = max(len(b.input_ids) - b.prompt_len for b in batch)
    logits = model(input_ids=ids.to(device), attention_mask=mask.to(device), position_ids=pos.to(device),
                   logits_to_keep=k + 1).logits  # [B, k+1, V]
    targets = labels[:, -k:].to(device)
    return F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)).float(), targets.reshape(-1), ignore_index=-100)


def _length_batches(examples: list, batch_size: int, rng: random.Random) -> list[list]:
    """Batches of similar length (less padding), in random order."""
    order = sorted(range(len(examples)), key=lambda i: len(examples[i].input_ids) + rng.random())
    batches = [[examples[i] for i in order[j : j + batch_size]] for j in range(0, len(order), batch_size)]
    rng.shuffle(batches)
    return batches


# ── training ────────────────────────────────────────────────────────────


def train(
    head: str,
    hf_id: str,
    chats: list[list[dict]],
    options: list[list[str]],
    labels: list[str],
    out_dir: str,
    *,
    epochs: float = 1.0,
    batch_size: int = 8,
    grad_accum: int = 1,
    lr: float = 2e-4,
    head_lr: float = 1e-3,
    lora_r: int = 16,
    pointer_rank: int = 512,
    max_len: int = 1024,
    seed: int = 0,
    on_log: Callable[[dict], None] | None = None,
    model=None,
    tok=None,
) -> dict:
    """Fine-tune one torso x head and save it to ``out_dir``.

    ``model``/``tok`` can be injected (the local shape test passes a tiny
    random model); otherwise they are loaded from ``hf_id``.
    """
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(seed)
    rng = random.Random(seed)
    device = device_and_dtype()[0]
    tok = tok or load_tokenizer(hf_id)
    base = model if model is not None else load_torso(hf_id, head)
    base.to(device)
    pad = tok.pad_token_id

    if head == "pointer":
        examples = [encode_pointer(tok, c, o, y) for c, o, y in zip(chats, options, labels)]
    else:
        examples = [encode_sft(tok, c, y) for c, y in zip(chats, labels)]
    dropped = sum(1 for e in examples if len(e.input_ids) > max_len)
    examples = [e for e in examples if len(e.input_ids) <= max_len]

    lora = LoraConfig(
        r=lora_r,
        lora_alpha=2 * lora_r,
        lora_dropout=0.05,
        target_modules="all-linear",
        task_type="FEATURE_EXTRACTION" if head == "pointer" else "CAUSAL_LM",
    )
    model = get_peft_model(base, lora)
    pointer = PointerHead(hidden_size(base), pointer_rank).to(device) if head == "pointer" else None

    groups = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": lr}]
    if pointer is not None:
        groups.append({"params": list(pointer.parameters()), "lr": head_lr})
    opt = torch.optim.AdamW(groups, weight_decay=0.0)
    steps_per_epoch = math.ceil(len(examples) / batch_size / grad_accum)
    total = max(1, int(steps_per_epoch * epochs))
    warmup = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * max(0.0, (total - s) / max(1, total - warmup))
    )

    model.train()
    history: list[dict] = []
    step, micro, running, seen = 0, 0, 0.0, 0
    started = time.time()
    while step < total:
        for batch in _length_batches(examples, batch_size, rng):
            if head == "pointer":
                scores = pointer_scores(model, pointer, batch, pad, device)
                y = torch.tensor([b.label for b in batch], device=device)
                loss = F.cross_entropy(scores, y)
            else:
                loss = sft_loss(model, batch, pad, device)
            (loss / grad_accum).backward()
            running += loss.item()
            seen += 1
            micro += 1
            if micro % grad_accum:
                continue
            torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 10 == 0 or step == total:
                rec = {"step": step, "loss": running / seen, "lr": sched.get_last_lr()[0],
                       "epoch": step / steps_per_epoch, "elapsed_s": time.time() - started}
                history.append(rec)
                running, seen = 0.0, 0
                if on_log:
                    on_log(rec)
            if step >= total:
                break

    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    info = {
        "train_examples": len(examples),
        "dropped_over_max_len": dropped,
        "steps": total,
        "train_seconds": time.time() - started,
        "trainable_params": n_params(model, trainable_only=True) + (n_params(pointer) if pointer else 0),
        "head_params": n_params(pointer) if pointer else _lm_head_params(base),
        "history": history,
        "pointer_rank": pointer_rank if pointer else None,
        "hidden_size": hidden_size(base),
    }
    if pointer is not None:
        torch.save(pointer.state_dict(), os.path.join(out_dir, HEAD_FILE))
    return info


def _lm_head_params(model) -> int:
    emb = model.get_output_embeddings()
    return emb.weight.numel() if emb is not None else 0


# ── inference ───────────────────────────────────────────────────────────


class DecisionModel:
    """A trained torso x head, loaded for inference.

    ``decide(chats)`` returns, per prompt: the decision, a probability per
    option (in `DECISIONS` order), the tokens it read and wrote, and — for the
    generative head — the raw text and whether it parsed.
    """

    def __init__(self, model_dir: str | None = None, *, base_model: str | None = None,
                 head: str | None = None, model=None, tok=None, pointer: PointerHead | None = None):
        self.device, dtype = device_and_dtype()
        manifest = {}
        if model_dir and os.path.exists(os.path.join(model_dir, MANIFEST)):
            with open(os.path.join(model_dir, MANIFEST)) as f:
                manifest = json.load(f)
        self.manifest = manifest
        self.head = head or manifest.get("head", "sft")
        base_model = base_model or manifest.get("base_model")
        self.tok = tok or load_tokenizer(model_dir or base_model)
        if model is None:
            base = load_torso(base_model, self.head, dtype=dtype, device=self.device)
            if model_dir:
                from peft import PeftModel

                base = PeftModel.from_pretrained(base, model_dir).merge_and_unload()
            model = base
        self.model = model.to(self.device).eval()
        self.pointer = pointer
        if self.head == "pointer" and self.pointer is None:
            self.pointer = PointerHead(hidden_size(self.model), manifest.get("pointer_rank") or 512)
            self.pointer.load_state_dict(torch.load(os.path.join(model_dir, HEAD_FILE), map_location="cpu"))
        if self.pointer is not None:
            self.pointer = self.pointer.to(self.device).eval()

    @torch.no_grad()
    def pointer_probs(self, chats, options: Sequence[str] = DECISIONS, batch_size: int = 16) -> list[dict]:
        out = []
        for i in range(0, len(chats), batch_size):
            exs = [encode_pointer(self.tok, c, options) for c in chats[i : i + batch_size]]
            scores = pointer_scores(self.model, self.pointer, exs, self.tok.pad_token_id, self.device)
            probs = torch.softmax(scores.float(), dim=-1).cpu().tolist()
            for e, p in zip(exs, probs):
                by_opt = dict(zip(options, p))
                vec = [by_opt[d] for d in DECISIONS]
                out.append({"probs": vec, "decision": DECISIONS[max(range(3), key=vec.__getitem__)],
                            "input_tokens": len(e.input_ids), "output_tokens": 0, "parsed": True})
        return out

    @torch.no_grad()
    def generate(self, chats, max_new_tokens: int = 12, batch_size: int = 16) -> list[dict]:
        """Greedy decoding, then parse. Probabilities are one-hot: that is
        what a JSON answer gives a caller."""
        out = []
        pad = self.tok.pad_token_id
        for i in range(0, len(chats), batch_size):
            seqs = [prompt_ids(self.tok, c) for c in chats[i : i + batch_size]]
            width = max(len(s) for s in seqs)
            # left-pad for generation
            ids = torch.full((len(seqs), width), pad, dtype=torch.long)
            mask = torch.zeros_like(ids)
            for j, s in enumerate(seqs):
                ids[j, width - len(s):] = torch.tensor(s)
                mask[j, width - len(s):] = 1
            gen = self.model.generate(
                input_ids=ids.to(self.device), attention_mask=mask.to(self.device),
                max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=pad,
                eos_token_id=self.tok.eos_token_id,
            )[:, width:]
            for s, g in zip(seqs, gen.cpu().tolist()):
                n_out = 0
                for t in g:
                    n_out += 1
                    if t == self.tok.eos_token_id:
                        break
                text = self.tok.decode(g[:n_out], skip_special_tokens=True)
                d = parse_decision(text)
                out.append({"probs": one_hot(d), "decision": d or "ask_user", "raw": text,
                            "parsed": d is not None, "input_tokens": len(s), "output_tokens": n_out})
        return out

    @torch.no_grad()
    def option_logprobs(self, chats, options: Sequence[str] = DECISIONS, batch_size: int = 8) -> list[dict]:
        """P(option) from the LM head: the likelihood of each full completion
        ``{"decision": "<option>"}``, renormalised over the options.

        This is how you get honest probabilities out of a generative model —
        but it costs one sequence per option, and it needs the candidate set
        up front, which is to say it needs a pointer head's inputs anyway.
        """
        out = []
        pad = self.tok.pad_token_id
        comps = [completion_ids(self.tok, o, eos=False) for o in options]
        for i in range(0, len(chats), batch_size):
            prompts = [prompt_ids(self.tok, c) for c in chats[i : i + batch_size]]
            seqs = [p + c for p in prompts for c in comps]
            k = max(len(c) for c in comps)
            ids, mask, pos = _left_pad(seqs, pad)
            logits = self.model(input_ids=ids.to(self.device), attention_mask=mask.to(self.device),
                                position_ids=pos.to(self.device), logits_to_keep=k + 1).logits
            logp = torch.log_softmax(logits[:, :-1].float(), dim=-1)  # predicts the last k tokens
            totals = []
            for r, seq in enumerate(seqs):
                n = len(comps[r % len(comps)])
                tgt = ids[r, -n:].to(self.device)
                totals.append(logp[r, k - n :].gather(1, tgt.unsqueeze(1)).sum().item())
            for j, p in enumerate(prompts):
                sc = totals[j * len(options) : (j + 1) * len(options)]
                probs = torch.softmax(torch.tensor(sc), dim=0).tolist()
                by_opt = dict(zip(options, probs))
                vec = [by_opt[d] for d in DECISIONS]
                out.append({"probs": vec, "decision": DECISIONS[max(range(3), key=vec.__getitem__)],
                            "input_tokens": len(p), "output_tokens": 0, "parsed": True})
        return out

    def decide(self, chats) -> list[dict]:
        """The head's native readout: pointer probabilities, or generate+parse."""
        if self.head == "pointer":
            return self.pointer_probs(chats)
        return self.generate(chats)

    def time_one(self, chat, readout: str, repeats: int = 1) -> float:
        """Wall-clock ms for one decision at batch size 1."""
        fn = {"pointer": self.pointer_probs, "json": self.generate, "logprob": self.option_logprobs}[readout]
        if self.device == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(repeats):
            fn([chat])
        if self.device == "cuda":
            torch.cuda.synchronize()
        return (time.perf_counter() - t) * 1000 / repeats
