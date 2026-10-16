"""LLM-backed arms: zero-shot, autoregressive SFT, and a Jev-style decision head.

The controlled comparison: **same torso, same LoRA, same prompt, same
data, same steps** -- only the output head differs.

- ``sft``  (autoregressive) the LM head writes ``{"memory": "3Gi", "cpu": 1}``
  token by token. Decoded two ways: greedy plus a margin of k cells (k tuned
  on val), and *sampled* -- N completions read as an empirical distribution
  and decoded by expected cost. Sampling is how a generative model gets a
  distribution at all, and it costs N decodes.
- ``jev``  (decision model) the LM head is dropped. A small head over the
  final prompt position's hidden state scores every memory cell and every
  CPU cell in ONE forward pass -- the shape TypeSafe's Jev and Strands
  Decider made popular: a typed choice from a fixed candidate set, with a
  probability for each candidate, no tokens generated. Temperature-scaled
  on val, then decoded by expected cost (``scoring.bayes_mem_index``),
  which is what a calibrated distribution buys you. ``jev-argmax`` decodes
  the same model by its single most likely cell, to show how much of the
  win is the decision rule.
- ``zeroshot`` a large hosted model, prompted, no training: "just ask an
  LLM". It carries the per-decision cost and latency of a frontier call.

This module imports torch lazily inside functions: the CPU stations import
it for `zeroshot` and must not need a GPU stack.
"""

from __future__ import annotations

import json
import math
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass

import numpy as np

# Module top, not inside zeroshot_predict: the code bundler only ships
# statically-imported modules (pilot-board-3: "cannot import name 'llm_client'").
from ..shared import llm_client
from . import features, grid, prompting, scoring

N_MEM, N_CPU = scoring.N_MEM, scoring.N_CPU


@dataclass
class TrainConfig:
    model_name: str = "Qwen/Qwen3-1.7B"
    max_len: int = 2048
    epochs: int = 3
    batch_size: int = 8
    grad_accum: int = 2
    lr_lora: float = 2e-4
    lr_head: float = 1e-3
    lora_r: int = 16
    lora_alpha: int = 32
    warmup_frac: float = 0.05
    cpu_loss_weight: float = 0.3
    n_samples: int = 8
    head: str = "classify"  # decision head: "classify", "pointer" or "classify+num"
    max_train_rows: int = 0  # 0 = all
    seed: int = 0


def _device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"  # local smoke runs


def _amp_dtype():
    """bf16 where the hardware has it; fp16 (+ a GradScaler) on T4-class GPUs."""
    import torch

    # Capability, not is_bf16_supported(): that counts *emulation* and says
    # True on a T4 (sm_75), where bf16 then runs ~15x slower than fp16
    # (rsb-v1-board-2: 20 optimizer steps in 35 minutes).
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8:
        return torch.float16
    return torch.bfloat16


def _autocast():
    import torch

    return torch.autocast(_device(), dtype=_amp_dtype())


def _scaler():
    import torch

    return torch.amp.GradScaler("cuda", enabled=_amp_dtype() == torch.float16)


def _sync() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


# ── tokenization ────────────────────────────────────────────────────────


def _prompt_ids(tok, row: dict, max_len: int) -> list[int]:
    text = tok.apply_chat_template(prompting.messages(row), tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    return ids[-max_len:]  # keep the end: the options and the answer slot


def _batches(items: list, size: int, key=None, shuffle_rng=None):
    order = sorted(range(len(items)), key=key) if key else list(range(len(items)))
    chunks = [order[i : i + size] for i in range(0, len(order), size)]
    if shuffle_rng is not None:
        shuffle_rng.shuffle(chunks)
    for c in chunks:
        yield [items[i] for i in c]


def _pad(seqs: list[list[int]], pad_id: int, left: bool = False):
    import torch

    n = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), n), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), n), dtype=torch.long)
    for i, s in enumerate(seqs):
        if left:
            ids[i, n - len(s) :], mask[i, n - len(s) :] = torch.tensor(s), 1
        else:
            ids[i, : len(s)], mask[i, : len(s)] = torch.tensor(s), 1
    return ids, mask


def _schedule(opt, total_steps: int, warmup_frac: float):
    import torch

    warm = max(1, int(total_steps * warmup_frac))

    def f(step):
        if step < warm:
            return (step + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total_steps - warm)))

    return torch.optim.lr_scheduler.LambdaLR(opt, f)


def _load(cfg: TrainConfig, causal: bool):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    cls = AutoModelForCausalLM if causal else AutoModel
    base = cls.from_pretrained(cfg.model_name, dtype=_amp_dtype(), attn_implementation="sdpa").to(_device())
    base.config.use_cache = False
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    base.enable_input_require_grads()
    lora = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=0.05,
                      target_modules=LORA_TARGETS, task_type="CAUSAL_LM" if causal else "FEATURE_EXTRACTION")
    return tok, get_peft_model(base, lora)


# ── the decision head ───────────────────────────────────────────────────
# Two head shapes over the same torso:
#
#   classify  LayerNorm + a linear readout per decision over the hidden state
#             at the answer slot (the prompt's last token): the grid is baked
#             into the head's output size.
#   pointer   Strands-Decider style: score the hidden state at EACH OPTION'S
#             OWN TOKEN in the prompt ("... 1.5Gi, 2Gi, 3Gi ...") against the
#             answer slot, q(answer) . k(option). The options are read, not
#             memorized, so the same head can rank a different candidate set.
#
# The prompt's suffix (options, instructions, chat template) is identical for
# every row and prompts are truncated from the left, so each option token sits
# at a FIXED offset from the end: `option_offsets` finds them once.


def option_offsets(tok) -> tuple[list[int], list[int]]:
    """Offsets from the last prompt token to each memory / cpu option's last token."""
    probe = {"task_code": "x = 1", "inputs_desc": "x: npy, 1 KiB, shape=(1,), dtype=uint8",
             "history_json": "[]"}
    text = tok.apply_chat_template(prompting.messages(probe), tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    spans = enc["offset_mapping"]
    n = len(enc["input_ids"])
    base = text.rindex("### Options\n")

    def last_token_of(char_end: int) -> int:
        for i, (a, b) in enumerate(spans):
            if a < char_end <= b:
                return i
        raise ValueError(f"no token covers char {char_end}")

    def offsets(line_prefix: str, options: list[str]) -> list[int]:
        pos = text.index(line_prefix, base) + len(line_prefix)
        out = []
        for opt in options:
            pos = text.index(opt, pos) + len(opt)
            out.append(n - 1 - last_token_of(pos))
        return out

    mem = offsets("memory: ", [grid.format_memory(m) for m in grid.MEMORY_GRID_MIB])
    cpu = offsets("cpu: ", [f"{c:g}" for c in grid.CPU_GRID])
    return mem, cpu


def _head_module(hidden: int, kind: str = "classify", offsets=None, num_stats=None):
    import torch

    class Classify(torch.nn.Module):
        """LayerNorm + two linear readouts over the answer-slot hidden state."""

        def __init__(self):
            super().__init__()
            self.norm = torch.nn.LayerNorm(hidden)
            # not `self.cpu`: that name is nn.Module.cpu()
            self.mem_head = torch.nn.Linear(hidden, N_MEM)
            self.cpu_head = torch.nn.Linear(hidden, N_CPU)

        def forward(self, H, last, num=None):
            h = self.norm(H[torch.arange(H.size(0), device=H.device), last].float())
            return self.mem_head(h), self.cpu_head(h)

    class NumClassify(torch.nn.Module):
        """The answer-slot state AND the scheduler-visible numeric features.

        The LM reads the code; the numbers (log input bytes, rows, history
        peaks -- exactly what the GBT arm sees) enter as numbers instead of
        as digit tokens a small LM has to parse. Standardized with the
        training split's statistics, which travel with the head.
        """

        def __init__(self, mean, std):
            super().__init__()
            self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
            self.register_buffer("std", torch.tensor(std, dtype=torch.float32))
            self.norm = torch.nn.LayerNorm(hidden)
            self.num = torch.nn.Sequential(torch.nn.Linear(len(mean), 128), torch.nn.GELU())
            self.trunk = torch.nn.Sequential(torch.nn.Linear(hidden + 128, 512), torch.nn.GELU())
            self.mem_head = torch.nn.Linear(512, N_MEM)
            self.cpu_head = torch.nn.Linear(512, N_CPU)

        def forward(self, H, last, num=None):
            h = self.norm(H[torch.arange(H.size(0), device=H.device), last].float())
            x = self.num((num.float() - self.mean) / self.std)
            z = self.trunk(torch.cat([h, x], dim=-1))
            return self.mem_head(z), self.cpu_head(z)

    class Pointer(torch.nn.Module):
        """q(answer slot) . k(option token), per decision."""

        def __init__(self, mem_off, cpu_off, d: int = 256):
            super().__init__()
            self.register_buffer("mem_off", torch.tensor(mem_off))
            self.register_buffer("cpu_off", torch.tensor(cpu_off))
            self.norm = torch.nn.LayerNorm(hidden)
            # keys are not "cpu": a ModuleDict key becomes an attribute, and
            # nn.Module.cpu() already owns that name
            self.q = torch.nn.ModuleDict({k: torch.nn.Linear(hidden, d) for k in ("memory", "cores")})
            self.k = torch.nn.ModuleDict({k: torch.nn.Linear(hidden, d) for k in ("memory", "cores")})
            self.scale = d ** -0.5

        def _score(self, H, last, off, key):
            b = torch.arange(H.size(0), device=H.device)
            a = self.norm(H[b, last].float())  # (B, D)
            opts = self.norm(H[b[:, None], last[:, None] - off[None, :]].float())  # (B, K, D)
            return (self.q[key](a)[:, None, :] * self.k[key](opts)).sum(-1) * self.scale

        def forward(self, H, last, num=None):
            return self._score(H, last, self.mem_off, "memory"), self._score(H, last, self.cpu_off, "cores")

    if kind == "pointer":
        return Pointer(*offsets)
    if kind == "classify+num":
        return NumClassify(*num_stats)
    return Classify()


def numeric_matrix(rows: list[dict]) -> np.ndarray:
    """The GBT arm's feature vector for each row (scheduler-visible only)."""
    names = features.feature_names()
    return np.array([[features.row_features(r).get(n, 0.0) for n in names] for r in rows], dtype=np.float32)


def _answer_states(model, ids, mask):
    """All hidden states, and each sequence's last real token (right-padded)."""
    out = model(input_ids=ids, attention_mask=mask)
    return out.last_hidden_state, mask.sum(1) - 1


def train_jev(train: list[dict], val: list[dict], cfg: TrainConfig, log=print):
    import torch

    torch.manual_seed(cfg.seed)
    tok, model = _load(cfg, causal=False)
    offsets = option_offsets(tok) if cfg.head == "pointer" else None
    num = numeric_matrix(train)
    stats = (num.mean(0).tolist(), (num.std(0) + 1e-3).tolist())
    head = _head_module(model.config.hidden_size, cfg.head, offsets, stats).to(_device())
    items = [(_prompt_ids(tok, r, cfg.max_len), r["label_mem_idx"], r["label_cpu_idx"], num[i])
             for i, r in enumerate(train)]
    params = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": cfg.lr_lora},
              {"params": head.parameters(), "lr": cfg.lr_head}]
    opt = torch.optim.AdamW(params, weight_decay=0.0)
    steps_per_epoch = math.ceil(len(items) / cfg.batch_size / cfg.grad_accum)
    sched = _schedule(opt, steps_per_epoch * cfg.epochs, cfg.warmup_frac)
    rng = np.random.default_rng(cfg.seed)
    scaler = _scaler()
    history = []
    model.train()
    step = 0
    for epoch in range(cfg.epochs):
        running = []
        for i, batch in enumerate(_batches(items, cfg.batch_size, key=lambda k: len(items[k][0]), shuffle_rng=rng)):
            ids, mask = _pad([b[0] for b in batch], tok.pad_token_id)
            ids, mask = ids.to(_device()), mask.to(_device())
            ym = torch.tensor([b[1] for b in batch], device=_device())
            yc = torch.tensor([b[2] for b in batch], device=_device())
            xb = torch.tensor(np.stack([b[3] for b in batch]), device=_device())
            with _autocast():
                H, last = _answer_states(model, ids, mask)
            lm, lc = head(H, last, xb)
            loss = torch.nn.functional.cross_entropy(lm, ym) + cfg.cpu_loss_weight * torch.nn.functional.cross_entropy(lc, yc)
            scaler.scale(loss / cfg.grad_accum).backward()
            running.append(float(loss.detach()))
            if (i + 1) % cfg.grad_accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_([p for g in params for p in g["params"]], 1.0)
                scaler.step(opt), scaler.update(), sched.step(), opt.zero_grad(set_to_none=True)
                step += 1
                if step % 20 == 0:
                    log(f"[jev] epoch {epoch} step {step} loss {np.mean(running[-40:]):.4f}")
        history.append({"epoch": epoch, "loss": float(np.mean(running))})
    model.eval()
    return tok, model, head, history


def jev_logits(tok, model, head, rows: list[dict], cfg: TrainConfig, batch_size: int = 16):
    import torch

    ids_all = [_prompt_ids(tok, r, cfg.max_len) for r in rows]
    num = numeric_matrix(rows)
    mem = np.zeros((len(rows), N_MEM), dtype=np.float32)
    cpu = np.zeros((len(rows), N_CPU), dtype=np.float32)
    order = sorted(range(len(rows)), key=lambda k: len(ids_all[k]))
    with torch.no_grad():
        for s in range(0, len(order), batch_size):
            idx = order[s : s + batch_size]
            ids, mask = _pad([ids_all[k] for k in idx], tok.pad_token_id)
            with _autocast():
                H, last = _answer_states(model, ids.to(_device()), mask.to(_device()))
            lm, lc = head(H, last, torch.tensor(num[idx], device=_device()))
            mem[idx], cpu[idx] = lm.float().cpu().numpy(), lc.float().cpu().numpy()
    return mem, cpu


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Scalar temperature minimizing val NLL (Guo et al. 2017)."""
    best = (1.0, float("inf"))
    for t in np.exp(np.linspace(np.log(0.3), np.log(5.0), 60)):
        z = logits / t
        z = z - z.max(1, keepdims=True)
        nll = float(-(z[np.arange(len(labels)), labels] - np.log(np.exp(z).sum(1))).mean())
        if nll < best[1]:
            best = (float(t), nll)
    return best[0]


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


# ── autoregressive SFT ──────────────────────────────────────────────────


def train_sft(train: list[dict], cfg: TrainConfig, log=print):
    import torch

    torch.manual_seed(cfg.seed)
    tok, model = _load(cfg, causal=True)
    end = tok.convert_tokens_to_ids("<|im_end|>")
    items = []
    for r in train:
        p = _prompt_ids(tok, r, cfg.max_len - 32)
        t = tok(prompting.target_text(r["label_mem_idx"], r["label_cpu_idx"]), add_special_tokens=False)["input_ids"] + [end]
        items.append((p, t))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr_lora, weight_decay=0.0)
    steps_per_epoch = math.ceil(len(items) / cfg.batch_size / cfg.grad_accum)
    sched = _schedule(opt, steps_per_epoch * cfg.epochs, cfg.warmup_frac)
    rng = np.random.default_rng(cfg.seed)
    scaler = _scaler()
    history = []
    model.train()
    step = 0
    for epoch in range(cfg.epochs):
        running = []
        for i, batch in enumerate(_batches(items, cfg.batch_size, key=lambda k: len(items[k][0]), shuffle_rng=rng)):
            # Left-pad so every completion ends at the last position, then ask
            # for logits on the last K positions only: the full-sequence logits
            # over a 151k vocabulary are ~7 GB at this batch (rsb-v1-board-2's
            # sft arm OOMed a 16 GB T4 on exactly that tensor). The decision
            # head never builds it at all.
            seqs = [p + t for p, t in batch]
            ids, mask = _pad(seqs, tok.pad_token_id, left=True)
            k = max(len(t) for _p, t in batch) + 1
            target = ids[:, -k + 1 :].clone()
            for j, (_p, t) in enumerate(batch):
                target[j, : (k - 1) - len(t)] = -100  # prompt tokens inside the window
            with _autocast():
                logits = model(input_ids=ids.to(_device()), attention_mask=mask.to(_device()),
                               logits_to_keep=k).logits
                loss = torch.nn.functional.cross_entropy(
                    logits[:, :-1].float().reshape(-1, logits.size(-1)), target.to(_device()).reshape(-1),
                    ignore_index=-100)
            scaler.scale(loss / cfg.grad_accum).backward()
            running.append(float(loss.detach()))
            if (i + 1) % cfg.grad_accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt), scaler.update(), sched.step(), opt.zero_grad(set_to_none=True)
                step += 1
                if step % 20 == 0:
                    log(f"[sft] epoch {epoch} step {step} loss {np.mean(running[-40:]):.4f}")
        history.append({"epoch": epoch, "loss": float(np.mean(running))})
    model.eval()
    return tok, model, history


def sft_generate(tok, model, rows: list[dict], cfg: TrainConfig, n: int = 1, batch_size: int = 16,
                 temperature: float = 1.0) -> list[list[str]]:
    import torch

    ids_all = [_prompt_ids(tok, r, cfg.max_len - 32) for r in rows]
    outs: list[list[str]] = [[] for _ in rows]
    order = sorted(range(len(rows)), key=lambda k: len(ids_all[k]))
    tok.padding_side = "left"
    with torch.no_grad():
        for s in range(0, len(order), batch_size):
            idx = order[s : s + batch_size]
            ids, mask = _pad([ids_all[k] for k in idx], tok.pad_token_id, left=True)
            gen = model.generate(input_ids=ids.to(_device()), attention_mask=mask.to(_device()), max_new_tokens=24,
                                 do_sample=n > 1, temperature=temperature if n > 1 else None,
                                 top_p=None, top_k=None, num_return_sequences=n,
                                 pad_token_id=tok.pad_token_id)
            texts = tok.batch_decode(gen[:, ids.shape[1] :], skip_special_tokens=True)
            for j, k in enumerate(idx):
                outs[k] = texts[j * n : (j + 1) * n]
    return outs


def samples_to_probs(texts: list[str], smoothing: float = 0.02) -> tuple[np.ndarray, int]:
    """N sampled answers -> (P(cell), modal cpu cell)."""
    counts = np.full(N_MEM, smoothing)
    cpus = Counter()
    for t in texts:
        a = prompting.parse_answer(t)
        if a:
            counts[a[0]] += 1
            cpus[a[1]] += 1
    cpu = cpus.most_common(1)[0][0] if cpus else grid.cpu_index_covering(1)
    return counts / counts.sum(), cpu


# ── latency ─────────────────────────────────────────────────────────────


def measure_latency(fn, rows: list[dict], n: int = 32) -> dict:
    """Median/p90 wall-clock of one decision at batch size 1."""
    times = []
    for r in rows[:n]:
        _sync()
        t0 = time.perf_counter()
        fn([r])
        _sync()
        times.append((time.perf_counter() - t0) * 1000)
    return {"p50_ms": float(np.median(times)), "p90_ms": float(np.quantile(times, 0.9)), "n": len(times)}


# ── zero-shot hosted LLM ────────────────────────────────────────────────


def zeroshot_predict(rows: list[dict], teacher: str = "qwen35-397b", concurrency: int = 16, log=print,
                     max_tokens: int = 1024) -> tuple[dict, dict]:
    """Ask a hosted model. Unparseable answers fall back to the grid's
    middle -- recorded as a parse failure, never silently."""
    url = llm_client.wait_until_ready(llm_client.resolve_teacher_candidates(teacher), deadline_s=1800)
    model = llm_client.served_model(url)
    stats = {"teacher": teacher, "model": model, "parse_failures": 0, "errors": 0, "latency_ms": [],
             "completion_tokens": [], "prompt_tokens": []}

    # Room to think. Asked for JSON only, a 27B model still reasons in prose
    # first; at max_tokens=64 that truncated 1,505 of 2,318 answers before the
    # JSON (rsb-v1-board-3). Letting it reason is the fair zero-shot
    # protocol -- and its token bill is part of what is being compared.
    def one(r):
        t0 = time.perf_counter()
        usage: dict = {}
        try:
            text = llm_client.chat(url, prompting.messages(r), max_tokens=max_tokens, temperature=0.0,
                                   timeout=600, model=model, usage=usage)
        except Exception as e:  # noqa: BLE001
            return r["run_id"], None, f"{type(e).__name__}: {e}", 0.0, usage
        return r["run_id"], text, "", (time.perf_counter() - t0) * 1000, usage

    preds = {}
    with ThreadPoolExecutor(concurrency) as pool:
        for i, (rid, text, err, ms, usage) in enumerate(pool.map(one, rows)):
            ans = prompting.parse_answer(text) if text else None
            if usage:
                stats["completion_tokens"].append(usage.get("completion_tokens", 0))
                stats["prompt_tokens"].append(usage.get("prompt_tokens", 0))
            if err:
                stats["errors"] += 1
            elif ans is None:
                stats["parse_failures"] += 1
            else:
                stats["latency_ms"].append(ms)
            m, c = ans or (grid.mem_index_covering(4096), grid.cpu_index_covering(1))
            preds[rid] = {"mem_idx": m, "cpu_idx": c, "raw": (text or err)[-200:]}
            if i % 100 == 0:
                log(f"[zeroshot] {i}/{len(rows)}")
    for k in ("completion_tokens", "prompt_tokens"):
        v = stats.pop(k)
        stats[f"mean_{k}"] = float(np.mean(v)) if v else float("nan")
    lat = stats.pop("latency_ms")
    stats["p50_ms"] = float(np.median(lat)) if lat else float("nan")
    stats["p90_ms"] = float(np.quantile(lat, 0.9)) if lat else float("nan")
    return preds, stats


def config_dict(cfg: TrainConfig) -> dict:
    return asdict(cfg)


def shift(preds: dict, k: int) -> dict:
    """Greedy predictions bumped up k memory cells -- a generative arm's margin knob."""
    return {rid: {**p, "mem_idx": min(N_MEM - 1, p["mem_idx"] + k)} for rid, p in preds.items()}


def dumps(o) -> str:
    return json.dumps(o, default=float)
