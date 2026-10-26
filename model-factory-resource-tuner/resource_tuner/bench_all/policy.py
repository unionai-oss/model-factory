"""The combined bench's learned policy: one decision model, three heads, two
ways to train it.

**The model** is the benches' Jev-style pointer decision model (an LLM torso
+ LoRA whose LM head is replaced by a head that scores each option *token*
in the prompt against the answer slot), with three decisions instead of
two: memory, CPU, and GPU-or-none. One forward pass gives a distribution
over every axis; no tokens are generated.

**Supervised** (``train_supervised``, the ``jev-pointer`` arm): cross-entropy
on each axis's oracle cell, masked where an axis was not measured (a GPU
task's host memory and cores come with its instance). This is what every
bench's decision arm does: imitate the cheapest-that-fits label, then
decode the calibrated distribution by each bench's decoder.

**RL** (``train_rl``, the ``rl-pointer`` arm): starting from the supervised
model, maximize the *joint* reward directly -- minus the run's regret ratio
under ``scoring.cost``, the same simulator the leaderboard scores with
(OOM retries, CPU throttling replayed against the measured series, CUDA
OOMs up the device ladder, a missing or a needless GPU). GRPO without
generation: for each prompt, K joint actions are sampled from the three
heads (one forward pass), each is scored by the simulator, advantages are
the group-normalized rewards, and the loss is the policy gradient plus a
KL penalty to the supervised policy. What imitation cannot see and RL can:
the axes interact (on a GPU node, memory and CPU are free; on a CPU node,
a GPU is the most expensive mistake there is), and under-asking costs more
than over-asking by an amount the label does not carry.

torch is imported inside functions, as in ``bench.llm_arms``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..bench.llm_arms import (
    TrainConfig,
    _answer_states,
    _autocast,
    _batches,
    _device,
    _load,
    _pad,
    _prompt_ids,
    _scaler,
    _schedule,
    fit_temperature,
    softmax,
)
from . import grid, prompting, scoring

AXES = ("memory", "cpu", "gpu")
SIZES = (grid.N_MEM, grid.N_CPU, grid.N_GPU)


@dataclass
class RLConfig:
    k_samples: int = 16  # joint actions per prompt (one forward pass)
    prompts: int = 6000  # training prompts for the RL phase
    batch_size: int = 8
    grad_accum: int = 2
    lr_lora: float = 3e-5
    lr_head: float = 1e-4
    kl_beta: float = 0.05
    epochs: int = 1
    seed: int = 0


def option_offsets(tok) -> list[list[int]]:
    """Offsets from the last prompt token to each option's last token, per axis."""
    probe = {"task_code": "x = 1", "inputs_desc": "x: npy, 1 KiB, shape=(1,), dtype=uint8", "history_json": "[]"}
    text = tok.apply_chat_template(prompting.messages(probe), tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    spans, n = enc["offset_mapping"], len(enc["input_ids"])
    base = text.rindex("### Options\n")

    def last_token_of(char_end: int) -> int:
        for i, (a, b) in enumerate(spans):
            if a < char_end <= b:
                return i
        raise ValueError(f"no token covers char {char_end}")

    out = []
    for key, options in zip(prompting.KEYS, prompting.OPTIONS):
        pos = text.index(f"{key}: ", base) + len(key) + 2
        offs = []
        for opt in options:
            pos = text.index(opt, pos) + len(opt)
            offs.append(n - 1 - last_token_of(pos))
        out.append(offs)
    return out


def head_module(hidden: int, offsets: list[list[int]], d: int = 256):
    import torch

    class TriPointer(torch.nn.Module):
        """q(answer slot) . k(option token), for each of the three decisions."""

        def __init__(self):
            super().__init__()
            for i, off in enumerate(offsets):
                self.register_buffer(f"off{i}", torch.tensor(off))
            self.norm = torch.nn.LayerNorm(hidden)
            self.q = torch.nn.ModuleList([torch.nn.Linear(hidden, d) for _ in offsets])
            self.k = torch.nn.ModuleList([torch.nn.Linear(hidden, d) for _ in offsets])
            self.scale = d ** -0.5

        def forward(self, H, last):
            b = torch.arange(H.size(0), device=H.device)
            a = self.norm(H[b, last].float())
            out = []
            for i in range(len(self.q)):
                off = getattr(self, f"off{i}")
                opts = self.norm(H[b[:, None], last[:, None] - off[None, :]].float())
                out.append((self.q[i](a)[:, None, :] * self.k[i](opts)).sum(-1) * self.scale)
            return out

    return TriPointer()


def labels(row: dict) -> tuple[int, int, int]:
    """Oracle cells; -1 where the axis is not scored (ignored by the loss)."""
    return int(row["label_mem"]), int(row["label_cpu"]), int(row["label_gpu"])


def _ids(tok, rows, cfg: TrainConfig) -> list[list[int]]:
    return [_prompt_ids(tok, r, cfg.max_len, prompting.SPEC) for r in rows]


def build(cfg: TrainConfig):
    tok, model = _load(cfg, causal=False)
    head = head_module(model.config.hidden_size, option_offsets(tok)).to(_device())
    return tok, model, head


def train_supervised(train: list[dict], cfg: TrainConfig, log=print):
    import torch

    torch.manual_seed(cfg.seed)
    tok, model, head = build(cfg)
    ids = _ids(tok, train, cfg)
    items = [(ids[i], *labels(r)) for i, r in enumerate(train)]
    params = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": cfg.lr_lora},
              {"params": head.parameters(), "lr": cfg.lr_head}]
    opt = torch.optim.AdamW(params, weight_decay=0.0)
    steps = math.ceil(len(items) / cfg.batch_size / cfg.grad_accum) * cfg.epochs
    sched = _schedule(opt, steps, cfg.warmup_frac)
    rng = np.random.default_rng(cfg.seed)
    scaler = _scaler()
    history, step = [], 0
    model.train()
    for epoch in range(cfg.epochs):
        running = []
        for i, batch in enumerate(_batches(items, cfg.batch_size, key=lambda k: len(items[k][0]), shuffle_rng=rng)):
            x, mask = _pad([b[0] for b in batch], tok.pad_token_id)
            with _autocast():
                H, last = _answer_states(model, x.to(_device()), mask.to(_device()))
            logits = head(H, last)
            loss = 0.0
            for a, lg in enumerate(logits):
                y = torch.tensor([b[1 + a] for b in batch], device=_device())
                if (y >= 0).any():
                    loss = loss + torch.nn.functional.cross_entropy(lg, y, ignore_index=-1)
            scaler.scale(loss / cfg.grad_accum).backward()
            running.append(float(loss.detach()))
            if (i + 1) % cfg.grad_accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_([p for g in params for p in g["params"]], 1.0)
                scaler.step(opt), scaler.update(), sched.step(), opt.zero_grad(set_to_none=True)
                step += 1
                if step % 20 == 0:
                    log(f"[sup] epoch {epoch} step {step}/{steps} loss {np.mean(running[-40:]):.4f}")
        history.append({"epoch": epoch, "loss": float(np.mean(running))})
    model.eval()
    return tok, model, head, history


def logits(tok, model, head, rows: list[dict], cfg: TrainConfig, batch_size: int = 16) -> list[np.ndarray]:
    import torch

    ids = _ids(tok, rows, cfg)
    out = [np.zeros((len(rows), n), dtype=np.float32) for n in SIZES]
    order = sorted(range(len(rows)), key=lambda k: len(ids[k]))
    with torch.no_grad():
        for s in range(0, len(order), batch_size):
            idx = order[s : s + batch_size]
            x, mask = _pad([ids[k] for k in idx], tok.pad_token_id)
            with _autocast():
                H, last = _answer_states(model, x.to(_device()), mask.to(_device()))
            for a, lg in enumerate(head(H, last)):
                out[a][idx] = lg.float().cpu().numpy()
    return out


def train_rl(tok, model, head, train: list[dict], cfg: TrainConfig, rl: RLConfig, log=print) -> list[dict]:
    """GRPO over the three heads, from the supervised policy, against `scoring.reward`."""
    import torch

    torch.manual_seed(rl.seed)
    rng = np.random.default_rng(rl.seed)
    rows = list(train)
    rng.shuffle(rows)
    rows = rows[: rl.prompts] if rl.prompts else rows
    # the reference policy: the supervised model's distributions, once, before any update
    model.eval()
    ref = [torch.tensor(softmax(z), dtype=torch.float32) for z in logits(tok, model, head, rows, cfg)]
    ids = _ids(tok, rows, cfg)
    items = list(range(len(rows)))
    params = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": rl.lr_lora},
              {"params": head.parameters(), "lr": rl.lr_head}]
    opt = torch.optim.AdamW(params, weight_decay=0.0)
    steps = math.ceil(len(items) / rl.batch_size / rl.grad_accum) * rl.epochs
    sched = _schedule(opt, steps, 0.05)
    scaler = _scaler()
    model.train()
    log_rows, step, window = [], 0, []
    for epoch in range(rl.epochs):
        for i, batch in enumerate(_batches(items, rl.batch_size, key=lambda k: len(ids[k]), shuffle_rng=rng)):
            x, mask = _pad([ids[k] for k in batch], tok.pad_token_id)
            with _autocast():
                H, last = _answer_states(model, x.to(_device()), mask.to(_device()))
            lgs = head(H, last)
            logps = [torch.log_softmax(lg.float(), -1) for lg in lgs]  # (B, n_axis)
            with torch.no_grad():
                acts = [torch.distributions.Categorical(logits=lp).sample((rl.k_samples,)).T for lp in logps]  # (B, K)
            acts_np = [a.cpu().numpy() for a in acts]
            R = np.zeros((len(batch), rl.k_samples), dtype=np.float32)
            for b, k in enumerate(batch):
                r = rows[k]
                for j in range(rl.k_samples):
                    R[b, j] = scoring.reward(r, int(acts_np[0][b, j]), int(acts_np[1][b, j]), int(acts_np[2][b, j]))
            adv = (R - R.mean(1, keepdims=True)) / (R.std(1, keepdims=True) + 1e-3)
            adv_t = torch.tensor(adv, device=_device())
            # a GPU task's memory and CPU do not move its reward: their log-probs are left out
            gpu_task = torch.tensor([rows[k]["source"] == "gpu" for k in batch], device=_device())
            lp_sum = torch.zeros_like(adv_t)
            kl = torch.zeros(len(batch), device=_device())
            for a, lp in enumerate(logps):
                taken = lp.gather(1, acts[a])  # (B, K)
                keep = torch.ones_like(gpu_task, dtype=torch.float32) if a == 2 else (~gpu_task).float()
                lp_sum = lp_sum + taken * keep[:, None]
                p_ref = ref[a][batch].to(_device())
                kl = kl + keep * (lp.exp() * (lp - torch.log(p_ref.clamp_min(1e-8)))).sum(-1)
            loss = -(adv_t * lp_sum).mean() + rl.kl_beta * kl.mean()
            scaler.scale(loss / rl.grad_accum).backward()
            window.append((float(R.mean()), float(kl.mean().detach())))
            if (i + 1) % rl.grad_accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_([p for g in params for p in g["params"]], 1.0)
                scaler.step(opt), scaler.update(), sched.step(), opt.zero_grad(set_to_none=True)
                step += 1
                if step % 10 == 0:
                    w = np.array(window[-20:])
                    entry = {"step": step, "mean_reward": float(w[:, 0].mean()), "kl": float(w[:, 1].mean())}
                    log_rows.append(entry)
                    log(f"[rl] step {step}/{steps} reward {entry['mean_reward']:.3f} kl {entry['kl']:.4f}")
    model.eval()
    return log_rows


def calibrate(val_logits: list[np.ndarray], val_rows: list[dict]) -> list[float]:
    """One temperature per axis, fit on the val rows where that axis is measured."""
    temps = []
    for a, z in enumerate(val_logits):
        y = np.array([labels(r)[a] for r in val_rows])
        keep = y >= 0
        temps.append(fit_temperature(z[keep], y[keep]) if keep.any() else 1.0)
    return temps


def probs(lgs: list[np.ndarray], temps: list[float]) -> list[np.ndarray]:
    return [softmax(z / t) for z, t in zip(lgs, temps)]
