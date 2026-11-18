"""Run a RightSize Bench pointer decision model (published on the Hugging Face Hub).

This file is copied as-is into each decision-model repo (``rightsize_pointer.py``),
so it depends only on torch, transformers, peft, safetensors and huggingface_hub.

    from rightsize_pointer import RightSizePointer

    model = RightSizePointer.from_pretrained("unionai/rightsize-rl-qwen3-1.7b")
    row = datasets.load_dataset("unionai/rightsize-bench", "jobs", split="test")[0]
    messages = [{"role": "system", "content": row["prompt_system"]},
                {"role": "user", "content": row["prompt_user"]}]
    model.request(messages)        # {"memory": "1.5Gi", "cpu": "1", "gpu": "none"}
    model.probabilities(messages)  # calibrated P(cell) for each axis

The model scores the options written at the end of the prompt (memory, cpu,
gpu). It is a Qwen3-1.7B torso with a LoRA adapter and three pointer heads: for
each axis, the hidden state at the prompt's last token is compared with the
hidden state at each option's last token. ``request`` is the argmax read-out
(the ``-argmax`` rows of the benchmark). The paper's cost-aware read-out, which
picks the memory cell with the lowest expected cost, lives in
``resource_tuner.bench_all.scoring.decode`` in
https://github.com/unionai-oss/model-factory.

Imports are inside functions so that the module can be imported (and copied)
without the model stack installed.
"""

from __future__ import annotations

import json
import os

CONFIG_FILE = "rightsize_config.json"
HEAD_FILE = "decision_head.safetensors"


def option_offsets(tok, chat_text: str, keys, options) -> list[list[int]]:
    """Offsets, counted back from the last prompt token, of each option's last token.

    Every RightSize prompt ends with the same options block and answer line, so
    these offsets are the same for every prompt a tokenizer encodes."""
    enc = tok(chat_text, add_special_tokens=False, return_offsets_mapping=True)
    spans, n = enc["offset_mapping"], len(enc["input_ids"])
    base = chat_text.rindex("### Options\n")

    def last_token_of(char_end: int) -> int:
        for i, (a, b) in enumerate(spans):
            if a < char_end <= b:
                return i
        raise ValueError(f"no token covers character {char_end}")

    out = []
    for key, opts in zip(keys, options):
        pos = chat_text.index(f"{key}: ", base) + len(key) + 2
        offs = []
        for opt in opts:
            pos = chat_text.index(opt, pos) + len(opt)
            offs.append(n - 1 - last_token_of(pos))
        out.append(offs)
    return out


def head_module(hidden: int, n_axes: int = 3, d: int = 256):
    """The three pointer heads: q(answer slot) . k(option token) / sqrt(d), per axis."""
    import torch

    class TriPointer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            for i in range(n_axes):
                self.register_buffer(f"off{i}", torch.zeros(1, dtype=torch.long))
            self.norm = torch.nn.LayerNorm(hidden)
            self.q = torch.nn.ModuleList([torch.nn.Linear(hidden, d) for _ in range(n_axes)])
            self.k = torch.nn.ModuleList([torch.nn.Linear(hidden, d) for _ in range(n_axes)])
            self.scale = d ** -0.5

        def load_state_dict(self, state, strict=True):  # offsets are buffers of varying length
            for i in range(n_axes):
                setattr(self, f"off{i}", state[f"off{i}"].clone())
            return super().load_state_dict(state, strict=strict)

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


class RightSizePointer:
    def __init__(self, tok, model, head, config: dict, device):
        self.tok, self.model, self.head, self.config, self.device = tok, model, head, config, device

    @classmethod
    def from_pretrained(cls, repo_or_path: str, revision: str | None = None, device: str | None = None,
                        dtype=None) -> "RightSizePointer":
        import torch
        from peft import PeftModel
        from safetensors.torch import load_file
        from transformers import AutoModel, AutoTokenizer

        path = repo_or_path
        if not os.path.isdir(path):
            from huggingface_hub import snapshot_download

            path = snapshot_download(repo_or_path, revision=revision)
        with open(os.path.join(path, CONFIG_FILE)) as f:
            config = json.load(f)
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if dtype is None:
            dtype = torch.bfloat16 if device == "cuda" else torch.float32
        tok = AutoTokenizer.from_pretrained(config["base_model"])
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        base = AutoModel.from_pretrained(config["base_model"], dtype=dtype, attn_implementation="sdpa")
        model = PeftModel.from_pretrained(base, path).to(device).eval()
        head = head_module(model.config.hidden_size, len(config["keys"]), config["head_dim"]).to(device)
        head.load_state_dict({k: v.to(device) for k, v in load_file(os.path.join(path, HEAD_FILE)).items()})
        head.eval()
        return cls(tok, model, head, config, device)

    def _chat_text(self, messages: list[dict]) -> str:
        return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                            enable_thinking=False)

    def logits(self, messages: list[dict]) -> list[list[float]]:
        """Uncalibrated scores for every option on every axis."""
        import torch

        ids = self.tok(self._chat_text(messages), add_special_tokens=False)["input_ids"][-self.config["max_len"]:]
        x = torch.tensor([ids], device=self.device)
        with torch.no_grad():
            H = self.model(input_ids=x, attention_mask=torch.ones_like(x)).last_hidden_state
            out = self.head(H, torch.tensor([len(ids) - 1], device=self.device))
        return [z[0].float().cpu().tolist() for z in out]

    def probabilities(self, messages: list[dict]) -> dict[str, list[float]]:
        """P(option) per axis, temperature-scaled on the logits (fitted on the validation split)."""
        import torch

        out = {}
        for key, z, t in zip(self.config["keys"], self.logits(messages), self.config["temperatures"]):
            out[key] = torch.softmax(torch.tensor(z) / t, dim=0).tolist()
        return out

    def request(self, messages: list[dict]) -> dict[str, str]:
        """The most likely option on each axis."""
        probs = self.probabilities(messages)
        return {key: opts[max(range(len(opts)), key=probs[key].__getitem__)]
                for key, opts in zip(self.config["keys"], self.config["options"])}
