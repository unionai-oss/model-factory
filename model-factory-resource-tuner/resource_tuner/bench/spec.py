"""What makes one RightSize benchmark different from another.

The three benches -- RightSizeMem (this package), RightSizeCPU
(``bench_cpu``) and RightSizeGPU (``bench_gpu``) -- share their LLM arms
(zero-shot, autoregressive SFT, the Jev-style decision head), their
trainer and their prompt layout. What differs is *which two decisions* an
arm makes and how they are scored:

    bench   primary decision (scored by cost)    secondary decision
    mem     memory request (17 cells)            CPU request (7 cells)
    cpu     peak CPU cores, 15 s window          mean CPU cores
    gpu     peak GPU memory (device ladder)      mean GPU utilization

A ``BenchSpec`` carries exactly that: the two option lists as the prompt
writes them, the row columns holding the oracle's answer, the prompt
itself, how an answer is parsed, and how a distribution over the primary
decision becomes one request (expected cost under that bench's cost
model). The shared LLM code reads everything bench-specific from here.

Pure logic: no flyte, no torch.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable


@dataclass(frozen=True)
class Decision:
    """One choice an arm makes, from a fixed candidate list."""

    key: str  # JSON key in the answer and the options line's prefix ("memory")
    options: tuple[str, ...]  # each candidate exactly as the prompt writes it
    label_col: str  # row column holding the oracle's index
    pred_key: str  # key of the index in a prediction dict
    default_idx: int  # what an unparseable answer falls back to
    parse_value: Callable[[object], int | None] | None = None  # JSON value -> index (snapping up)

    def parse(self, value: object) -> int | None:
        if self.parse_value is not None:
            return self.parse_value(value)
        return snap_up(value, [float(o) for o in self.options])


@dataclass(frozen=True)
class BenchSpec:
    name: str
    title: str  # "RightSizeMem Bench"
    system: str
    a: Decision  # primary: decoded by expected cost, the headline metric
    b: Decision  # secondary: argmax
    user_prompt: Callable[[dict], str]
    decode: Callable[[object], int]  # P(primary cell) -> chosen cell
    headline: Callable[[list[dict], dict[str, dict]], float]  # lower is better; tunes margins
    row_features: Callable[[dict], dict[str, float]]
    feature_names: Callable[[], list[str]]
    probs_key: str = "mem_probs"
    target_text: Callable[[int, int], str] | None = None
    parse_answer: Callable[[str], tuple[int, int] | None] | None = None
    notes: dict = field(default_factory=dict)

    def messages(self, row: dict) -> list[dict]:
        user = with_hint(self.user_prompt(row), row.get("hint") or "")
        return [{"role": "system", "content": self.system}, {"role": "user", "content": user}]

    def target(self, a_idx: int, b_idx: int) -> str:
        if self.target_text is not None:
            return self.target_text(a_idx, b_idx)
        return json.dumps({self.a.key: self.a.options[a_idx], self.b.key: self.b.options[b_idx]})

    def parse(self, text: str) -> tuple[int, int] | None:
        if self.parse_answer is not None:
            return self.parse_answer(text)
        return parse_json_answer(text, self.a, self.b)

    def labels(self, row: dict) -> tuple[int, int]:
        return int(row[self.a.label_col]), int(row[self.b.label_col])

    def pred(self, a_idx: int, b_idx: int, probs=None, raw: str = "") -> dict:
        out = {self.a.pred_key: int(a_idx), self.b.pred_key: int(b_idx)}
        if probs is not None:
            out[self.probs_key] = probs
        if raw:
            out["raw"] = raw
        return out


# ── shared helpers ──────────────────────────────────────────────────────

HINT_HEADER = "### A tree model's estimate"


def with_hint(prompt: str, hint: str) -> str:
    """A stacked arm's prompt: the trees' estimate (``bench.hybrid``) just
    before the options. The suffix from "### Options" on is untouched, so the
    pointer head's option offsets (counted from the end) still hold."""
    if not hint:
        return prompt
    i = prompt.rindex("### Options")
    return f"{prompt[:i]}{HINT_HEADER}\n{hint}\n\n{prompt[i:]}"


_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
_JSON = re.compile(r"\{[^{}]*\}")


def snap_up(value: object, cells: list[float]) -> int | None:
    """A free-form number (or "2.5", "40%", "6 GiB") -> the smallest cell >= it.

    Snapping up is the generous reading: an arm is never penalized for a
    value between two cells."""
    m = _NUMBER.search(str(value))
    if not m:
        return None
    x = float(m.group(0))
    if x < 0:
        return None
    for i, c in enumerate(cells):
        if x <= c + 1e-9:
            return i
    return len(cells) - 1


def parse_json_answer(text: str, a: Decision, b: Decision) -> tuple[int, int] | None:
    """The last JSON object in `text` that names a usable primary answer."""
    for m in reversed(_JSON.findall(text or "")):
        try:
            obj = json.loads(m)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict) or a.key not in obj:
            continue
        ai = a.parse(obj[a.key])
        if ai is None:
            continue
        bi = b.parse(obj[b.key]) if b.key in obj else None
        return ai, b.default_idx if bi is None else bi
    return None


def options_block(a: Decision, b: Decision) -> str:
    return f"{a.key}: {', '.join(a.options)}\n{b.key}: {', '.join(b.options)}"


def answer_instruction(a: Decision, b: Decision) -> str:
    return f'Answer with JSON only: {{"{a.key}": "<one {a.key} option>", "{b.key}": "<one {b.key} option>"}}'
