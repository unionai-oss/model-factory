"""Turning a decision prompt into token ids — for both heads.

Pointer: the ids of the rendered chat (system + user, ending at the assistant
turn), plus the token index of the **end of each option line** and the
**answer position** (the last token). Causal attention means the option
positions have read the whole state and question by the time they are reached,
and the answer position has read everything, including all options.

Generative: the same prompt ids, followed by the completion ids
``{"decision": "..."}<eos>``; loss is taken on the completion only.

No new special tokens: the pointer reads positions in ordinary text, so no
embedding has to be resized or trained from scratch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..harness.prompt import completion, option_markers


def render_prompt(tok, chat: list[dict[str, str]]) -> str:
    """The chat as the torso's template renders it, ready for the answer.

    ``enable_thinking=False`` matters for Qwen3 (it closes an empty think
    block); other templates ignore the variable.
    """
    return tok.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


@dataclass
class PointerExample:
    input_ids: list[int]
    option_pos: list[int]
    answer_pos: int
    label: int  # index into the options this prompt listed


def _token_at(offsets: Sequence[tuple[int, int]], char_idx: int) -> int:
    for t, (s, e) in enumerate(offsets):
        if s <= char_idx < e:
            return t
    # a char inside no token's span (some tokenizers report empty spans for
    # merged whitespace): take the last token that starts before it
    best = 0
    for t, (s, _) in enumerate(offsets):
        if s <= char_idx:
            best = t
    return best


def encode_pointer(tok, chat, options: Sequence[str], label: str | None = None) -> PointerExample:
    text = render_prompt(tok, chat)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]
    base = text.rfind("Options:")
    if base < 0:
        raise ValueError("prompt has no Options: block")
    positions = []
    for marker in option_markers(options):
        start = text.find("\n" + marker, base)
        if start < 0:
            raise ValueError(f"option {marker!r} not found in prompt")
        last_char = start + len(marker)  # +1 for the newline, -1 for last char
        positions.append(_token_at(offsets, last_char))
    if len(set(positions)) != len(positions):
        raise ValueError(f"options collapsed onto the same token: {positions}")
    y = list(options).index(label) if label is not None else -1
    return PointerExample(list(ids), positions, len(ids) - 1, y)


@dataclass
class SFTExample:
    input_ids: list[int]
    #: -100 on prompt tokens: only the completion is learned.
    labels: list[int]
    prompt_len: int


def completion_ids(tok, decision: str, eos: bool = True) -> list[int]:
    ids = tok(completion(decision), add_special_tokens=False)["input_ids"]
    if eos and tok.eos_token_id is not None:
        ids = ids + [tok.eos_token_id]
    return list(ids)


def encode_sft(tok, chat, label: str) -> SFTExample:
    prompt = tok(render_prompt(tok, chat), add_special_tokens=False)["input_ids"]
    comp = completion_ids(tok, label)
    return SFTExample(list(prompt) + comp, [-100] * len(prompt) + comp, len(prompt))


def prompt_ids(tok, chat) -> list[int]:
    return list(tok(render_prompt(tok, chat), add_special_tokens=False)["input_ids"])
