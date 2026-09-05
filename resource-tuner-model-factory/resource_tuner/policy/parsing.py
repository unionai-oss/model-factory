"""Completion → validated Proposal.

Lenient on wrapping (models fence JSON, prepend prose, emit <think> blocks —
Qwen3-family models especially), strict on content (validate_proposal
rejects unknown keys and junk values). The split matters for RL: recovering
the JSON from sloppy wrapping keeps early training from starving on format
noise, while content strictness keeps the action space honest.
"""

from __future__ import annotations

import json
import re

from .actions import InvalidProposal, Proposal, validate_proposal

_THINK_RE = re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL)
# First {...} block without nested braces — Resources kwargs are flat.
_JSON_RE = re.compile(r"\{[^{}]*\}")


def extract_proposal(completion: str) -> Proposal:
    """Parse a model completion into a Proposal; raises InvalidProposal."""
    text = _THINK_RE.sub("", completion)
    m = _JSON_RE.search(text)
    if not m:
        raise InvalidProposal("no JSON object in completion")
    try:
        raw = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        raise InvalidProposal(f"bad JSON: {e}")
    return validate_proposal(raw)


def try_extract_proposal(completion: str) -> Proposal | None:
    try:
        return extract_proposal(completion)
    except InvalidProposal:
        return None


def format_credit(completion: str) -> float:
    """Graded parseability in [0, 1] — the format ladder as a REWARD SIGNAL.

    A flat zero for every invalid completion gives the policy no gradient
    toward JSON-ness (round 7/8: 28%→10% of completions unparseable, all
    scored identically to prose). Rungs:

        0.00  no {...} object anywhere
        0.25  a {...} block exists but is not valid JSON
        0.50  valid JSON, wrong schema (unknown keys, junk values)
        0.75  schema-valid except the gpu field (the round-7/8 failure)
        1.00  fully valid proposal

    Multiply by rewards.FORMAT_REWARD for the reward-scale bonus.
    """
    text = _THINK_RE.sub("", completion)
    m = _JSON_RE.search(text)
    if not m:
        return 0.0
    try:
        raw = json.loads(m.group(0))
    except json.JSONDecodeError:
        return 0.25
    try:
        validate_proposal(raw)
        return 1.0
    except InvalidProposal:
        pass
    # One rung higher when ONLY the gpu value is the problem: cpu/memory
    # parse fine without it.
    if isinstance(raw, dict) and "gpu" in raw:
        try:
            validate_proposal({k: v for k, v in raw.items() if k != "gpu"})
            return 0.75
        except InvalidProposal:
            pass
    return 0.5
