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
