"""The one prompt every arm sees: training, eval, and the serving app.

A decision prompt has three parts, which is the Jev / Strands Decider shape:

    state     what is being proposed, and where (the observation)
    question  "may this tool call run?"
    options   the candidate answers, one per line

The **generative** arm answers by emitting ``{"decision": "<option>"}``. The
**pointer** arm never generates: it reads the hidden state at the end of each
option line and at the final ("answer") position, and scores each option
against the answer — one forward pass, zero output tokens. Both arms train on
the identical prompt, so the only thing that differs between them is the head.
"""

from __future__ import annotations

import json
import random
from typing import Any, Mapping, Sequence

from .policy import DECISIONS, POLICY_TEXT, Observation

QUESTION = "Should the coding agent's proposed tool call run?"


def answer_instruction() -> str:
    return 'Answer with JSON only: {"decision": "<one of the options>"}.'


def render_state(obs: Observation) -> str:
    lines = [f"tool: {obs.tool}"]
    if obs.command is not None:
        lines.append(f"command: {obs.command}")
    if obs.path is not None:
        lines.append(f"path: {obs.path}")
    if obs.resolved_path is not None:
        lines.append(f"resolved_path: {obs.resolved_path}")
    if obs.url is not None:
        lines.append(f"url: {obs.url}")
    lines += [
        f"workspace_root: {obs.workspace_root}",
        f"cwd: {obs.cwd}",
        f"home: {obs.home}",
        f"git_branch: {obs.git_branch}",
        f"permission_mode: {obs.permission_mode}",
        f"user_task: {obs.user_task}",
    ]
    return "\n".join(lines)


def render_user(obs: Observation, options: Sequence[str] = DECISIONS) -> str:
    opts = "\n".join(f"- {o}" for o in options)
    return f"{render_state(obs)}\n\nQuestion: {QUESTION}\nOptions:\n{opts}"


def build_chat(
    obs: Observation | Mapping[str, Any], options: Sequence[str] = DECISIONS
) -> list[dict[str, str]]:
    if not isinstance(obs, Observation):
        obs = Observation.from_dict(obs)
    return [
        {"role": "system", "content": f"{POLICY_TEXT}\n{answer_instruction()}"},
        {"role": "user", "content": render_user(obs, options)},
    ]


def option_markers(options: Sequence[str] = DECISIONS) -> list[str]:
    """The exact text of each option line; the pointer head reads the hidden
    state at the last token of each."""
    return [f"- {o}" for o in options]


def completion(decision: str) -> str:
    """The generative arm's target."""
    return json.dumps({"decision": decision})


def shuffled_options(rng: random.Random) -> list[str]:
    """A random option order, used in training so the pointer head learns to
    score an option's *content*, not its slot."""
    opts = list(DECISIONS)
    rng.shuffle(opts)
    return opts


def parse_decision(text: str) -> str | None:
    """The decision in a generative completion, or None if there isn't one.

    Forgiving about packaging (prose, fences), strict about content: the value
    must be exactly one of the options.
    """
    if not text:
        return None
    start = text.find("{")
    while start >= 0:
        depth = 0
        for j in range(start, len(text)):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start : j + 1])
                    except ValueError:
                        break
                    d = obj.get("decision") if isinstance(obj, dict) else None
                    return d if d in DECISIONS else None
        start = text.find("{", start + 1)
    return None
