"""Parsing a model's output into a tool call, and grading it.

Small instruct models wrap JSON in prose and code fences no matter how firmly
the prompt says not to, so `parse_call` is deliberately forgiving about the
packaging and strict about the content: it will dig a JSON object out of a
fenced block or a sentence, but it will not guess a tool name, invent
arguments, or accept a call to a tool that is not in the catalog.

That split matters for what the metrics mean. `json_valid` measures whether
the model produced a well-formed call at all (a formatting skill, which
fine-tuning fixes almost immediately). `tool_accuracy` and `exact_accuracy`
measure the decision itself. Reporting them separately is what stops a
format win from being mistaken for a reasoning win.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .tools import TOOL_NAMES, ToolCall, args_match

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _candidates(text: str) -> Iterable[str]:
    """Substrings of ``text`` that might be the JSON object, best guess first."""
    stripped = text.strip()
    if stripped:
        yield stripped
    for block in _FENCE_RE.findall(text):
        block = block.strip()
        if block:
            yield block
    # Balanced-brace scan: the first {...} that nests correctly. A regex
    # cannot do this, and `args` is itself an object, so the naive
    # "first { to first }" slice truncates every real call.
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                yield text[start : i + 1]
                start = -1
            elif depth < 0:
                depth = 0


def parse_call(text: str) -> ToolCall | None:
    """The tool call in ``text``, or None if there is not a valid one."""
    if not text:
        return None
    for candidate in _candidates(text):
        try:
            obj = json.loads(candidate)
        except (ValueError, TypeError):
            continue
        call = ToolCall.from_obj(obj)
        if call is not None:
            return call
    return None


@dataclass(frozen=True)
class Grade:
    """How one prediction scored against one oracle label."""

    episode_id: str
    expected_tool: str
    predicted_tool: str | None
    json_valid: bool
    tool_correct: bool
    exact_correct: bool
    hard: bool
    raw: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "expected_tool": self.expected_tool,
            "predicted_tool": self.predicted_tool or "",
            "json_valid": self.json_valid,
            "tool_correct": self.tool_correct,
            "exact_correct": self.exact_correct,
            "hard": self.hard,
        }


def grade_one(
    episode_id: str, expected_json: str, completion: str, hard: bool = False
) -> Grade:
    """Grade one completion against one oracle label (serialized ToolCall)."""
    expected = ToolCall.from_obj(json.loads(expected_json))
    if expected is None:  # pragma: no cover - labels come from the oracle
        raise ValueError(f"episode {episode_id}: label is not a valid tool call")
    got = parse_call(completion)
    if got is None:
        return Grade(
            episode_id=episode_id,
            expected_tool=expected.tool,
            predicted_tool=None,
            json_valid=False,
            tool_correct=False,
            exact_correct=False,
            hard=hard,
            raw=completion[:400],
        )
    tool_correct = got.tool == expected.tool
    return Grade(
        episode_id=episode_id,
        expected_tool=expected.tool,
        predicted_tool=got.tool,
        json_valid=True,
        tool_correct=tool_correct,
        # Arguments are only meaningful if the tool is right, so an exact hit
        # requires both. Grading args on a wrong tool would compare against a
        # schema the model was not even aiming at.
        exact_correct=tool_correct and args_match(expected.args, got.args, expected.tool),
        hard=hard,
        raw=completion[:400],
    )


def summarize(grades: Sequence[Grade]) -> dict[str, Any]:
    """Aggregate metrics over graded episodes.

    `exact_accuracy` is the headline — it is what the champion is chosen on,
    because a right tool with wrong arguments is still a wrong action.
    """
    n = len(grades)
    if n == 0:
        return {
            "n": 0,
            "json_valid_rate": 0.0,
            "tool_accuracy": 0.0,
            "exact_accuracy": 0.0,
            "n_hard": 0,
            "hard_exact_accuracy": 0.0,
            "per_tool": {},
            "confusion": {},
        }
    hard = [g for g in grades if g.hard]
    per_tool: dict[str, dict[str, Any]] = {}
    for tool in TOOL_NAMES:
        subset = [g for g in grades if g.expected_tool == tool]
        per_tool[tool] = {
            "n": len(subset),
            "tool_accuracy": _mean(g.tool_correct for g in subset),
            "exact_accuracy": _mean(g.exact_correct for g in subset),
        }
    confusion: dict[str, int] = {}
    for g in grades:
        if not g.tool_correct:
            key = f"{g.expected_tool}->{g.predicted_tool or 'unparseable'}"
            confusion[key] = confusion.get(key, 0) + 1
    return {
        "n": n,
        "json_valid_rate": _mean(g.json_valid for g in grades),
        "tool_accuracy": _mean(g.tool_correct for g in grades),
        "exact_accuracy": _mean(g.exact_correct for g in grades),
        "n_hard": len(hard),
        "hard_exact_accuracy": _mean(g.exact_correct for g in hard),
        "per_tool": per_tool,
        # Most-confused pairs first: this is the single most useful thing on
        # the scorecard when a model is only partly right.
        "confusion": dict(sorted(confusion.items(), key=lambda kv: -kv[1])[:12]),
    }


def _mean(values: Iterable[bool]) -> float:
    items = list(values)
    return sum(1 for v in items if v) / len(items) if items else 0.0


def grade_batch(
    episode_ids: Sequence[str],
    expected: Sequence[str],
    completions: Sequence[str],
    hard: Sequence[bool] | None = None,
) -> list[Grade]:
    flags = list(hard) if hard is not None else [False] * len(episode_ids)
    return [
        grade_one(eid, exp, comp, bool(h))
        for eid, exp, comp, h in zip(episode_ids, expected, completions, flags)
    ]


def majority_baseline(labels: Sequence[str]) -> Mapping[str, Any]:
    """Accuracy of always guessing the most common tool.

    The floor any model has to clear to have learned anything. Without it, a
    64% exact-accuracy scorecard looks like a result rather than a coin flip
    on a skewed split.
    """
    if not labels:
        return {"tool": "", "accuracy": 0.0}
    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    tool, count = max(counts.items(), key=lambda kv: kv[1])
    return {"tool": tool, "accuracy": count / len(labels)}
