"""The agentic harness's tool catalog.

A deliberately small, legible catalog for a support-triage agent. Six tools,
each with a flat argument schema, because the point of this factory is the
*decision* (which tool, which arguments) and not the tool implementations.

The catalog is data, not code: the prompt builder renders it, the oracle
policy returns calls against it, and the scorer validates calls against it.
One definition, three consumers — so a tool cannot drift between what the
model is told, what it is trained on, and what it is graded against.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class Tool:
    """One callable tool: a name, a description, and typed arguments."""

    name: str
    description: str
    #: argument name -> short type/meaning, rendered into the prompt
    args: Mapping[str, str] = field(default_factory=dict)

    @property
    def required(self) -> tuple[str, ...]:
        return tuple(self.args)


LOOKUP_ORDER = Tool(
    name="lookup_order",
    description="Fetch an order's status, total and return window. Call this first when the customer names an order you have not looked up yet.",
    args={"order_id": "the order id from the customer's message"},
)

CHECK_SHIPPING = Tool(
    name="check_shipping",
    description="Fetch live carrier tracking for a shipped order.",
    args={"tracking_number": "the order's tracking number"},
)

ISSUE_REFUND = Tool(
    name="issue_refund",
    description="Refund a delivered order that is inside its return window and under the auto-approval limit.",
    args={"order_id": "the order id", "amount_cents": "refund amount in cents"},
)

ESCALATE_TO_HUMAN = Tool(
    name="escalate_to_human",
    description="Hand off to a human agent. Use when a refund exceeds the auto-approval limit, the return window has closed, or the customer is reporting damage or a safety issue.",
    args={"reason": "one short phrase naming why a human is needed"},
)

SEARCH_FAQ = Tool(
    name="search_faq",
    description="Search the help centre for a general policy question that is not about a specific order.",
    args={"query": "the customer's question, as a search query"},
)

REPLY = Tool(
    name="reply",
    description="Answer the customer directly. Terminal: use only when no other tool is needed.",
    args={"message": "the reply to send"},
)

CATALOG: tuple[Tool, ...] = (
    LOOKUP_ORDER,
    CHECK_SHIPPING,
    ISSUE_REFUND,
    ESCALATE_TO_HUMAN,
    SEARCH_FAQ,
    REPLY,
)

TOOLS_BY_NAME: dict[str, Tool] = {t.name: t for t in CATALOG}

#: Tool names, in catalog order. The scorer reports per-tool accuracy over
#: exactly this list so a model that silently stops using one is visible.
TOOL_NAMES: tuple[str, ...] = tuple(t.name for t in CATALOG)


@dataclass(frozen=True)
class ToolCall:
    """A decision: which tool, with which arguments."""

    tool: str
    args: Mapping[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        # Sorted keys so a call has one canonical serialization — the training
        # targets and the exact-match scorer both depend on that.
        return json.dumps({"tool": self.tool, "args": dict(self.args)}, sort_keys=True)

    @classmethod
    def from_obj(cls, obj: Any) -> "ToolCall | None":
        """A ToolCall from already-parsed JSON, or None if it is not one.

        Returns None rather than raising: a malformed call is a *score* of zero
        for that episode, not a failure of the eval run.
        """
        if not isinstance(obj, dict):
            return None
        tool = obj.get("tool")
        if not isinstance(tool, str) or tool not in TOOLS_BY_NAME:
            return None
        args = obj.get("args")
        if args is None:
            args = {}
        if not isinstance(args, dict):
            return None
        return cls(tool=tool, args=args)


def render_catalog() -> str:
    """The tool catalog as the model sees it in its system prompt."""
    lines = []
    for t in CATALOG:
        arg_desc = ", ".join(f"{k} ({v})" for k, v in t.args.items()) or "no arguments"
        lines.append(f"- {t.name}({', '.join(t.args)}): {t.description} Arguments: {arg_desc}.")
    return "\n".join(lines)


def args_match(expected: Mapping[str, Any], got: Mapping[str, Any], tool: str) -> bool:
    """Whether a predicted call's arguments match the oracle's.

    Only the tool's *declared* arguments are compared, and only on the values
    the oracle actually pins. `reply`'s message and `escalate_to_human`'s
    reason are free text — any non-empty string is accepted, because grading
    those on exact wording would measure phrasing, not the decision.
    """
    spec = TOOLS_BY_NAME.get(tool)
    if spec is None:
        return False
    for key in spec.required:
        want = expected.get(key)
        have = got.get(key)
        if isinstance(want, str) and want == FREE_TEXT:
            if not isinstance(have, str) or not have.strip():
                return False
            continue
        if _norm(want) != _norm(have):
            return False
    return True


#: Sentinel for an oracle argument whose exact value is not graded.
FREE_TEXT = "<free-text>"


def _norm(value: Any) -> Any:
    """Compare 1200, "1200" and " 1200 " as equal; everything else verbatim."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        try:
            return float(stripped)
        except ValueError:
            return stripped.casefold()
    return value
