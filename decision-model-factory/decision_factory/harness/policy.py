"""The oracle policy: the decision the model is supposed to learn.

This is the whole reason the factory needs no LLM judge. The decision problem
is defined by a deterministic function from an observation to a tool call, so
every synthetic episode comes with an exactly-correct label and evaluation is
exact match rather than a rubric. A model is right or it is not.

The policy is written to be *readable*, in the order a support agent would
actually reason, and each branch is a rule someone could state in a sentence:

1. Damage or a safety issue is a human's problem, always.
2. A general policy question with no order attached goes to the help centre.
3. An order you have not looked up yet must be looked up first.
4. "Where is it" on a shipped order means check the carrier.
5. A refund request is auto-approved only if the order is delivered, inside
   its return window, and under the auto-approval limit.
6. Anything else about an order that you cannot act on goes to a human.
7. Otherwise, answer directly.

Rule 5 is the interesting one: it is a conjunction of three conditions, two of
them numeric thresholds. That is what makes the problem non-trivial for a
small model — it has to compare numbers, not just pattern-match keywords.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from .tools import FREE_TEXT, ToolCall

#: Refunds at or below this are auto-approved; above it a human decides.
AUTO_REFUND_LIMIT_CENTS = 10_000

#: Returns are accepted up to this many days after delivery.
RETURN_WINDOW_DAYS = 30


@dataclass(frozen=True)
class Observation:
    """What the agent can see when it has to decide.

    Flat and JSON-serializable on purpose: this is what gets rendered into the
    prompt, so anything the policy reads has to be visible to the model too.
    Otherwise the labels would depend on information the model never sees and
    the ceiling would be below 100% for reasons that look like model error.
    """

    customer_message: str
    #: Intent the message expresses. Part of the observation (not something
    #: the model must infer from scratch) so the task stays a *decision*
    #: problem rather than an intent-classification problem.
    intent: str  # refund | where_is_it | policy_question | damage | chitchat
    order_id: str | None = None
    #: Set once lookup_order has been called for this order.
    order_looked_up: bool = False
    order_status: str | None = None  # processing | shipped | delivered
    order_total_cents: int | None = None
    days_since_delivery: int | None = None
    tracking_number: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


INTENTS: tuple[str, ...] = ("refund", "where_is_it", "policy_question", "damage", "chitchat")

ORDER_STATUSES: tuple[str, ...] = ("processing", "shipped", "delivered")


def decide(obs: Observation) -> ToolCall:
    """The correct tool call for an observation. Total: never returns None."""
    # 1. Damage / safety: a human, regardless of anything else.
    if obs.intent == "damage":
        return ToolCall("escalate_to_human", {"reason": FREE_TEXT})

    # 2. A general question with no order attached.
    if obs.intent == "policy_question" and not obs.order_id:
        return ToolCall("search_faq", {"query": FREE_TEXT})

    # 3. An order in play that has not been looked up yet.
    if obs.order_id and not obs.order_looked_up:
        return ToolCall("lookup_order", {"order_id": obs.order_id})

    # 4. "Where is it" on something already shipped.
    if obs.intent == "where_is_it":
        if obs.order_status == "shipped" and obs.tracking_number:
            return ToolCall("check_shipping", {"tracking_number": obs.tracking_number})
        if obs.order_status == "delivered":
            return ToolCall("reply", {"message": FREE_TEXT})
        # Still processing (or shipped with no tracking yet): nothing to check.
        return ToolCall("reply", {"message": FREE_TEXT})

    # 5. Refund: three conditions, all of which must hold.
    if obs.intent == "refund":
        refundable = (
            obs.order_status == "delivered"
            and obs.days_since_delivery is not None
            and obs.days_since_delivery <= RETURN_WINDOW_DAYS
            and obs.order_total_cents is not None
            and obs.order_total_cents <= AUTO_REFUND_LIMIT_CENTS
        )
        if refundable:
            return ToolCall(
                "issue_refund",
                {"order_id": obs.order_id, "amount_cents": obs.order_total_cents},
            )
        return ToolCall("escalate_to_human", {"reason": FREE_TEXT})

    # 6. A policy question about a specific, looked-up order.
    if obs.intent == "policy_question":
        return ToolCall("search_faq", {"query": FREE_TEXT})

    # 7. Chitchat and anything else.
    return ToolCall("reply", {"message": FREE_TEXT})


def why(obs: Observation) -> str:
    """One-line rationale for the oracle's choice, for the data card.

    Not used as a training target — the models are trained to emit a call, not
    a justification — but it makes a sampled episode table readable by a human
    checking that the labels are sane.
    """
    if obs.intent == "damage":
        return "damage/safety always goes to a human"
    if obs.intent == "policy_question" and not obs.order_id:
        return "general question, no order attached"
    if obs.order_id and not obs.order_looked_up:
        return "order named but not yet looked up"
    if obs.intent == "where_is_it":
        if obs.order_status == "shipped" and obs.tracking_number:
            return "shipped with tracking, so check the carrier"
        if obs.order_status == "delivered":
            return "already delivered, so just tell them"
        return f"status {obs.order_status!r}, nothing to track yet"
    if obs.intent == "refund":
        reasons = []
        if obs.order_status != "delivered":
            reasons.append(f"not delivered ({obs.order_status})")
        if obs.days_since_delivery is not None and obs.days_since_delivery > RETURN_WINDOW_DAYS:
            reasons.append(f"{obs.days_since_delivery}d > {RETURN_WINDOW_DAYS}d window")
        if obs.order_total_cents is not None and obs.order_total_cents > AUTO_REFUND_LIMIT_CENTS:
            reasons.append(f"{obs.order_total_cents}c > {AUTO_REFUND_LIMIT_CENTS}c limit")
        return "; ".join(reasons) if reasons else "delivered, in window, under limit"
    if obs.intent == "policy_question":
        return "policy question about a known order"
    return "nothing to do but answer"


SYSTEM_PROMPT_HEADER = """You are the routing brain of a customer-support agent.
Given the current observation, choose exactly ONE tool to call next.

Tools:
{catalog}

Policy:
- Damage or safety issues always go to a human.
- An order mentioned by the customer must be looked up before you act on it.
- Refunds are auto-approved only when the order is delivered, no more than \
{window} days since delivery, and the total is at most {limit} cents. \
Otherwise escalate to a human.
- General questions that are not about one specific order go to the help centre.

Answer with JSON only, in the form {{"tool": "<name>", "args": {{...}}}}. \
No prose, no code fences."""


def system_prompt() -> str:
    """The system prompt shared by training, eval and the serving app."""
    from .tools import render_catalog

    return SYSTEM_PROMPT_HEADER.format(
        catalog=render_catalog(),
        window=RETURN_WINDOW_DAYS,
        limit=AUTO_REFUND_LIMIT_CENTS,
    )


def user_prompt(obs: Observation | Mapping[str, Any]) -> str:
    """The observation as the model sees it."""
    import json

    d = obs.to_dict() if isinstance(obs, Observation) else dict(obs)
    message = d.pop("customer_message", "")
    state = {k: v for k, v in d.items() if v is not None}
    return (
        f'Customer says: "{message}"\n'
        f"Observation: {json.dumps(state, sort_keys=True)}\n"
        "Which tool do you call?"
    )


def build_chat(obs: Observation | Mapping[str, Any]) -> list[dict[str, str]]:
    """The full chat the model is prompted with. One definition, used by
    training, eval and serving so the three cannot drift."""
    return [
        {"role": "system", "content": system_prompt()},
        {"role": "user", "content": user_prompt(obs)},
    ]
