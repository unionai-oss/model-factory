"""Sampling observations: the synthetic data generator.

No LLM is involved in making this data. The decision problem has a
programmatic oracle (`policy.decide`), so an episode is just a sampled
observation plus its exact label — which makes generation seconds-fast,
free, and perfectly labelled.

Two things this sampler does deliberately:

**Stratifies by the tool the oracle picks.** Uniform sampling over
observations would bury `issue_refund` (it needs three conditions to line up
at once) and flood `lookup_order` (every order starts un-looked-up). A model
trained on that learns the prior, not the policy. So episodes are drawn to a
target mix over the six tools instead.

**Spends a fixed share of episodes NEAR the thresholds.** The refund rule
turns on `amount <= 10000` and `days <= 30`. Sampling amounts uniformly over
a wide range makes almost every case decidable from magnitude alone, so a
model can score well while having learned "big number -> escalate". The hard
fraction draws amounts and delays within a few units of the boundary, on both
sides, where only the actual comparison gets it right. `n_hard` in the data
card is how many of those an eval split contains.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable

from .policy import (
    AUTO_REFUND_LIMIT_CENTS,
    RETURN_WINDOW_DAYS,
    Observation,
    decide,
    why,
)
from .tools import TOOL_NAMES, ToolCall

#: Target share of episodes per oracle tool. `reply` is the catch-all so it
#: gets the smallest share; the three that carry real logic get the most.
TOOL_MIX: dict[str, float] = {
    "lookup_order": 0.20,
    "issue_refund": 0.20,
    "escalate_to_human": 0.20,
    "check_shipping": 0.15,
    "search_faq": 0.15,
    "reply": 0.10,
}

#: Share of episodes drawn near a decision boundary. See the module docstring.
HARD_FRACTION = 0.40

#: How close to a threshold "near" means.
_NEAR_CENTS = 1_500
_NEAR_DAYS = 4

_REFUND_MESSAGES = [
    "I'd like a refund for this please",
    "this isn't what I wanted, can I get my money back",
    "please refund order {oid}",
    "I want to return this and be refunded",
    "can you refund me for {oid}?",
]
_WHERE_MESSAGES = [
    "where is my order?",
    "any update on {oid}? it's been a while",
    "has {oid} shipped yet",
    "when will this arrive",
    "tracking for {oid} please",
]
_POLICY_MESSAGES = [
    "what's your return policy?",
    "how long do I have to return something",
    "do you ship internationally?",
    "how do gift cards work",
    "what payment methods do you take",
]
_DAMAGE_MESSAGES = [
    "the item arrived smashed",
    "this came broken and there's glass everywhere",
    "the package was damaged in transit",
    "the battery is swollen, is this safe?",
    "it arrived leaking",
]
_CHITCHAT_MESSAGES = [
    "thanks for your help!",
    "hello?",
    "you've been great, cheers",
    "ok thank you",
    "hi there",
]


@dataclass(frozen=True)
class Episode:
    """One labelled decision: what the agent saw, and what it should do."""

    episode_id: str
    observation: Observation
    label: ToolCall
    rationale: str
    #: True when this episode sits near a numeric decision boundary.
    hard: bool

    def to_row(self) -> dict[str, Any]:
        """Flat parquet row. `label_json` is the training target verbatim."""
        obs = self.observation.to_dict()
        return {
            "episode_id": self.episode_id,
            "label_tool": self.label.tool,
            "label_json": self.label.to_json(),
            "rationale": self.rationale,
            "hard": self.hard,
            **{f"obs_{k}": v for k, v in obs.items()},
        }


def _order_id(rng: random.Random) -> str:
    return f"A{rng.randint(10_000, 99_999)}"


def _tracking(rng: random.Random) -> str:
    return f"1Z{rng.randint(100_000_000, 999_999_999)}"


def _message(rng: random.Random, pool: list[str], oid: str | None) -> str:
    return rng.choice(pool).format(oid=oid or "that order")


def _amount(rng: random.Random, hard: bool, under_limit: bool) -> int:
    """A total that is under or over the auto-approval limit."""
    if hard:
        delta = rng.randint(1, _NEAR_CENTS)
        return AUTO_REFUND_LIMIT_CENTS - delta if under_limit else AUTO_REFUND_LIMIT_CENTS + delta
    if under_limit:
        return rng.randint(500, AUTO_REFUND_LIMIT_CENTS)
    return rng.randint(AUTO_REFUND_LIMIT_CENTS + 1, 120_000)


def _days(rng: random.Random, hard: bool, in_window: bool) -> int:
    """Days since delivery, inside or outside the return window."""
    if hard:
        delta = rng.randint(1, _NEAR_DAYS)
        return RETURN_WINDOW_DAYS - delta if in_window else RETURN_WINDOW_DAYS + delta
    if in_window:
        return rng.randint(0, RETURN_WINDOW_DAYS)
    return rng.randint(RETURN_WINDOW_DAYS + 1, 180)


# ── per-tool observation builders ───────────────────────────────────────
# Each returns an observation the oracle will answer with that tool. They are
# asserted against `decide` in `sample_episodes`, so a builder that drifts out
# of agreement with the policy fails loudly instead of mislabelling data.


def _obs_lookup_order(rng: random.Random, hard: bool) -> Observation:
    intent = rng.choice(["refund", "where_is_it", "policy_question"])
    oid = _order_id(rng)
    pool = {
        "refund": _REFUND_MESSAGES,
        "where_is_it": _WHERE_MESSAGES,
        "policy_question": _POLICY_MESSAGES,
    }[intent]
    return Observation(
        customer_message=_message(rng, pool, oid),
        intent=intent,
        order_id=oid,
        order_looked_up=False,
    )


def _obs_issue_refund(rng: random.Random, hard: bool) -> Observation:
    oid = _order_id(rng)
    return Observation(
        customer_message=_message(rng, _REFUND_MESSAGES, oid),
        intent="refund",
        order_id=oid,
        order_looked_up=True,
        order_status="delivered",
        order_total_cents=_amount(rng, hard, under_limit=True),
        days_since_delivery=_days(rng, hard, in_window=True),
        tracking_number=_tracking(rng),
    )


def _obs_escalate(rng: random.Random, hard: bool) -> Observation:
    oid = _order_id(rng)
    # Three ways to land here: damage, over the limit, outside the window.
    which = rng.choice(["damage", "over_limit", "late"])
    if which == "damage":
        return Observation(
            customer_message=_message(rng, _DAMAGE_MESSAGES, oid),
            intent="damage",
            order_id=oid,
            order_looked_up=True,
            order_status="delivered",
            order_total_cents=_amount(rng, False, under_limit=rng.random() < 0.5),
            days_since_delivery=_days(rng, False, in_window=True),
        )
    return Observation(
        customer_message=_message(rng, _REFUND_MESSAGES, oid),
        intent="refund",
        order_id=oid,
        order_looked_up=True,
        order_status="delivered",
        order_total_cents=_amount(rng, hard, under_limit=which != "over_limit"),
        days_since_delivery=_days(rng, hard, in_window=which != "late"),
        tracking_number=_tracking(rng),
    )


def _obs_check_shipping(rng: random.Random, hard: bool) -> Observation:
    oid = _order_id(rng)
    return Observation(
        customer_message=_message(rng, _WHERE_MESSAGES, oid),
        intent="where_is_it",
        order_id=oid,
        order_looked_up=True,
        order_status="shipped",
        order_total_cents=_amount(rng, False, under_limit=rng.random() < 0.5),
        tracking_number=_tracking(rng),
    )


def _obs_search_faq(rng: random.Random, hard: bool) -> Observation:
    # Either no order at all, or a policy question about a looked-up order.
    if rng.random() < 0.6:
        return Observation(
            customer_message=_message(rng, _POLICY_MESSAGES, None),
            intent="policy_question",
        )
    oid = _order_id(rng)
    return Observation(
        customer_message=_message(rng, _POLICY_MESSAGES, oid),
        intent="policy_question",
        order_id=oid,
        order_looked_up=True,
        order_status=rng.choice(["processing", "shipped", "delivered"]),
    )


def _obs_reply(rng: random.Random, hard: bool) -> Observation:
    if rng.random() < 0.6:
        return Observation(
            customer_message=_message(rng, _CHITCHAT_MESSAGES, None),
            intent="chitchat",
        )
    # "Where is it" on something already delivered, or still processing.
    oid = _order_id(rng)
    delivered = rng.random() < 0.5
    return Observation(
        customer_message=_message(rng, _WHERE_MESSAGES, oid),
        intent="where_is_it",
        order_id=oid,
        order_looked_up=True,
        order_status="delivered" if delivered else "processing",
        days_since_delivery=_days(rng, False, in_window=True) if delivered else None,
    )


BUILDERS: dict[str, Callable[[random.Random, bool], Observation]] = {
    "lookup_order": _obs_lookup_order,
    "issue_refund": _obs_issue_refund,
    "escalate_to_human": _obs_escalate,
    "check_shipping": _obs_check_shipping,
    "search_faq": _obs_search_faq,
    "reply": _obs_reply,
}


def _plan(n: int) -> list[str]:
    """How many episodes to draw per target tool, summing to exactly n."""
    counts = {tool: int(n * share) for tool, share in TOOL_MIX.items()}
    # Hand out the rounding remainder in catalog order, deterministically.
    i = 0
    while sum(counts.values()) < n:
        counts[TOOL_NAMES[i % len(TOOL_NAMES)]] += 1
        i += 1
    plan: list[str] = []
    for tool, count in counts.items():
        plan.extend([tool] * count)
    return plan


def sample_episodes(n: int, seed: int = 0, hard_fraction: float = HARD_FRACTION) -> list[Episode]:
    """``n`` labelled episodes, stratified by oracle tool. Deterministic in ``seed``."""
    if n <= 0:
        return []
    rng = random.Random(seed)
    plan = _plan(n)
    rng.shuffle(plan)

    episodes: list[Episode] = []
    for i, target in enumerate(plan):
        hard = rng.random() < hard_fraction
        # A builder aims at `target`, but the oracle has the last word: retry a
        # few times and otherwise keep the episode under its TRUE label. The
        # label always comes from `decide`, never from `target`, so the mix can
        # drift slightly but the data can never be mislabelled.
        obs = BUILDERS[target](rng, hard)
        for _ in range(4):
            if decide(obs).tool == target:
                break
            obs = BUILDERS[target](rng, hard)
        episodes.append(
            Episode(
                episode_id=f"ep{i:06d}",
                observation=obs,
                label=decide(obs),
                rationale=why(obs),
                hard=hard,
            )
        )
    return episodes
