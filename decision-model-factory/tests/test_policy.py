"""The oracle policy. These tests ARE the specification of the task.

If the policy is wrong, every label in every dataset is wrong and the models
learn the bug — so the rules get stated here independently of the
implementation, especially the threshold edges.
"""

import pytest

from decision_factory.harness.policy import (
    AUTO_REFUND_LIMIT_CENTS,
    RETURN_WINDOW_DAYS,
    Observation,
    build_chat,
    decide,
    system_prompt,
)
from decision_factory.harness.tools import TOOL_NAMES


def delivered(**kw) -> Observation:
    """A looked-up, delivered order asking for a refund; override as needed."""
    base = dict(
        customer_message="refund please",
        intent="refund",
        order_id="A12345",
        order_looked_up=True,
        order_status="delivered",
        order_total_cents=5_000,
        days_since_delivery=5,
    )
    base.update(kw)
    return Observation(**base)


def test_damage_always_escalates_even_when_refundable():
    # Rule 1 outranks the refund rule: a cheap, in-window, delivered order
    # still goes to a human if the customer reports damage.
    obs = delivered(intent="damage", customer_message="arrived smashed")
    assert decide(obs).tool == "escalate_to_human"


def test_an_unlooked_up_order_is_looked_up_first():
    obs = delivered(order_looked_up=False)
    call = decide(obs)
    assert call.tool == "lookup_order"
    assert call.args["order_id"] == "A12345"


def test_refund_is_approved_when_all_three_conditions_hold():
    call = decide(delivered(order_total_cents=4_200, days_since_delivery=3))
    assert call.tool == "issue_refund"
    assert call.args == {"order_id": "A12345", "amount_cents": 4_200}


@pytest.mark.parametrize(
    "kw, why",
    [
        ({"order_status": "shipped"}, "not delivered"),
        ({"order_total_cents": AUTO_REFUND_LIMIT_CENTS + 1}, "over the limit"),
        ({"days_since_delivery": RETURN_WINDOW_DAYS + 1}, "outside the window"),
    ],
)
def test_refund_escalates_when_any_condition_fails(kw, why):
    assert decide(delivered(**kw)).tool == "escalate_to_human", why


def test_the_limit_and_window_are_inclusive_boundaries():
    # The exact threshold is ALLOWED. This is the single most likely place for
    # an off-by-one to silently flip ~half the near-boundary episodes.
    at_limit = delivered(
        order_total_cents=AUTO_REFUND_LIMIT_CENTS, days_since_delivery=RETURN_WINDOW_DAYS
    )
    assert decide(at_limit).tool == "issue_refund"
    just_over_amount = delivered(order_total_cents=AUTO_REFUND_LIMIT_CENTS + 1)
    just_over_days = delivered(days_since_delivery=RETURN_WINDOW_DAYS + 1)
    assert decide(just_over_amount).tool == "escalate_to_human"
    assert decide(just_over_days).tool == "escalate_to_human"


def test_where_is_it_checks_shipping_only_with_tracking():
    shipped = delivered(
        intent="where_is_it", order_status="shipped", tracking_number="1Z999"
    )
    call = decide(shipped)
    assert call.tool == "check_shipping"
    assert call.args["tracking_number"] == "1Z999"

    # Shipped but no tracking yet: there is nothing to look up, so answer.
    no_tracking = delivered(intent="where_is_it", order_status="shipped")
    assert decide(no_tracking).tool == "reply"


def test_where_is_it_on_a_delivered_order_just_replies():
    assert decide(delivered(intent="where_is_it")).tool == "reply"


def test_a_general_question_with_no_order_goes_to_the_faq():
    obs = Observation(customer_message="what's your return policy?", intent="policy_question")
    assert decide(obs).tool == "search_faq"


def test_a_policy_question_about_a_known_order_still_goes_to_the_faq():
    assert decide(delivered(intent="policy_question")).tool == "search_faq"


def test_chitchat_replies():
    obs = Observation(customer_message="thanks!", intent="chitchat")
    assert decide(obs).tool == "reply"


def test_decide_is_total_over_the_intent_space():
    # Every intent must produce a call from the catalog; a None or an unknown
    # tool would mean unlabelled training data.
    from decision_factory.harness.policy import INTENTS, ORDER_STATUSES

    for intent in INTENTS:
        for status in (None, *ORDER_STATUSES):
            for looked_up in (False, True):
                obs = Observation(
                    customer_message="x",
                    intent=intent,
                    order_id="A1" if status else None,
                    order_looked_up=looked_up,
                    order_status=status,
                    order_total_cents=1_000 if status else None,
                    days_since_delivery=1 if status == "delivered" else None,
                )
                call = decide(obs)
                assert call.tool in TOOL_NAMES


def test_the_prompt_tells_the_model_the_thresholds_it_is_graded_on():
    # The policy is learnable only if the numbers it turns on are visible.
    prompt = system_prompt()
    assert str(AUTO_REFUND_LIMIT_CENTS) in prompt
    assert str(RETURN_WINDOW_DAYS) in prompt
    for name in TOOL_NAMES:
        assert name in prompt


def test_the_chat_shows_the_model_everything_the_policy_reads():
    # A field the oracle branches on but the prompt hides would put the
    # accuracy ceiling below 100% for reasons that look like model error.
    obs = delivered(order_total_cents=4_242, days_since_delivery=7)
    rendered = " ".join(m["content"] for m in build_chat(obs))
    assert "4242" in rendered
    assert "delivered" in rendered
    assert "A12345" in rendered
