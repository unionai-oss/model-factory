"""Parsing model output into a tool call, and grading it.

The parser is the part most likely to flatter a model (by rescuing malformed
output) or libel one (by failing to find a call that is right there), so both
directions are pinned here.
"""

from decision_factory.harness.scoring import (
    grade_one,
    majority_baseline,
    parse_call,
    summarize,
)
from decision_factory.harness.tools import FREE_TEXT, ToolCall


def test_parses_a_bare_json_call():
    call = parse_call('{"tool": "lookup_order", "args": {"order_id": "A1"}}')
    assert call == ToolCall("lookup_order", {"order_id": "A1"})


def test_parses_a_fenced_call():
    raw = 'Sure!\n```json\n{"tool": "reply", "args": {"message": "hi"}}\n```\n'
    assert parse_call(raw).tool == "reply"


def test_parses_a_call_embedded_in_prose():
    raw = 'I think we should call {"tool": "search_faq", "args": {"query": "returns"}} here.'
    assert parse_call(raw).tool == "search_faq"


def test_nested_args_survive_the_brace_scan():
    # A "first { to first }" slice truncates this; `args` is itself an object,
    # so the scan has to balance braces.
    raw = 'blah {"tool": "issue_refund", "args": {"order_id": "A1", "amount_cents": 500}} blah'
    call = parse_call(raw)
    assert call.args == {"order_id": "A1", "amount_cents": 500}


def test_rejects_a_tool_outside_the_catalog():
    # Hallucinating a plausible-but-nonexistent tool is a wrong decision, not
    # a parse to be salvaged.
    assert parse_call('{"tool": "delete_account", "args": {}}') is None


def test_rejects_non_json_and_empty_output():
    assert parse_call("I would look up the order.") is None
    assert parse_call("") is None
    assert parse_call("{not json}") is None


def test_missing_args_parse_as_empty_not_as_a_failure():
    call = parse_call('{"tool": "reply"}')
    assert call == ToolCall("reply", {})


def test_unparseable_output_scores_zero_without_raising():
    g = grade_one("ep1", ToolCall("reply", {"message": FREE_TEXT}).to_json(), "no idea")
    assert (g.json_valid, g.tool_correct, g.exact_correct) == (False, False, False)
    assert g.predicted_tool is None


def test_right_tool_wrong_args_is_not_an_exact_hit():
    expected = ToolCall("issue_refund", {"order_id": "A1", "amount_cents": 500}).to_json()
    g = grade_one("ep1", expected, '{"tool": "issue_refund", "args": {"order_id": "A1", "amount_cents": 999}}')
    assert g.tool_correct is True
    assert g.exact_correct is False


def test_numeric_arguments_compare_across_string_and_int():
    # A model that emits "500" instead of 500 made the right decision.
    expected = ToolCall("issue_refund", {"order_id": "A1", "amount_cents": 500}).to_json()
    g = grade_one("ep1", expected, '{"tool": "issue_refund", "args": {"order_id": "A1", "amount_cents": "500"}}')
    assert g.exact_correct is True


def test_free_text_arguments_accept_any_nonempty_string():
    expected = ToolCall("escalate_to_human", {"reason": FREE_TEXT}).to_json()
    ok = grade_one("ep1", expected, '{"tool": "escalate_to_human", "args": {"reason": "over limit"}}')
    assert ok.exact_correct is True
    # ...but not an empty one: that is a malformed call, not a phrasing choice.
    blank = grade_one("ep2", expected, '{"tool": "escalate_to_human", "args": {"reason": "  "}}')
    assert blank.exact_correct is False


def test_wrong_tool_never_counts_as_exact_even_with_matching_args():
    expected = ToolCall("reply", {"message": FREE_TEXT}).to_json()
    g = grade_one("ep1", expected, '{"tool": "search_faq", "args": {"query": "hello"}}')
    assert g.exact_correct is False


def _grade(expected_tool, predicted, hard=False, args=None):
    expected = ToolCall(expected_tool, args or {"message": FREE_TEXT}).to_json()
    return grade_one("ep", expected, predicted, hard)


def test_summarize_separates_format_from_decision():
    grades = [
        _grade("reply", '{"tool": "reply", "args": {"message": "hi"}}'),
        _grade("reply", "garbage"),
        _grade("reply", '{"tool": "search_faq", "args": {"query": "x"}}'),
        _grade("reply", '{"tool": "reply", "args": {"message": "yo"}}'),
    ]
    s = summarize(grades)
    assert s["n"] == 4
    assert s["json_valid_rate"] == 0.75  # 3 of 4 produced a parseable call
    assert s["tool_accuracy"] == 0.5  # 2 of 4 picked the right tool
    assert s["exact_accuracy"] == 0.5


def test_summarize_reports_hard_subset_separately():
    grades = [
        _grade("reply", '{"tool": "reply", "args": {"message": "a"}}', hard=True),
        _grade("reply", "garbage", hard=True),
        _grade("reply", '{"tool": "reply", "args": {"message": "b"}}', hard=False),
    ]
    s = summarize(grades)
    assert s["n_hard"] == 2
    assert s["hard_exact_accuracy"] == 0.5
    assert s["exact_accuracy"] > s["hard_exact_accuracy"]


def test_summarize_names_the_most_common_confusion():
    grades = [_grade("reply", '{"tool": "search_faq", "args": {"query": "x"}}') for _ in range(3)]
    s = summarize(grades)
    assert s["confusion"]["reply->search_faq"] == 3


def test_summarize_handles_an_empty_split():
    s = summarize([])
    assert s["n"] == 0 and s["exact_accuracy"] == 0.0


def test_per_tool_covers_the_whole_catalog():
    # A model that stops emitting a tool entirely should show up as a 0-row,
    # not vanish from the table.
    from decision_factory.harness.tools import TOOL_NAMES

    s = summarize([_grade("reply", '{"tool": "reply", "args": {"message": "a"}}')])
    assert set(s["per_tool"]) == set(TOOL_NAMES)
    assert s["per_tool"]["issue_refund"]["n"] == 0


def test_majority_baseline_is_the_floor_to_beat():
    labels = ["reply", "reply", "reply", "search_faq"]
    base = majority_baseline(labels)
    assert base["tool"] == "reply"
    assert base["accuracy"] == 0.75
    assert majority_baseline([])["accuracy"] == 0.0
