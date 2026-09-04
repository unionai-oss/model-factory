"""Prompt rendering: the policy's input contract."""

from resource_tuner.policy.prompts import SYSTEM_PROMPT, render_messages

CODE = "import flyte\n\nasync def transform(): ..."


def test_messages_shape_and_content():
    msgs = render_messages(CODE, "input: 1M rows")
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[0]["content"] == SYSTEM_PROMPT
    assert CODE.rstrip() in msgs[1]["content"]
    assert "input: 1M rows" in msgs[1]["content"]


def test_system_prompt_states_the_output_contract():
    # The parser accepts only cpu/memory/gpu; the prompt must say so.
    for token in ('"cpu"', '"memory"', "gpu", "Mi", "Gi"):
        assert token in SYSTEM_PROMPT


def test_prior_and_history_render_only_when_given():
    bare = render_messages(CODE, "p")[1]["content"]
    assert "prior" not in bare and "Recent runs" not in bare

    rich = render_messages(
        CODE,
        "p",
        prior={"cpu": 1, "memory": "2Gi"},
        history=[{"resources": {"cpu": 1}, "peak": "900Mi", "ok": True}],
    )[1]["content"]
    assert "Author-declared prior" in rich and "2Gi" in rich
    assert "Recent runs" in rich and "success=True" in rich
