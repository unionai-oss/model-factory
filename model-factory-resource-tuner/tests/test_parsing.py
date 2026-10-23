"""Completion parsing: lenient wrapping, strict content."""

from resource_tuner.policy.actions import Proposal
from resource_tuner.policy.parsing import try_extract_proposal


def test_bare_json():
    assert try_extract_proposal('{"cpu": 2, "memory": "4Gi"}') == Proposal(2, 4096)


def test_fenced_and_prosed_json():
    fenced = 'Sure! Here you go:\n```json\n{"cpu": 1, "memory": "512Mi"}\n```\nDone.'
    assert try_extract_proposal(fenced) == Proposal(1, 512)


def test_think_block_is_ignored():
    # Qwen3-family models emit <think> even when told not to; the JSON
    # inside reasoning must not be mistaken for the answer.
    text = (
        '<think>maybe {"cpu": 16, "memory": "64Gi"}? no, smaller.</think>\n'
        '{"cpu": 1, "memory": "1Gi"}'
    )
    assert try_extract_proposal(text) == Proposal(1, 1024)


def test_unclosed_think_block_yields_nothing():
    assert try_extract_proposal('<think>{"cpu": 1, "memory": "1Gi"}') is None


def test_invalid_content_is_not_laundered():
    assert try_extract_proposal('{"cpu": 1, "memory": "1Gi", "disk": "9Gi"}') is None
    assert try_extract_proposal("no json here") is None
    assert try_extract_proposal('{"cpu": 1}') is None
