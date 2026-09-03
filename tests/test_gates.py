"""HITL gate semantics: auto-approve bypass and timeout-means-reject."""

import asyncio
import types

import flyte
import pytest

from model_factory.shared.gates import gate


def test_auto_approve_never_creates_a_condition(monkeypatch):
    """CI/smoke runs must not park on a condition the cluster will never
    signal — auto_approve short-circuits before flyte is touched."""

    def boom(*a, **k):
        raise AssertionError("auto_approve gate reached flyte.new_condition")

    monkeypatch.setattr(flyte, "new_condition", types.SimpleNamespace(aio=boom), raising=False)
    assert asyncio.run(gate("g", "prompt", auto_approve=True)) is True


def _patched_condition(monkeypatch, wait):
    async def new_condition(name, **kwargs):
        return types.SimpleNamespace(wait=types.SimpleNamespace(aio=wait))

    monkeypatch.setattr(
        flyte, "new_condition", types.SimpleNamespace(aio=new_condition), raising=False
    )


@pytest.mark.parametrize("signal,expected", [(True, True), (False, False)])
def test_gate_returns_the_human_signal(monkeypatch, signal, expected):
    async def wait():
        return signal

    _patched_condition(monkeypatch, wait)
    assert asyncio.run(gate("g", "prompt", auto_approve=False)) is expected


def test_gate_timeout_means_reject(monkeypatch):
    """An unanswered gate must resolve to False, not crash the run."""

    async def wait():
        raise flyte.errors.ConditionTimedoutError("nobody answered")

    _patched_condition(monkeypatch, wait)
    assert asyncio.run(gate("g", "prompt", auto_approve=False)) is False
