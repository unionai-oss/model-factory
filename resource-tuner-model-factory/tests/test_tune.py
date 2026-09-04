"""@tune.resources: interception, override, and the DEGRADED fallback."""

import asyncio

import pytest

from resource_tuner import tune


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("RT_TUNE_URL", raising=False)


class FakeTask:
    """Duck-typed flyte task: records how it was invoked."""

    def __init__(self):
        self.calls = []
        self.name = "fake.task"

    @property
    def func(self):
        def body():  # pragma: no cover - source only
            return 1

        return body

    async def __call__(self, *a, **k):
        self.calls.append(("plain", None))
        return "ran"

    def override(self, resources=None):
        outer = self

        class Overridden:
            async def __call__(self, *a, **k):
                outer.calls.append(("override", resources))
                return "ran-tuned"

        return Overridden()


def test_service_url_defaults_to_svc_dns_and_env_overrides(monkeypatch):
    assert tune.service_url().startswith("http://rt-tune.resource-tuner-model-factory")
    monkeypatch.setenv("RT_TUNE_URL", "http://localhost:9000/")
    assert tune.service_url() == "http://localhost:9000"


def test_tuned_invocation_overrides_resources(monkeypatch):
    task = FakeTask()
    tuned = tune.resources(task)
    monkeypatch.setattr(
        tune, "request_proposal", lambda *a, **k: {"cpu": 1, "memory": "512Mi"}
    )
    out = asyncio.run(tuned())
    assert out == "ran-tuned"
    kind, res = task.calls[0]
    assert kind == "override"
    assert res.memory == "512Mi"


def test_service_failure_degrades_to_the_prior(monkeypatch):
    """Tuning may slow a call; it must never break one."""
    task = FakeTask()
    tuned = tune.resources(task)
    monkeypatch.setattr(tune, "request_proposal", lambda *a, **k: None)
    assert asyncio.run(tuned()) == "ran"
    assert task.calls[0][0] == "plain"


def test_disabled_decorator_is_a_passthrough(monkeypatch):
    task = FakeTask()
    tuned = tune.resources(enabled=False)(task)
    monkeypatch.setattr(
        tune, "request_proposal",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not be called")),
    )
    assert asyncio.run(tuned()) == "ran"


def test_explicit_override_outbids_tuning():
    task = FakeTask()
    tuned = tune.resources(task)
    assert asyncio.run(tuned.override(resources="explicit")()) == "ran-tuned"
    assert task.calls[0] == ("override", "explicit")
