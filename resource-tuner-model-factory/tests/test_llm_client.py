"""Teacher endpoint resolution."""

import pytest

from resource_tuner.shared.llm_client import TeacherError, resolve_teacher


def test_named_teachers_resolve_to_internal_service_dns():
    # In-cluster callers must use svc DNS: the public app URL sits behind
    # the OIDC gateway and answers pods with a login redirect.
    url = resolve_teacher("qwen38-27b")
    assert url == "http://qwen38-27b.llm-service-development.svc.cluster.local"
    assert resolve_teacher("glm-5-2").startswith("http://glm-5-2.")


def test_default_and_explicit_url_pass_through():
    assert resolve_teacher(None).startswith("http://qwen38-27b.")
    assert resolve_teacher("http://localhost:8080/") == "http://localhost:8080"


def test_env_override_wins(monkeypatch):
    monkeypatch.setenv("RT_TEACHER_URL", "http://tunnel:9999/")
    assert resolve_teacher("glm-5-2") == "http://tunnel:9999"


def test_unknown_teacher_fails_loudly():
    with pytest.raises(TeacherError, match="unknown teacher"):
        resolve_teacher("gpt-99")
