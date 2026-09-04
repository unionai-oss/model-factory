"""Teacher endpoint resolution + auth headers."""

import pytest

from resource_tuner.shared.llm_client import (
    TeacherError,
    _headers,
    resolve_teacher,
    resolve_teacher_candidates,
)


@pytest.fixture(autouse=True)
def _no_ambient_env(monkeypatch):
    monkeypatch.delenv("LLM_SERVICE_API_KEY", raising=False)
    monkeypatch.delenv("RT_TEACHER_URL", raising=False)
    monkeypatch.delenv("KUBERNETES_SERVICE_HOST", raising=False)


def test_in_cluster_prefers_svc_dns_with_public_fallback(monkeypatch):
    """Task pods get 403 at the public app gateway regardless of bearer —
    in-cluster must try svc DNS first, public+key second."""
    monkeypatch.setenv("KUBERNETES_SERVICE_HOST", "10.0.0.1")
    monkeypatch.setenv("LLM_SERVICE_API_KEY", "sekrit")
    urls = resolve_teacher_candidates("qwen38-27b")
    assert urls[0].startswith("http://qwen38-27b.llm-service-development.svc")
    assert urls[1].startswith("https://qwen38-27b-llm-service-development.apps")


def test_keyless_resolution_uses_internal_service_dns():
    # Without the API key, in-cluster svc DNS is the only reachable path
    # (the public app URL is OIDC-gated).
    url = resolve_teacher("qwen38-27b")
    assert url == "http://qwen38-27b.llm-service-development.svc.cluster.local"


def test_api_key_switches_to_public_endpoint_with_bearer(monkeypatch):
    monkeypatch.setenv("LLM_SERVICE_API_KEY", "sekrit")
    url = resolve_teacher("qwen38-27b")
    assert url == "https://qwen38-27b-llm-service-development.apps.demo.hosted.unionai.cloud"
    assert resolve_teacher("glm-5-2").startswith("https://glm-5-2-")
    assert _headers()["Authorization"] == "Bearer sekrit"


def test_no_key_means_no_auth_header():
    assert "Authorization" not in _headers()


def test_default_and_explicit_url_pass_through():
    assert resolve_teacher(None).startswith("http://qwen38-27b.")
    assert resolve_teacher("http://localhost:8080/") == "http://localhost:8080"


def test_env_override_wins_even_over_api_key(monkeypatch):
    monkeypatch.setenv("LLM_SERVICE_API_KEY", "sekrit")
    monkeypatch.setenv("RT_TEACHER_URL", "http://tunnel:9999/")
    assert resolve_teacher("glm-5-2") == "http://tunnel:9999"


def test_unknown_teacher_fails_loudly():
    with pytest.raises(TeacherError, match="unknown teacher"):
        resolve_teacher("gpt-99")
