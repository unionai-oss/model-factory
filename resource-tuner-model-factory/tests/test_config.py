"""Config: profile resolution, env propagation, and the model ladder."""

import importlib

import pytest

from resource_tuner import config


@pytest.fixture(autouse=True)
def _restore_config(monkeypatch):
    yield
    for k in list(config.cluster_env_vars()) + ["RT_MODEL", "RT_USE_SECRETS"]:
        monkeypatch.delenv(k, raising=False)
    importlib.reload(config)


def test_unknown_profile_fails_loudly():
    with pytest.raises(ValueError, match="unknown profile"):
        config.get_profile("nope")


def test_profile_registry_names_match():
    assert all(name == p.name for name, p in config.PROFILES.items())


def test_smoke_profile_is_the_fundamentals_first_rung():
    smoke = config.get_profile("smoke")
    # Stage A only: prove reward goes up before composite rewards exist.
    assert smoke.reward_stage == "success"
    assert smoke.cluster_episode_fraction == 0.0  # sim-only training
    assert smoke.eval_cluster_episodes > 0  # but real validation episodes


def test_composite_profiles_use_stage_b():
    assert config.get_profile("dev").reward_stage == "composite"
    assert config.get_profile("full").reward_stage == "composite"


def test_default_model_is_text_only():
    """Qwen3.5 is multimodal-only and TRL GRPO fails on the arch (trl#5269);
    until that closes the DEFAULT must be a text-only base."""
    assert "3.5" not in config.DEFAULT_MODEL
    # ...but the requested Qwen3.5 ladder stays wired for the swap.
    assert any("Qwen3.5" in m for m in config.MODEL_LADDER.values())


def test_cluster_env_vars_round_trip(monkeypatch):
    """A container re-importing config with these env vars resolves the
    same values the deploy did (children re-derive specs at runtime)."""
    monkeypatch.setenv("RT_TRAIN_GPU", "A10G:1")
    monkeypatch.setenv("RT_MODEL", "Qwen/Qwen3-0.6B")
    importlib.reload(config)
    propagated = config.cluster_env_vars()
    assert propagated["RT_TRAIN_GPU"] == "A10G:1"

    for k, v in propagated.items():
        monkeypatch.setenv(k, v)
    importlib.reload(config)
    assert config.TRAIN_GPU == "A10G:1"
    assert config.cluster_env_vars() == propagated


def test_train_resources_reflect_env(monkeypatch):
    monkeypatch.setenv("RT_TRAIN_GPU", "A10G:1")
    monkeypatch.setenv("RT_TRAIN_MEMORY", "24Gi")
    importlib.reload(config)
    res = config.train_resources()
    assert res.gpu == "A10G:1" and res.memory == "24Gi"
