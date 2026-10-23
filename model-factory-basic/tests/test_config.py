"""Cluster/factory profile resolution and env-var propagation.

The trap these guard: child task specs are re-derived INSIDE the parent's
container, so any sizing knob that doesn't travel through
``cluster_env_vars()`` silently falls back to the default profile in the
child — the task then queues forever on an accelerator the tenant doesn't
have (see config.cluster_env_vars docstring).
"""

import importlib

import pytest

from model_factory import config


def _reload_with_env(monkeypatch, **env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    importlib.reload(config)
    return config


@pytest.fixture(autouse=True)
def _restore_config(monkeypatch):
    """Reload config with a clean environment after each test."""
    yield
    for k in list(config.cluster_env_vars()) + ["MF_USE_SECRETS", "MF_CLUSTER"]:
        monkeypatch.delenv(k, raising=False)
    importlib.reload(config)


def test_unknown_cluster_and_profile_fail_loudly():
    with pytest.raises(ValueError, match="unknown cluster"):
        config.get_cluster("nope")
    with pytest.raises(ValueError, match="unknown profile"):
        config.get_profile("nope")


def test_profile_registry_keys_match_profile_names():
    assert all(name == p.name for name, p in config.PROFILES.items())
    assert all(name == c.name for name, c in config.CLUSTERS.items())


def test_smoke_profile_never_blocks_promotion():
    # Smoke is the CI/E2E profile; a positive margin would make every smoke
    # run stall at the promotion gate.
    assert config.get_profile("smoke").promotion_margin < 0


def test_cluster_env_vars_cover_every_sizing_knob():
    """Every MF_* override the module reads for task sizing must propagate."""
    keys = set(config.cluster_env_vars())
    assert keys == {
        "MF_CLUSTER",
        "MF_GPU",
        "MF_INFERENCE_GPU",
        "MF_GPU_CPU",
        "MF_GPU_MEMORY",
        "MF_GPU_DISK",
        "MF_CPU",
        "MF_CPU_MEMORY",
        "MF_INFERENCE_CPU",
        "MF_INFERENCE_MEMORY",
        "MF_INFERENCE_DISK",
    }


def test_cluster_env_vars_round_trip(monkeypatch):
    """A container that re-imports config with these env vars resolves the
    exact same values the deploy did — including per-field overrides."""
    cfg = _reload_with_env(monkeypatch, MF_GPU="V100:4", MF_INFERENCE_GPU="T4:1")
    propagated = cfg.cluster_env_vars()

    cfg2 = _reload_with_env(monkeypatch, **propagated)
    assert cfg2.GPU == "V100:4"
    assert cfg2.cluster_env_vars() == propagated


def test_env_overrides_reach_resources(monkeypatch):
    cfg = _reload_with_env(
        monkeypatch, MF_INFERENCE_CPU="4", MF_INFERENCE_MEMORY="12Gi", MF_INFERENCE_GPU="L4:1"
    )
    res = cfg.inference_resources()
    assert res.cpu == 4
    assert res.memory == "12Gi"
    assert res.gpu == "L4:1"


def test_serving_sized_for_serving_pool_not_training_pool():
    """Regression: the serving app once requested training-node sizing on the
    small serving instance — an unschedulable request, and when it did fit
    (same accelerator as training) it starved every GPU stage."""
    demo = config.get_cluster("demo")
    assert demo.inference_gpu != demo.gpu
    assert demo.inference_task_cpu < demo.gpu_task_cpu

    inf, gpu = config.inference_resources(), config.gpu_resources()
    assert (inf.cpu, inf.memory, inf.gpu) != (gpu.cpu, gpu.memory, gpu.gpu)


def test_wandb_follows_secrets_flag(monkeypatch):
    cfg = _reload_with_env(monkeypatch, MF_USE_SECRETS="1")
    assert cfg.get_profile("smoke").wandb_enabled
    cfg = _reload_with_env(monkeypatch, MF_USE_SECRETS="0")
    assert not cfg.get_profile("smoke").wandb_enabled
