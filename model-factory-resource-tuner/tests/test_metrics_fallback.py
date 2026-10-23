"""Pod-metrics wrapper: graceful degradation when the private plugin is
absent (the default in CI and plugin-less deploys)."""

import asyncio

from resource_tuner.environment import metrics


def test_peak_helper():
    class Series:
        def __init__(self, values):
            self.values = values

    assert metrics._peak([Series([(0, 1.0), (1, 3.0)]), Series([(0, 2.0)])]) == 3.0
    assert metrics._peak([]) is None
    assert metrics._peak([Series([])]) is None


def test_missing_plugin_degrades_not_raises(monkeypatch):
    """Without flyteplugins-union, callers must get a no-data answer, never
    an ImportError — episode scoring falls back to harness rusage."""
    import builtins

    real_import = builtins.__import__

    def block_plugin(name, *args, **kwargs):
        if name.startswith("flyteplugins"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", block_plugin)
    assert metrics.metrics_available() is False
    out = asyncio.run(metrics.pod_peaks_for_action("run-x"))
    assert out["peak_memory_mib"] is None and out["peak_cpu"] is None
    assert "not installed" in out["error"]
