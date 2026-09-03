"""Lineage app helpers: console-link building and run→station attribution.

Companion to test_assets_source.py: even after assets stops leaking display
strings, the lineage page must refuse to build a console URL from one — a
missing link beats a link that 404s.
"""

import types

from model_factory.contracts import ARTIFACT_CHECKPOINT, ARTIFACT_RL_DATASET
from model_factory.lineage_app import _run_url, _station_for_task


def _fake_client(monkeypatch):
    import flyte._initialize

    def run_url(project, domain, run_name):
        return f"https://console.example.com/{project}/{domain}/runs/{run_name}"

    monkeypatch.setattr(
        flyte._initialize,
        "get_client",
        lambda: types.SimpleNamespace(console=types.SimpleNamespace(run_url=run_url)),
    )


def test_run_url_builds_a_link_for_a_bare_identifier(monkeypatch):
    _fake_client(monkeypatch)
    url = _run_url("model-factory", "development", "uwbwvdrsf2gzj27gmvgp")
    assert url.endswith("/runs/uwbwvdrsf2gzj27gmvgp")


def test_run_url_refuses_display_strings(monkeypatch):
    """`Artifact.source` renders as "run <run>/<action> (attempt 1)"; if that
    leaks in here it must produce no link, not an urlencoded 404."""
    _fake_client(monkeypatch)
    assert _run_url("p", "d", "") == ""
    assert _run_url("p", "d", "run uwbw/5jow (attempt 1)") == ""
    assert _run_url("p", "d", "uwbw/5jow") == ""


def test_run_url_swallows_a_missing_client():
    # Outside a flyte context get_client() raises; the page renders unlinked.
    assert _run_url("p", "d", "uwbwvdrsf2gzj27gmvgp") == ""


def test_station_for_task_maps_producers_to_artifacts():
    assert _station_for_task("mf-cpu.publish_dataset") == ARTIFACT_RL_DATASET
    assert _station_for_task("trainer.train_grpo") == ARTIFACT_CHECKPOINT
    assert _station_for_task("some-unrelated.task") == ""
    assert _station_for_task("") == ""
