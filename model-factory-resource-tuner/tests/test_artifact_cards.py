"""Artifact-card section builders (pure: pandas in, JSON-able dicts out)."""

import json

import pandas as pd

from resource_tuner.lineage_app import _corpus_sections, _kv_section, _table_section


def _corpus_df(n=10):
    return pd.DataFrame(
        {
            "family": ["etl", "ml_training"] * (n // 2),
            "split": ["train"] * (n - 2) + ["heldout"] * 2,
            "true_peak_memory_mib": [100.0 * (i + 1) for i in range(n)],
            "true_cpu_cores": [1.0] * n,
            "duration_s": [60] * n,
            "params_json": [
                json.dumps({"archetype": i % 3, "label_source": "fitted" if i % 2 else "measured"})
                for i in range(n)
            ],
        }
    )


def test_corpus_sections_shape_and_stats():
    sections = _corpus_sections(_corpus_df(), pd.Series([500, 700, 900] * 4)[:10])
    assert sections[0]["kv"]["rows"] == "10"
    assert "train:8" in sections[0]["kv"]["splits"]
    assert "measured" in sections[0]["kv"]["label sources"]
    peaks = sections[1]["table"]
    assert peaks["headers"][0] == "p5" and int(peaks["rows"][0][-1]) == 1000
    # everything must be JSON-serializable strings for the API
    json.dumps(sections)


def test_section_helpers_stringify():
    assert _kv_section("H", {"a": 1})["kv"] == {"a": "1"}
    t = _table_section("H", ["x"], [[1.5]])
    assert t["table"]["rows"] == [["1.5"]]
