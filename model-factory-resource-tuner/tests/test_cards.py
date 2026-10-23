"""Artifact-card renderers (pure: dicts in, markdown out) + publish wiring.

The renderers run at the tail end of multi-hour training and corpus runs,
on whatever the producer happens to be holding — so the tests that matter
are the ones that feed them PARTIAL state and assert they still render.
"""

import json
import tempfile

import pytest

from resource_tuner.contracts import CORPUS_COLUMN_DOCS, CORPUS_COLUMNS
from resource_tuner.shared import cards


def _corpus_parquet(n=40):
    import pandas as pd

    rows = []
    for i in range(n):
        rows.append(
            {
                "task_id": f"t{i}",
                "family": ["etl", "ml_training", "data_science", "batch_inference"][i % 4],
                "source_code": "def f():\n    pass\n" * (i % 5 + 1),
                "harness_code": "def f():\n    pass\n",
                "input_profile": "1M rows",
                "params_json": json.dumps(
                    {"archetype": i % 3, "label_source": "measured" if i % 2 else "fitted"}
                ),
                "generator": "template" if i % 3 else "qwen38-27b",
                "prior_json": "",
                "history_json": "",
                "true_peak_memory_mib": 100.0 * (i + 1),
                "true_cpu_cores": 1.0 + (i % 4),
                "true_gpu_mem_mib": 8000.0 if i % 10 == 0 else 0.0,
                "duration_s": 30 + i,
                "split": "heldout" if i >= n - 8 else "train",
            }
        )
    path = tempfile.mktemp(suffix=".parquet")
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


# ── corpus ──────────────────────────────────────────────────────────────


def test_corpus_stats_reads_facets_quantiles_and_label_sources():
    stats = cards.corpus_stats(_corpus_parquet())
    assert stats["rows"] == 40
    assert stats["split"] == {"train": 32, "heldout": 8}
    assert set(stats["family"]) == {"etl", "ml_training", "data_science", "batch_inference"}
    assert stats["generator"]["qwen38-27b"] > 0
    assert stats["gpu_rows"] == 4
    assert set(stats["label_sources"]) == {"measured", "fitted"}
    mem = stats["quantiles"]["true_peak_memory_mib"]
    assert mem["max"] == 4000.0 and mem["p5"] < mem["p50"] < mem["p95"]
    assert stats["by_family"]["family_count"] == [10, 10, 10, 10]


def test_corpus_card_documents_every_schema_column():
    path = _corpus_parquet()
    card = cards.corpus_card(
        artifact="tuning-task-corpus",
        stats=cards.corpus_stats(path),
        provenance={"produced_by": "build_task_corpus", "seed": 0},
        intended_use="Train on split=train only.",
    )
    for column, doc in CORPUS_COLUMN_DOCS.items():
        assert column in card, f"{column} missing from card schema table"
        assert cards._cell(doc)[:30] in card
    # provenance keys are rendered as prose, not snake_case
    assert "produced by" in card and "build_task_corpus" in card
    assert "Train on split=train only." in card
    assert "heldout 8" in card and "train 32" in card


def test_corpus_card_survives_empty_stats():
    """A stats failure costs a thinner card, never the publish."""
    card = cards.corpus_card(artifact="synthetic-task-corpus", stats={}, provenance={})
    assert card.startswith("# synthetic-task-corpus")
    # with no column list, the schema still documents the full contract
    assert all(c in card for c in CORPUS_COLUMNS)


def test_corpus_card_flags_undocumented_columns():
    stats = {"rows": 1, "columns": CORPUS_COLUMNS + ["mystery_col"]}
    assert "Undocumented extra columns: mystery_col" in cards.corpus_card(
        artifact="tuning-task-corpus", stats=stats, provenance={}
    )


def test_corpus_stats_safe_swallows_a_broken_path():
    assert cards.corpus_stats_safe("/nonexistent/corpus.parquet") == {}


def test_corpus_attrs_are_flat_strings():
    attrs = cards.corpus_attrs(
        cards.corpus_stats(_corpus_parquet()),
        {"produced_by": "build_task_corpus", "seed": 0, "teachers": "qwen38-27b"},
    )
    assert attrs["rows"] == "40" and attrs["train_rows"] == "32"
    assert attrs["generators"] == "qwen38-27b,template"
    assert all(isinstance(k, str) and isinstance(v, str) for k, v in attrs.items())


# ── checkpoint ──────────────────────────────────────────────────────────


def _manifest():
    return {
        "base_model": "Qwen/Qwen3.5-4B",
        "profile": "dev",
        "corpus_path": "s3://bucket/corpus.parquet",
        "reward_stage": "composite",
        "reward_shape": {"fit": 1.0, "waste": 0.5},
        "max_steps": 150,
        "use_lora": True,
        "gbt_hint": False,
        "hypothesis_description": "graded JSON reward lifts validity",
        "final_metrics": {
            "mean_reward_first": 0.1,
            "mean_reward_last": 0.82,
            "logged_steps": 150,
        },
    }


def test_checkpoint_card_carries_provenance_and_reward_shape():
    card = cards.checkpoint_card(_manifest())
    assert "Qwen/Qwen3.5-4B" in card
    assert "s3://bucket/corpus.parquet" in card
    assert "0.100 → 0.820" in card
    assert "| fit | 1.0 |" in card
    assert "graded JSON reward lifts validity" in card
    assert "Unevaluated at publish time" in card


def test_intermediate_checkpoint_card_warns_against_promotion():
    card = cards.checkpoint_card({"profile": "dev", "reward_stage": "composite"}, step=40)
    assert "tuner-checkpoint-intermediate" in card
    assert "not for promotion" in card.lower()
    assert "step 40" in card
    assert cards.checkpoint_attrs({}, step=40)["intermediate"] == "true"


def test_checkpoint_attrs_identify_the_artifact_as_a_model():
    attrs = cards.checkpoint_attrs(_manifest())
    assert attrs["architecture"] == "Qwen/Qwen3.5-4B"
    assert attrs["framework"] == "peft" and attrs["serial_format"] == "safetensors"
    assert attrs["task"] == "resource-proposal"
    assert attrs["mean_reward_last"] == "0.8200"
    # a full fine-tune is not PEFT
    assert cards.checkpoint_attrs({"use_lora": False})["framework"] == "transformers"


# ── reports ─────────────────────────────────────────────────────────────


def test_eval_report_card_states_the_gate_and_both_baselines():
    report = {
        "auto_gate_passed": False,
        "base_model": "Qwen/Qwen3.5-4B",
        "reward_stage": "composite",
        "n_contexts": 512,
        "schema_validity": 1.0,
        "success_rate": 0.98,
        "median_overprovision_pct": 47.0,
        "baseline_success_rate": 0.64,
        "baseline_median_overprovision_pct": 25.0,
        "ml_baseline_success_rate": 0.99,
        "policy_cost_per_task_hr": 0.2231,
        "baseline_cost_per_task_hr": 0.2198,
        "ml_baseline_cost_per_task_hr": None,
        "dollars_saved_per_1k_task_hrs": -3.3,
        "cluster_episodes": [{"task_id": "t1"}, {"task_id": "t2"}],
        "checkpoint_path": "s3://bucket/ckpt",
        "gpu_contexts": 12,
        "gpu_success_rate": 0.5,
    }
    card = cards.eval_report_card(report)
    assert "**Gate: FAIL**" in card
    assert "98%" in card and "64%" in card and "47%" in card
    assert "$0.2231" in card
    assert "– |" in card  # the missing ML cost renders as a dash, not None
    assert "None" not in card
    attrs = cards.eval_report_attrs(report)
    assert attrs["gate"] == "FAIL" and attrs["cluster_episodes"] == "2"


def test_eval_report_card_renders_from_a_nearly_empty_report():
    card = cards.eval_report_card({})
    assert "**Gate: FAIL**" in card and "None" not in card


def test_ab_report_card_reports_real_pod_outcomes():
    report = {
        "n_tasks": 12,
        "prior": {"cpu": 2, "memory": "4Gi"},
        "prior_oom_rate": 0.13,
        "tuned_oom_rate": 0.0,
        "prior_fit_rate": 0.87,
        "tuned_fit_rate": 1.0,
        "prior_median_overprovision_pct": 54.0,
        "tuned_median_overprovision_pct": 47.0,
        "episodes": [{"task_id": "t1"}],
    }
    card = cards.ab_report_card(report)
    assert "13%" in card and "0%" in card and "54%" in card
    assert '"memory": "4Gi"' in card
    assert cards.ab_report_attrs(report)["tuned_oom_rate"] == "0.000"


def test_ml_baseline_card_scores_against_the_rule_baseline():
    manifest = {
        "estimator": "quantile-gbt",
        "quantiles": {"memory": 0.9, "cpu": 0.9, "gpu_mem": 0.95},
        "n_train": 90_000,
        "n_features": 27,
        "corpus": "s3://bucket/corpus.parquet",
        "sim_heldout": {
            "ml": {"fit": 0.99, "cost_per_task_hr": 0.21, "median_mem_waste_pct": 60.0},
            "rule_baseline": {"fit": 0.64, "cost_per_task_hr": 0.22},
        },
    }
    card = cards.ml_baseline_card(manifest)
    assert "99%" in card and "64%" in card and "90,000" in card
    assert "$0.2100" in card
    assert "None" not in card
    assert cards.ml_baseline_attrs(manifest)["model_type"] == "quantile-gbt"


# ── formatting invariants ───────────────────────────────────────────────


def test_table_cells_never_break_the_markdown_row():
    t = cards.table(["a", "b"], [["x|y", "line\nbreak"]])
    assert t.count("\n") == 2  # header, separator, one row
    assert "x\\|y" in t


def test_table_of_nothing_renders_nothing():
    assert cards.table(["a"], []) == ""


def test_kv_drops_empty_values():
    assert cards.kv({"a": 1, "b": None, "c": "", "d": []}) == "- **a**: 1"


# ── wiring ──────────────────────────────────────────────────────────────


def test_publish_passes_card_and_attrs_through(monkeypatch):
    """Patch the real module's attributes, not sys.modules: `publish` does
    `import flyte.artifacts as artifacts`, which resolves the already-imported
    submodule attribute and walks straight past a sys.modules stand-in."""
    import flyte.artifacts as artifacts

    import resource_tuner.contracts as contracts

    seen = {}

    def _fake_meta(**kw):
        seen.update(kw)
        return "META"

    monkeypatch.setattr(artifacts, "Metadata", _fake_meta)
    monkeypatch.setattr(artifacts, "new", lambda obj, meta: (obj, meta))

    out = contracts.publish(
        "obj", "a-name", "desc", kind="model", attrs={"k": "v"}, card="CARD"
    )
    assert out == ("obj", "META")
    assert seen["attrs"] == {"k": "v"} and seen["card"] == "CARD"
    assert seen["kind"] == "model" and seen["name"] == "a-name"
    # no attrs → None, so an empty mapping never reaches the control plane
    contracts.publish("obj", "a-name")
    assert seen["attrs"] is None and seen["card"] is None


@pytest.mark.parametrize(
    "module,artifact",
    [
        ("resource_tuner.training.stations", "ARTIFACT_TASK_CORPUS"),
        ("resource_tuner.training.stations", "ARTIFACT_SYNTHETIC"),
        ("resource_tuner.training.stations", "ARTIFACT_AB_REPORT"),
        ("resource_tuner.training.grpo", "ARTIFACT_TUNER_CHECKPOINT"),
        ("resource_tuner.training.grpo", "ARTIFACT_TUNER_CHECKPOINT_INTERMEDIATE"),
        ("resource_tuner.training.evaluate", "ARTIFACT_EVAL_REPORT"),
        ("resource_tuner.training.ml_station", "ARTIFACT_ML_BASELINE"),
    ],
)
def test_every_publish_site_attaches_a_card(module, artifact):
    """Guards the actual ask: no artifact ships without a card.

    Source-level because the publishes live at the end of GPU/cluster
    tasks — asserting on the call text is what can run in CI.
    """
    import ast
    import importlib
    import inspect

    tree = ast.parse(inspect.getsource(importlib.import_module(module)))
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "publish"
        and any(
            isinstance(a, ast.Name) and a.id == artifact
            for a in list(n.args) + [k.value for k in n.keywords]
        )
    ]
    assert calls, f"no publish of {artifact} found in {module}"
    for call in calls:
        kwargs = {k.arg for k in call.keywords}
        assert "card" in kwargs, f"{artifact} published without a card"
        assert "attrs" in kwargs, f"{artifact} published without attrs"
