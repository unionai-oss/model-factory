"""Task generation: determinism, split hygiene, and — crucially — that the
generated code actually runs and its footprint model points the right way."""

import json

from resource_tuner.contracts import CORPUS_COLUMNS
from resource_tuner.taskgen.corpus import build_corpus
from resource_tuner.taskgen.templates import (
    FAMILIES,
    _render_harness,
    generate_task,
)


def test_generation_is_deterministic():
    a, b = generate_task("etl", seed=7), generate_task("etl", seed=7)
    assert a == b
    assert generate_task("etl", seed=8) != a


def test_all_families_render_policy_and_harness_views():
    for family in FAMILIES:
        t = generate_task(family, seed=1)
        assert "@env.task" in t.source_code and "flyte.TaskEnvironment" in t.source_code
        assert "flyte" not in t.harness_code  # harness view must run without flyte
        assert "def run()" in t.harness_code
        assert t.true_peak_memory_mib > 0 and t.true_cpu_cores >= 1
        compile(t.source_code, t.task_id, "exec")
        compile(t.harness_code, t.task_id, "exec")


def test_harness_code_executes_with_tiny_params():
    """Run each family's workload for real (tiny sizes, zero hold time) —
    a template that renders but crashes would fail every cluster episode."""
    tiny = {
        "data_engineering": {"rows": 200, "cols": 4, "segments": 10, "duration_s": 0},
        "data_science": {
            "n_samples": 200, "n_features": 8, "n_estimators": 3, "n_jobs": 1, "duration_s": 0,
        },
        "ml_training": {"hidden": 16, "depth": 2, "batch_size": 8, "duration_s": 0},
        "batch_inference": {
            "dim": 16, "out_dim": 32, "batch_size": 16, "n_batches": 2, "duration_s": 0,
        },
        "etl": {"n_records": 100, "duration_s": 0},
    }
    for family, params in tiny.items():
        code = _render_harness(FAMILIES[family], params)
        ns: dict = {}
        exec(compile(code, family, "exec"), ns)
        result = ns["run"]()
        assert isinstance(result, dict) and result, family


def test_footprint_grows_with_input_size():
    fam = FAMILIES["data_engineering"]
    small, _ = fam.footprint({"rows": 10_000, "cols": 8})
    large, _ = fam.footprint({"rows": 10_000_000, "cols": 8})
    assert large > small * 10


def test_corpus_splits_are_disjoint_and_schema_complete():
    records = build_corpus(20, 10, seed=3)
    assert len(records) == 30
    assert all(set(r) == set(CORPUS_COLUMNS) for r in records)
    train = {r["task_id"] for r in records if r["split"] == "train"}
    heldout = {r["task_id"] for r in records if r["split"] == "heldout"}
    assert len(train) == 20 and len(heldout) == 10
    assert not train & heldout
    for r in records:
        json.loads(r["params_json"])  # params survive a round-trip
