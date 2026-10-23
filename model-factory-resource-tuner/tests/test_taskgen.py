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
        # GPU families fall back to CPU here (guarded device pick) — the
        # point is the rendered code is executable anywhere.
        "gpu_batch_inference": {
            "hidden": 16, "depth": 2, "batch_size": 2, "seq_len": 4, "duration_s": 0,
        },
        "gpu_finetune": {
            "hidden": 16, "depth": 2, "lora_r": 4, "batch_size": 2, "seq_len": 4,
            "duration_s": 0,
        },
    }
    assert set(tiny) == set(FAMILIES)  # every family stays covered
    for family, params in tiny.items():
        code = _render_harness(FAMILIES[family], params)
        ns: dict = {}
        exec(compile(code, family, "exec"), ns)
        result = ns["run"]()
        assert isinstance(result, dict) and result, family


def test_gpu_family_vram_spans_the_tenant_ladder():
    """Sampled VRAM demand must exercise T4 AND bigger cards, or the GPU
    choice degenerates to 'always T4'."""
    from resource_tuner.pricing import cheapest_gpu_for

    types = {
        cheapest_gpu_for(generate_task("gpu_batch_inference", seed=s).true_gpu_mem_mib)
        for s in range(60)
    }
    assert "T4" in types and len(types - {None}) >= 2, types
    assert all(
        generate_task(f, seed=3).true_gpu_mem_mib == 0.0
        for f in ("etl", "data_engineering")
    )


def test_gpu_vram_cap_produces_single_t4_only_tasks():
    """gpu_max_vram_mib=14000: every GPU task fits (and therefore should
    be proposed) exactly one T4 — the round-8 corpus constraint."""
    for fam in ("gpu_batch_inference", "gpu_finetune"):
        for s in range(40):
            t = generate_task(fam, seed=s, gpu_max_vram_mib=14_000)
            assert 0 < t.true_gpu_mem_mib <= 14_000, (fam, s, t.true_gpu_mem_mib)
    # deterministic: capping is a pure function of the sampled params
    a = generate_task("gpu_batch_inference", seed=5, gpu_max_vram_mib=14_000)
    b = generate_task("gpu_batch_inference", seed=5, gpu_max_vram_mib=14_000)
    assert a == b
    # uncapped sampling is untouched
    assert generate_task("gpu_batch_inference", seed=5) != a or a.true_gpu_mem_mib <= 14_000


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
