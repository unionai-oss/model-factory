"""RightSizeCPU Bench: the properties its leaderboard rests on.

Mirrors test_bench.py: the information boundary, the cost model's
ordering, splits by task, cross-module call shapes, and that the label is
what the measurement says (a 15 s window of a per-second series).
"""

from __future__ import annotations

import json
import os
from random import Random

import numpy as np
import pytest

from resource_tuner.bench import common
from resource_tuner.bench.workloads import render_harness, render_task, scale_input
from resource_tuner.bench_cpu import arms, dataset, features, grid, measure, scoring
from resource_tuner.bench_cpu.spec import CPU
from resource_tuner.bench_cpu.workloads import FAMILIES, HELDOUT_FAMILIES, make_task


def _fake_measure(row: dict, rng: Random) -> dict:
    knobs = json.loads(row["knobs_json"])
    pool = knobs.get("threads") or knobs.get("processes") or 1
    peak = min(8.0, pool * rng.uniform(0.5, 1.0)) if row["driver"] != "serial" else 1.0
    if knobs.get("work") == "pure_python":
        peak = 1.0
    series = [peak * rng.uniform(0.6, 1.0) for _ in range(70)]
    series[20:40] = [peak] * 20
    sizes = {s["name"]: 1000 * (s.get("rows") or s["shape"][0]) for s in json.loads(row["inputs_json"])}
    return {"ok": True, "input_bytes": sizes, "peak_cores": grid.peak_window(series),
            "mean_cores": float(np.mean(series)), "peak_cores_1s": max(series), "cpu_series": series,
            "wall_s": 70.0, "cpu_source": "fake"}


@pytest.fixture(scope="module")
def corpus():
    rng = Random(0)
    rows = dataset.generate_runs(10, seed=1)
    return dataset.attach_history([dataset.attach_measurement(r, _fake_measure(r, rng)) for r in rows])


@pytest.fixture(scope="module")
def splits(corpus):
    return common.usable_by_split(corpus)


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_every_family_renders_code_that_compiles(family):
    for seed in range(4):
        t = make_task(family, seed)
        compile(render_task(t), "task", "exec")
        compile(render_harness(t), "harness", "exec")


def test_peak_window_is_the_busiest_15_seconds():
    s = [1.0] * 30 + [4.0] * 15 + [1.0] * 30
    assert grid.peak_window(s) == pytest.approx(4.0)
    assert grid.peak_window([4.0] * 5 + [0.0] * 60) == pytest.approx(20 / 15)  # a 5 s spike is not a 15 s peak
    assert grid.peak_window([2.0] * 6) == pytest.approx(2.0)  # shorter than a window: its mean


def test_per_second_absorbs_a_short_final_interval():
    stamps = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (2.05, 2.05)]
    s = measure.per_second(stamps)
    assert max(s) < 1.5 and len(s) == 2


def test_oracle_cell_is_within_margin():
    for peak in np.linspace(0.05, 7.9, 300):
        assert grid.within_margin(grid.CPU_GRID[grid.label_peak_index(peak)], peak), peak


def test_splits_are_by_task(corpus):
    by_task = {}
    for r in corpus:
        by_task.setdefault(r["task_id"], set()).add(r["split"])
    assert all(len(s) == 1 for s in by_task.values())
    assert {r["family"] for r in corpus if r["split"] == "ood"} == set(HELDOUT_FAMILIES)


def test_features_and_prompt_cannot_see_the_answer(corpus):
    r = dict(corpus[5])
    poisoned = {**r, "knobs_json": '{"threads": 99}', "family": "x", "driver": "y", "peak_cores": 99.0,
                "mean_cores": 99.0, "label_peak_idx": 0, "label_mean_idx": 0, "cpu_series_json": "[99]"}
    assert features.row_features(r) == features.row_features(poisoned)
    assert CPU.user_prompt(r) == CPU.user_prompt(poisoned)


def test_throttling_costs_more_than_the_oracle(splits):
    rows = splits["test"]
    oracle = {r["run_id"]: {"peak_idx": r["label_peak_idx"], "mean_idx": r["label_mean_idx"]} for r in rows}
    s = scoring.score(rows, oracle)
    assert s["cpu_regret"] == pytest.approx(0.0, abs=1e-9) and s["within_margin"] == 1.0
    low = {k: {**v, "peak_idx": 0} for k, v in oracle.items()}
    high = {k: {**v, "peak_idx": scoring.N - 1} for k, v in oracle.items()}
    assert scoring.score(rows, low)["cpu_regret"] > 0 and scoring.score(rows, low)["under_rate"] > 0.5
    assert scoring.score(rows, high)["cpu_regret"] > 0


def test_throttled_seconds():
    s = np.array([2.0, 2.0, 0.5])
    assert scoring.throttled_seconds(s, 4.0) == 3.0
    assert scoring.throttled_seconds(s, 1.0) == 5.0


def test_decode_picks_the_cell_within_margin_of_both_modes():
    p = np.zeros(scoring.N)
    p[6], p[8] = 0.5, 0.5  # peak in (1.5, 1.75] or (2, 2.5]
    assert grid.CPU_GRID[scoring.decode_index(p)] == 2.0


def test_cost_optimal_decoding_under_predicts():
    """Why accuracy, not cost, is the CPU headline: under throttling-only cost,
    asking below the peak is cheaper."""
    p = np.zeros(scoring.N)
    p[8] = 1.0
    assert scoring.bayes_index(p) < 8 <= scoring.decode_index(p)


@pytest.mark.parametrize("arm", sorted(arms.ARMS))
def test_every_classical_arm_fits_and_predicts(arm, splits):
    fit, predict = arms.ARMS[arm]
    state = fit(splits["train"], splits["val"])
    preds = predict(state, splits["test"])
    s = scoring.score(splits["test"], preds)
    assert s["n"] == len(splits["test"]) and 0 <= s["within_margin"] <= 1


def test_code_rule_reads_declared_pools():
    assert arms.code_rule_cores('con.execute("SET threads = 6")') == 6
    assert arms.code_rule_cores("RandomForestClassifier(n_estimators=4, n_jobs=8)") == 8
    assert arms.code_rule_cores("df.groupby('a').sum()") == 1


def test_spec_round_trips():
    for a in range(len(CPU.a.options)):
        for b in (0, 7, 15):
            assert CPU.parse(CPU.target(a, b)) == (a, b)
    assert CPU.parse('reasoning... {"peak_cpu": 3.2, "mean_cpu": "1.1"}') == (grid.index_covering(3.2), grid.index_covering(1.1))


def test_cross_module_shapes(splits, corpus):
    import pandas as pd

    rows = splits["val"] + splits["test"]
    st = arms.fit_gbt_dist(splits["train"], splits["val"])
    preds = arms.predict_gbt_dist(st, rows)
    by_variant = common.preds_by_variant(pd.DataFrame(common.pred_rows("gbt-dist", preds, rows, CPU)), CPU)
    card = scoring.score_slices([r for r in corpus if dataset.usable(r)], by_variant["gbt-dist"])
    assert card["split=test"]["n"] > 0 and "brier" in card["split=test"]
    assert scoring.frontier(splits["test"], by_variant["gbt-dist"])


def test_factory_graph():
    from resource_tuner.bench_cpu.factory import ARTIFACT_ARM, PARTITION_ARM, PARTITION_RELEASE, build_graph

    g = build_graph()
    assert g.validate() == []
    assert set(g._nodes[ARTIFACT_ARM].partitions) == {PARTITION_RELEASE, PARTITION_ARM}


@pytest.mark.parametrize("pkg", ["resource_tuner.bench_cpu", "resource_tuner.bench_gpu"])
def test_no_lazy_package_imports(pkg):
    import ast
    import importlib
    import pathlib

    offenders = []
    for path in pathlib.Path(importlib.import_module(pkg).__file__).parent.glob("*.py"):
        for fn in ast.walk(ast.parse(path.read_text())):
            if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for node in ast.walk(fn):
                    if isinstance(node, ast.ImportFrom) and node.level > 0:
                        offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, offenders


def test_a_real_workload_is_measured(monkeypatch):
    """The harness end to end: a 2-process pool should read ~2 cores."""
    monkeypatch.setenv("TIME_BUDGET_S", "3")
    t = make_task("process_pool", 3)
    m = measure.measure_run(render_harness(t), [scale_input(s, 0.05) for s in t.inputs], seed=0, timeout_s=60)
    assert m["ok"], m["error"]
    assert m["wall_s"] >= 3 and len(m["cpu_series"]) >= 3
    assert 0.3 < m["peak_cores"] <= t.knobs["processes"] + 1.5


def test_pointer_offsets_land_on_cpu_options(corpus):
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer

    from resource_tuner.bench.llm_arms import _prompt_ids, option_offsets

    try:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    except Exception:  # noqa: BLE001 -- offline
        pytest.skip("tokenizer not available")
    a_off, b_off = option_offsets(tok, CPU)
    for r in corpus[:20]:
        ids = _prompt_ids(tok, r, 8192, CPU)
        n = len(ids)
        for offs, opts in ((a_off, CPU.a.options), (b_off, CPU.b.options)):
            for off, o in zip(offs, opts):
                assert tok.decode([ids[n - 1 - off]]) in o


def test_time_budget_env_is_read_by_every_family():
    for fam in FAMILIES:
        assert 'os.environ.get("TIME_BUDGET_S"' in render_harness(make_task(fam, 0))
    assert os.environ.get("TIME_BUDGET_S") is None
