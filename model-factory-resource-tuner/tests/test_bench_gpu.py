"""RightSizeGPU Bench: the properties its leaderboard rests on."""

from __future__ import annotations

import json
from random import Random

import numpy as np
import pytest

from resource_tuner.bench import common
from resource_tuner.bench.workloads import render_harness, render_task, scale_input
from resource_tuner.bench_gpu import arms, dataset, features, grid, measure, scoring
from resource_tuner.bench_gpu.spec import GPU
from resource_tuner.bench_gpu.workloads import FAMILIES, HELDOUT_FAMILIES, make_task


def _fake_measure(row: dict, rng: Random) -> dict:
    sizes = {s["name"]: 4 * int(np.prod(s.get("shape") or [s.get("rows", 1), 4])) for s in json.loads(row["inputs_json"])}
    gib = 0.5 + sum(sizes.values()) / 2**30 * rng.uniform(1, 6) + rng.choice([0.5, 2, 6, 12])
    return {"ok": True, "input_bytes": sizes, "peak_gpu_mib": gib * 1024, "mean_util": rng.uniform(5, 98),
            "t_peak_frac": 0.1, "wall_s": 160.0, "device_name": "Tesla T4", "util_windows": [50.0] * 5}


@pytest.fixture(scope="module")
def corpus():
    rng = Random(0)
    rows = dataset.generate_runs(12, seed=1)
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


def test_device_ladder():
    assert grid.DEVICES[grid.device_for(10)][0] == "T4"
    assert grid.DEVICES[grid.device_for(20)][0] == "L4"  # the A10G holds as much and costs more
    assert grid.DEVICES[grid.device_for(30)][0] == "L40S"
    assert grid.DEVICES[grid.next_device_up(0)][0] == "L4"
    assert grid.next_device_up(3) is None


def test_oom_retries_up_the_ladder():
    t4 = grid.mem_index_covering(4)
    c_fit, att_fit, _ = scoring.simulate(t4, 3.0, 0.1)
    c_oom, att_oom, failed = scoring.simulate(t4, 18.0, 0.1)
    assert att_fit == 1 and att_oom == 2 and not failed
    assert c_oom > scoring.oracle_cost(18.0) > c_fit


def test_oracle_has_zero_regret(splits):
    rows = splits["test"]
    oracle = {r["run_id"]: {"gmem_idx": r["label_gmem_idx"], "util_idx": r["label_util_idx"]} for r in rows}
    s = scoring.score(rows, oracle)
    assert s["gpu_regret"] <= 1e-9 and s["oom_rate"] == 0.0 and s["util_within_10"] == 1.0


def test_features_and_prompt_cannot_see_the_answer(corpus):
    r = dict(corpus[3])
    poisoned = {**r, "knobs_json": "{}", "family": "x", "driver": "y", "peak_gpu_mib": 1e9, "mean_util": 99.0,
                "label_gmem_idx": 0, "label_util_idx": 0, "measured_on": "H100"}
    assert features.row_features(r) == features.row_features(poisoned)
    assert GPU.user_prompt(r) == GPU.user_prompt(poisoned)


def test_heldout_families_are_only_ood(corpus):
    assert {r["family"] for r in corpus if r["split"] == "ood"} == set(HELDOUT_FAMILIES)


@pytest.mark.parametrize("arm", sorted(arms.ARMS))
def test_every_classical_arm_fits_and_predicts(arm, splits):
    fit, predict = arms.ARMS[arm]
    preds = predict(fit(splits["train"], splits["val"]), splits["test"])
    assert scoring.score(splits["test"], preds)["n"] == len(splits["test"])


def test_spec_round_trips():
    for a in range(len(GPU.a.options)):
        for b in range(len(GPU.b.options)):
            assert GPU.parse(GPU.target(a, b)) == (a, b)
    assert GPU.parse('{"gpu_memory": "5000 MiB", "gpu_util": "73%"}') == (grid.mem_index_covering(5000 / 1024), 7)


def test_cross_module_shapes(splits, corpus):
    import pandas as pd

    rows = splits["val"] + splits["test"]
    preds = arms.predict_gbt_dist(arms.fit_gbt_dist(splits["train"], splits["val"]), rows)
    by_variant = common.preds_by_variant(pd.DataFrame(common.pred_rows("gbt-dist", preds, rows, GPU)), GPU)
    card = scoring.score_slices([r for r in corpus if dataset.usable(r)], by_variant["gbt-dist"])
    assert card["split=test"]["n"] > 0 and "brier" in card["split=test"]
    assert scoring.frontier(splits["test"], by_variant["gbt-dist"])


def test_factory_graph():
    from resource_tuner.bench_gpu.factory import ARTIFACT_ARM, PARTITION_ARM, PARTITION_RELEASE, build_graph

    g = build_graph()
    assert g.validate() == []
    assert set(g._nodes[ARTIFACT_ARM].partitions) == {PARTITION_RELEASE, PARTITION_ARM}


def test_token_inputs_stay_inside_the_vocabulary():
    from pathlib import Path
    import tempfile

    from resource_tuner.bench.measure import write_input

    spec = {**make_task("gpt_train", 1).inputs[0]}
    spec = scale_input(spec, 0.01)
    with tempfile.TemporaryDirectory() as d:
        x = np.load(write_input(spec, Path(d), 0))
    assert x.min() >= 0 and x.max() < spec["high"]


def test_dcgm_windows():
    samples = [(t * 0.5, 0.0, 100.0 if t < 60 else 0.0) for t in range(120)]
    assert measure.dcgm_windows(samples) == [100.0, 0.0]


def test_a_real_workload_runs_without_a_gpu(monkeypatch):
    monkeypatch.setenv("TIME_BUDGET_S", "2")
    t = make_task("mlp_tabular", 2)
    m = measure.measure_run(render_harness(t), [scale_input(s, 0.01) for s in t.inputs], seed=0, timeout_s=120)
    assert m["ok"], m["error"]


def test_pointer_offsets_land_on_gpu_options(corpus):
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer

    from resource_tuner.bench.llm_arms import _prompt_ids, option_offsets

    try:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    except Exception:  # noqa: BLE001 -- offline
        pytest.skip("tokenizer not available")
    a_off, b_off = option_offsets(tok, GPU)
    for r in corpus[:20]:
        ids = _prompt_ids(tok, r, 8192, GPU)
        n = len(ids)
        for offs, opts in ((a_off, GPU.a.options), (b_off, GPU.b.options)):
            for off, o in zip(offs, opts):
                # the option's last character may share a token with the separator ("%,")
                assert tok.decode([ids[n - 1 - off]]).rstrip(", \n") in o


def test_a_device_bound_peak_is_not_scored_for_memory_accuracy(splits):
    rows = [dict(r) for r in splits["test"][:6]]
    for r in rows:
        r["fills_device"] = True
    preds = {r["run_id"]: {"gmem_idx": r["label_gmem_idx"], "util_idx": 0} for r in rows}
    s = scoring.score(rows, preds)
    assert s["n_mem_scored"] == 0 and s["device_exact"] == 1.0
