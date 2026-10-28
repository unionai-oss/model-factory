"""The trees x LLM hybrids (round 21): stacked hints and pooled distributions.

What the leaderboard's hybrid rows rest on: a training run's hint never comes
from trees that saw its own task, the hint sits where the pointer head's
option offsets do not move, a pool at w=1 is the trees and at w=0 the LLM,
and every station's call shape (GPU, CPU and the combined bench) holds.
"""

from __future__ import annotations

import json
from random import Random

import numpy as np
import pandas as pd
import pytest

from resource_tuner.bench import common, hybrid
from resource_tuner.bench.spec import HINT_HEADER, with_hint
from resource_tuner.bench_gpu import arms as gpu_arms
from resource_tuner.bench_gpu import dataset as gpu_dataset
from resource_tuner.bench_gpu import scoring as gpu_scoring
from resource_tuner.bench_gpu.spec import GPU


def _fake(row: dict, rng: Random) -> dict:
    sizes = {s["name"]: 4 * int(np.prod(s.get("shape") or [s.get("rows", 1), 4])) for s in json.loads(row["inputs_json"])}
    gib = 0.5 + sum(sizes.values()) / 2**30 * rng.uniform(1, 6) + rng.choice([0.5, 2, 6, 12, 30])
    return {"ok": True, "input_bytes": sizes, "peak_gpu_mib": gib * 1024, "mean_util": rng.uniform(5, 98),
            "t_peak_frac": 0.1, "wall_s": 60.0, "device_name": "Tesla T4"}


@pytest.fixture(scope="module")
def gpu_rows():
    rng = Random(0)
    rows = gpu_dataset.generate_runs(10, seed=3)
    return gpu_dataset.attach_history([gpu_dataset.attach_measurement(r, _fake(r, rng)) for r in rows])


@pytest.fixture(scope="module")
def gpu_splits(gpu_rows):
    return common.usable_by_split(gpu_rows)


def test_crossfit_never_predicts_a_run_from_its_own_task(gpu_splits):
    """Each training run's prediction comes from trees fit without its task's runs."""
    seen: dict[str, set] = {}

    def fit(train, val):
        return {"tasks": {r["task_id"] for r in train}}

    def predict(state, rows):
        for r in rows:
            seen[r["run_id"]] = state["tasks"]
        return {r["run_id"]: {} for r in rows}

    train = gpu_splits["train"]
    out = hybrid.crossfit(train, gpu_splits["val"], gpu_splits["test"], fit, predict)
    assert set(out) == {r["run_id"] for r in train + gpu_splits["test"]}
    for r in train:
        assert r["task_id"] not in seen[r["run_id"]]
    for r in gpu_splits["test"]:  # eval rows: trees fit on all of train
        assert seen[r["run_id"]] == {x["task_id"] for x in train}


def test_hint_sits_before_the_options_and_leaves_the_suffix(gpu_rows):
    r = gpu_rows[0]
    plain = GPU.messages(r)[1]["content"]
    pred = {"gmem_idx": 4, "util_idx": 7, "gmem_probs": [0.0] * 3 + [0.2, 0.5, 0.3] + [0.0] * 11}
    hinted = GPU.messages({**r, "hint": hybrid.hint_text(pred, GPU)})[1]["content"]
    i = plain.rindex("### Options")
    assert hinted.endswith(plain[i:]) and hinted.startswith(plain[:i])
    assert HINT_HEADER in hinted and "4GiB 50%" in hinted and "their pick: 4GiB" in hinted
    assert with_hint(plain, "") == plain


def test_pool_weight_one_is_the_trees_and_zero_the_llm():
    a, b = np.array([0.7, 0.2, 0.1]), np.array([0.1, 0.3, 0.6])
    assert np.allclose(hybrid.log_pool(a, b, 1.0), a, atol=1e-6)
    assert np.allclose(hybrid.log_pool(a, b, 0.0), b, atol=1e-6)
    assert hybrid.log_pool(a, b, 0.5).sum() == pytest.approx(1.0)


def test_gpu_pool_variants_end_to_end(gpu_splits, gpu_rows):
    """gbt-dist and a fake distributional LLM variant -> pooled variants that score."""
    train, val = gpu_splits["train"], gpu_splits["val"]
    eval_rows = val + gpu_splits["test"] + gpu_splits["ood"]
    trees = gpu_arms.predict_gbt_dist(gpu_arms.fit_gbt_dist(train, val), eval_rows)
    rng = np.random.default_rng(0)
    llm = {}
    for r in eval_rows:
        p = rng.dirichlet(np.ones(len(GPU.a.options)))
        p[r["label_gmem_idx"]] += 1.0
        p /= p.sum()
        llm[r["run_id"]] = {"gmem_idx": int(p.argmax()), "util_idx": int(r["label_util_idx"]), "gmem_probs": p.tolist()}
    table = common.pred_rows("gbt-dist", trees, eval_rows, GPU) + common.pred_rows("jev-pointer", llm, eval_rows, GPU)
    table += common.pred_rows("jev-pointer-argmax", llm, eval_rows, GPU)
    variants = common.preds_by_variant(pd.DataFrame(table), GPU)
    assert hybrid.poolable(variants, GPU) == ["jev-pointer"]
    pooled, knobs = hybrid.pool_variants(eval_rows, variants, GPU)
    assert set(pooled) == {"pool-jev-pointer", "pool-jev-pointer-argmax"}
    assert 0.0 <= knobs["pool-jev-pointer"]["w_trees"] <= 1.0
    usable = [r for r in gpu_rows if gpu_dataset.usable(r)]
    card = gpu_scoring.score_slices(usable, pooled["pool-jev-pointer"])
    assert card["split=test"]["n"] == len(gpu_splits["test"])
    # the tuned weight is no worse on val than either member alone (w = 1 and w = 0 are on the grid),
    # each read by the bench's decoder as the pool is
    decoded = {rid: {**p, "gmem_idx": GPU.decode(p["gmem_probs"])} for rid, p in llm.items()}
    head = GPU.headline
    assert knobs["pool-jev-pointer"]["val_headline"] <= min(head(val, trees), head(val, decoded)) + 1e-9


def test_stacked_arm_names_map_to_their_base():
    assert hybrid.base_arm("stack-jev-pointer-4b") == "jev-pointer-4b"
    assert hybrid.base_arm("sft") == "sft" and not hybrid.is_stacked("sft")
    from resource_tuner.bench_cpu import stations as cpu_stations
    from resource_tuner.bench_gpu import stations as gpu_stations

    for st in (cpu_stations, gpu_stations):
        assert {hybrid.base_arm(a) for a in st.STACK_ARMS} <= set(st.GPU_ARMS)


def test_cpu_hints_and_pool(monkeypatch):
    """The CPU bench's spec carries hints and pools through the same helpers."""
    from resource_tuner.bench_cpu import arms as cpu_arms
    from resource_tuner.bench_cpu import dataset as cpu_dataset
    from resource_tuner.bench_cpu import grid as cpu_grid
    from resource_tuner.bench_cpu.spec import CPU

    rng = Random(1)

    def fake(row):
        knobs = json.loads(row["knobs_json"])
        peak = min(6.0, (knobs.get("threads") or knobs.get("processes") or 1) * rng.uniform(0.5, 1.0))
        series = [peak * rng.uniform(0.6, 1.0) for _ in range(40)]
        sizes = {s["name"]: 1000 * (s.get("rows") or s["shape"][0]) for s in json.loads(row["inputs_json"])}
        return {"ok": True, "input_bytes": sizes, "peak_cores": cpu_grid.peak_window(series),
                "mean_cores": float(np.mean(series)), "cpu_series": series, "wall_s": 40.0, "peak_rss_mib": 500.0,
                "cpu_source": "fake"}

    rows = cpu_dataset.generate_runs(6, seed=2)
    rows = cpu_dataset.attach_history([cpu_dataset.attach_measurement(r, fake(r)) for r in rows])
    splits = common.usable_by_split(rows)
    eval_rows = splits["val"] + splits["test"]
    fit, predict = cpu_arms.ARMS[hybrid.ANCHOR]
    preds = hybrid.crossfit(splits["train"], splits["val"], eval_rows, fit, predict)
    hinted = hybrid.with_hints(splits["train"] + eval_rows, preds, CPU)
    assert all(r["hint"] for r in hinted)
    assert HINT_HEADER in CPU.messages(hinted[0])[1]["content"]


def test_combined_hints_and_pool():
    from resource_tuner.bench_all import arms as all_arms
    from resource_tuner.bench_all import prompting, stations
    from resource_tuner.bench import dataset as mem_dataset
    from resource_tuner.bench_all import dataset
    from resource_tuner.bench_cpu import dataset as cpu_dataset
    from tests.test_bench_all import _cpu_measure, _gpu_measure, _mem_measure

    rng = Random(0)
    parts = []
    for ds, fake, n in ((mem_dataset, _mem_measure, 4), (cpu_dataset, _cpu_measure, 4), (gpu_dataset, _gpu_measure, 6)):
        rows = ds.generate_runs(n, seed=1)
        parts.append(ds.attach_history([ds.attach_measurement(r, fake(r, rng)) for r in rows]))
    rows = dataset.unify(*parts)
    splits = dataset.by_split(rows)
    eval_rows = [r for s in ("val", "test", "ood") for r in splits[s]]
    fit, predict = all_arms.ARMS[hybrid.ANCHOR]
    preds = hybrid.crossfit(splits["train"], splits["val"], eval_rows, fit, predict)
    hints = stations.all_hints(preds)
    r = next(x for x in splits["train"] if x["run_id"] in hints)
    msg = prompting.messages({**r, "hint": hints[r["run_id"]]})[1]["content"]
    plain = prompting.messages(r)[1]["content"]
    assert msg.endswith(plain[plain.rindex("### Options"):]) and HINT_HEADER in msg
    # pool the trees with a copy of themselves under another name: a real member's call shape
    split_of = {x["run_id"]: x["split"] for x in eval_rows}
    trees = {rid: p for rid, p in preds.items() if rid in split_of}
    table = stations._pred_table("gbt-dist", trees, split_of) + stations._pred_table("jev-pointer", trees, split_of)
    variants = stations._probs_of(pd.DataFrame(table))
    usable = [x for x in rows if dataset.usable(x)]
    pooled, knobs = stations.pool_all(usable, variants)
    assert set(knobs) == {"pool-jev-pointer"}
    assert {t["variant"] for t in pooled} == {"pool-jev-pointer", "pool-jev-pointer-argmax"}


def test_pointer_offsets_hold_with_a_hint(gpu_rows):
    """The pointer head reads each option at a fixed offset from the prompt's end;
    a stacked arm's hint must not move them (GPU spec and the combined policy)."""
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer

    from resource_tuner.bench.llm_arms import _prompt_ids, option_offsets
    from resource_tuner.bench_all import policy, prompting

    try:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    except Exception:  # noqa: BLE001 -- offline
        pytest.skip("tokenizer not available")
    pred = {"gmem_idx": 12, "util_idx": 3, "gmem_probs": [0.0] * 11 + [0.3, 0.6, 0.1] + [0.0] * 3}
    a_off, b_off = option_offsets(tok, GPU)
    for r in gpu_rows[:8]:
        ids = _prompt_ids(tok, {**r, "hint": hybrid.hint_text(pred, GPU)}, 8192, GPU)
        n = len(ids)
        for offs, opts in ((a_off, GPU.a.options), (b_off, GPU.b.options)):
            for off, o in zip(offs, opts):
                assert tok.decode([ids[n - 1 - off]]).rstrip(", \n") in o
    hint = prompting.hint_text((np.ones(17), np.ones(16), np.ones(18)), (3, 2, 0))
    offs = policy.option_offsets(tok)
    for r in gpu_rows[:4]:
        row = {**r, "history_json": "[]", "hint": hint}
        ids = _prompt_ids(tok, row, 8192, prompting.SPEC)
        for axis_offs, opts in zip(offs, prompting.OPTIONS):
            for off, o in zip(axis_offs, opts):
                assert tok.decode([ids[len(ids) - 1 - off]]).strip() in o
