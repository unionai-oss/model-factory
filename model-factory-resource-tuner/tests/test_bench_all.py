"""RightSize Bench (combined): the properties its leaderboard and its RL reward rest on.

The joint cost model's ordering (each axis fails the way its own bench says),
the information boundary, the joined corpus, every arm's call shape, and a
supervised + RL step of the three-head decision model on a tiny random torso
-- the shapes that would otherwise only fail on the cluster, hours in.
"""

from __future__ import annotations

import json
import math
from random import Random

import numpy as np
import pytest

from resource_tuner.bench import dataset as mem_dataset
from resource_tuner.bench_all import arms, dataset, features, grid, prompting, scoring
from resource_tuner.bench_cpu import dataset as cpu_dataset
from resource_tuner.bench_cpu import grid as cpu_grid
from resource_tuner.bench_gpu import dataset as gpu_dataset


def _mem_measure(row, rng):
    inputs = json.loads(row["inputs_json"])
    nbytes = {s["name"]: int(math.prod(s["shape"]) * 8) if s["fmt"] == "npy" else int(s["rows"] * len(s["columns"]) * 8)
              for s in inputs}
    return {"ok": True, "input_bytes": nbytes, "peak_rss_mib": 150 + 3 * sum(nbytes.values()) / 2**20 * rng.uniform(0.9, 1.1),
            "avg_cores": rng.choice([1.0, 2.0]), "wall_s": 3.0, "t_peak_frac": 0.5}


def _cpu_measure(row, rng):
    knobs = json.loads(row["knobs_json"])
    peak = min(6.0, (knobs.get("threads") or knobs.get("processes") or 1) * rng.uniform(0.5, 1.0))
    series = [peak * rng.uniform(0.6, 1.0) for _ in range(40)]
    series[10:30] = [peak] * 20
    sizes = {s["name"]: 1000 * (s.get("rows") or s["shape"][0]) for s in json.loads(row["inputs_json"])}
    return {"ok": True, "input_bytes": sizes, "peak_cores": cpu_grid.peak_window(series), "mean_cores": float(np.mean(series)),
            "cpu_series": series, "wall_s": 40.0, "peak_rss_mib": rng.uniform(100, 3000), "cpu_source": "fake"}


def _gpu_measure(row, rng):
    sizes = {s["name"]: 4 * int(np.prod(s.get("shape") or [s.get("rows", 1), 4])) for s in json.loads(row["inputs_json"])}
    gib = 0.5 + rng.choice([0.5, 2, 6, 12, 20])
    return {"ok": True, "input_bytes": sizes, "peak_gpu_mib": gib * 1024, "mean_util": rng.uniform(5, 98),
            "t_peak_frac": 0.1, "wall_s": 90.0, "device_name": "Tesla T4", "mean_cpu_cores": 1.2}


@pytest.fixture(scope="module")
def corpus():
    rng = Random(0)
    parts = []
    for ds, fake, n in ((mem_dataset, _mem_measure, 4), (cpu_dataset, _cpu_measure, 4), (gpu_dataset, _gpu_measure, 6)):
        rows = ds.generate_runs(n, seed=1)
        parts.append(ds.attach_history([ds.attach_measurement(r, fake(r, rng)) for r in rows]))
    return dataset.unify(*parts)


@pytest.fixture(scope="module")
def splits(corpus):
    return dataset.by_split(corpus)


def test_corpus_joins_all_three_with_disjoint_ids(corpus):
    assert {r["source"] for r in corpus} == {"mem", "cpu", "gpu"}
    assert len({r["run_id"] for r in corpus}) == len(corpus)
    for r in corpus:
        if r["source"] == "gpu":
            assert r["label_mem"] == r["label_cpu"] == -1 and r["label_gpu"] >= 1
        else:
            assert r["label_gpu"] == grid.NO_GPU and r["label_mem"] >= 0 and r["label_cpu"] >= 0


def test_oracle_has_zero_regret(splits):
    rows = splits["test"] + splits["ood"]
    preds = {r["run_id"]: dict(zip(("mem_idx", "cpu_idx", "gpu_idx"), scoring.oracle_action(r))) for r in rows}
    s = scoring.score(rows, preds)
    assert s["regret"] == pytest.approx(0.0, abs=1e-9) and s["all_right"] == 1.0 and s["first_attempt_ok"] == 1.0


def test_each_axis_fails_the_way_its_bench_says(corpus):
    cpu_row = next(r for r in corpus if r["source"] == "cpu" and r["ok"] and r["label_mem"] > 2 and r["label_cpu"] > 2)
    gpu_row = next(r for r in corpus if r["source"] == "gpu" and r["ok"] and r["label_gpu"] > 2)
    m, c, g = scoring.oracle_action(cpu_row)
    base = scoring.cost(cpu_row, m, c, g).cost
    oom = scoring.cost(cpu_row, m - 2, c, g)
    assert oom.cost > base and oom.attempts > 1 and not oom.first_ok
    assert scoring.cost(cpu_row, m, c - 2, g).cost > base  # throttled: late
    assert scoring.cost(cpu_row, m, c, 1).cost > 3 * base  # a GPU for a CPU task
    _, _, gg = scoring.oracle_action(gpu_row)
    gbase = scoring.cost(gpu_row, 0, 0, gg).cost
    none = scoring.cost(gpu_row, 0, 0, grid.NO_GPU)
    assert none.cost > gbase and not none.first_ok  # no GPU: fails at start, retried
    # on a GPU node, host memory and cores are the instance's: they do not move the cost
    assert scoring.cost(gpu_row, grid.N_MEM - 1, grid.N_CPU - 1, gg).cost == pytest.approx(gbase)
    assert scoring.reward(cpu_row, m, c, g) == pytest.approx(-1.0)
    assert scoring.reward(cpu_row, m, c, 1) >= -scoring.REWARD_CAP


def test_features_and_prompt_cannot_see_the_answer(corpus):
    r = next(r for r in corpus if r["source"] == "cpu")
    leaked = dict(r, label_mem=0, label_cpu=0, need_mib=1.0, need_cores=0.1, source="gpu", family="x", driver="y")
    assert features.row_features(r) == features.row_features(leaked)
    assert prompting.user_prompt(r) == prompting.user_prompt(leaked)
    assert set(features.feature_names()) == set(features.row_features(r))


def test_parse_round_trips_every_option():
    for m in (0, 5, grid.N_MEM - 1):
        for c in (0, 7, grid.N_CPU - 1):
            for g in range(grid.N_GPU):
                assert prompting.parse_answer("sure: " + prompting.target_text(m, c, g)) == (m, c, g)
    assert prompting.parse_answer('{"memory": "3Gi", "cpu": 2}') == (grid.mem_cell(3072), grid.cpu_cell(2), grid.NO_GPU)
    assert prompting.parse_answer("no json here") is None


def test_balanced_caps_each_source(corpus):
    b = dataset.balanced(corpus, 5)
    assert all(sum(r["source"] == s for r in b) <= 5 for s in dataset.SOURCES)


@pytest.mark.parametrize("arm", sorted(arms.ARMS))
def test_every_classical_arm_fits_and_predicts(arm, splits):
    fit, predict = arms.ARMS[arm]
    state = fit(splits["train"], splits["val"] or splits["train"])
    rows = splits["test"] + splits["ood"]
    preds = predict(state, rows)
    assert set(preds) == {r["run_id"] for r in rows}
    s = scoring.score(rows, preds)
    assert s["n"] == len(rows) and np.isfinite(s["regret"])
    for p in preds.values():
        assert 0 <= p["mem_idx"] < grid.N_MEM and 0 <= p["cpu_idx"] < grid.N_CPU and 0 <= p["gpu_idx"] < grid.N_GPU
    from resource_tuner.bench_all.stations import knobs

    json.dumps(knobs(state))  # fit_all_arm writes these into the arm's meta (rsa-full-2 failed here)


def test_rules_ask_for_a_gpu_exactly_when_the_code_uses_cuda(splits):
    state = arms.fit_rules(splits["train"], splits["val"] or splits["train"])
    rows = splits["test"] + splits["ood"]
    preds = arms.predict_rules(state, rows)
    for r in rows:
        assert (preds[r["run_id"]]["gpu_idx"] != grid.NO_GPU) == arms.mentions_cuda(r["task_code"])
    # and the code tells them apart: every GPU-bench task mentions CUDA, no CPU-bench task does
    assert all(arms.mentions_cuda(r["task_code"]) == (r["source"] == "gpu") for r in rows)


def test_slices_cover_every_source(splits):
    rows = splits["val"] + splits["test"] + splits["ood"]
    preds = {r["run_id"]: dict(zip(("mem_idx", "cpu_idx", "gpu_idx"), scoring.oracle_action(r))) for r in rows}
    sl = scoring.score_slices(rows, preds)
    assert "test/source=gpu" in sl and "ood/source=cpu" in sl


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    pytest.importorskip("peft")
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
        conf = transformers.AutoConfig.from_pretrained("Qwen/Qwen3-1.7B")
    except Exception:  # noqa: BLE001 -- offline
        pytest.skip("Qwen3 tokenizer not available")
    small = dict(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=2,
                 num_key_value_heads=1, head_dim=32)
    if getattr(conf, "layer_types", None):
        small["layer_types"] = conf.layer_types[:2]
    conf = type(conf)(**{**conf.to_dict(), **small})
    d = tmp_path_factory.mktemp("tiny-qwen3")
    transformers.AutoModel.from_config(conf).save_pretrained(d)
    tok.save_pretrained(d)
    return str(d)


def test_policy_offsets_land_on_options(tiny_model, corpus):
    from transformers import AutoTokenizer

    from resource_tuner.bench.llm_arms import _prompt_ids
    from resource_tuner.bench_all.policy import option_offsets

    tok = AutoTokenizer.from_pretrained(tiny_model)
    offs = option_offsets(tok)
    for r in corpus[:: max(1, len(corpus) // 10)]:
        ids = _prompt_ids(tok, r, 8192, prompting.SPEC)
        n = len(ids)
        for axis_offs, opts in zip(offs, prompting.OPTIONS):
            for off, o in zip(axis_offs, opts):
                assert tok.decode([ids[n - 1 - off]]).strip() in o


def test_supervised_then_rl_end_to_end(tiny_model, splits, monkeypatch):
    """train_supervised -> logits -> calibrate -> decode, then train_rl: the call shapes the T4 stations use."""
    import torch

    from resource_tuner.bench import llm_arms
    from resource_tuner.bench.llm_arms import TrainConfig
    from resource_tuner.bench_all import policy, stations

    monkeypatch.setattr(llm_arms, "_device", lambda: "cpu")
    monkeypatch.setattr(policy, "_device", lambda: "cpu")
    monkeypatch.setattr(llm_arms, "_amp_dtype", lambda: torch.float32)
    monkeypatch.setattr(llm_arms, "_autocast", lambda: torch.autocast("cpu", enabled=False))
    monkeypatch.setattr(policy, "_autocast", lambda: torch.autocast("cpu", enabled=False))
    cfg = TrainConfig(model_name=tiny_model, epochs=1, batch_size=4, grad_accum=1, max_len=1024, head="pointer", bench="all")
    train = dataset.balanced(splits["train"], 6)
    tok, model, head, hist = policy.train_supervised(train, cfg, log=lambda *_: None)
    assert hist and np.isfinite(hist[0]["loss"])
    eval_rows = (splits["val"] or splits["train"])[:6] + splits["test"][:6]
    eval_rows = [dict(r, split="val") for r in eval_rows[:6]] + eval_rows[6:]
    meta: dict = {}
    table = stations._policy_predictions(tok, model, head, eval_rows, cfg, "jev-pointer", meta)
    assert {t["variant"] for t in table} == {"jev-pointer", "jev-pointer-argmax"}
    assert len(meta["temperatures"]) == 3 and meta["gpu_threshold"] in arms.GPU_THRESHOLDS
    rl = policy.RLConfig(k_samples=4, prompts=8, batch_size=4, grad_accum=1)
    curve = policy.train_rl(tok, model, head, train, cfg, rl, log=lambda *_: None)
    assert isinstance(curve, list)
    lgs = policy.logits(tok, model, head, eval_rows, cfg)
    assert [z.shape[1] for z in lgs] == [grid.N_MEM, grid.N_CPU, grid.N_GPU]


def test_factory_graph():
    from resource_tuner.bench_all.factory import rightsize_all_bench

    assert not rightsize_all_bench.validate()


def test_cache_versions_hash_only_shipped_modules():
    """The bundler ships what a deploy imports; a cache version that hashes any
    other file dies on the cluster (rsa-full-1: bench/measure.py)."""
    import subprocess
    import sys

    code = ("import sys, resource_tuner.bench_all.stations as s, resource_tuner.bench.versioning as v\n"
            "s.all_corpus_cache(); s.all_arms_cache()\n"
            "names = [n if '/' in n else 'bench/' + n for n in (*v._ALL_CORPUS, *v._ALL_ARMS)]\n"
            "missing = [n for n in names if 'resource_tuner.' + n.replace('/', '.') not in sys.modules]\n"
            "print(missing)\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert out == "[]", out
