"""Publishing RightSize Bench to the Hugging Face Hub (round 24): ``bench_all.hub``.

What a release rests on: the cheapest-request oracle never costs more than the
label and is never beaten by the label; every published pod row scores with
the repo's simulator unchanged; predictions and the leaderboard agree; the
cards carry valid Hub metadata; and the standalone inference file shipped in
each model repo reproduces the training code's scores.
"""

from __future__ import annotations

import json
import os

import pandas as pd
import pytest
import yaml

from resource_tuner.bench_all import dataset, hub, hub_cards, hub_pointer, prompting, scoring

from test_bench_all import corpus, tiny_model  # noqa: F401 -- the fake-measured corpus and a tiny Qwen3


@pytest.fixture(scope="module")
def twinned(corpus):  # noqa: F811
    return [r for r in dataset.with_k8s(corpus) if dataset.usable(r)]


def test_cheapest_request_is_never_dearer_than_the_label(twinned):
    for r in twinned:
        m, c, g = scoring.cheapest_action(r)
        assert scoring.min_cost(r) <= scoring.cost(r, *scoring.oracle_action(r)).cost + 1e-12
        if r["source"] != "gpu":
            assert g == 0  # a CPU task's cheapest request has no GPU
        assert scoring.score([r], {r["run_id"]: {"mem_idx": m, "cpu_idx": c, "gpu_idx": g}},
                             oracle="cheapest")["regret"] == pytest.approx(0.0, abs=1e-9)


def test_pod_records_score_unchanged(twinned):
    recs = [hub.pod_record(r) for r in twinned]
    assert {r["platform"] for r in recs} == {"job", "flyte"}
    assert {r["split"] for r in recs} <= {"train", "validation", "test", "unseen"}
    for rec, r in zip(recs, twinned):
        assert rec["prompt_user"] == prompting.user_prompt(r)
        assert (rec["label_memory"] is None) == (r["label_mem"] < 0)
    labels = {r["run_id"]: dict(zip(("mem_idx", "cpu_idx", "gpu_idx"), scoring.oracle_action(r))) for r in twinned}
    assert scoring.score(recs, labels)["regret"] == pytest.approx(scoring.score(twinned, labels)["regret"])


def test_predictions_and_leaderboard_agree(twinned):
    picks = {1: {"rules": {r["run_id"]: scoring.oracle_action(r) for r in twinned},
                 "static": {r["run_id"]: (8, 3, 0) for r in twinned}}}
    preds = hub.predictions_table(twinned, picks)
    assert set(preds["approach"]) == {"rules", "static"}
    board = hub.leaderboard_table(preds)
    for (a, s, p), part in preds.groupby(["approach", "split", "platform"]):
        row = board[(board["approach"] == a) & (board["split"] == s) & (board["platform"] == p)].iloc[0]
        assert row["regret"] == pytest.approx(part["regret"].mean())
    assert (preds["regret"] >= -1e-9).all()


def test_split_files_follow_the_hub_layout(tmp_path, twinned):
    df = pd.DataFrame([hub.pod_record(r) for r in twinned])
    files = hub.write_split_files(df, str(tmp_path), "pods")
    for f in files:
        parts = f.split("/")
        assert parts[:2] == ["data", "pods"] and parts[2] in ("job", "flyte")
        assert parts[3].endswith("-00000-of-00001.parquet")


def _front_matter(card: str) -> dict:
    assert card.startswith("---\n")
    return yaml.safe_load(card.split("---\n")[1])


def test_cards_carry_hub_metadata(twinned):
    preds = hub.predictions_table(twinned, {1: {"gbt-dist": {r["run_id"]: scoring.oracle_action(r) for r in twinned}}})
    board = hub.leaderboard_table(preds)
    files = {"pods": ["data/pods/job/train-00000-of-00001.parquet", "data/pods/flyte/test-00000-of-00001.parquet"],
             "features": ["data/features/job/test-00000-of-00001.parquet"],
             "counterfactual": ["data/counterfactual/test-00000-of-00001.parquet"],
             "predictions": ["data/predictions/test-00000-of-00001.parquet"]}
    stats = {"release": "k8s-2", "pods": 2, "runs": 1, "families": 1, "unseen_families": ["duckdb_sql"],
             "by_split": {"job/train": 1}, "counterfactual_pods": 1, "prediction_rows": len(preds), "seeds": [1],
             "files": files}
    meta = _front_matter(hub_cards.dataset_card(stats, board))
    configs = {c["config_name"]: c for c in meta["configs"]}
    assert configs["pods"]["default"] is True
    assert {"jobs", "flyte_tasks", "features", "counterfactual", "predictions", "leaderboard"} <= set(configs)
    assert [d["split"] for d in configs["pods"]["data_files"]] == ["train", "test"]
    assert meta["license"] == "cc-by-4.0"

    cfg = {"release": "k8s-2", "approach": "rl-pointer", "seed": 1, "base_model": "Qwen/Qwen3-1.7B", "max_len": 3072,
           "keys": list(prompting.KEYS), "temperatures": [1.0, 1.0, 1.0], "train_platforms": "all", "n_train": 10,
           "lora": {"r": 16, "alpha": 32}, "rl": {"k_samples": 16}, "verify": {"checked": 2, "agree": 2}}
    metrics = {"rl-pointer": {"test/job": {"regret": 0.1, "n": 5}}}
    meta = _front_matter(hub_cards.pointer_card(cfg, metrics))
    assert meta["base_model"] == "Qwen/Qwen3-1.7B" and meta["library_name"] == "peft"
    assert meta["model-index"][0]["results"][0]["metrics"][0]["value"] == 10.0
    card = hub_cards.pointer_card(cfg, metrics)
    assert "unionai/rightsize-rl-qwen3-1.7b" in card and "{repo_id}" not in card

    tcfg = {"release": "k8s-2", "levels": [0.05, 0.95], "gpu_threshold": 0.3, "refit_matches_published": 1.0}
    assert _front_matter(hub_cards.trees_card(tcfg, {"gbt-dist": {"test/job": {"regret": 0.2, "n": 3}}}))["library_name"] == "sklearn"


def test_published_inference_file_matches_the_training_code(tiny_model, twinned, tmp_path, monkeypatch):  # noqa: F811
    """Save a tiny decision model the way ``stage_pointer`` does, reload it with
    ``rightsize_pointer.py``, and compare its scores with ``policy.logits``."""
    import numpy as np
    import torch
    from peft import LoraConfig, get_peft_model
    from safetensors.torch import save_file
    from transformers import AutoModel, AutoTokenizer

    from resource_tuner.bench.llm_arms import LORA_TARGETS, TrainConfig
    from resource_tuner.bench_all import policy

    import contextlib

    from resource_tuner.bench_all import policy as policy_mod

    # both sides in fp32 on the CPU, so any difference is the code's, not the precision's
    monkeypatch.setattr(policy_mod, "_device", lambda: "cpu")
    monkeypatch.setattr(policy_mod, "_autocast", contextlib.nullcontext)
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(tiny_model)
    base = AutoModel.from_pretrained(tiny_model, dtype=torch.float32)
    model = get_peft_model(base, LoraConfig(r=4, lora_alpha=8, target_modules=LORA_TARGETS, init_lora_weights=False,
                                            task_type="FEATURE_EXTRACTION")).eval()
    head = policy.head_module(model.config.hidden_size, policy.option_offsets(tok)).eval()
    d = tmp_path / "repo"
    model.save_pretrained(d)
    save_file({k: v.contiguous() for k, v in head.state_dict().items()}, os.path.join(d, hub_pointer.HEAD_FILE))
    with open(os.path.join(d, hub_pointer.CONFIG_FILE), "w") as f:
        json.dump({"base_model": tiny_model, "max_len": 3072, "head_dim": 256, "keys": list(prompting.KEYS),
                   "options": [list(o) for o in prompting.OPTIONS], "temperatures": [1.0, 1.0, 1.0]}, f)

    rp = hub_pointer.RightSizePointer.from_pretrained(str(d), device="cpu", dtype=torch.float32)
    cfg = TrainConfig(model_name=tiny_model, head="pointer", bench="all", max_len=3072)
    rows = [twinned[0], twinned[-1]]
    want = policy.logits(tok, model, head, rows, cfg, batch_size=1)
    for i, r in enumerate(rows):
        got = rp.logits(prompting.messages(r))
        for a in range(3):
            np.testing.assert_allclose(got[a], want[a][i], rtol=1e-4, atol=1e-4)
        req = rp.request(prompting.messages(r))
        assert set(req) == set(prompting.KEYS)
