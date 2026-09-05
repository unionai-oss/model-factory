"""Round 11: graded format reward, capacity/full-FT profiles, GBT-hint
composition."""

import pytest

from resource_tuner import pricing
from resource_tuner.config import PROFILES, R11_FULLFT, R11_GBT, R11_R64
from resource_tuner.policy.parsing import format_credit
from resource_tuner.policy.prompts import render_messages
from resource_tuner.rewards.rewards import FORMAT_REWARD
from resource_tuner.taskgen.corpus import build_corpus
from resource_tuner.training.grpo import _record_to_row, make_reward_fn
from resource_tuner.training.ml_baseline import MLBaseline, out_of_fold_hints


# ── graded parseability ─────────────────────────────────────────────────


def test_format_credit_ladder():
    assert format_credit("I think 4 CPUs should do") == 0.0
    assert format_credit('{"cpu": 2, "memory": }') == 0.25  # braces, bad JSON
    assert format_credit('{"cores": 2, "ram": "4Gi"}') == 0.5  # JSON, wrong schema
    assert format_credit('{"cpu": 2, "memory": "4Gi", "gpu": "H100:1"}') == 0.75
    assert format_credit('{"cpu": 2, "memory": "4Gi", "gpu": "T4:1"}') == 1.0
    assert format_credit('<think>hmm</think>{"cpu": 1, "memory": "1Gi"}') == 1.0


def test_reward_fn_grades_invalid_completions():
    fn = make_reward_fn("c-cost", num_generations=0, max_steps=10)
    rewards = fn(
        ["no json here", '{"cores": 2, "ram": "4Gi"}',
         '{"cpu": 2, "memory": "4Gi", "gpu": "H100:1"}'],
        true_peak_memory_mib=[500.0] * 3,
        true_cpu_cores=[1.0] * 3,
        duration_s=[60] * 3,
    )
    assert rewards[0] == 0.0
    assert rewards[1] == pytest.approx(0.5 * FORMAT_REWARD)
    assert rewards[2] == pytest.approx(0.75 * FORMAT_REWARD)
    assert rewards[0] < rewards[1] < rewards[2]  # a gradient, not a cliff


# ── profiles ────────────────────────────────────────────────────────────


def test_round11_profiles():
    assert R11_R64.lora_r == 64 and R11_R64.lora_mlp and R11_R64.use_lora
    assert R11_GBT.gbt_hint and R11_GBT.use_lora
    assert not R11_FULLFT.use_lora and not R11_FULLFT.use_qlora
    assert "0.6B" in R11_FULLFT.base_model
    for p in (R11_R64, R11_GBT, R11_FULLFT):
        assert p.reward_stage == "c-cost" and p.name in PROFILES
        assert (p.max_steps, p.train_contexts) == (300, 4096)  # r8-comparable


# ── GBT-hint composition ────────────────────────────────────────────────


def test_prompt_renders_ml_estimate():
    msgs = render_messages(
        "def f(): ...", "input: x", ml_estimate={"cpu": 2, "memory": "4Gi"}
    )
    user = msgs[1]["content"]
    assert "Statistical estimate" in user and "'memory': '4Gi'" in user
    assert "Statistical estimate" not in render_messages("def f(): ...", "x")[1]["content"]


def test_out_of_fold_hints_cover_and_stay_deterministic():
    records = build_corpus(180, 0, seed=9)
    a = out_of_fold_hints(records, k=3)
    b = out_of_fold_hints(records, k=3)
    assert a == b
    covered = sum(1 for h in a if h is not None)
    assert covered / len(records) > 0.95
    # tiny splits refuse to fold rather than leak
    assert out_of_fold_hints(records[:10], k=3) == [None] * 10


def test_record_row_uses_hint_for_prompt_and_reward_reference():
    records = build_corpus(120, 0, seed=9)
    ml = MLBaseline().fit(records)
    hint = ml.propose(records[0])
    row = _record_to_row(records[0], baselines=None, ml_hint=hint)
    assert "Statistical estimate" in row["prompt"][1]["content"]
    assert row["baseline_cost_per_hr"] == pytest.approx(
        pricing.dollars_per_hr(hint.cpu, hint.memory_mib, hint.gpu_type, hint.gpu)
    )
    plain = _record_to_row(records[0], baselines=None)
    assert plain["baseline_cost_per_hr"] is None
    assert "Statistical estimate" not in plain["prompt"][1]["content"]
