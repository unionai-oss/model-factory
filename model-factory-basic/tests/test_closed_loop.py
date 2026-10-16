"""The pure logic the loop rests on: data windows, leak checks, the curriculum,
the comparison and the release decision."""

from datetime import date, datetime

import pandas as pd
import pytest

from model_factory.contracts import find_leaks, heldout_fingerprint, question_key
from model_factory.data_engineering.synthetic import pick_seeds
from model_factory.data_engineering.tasks import day_window, normalize_rows
from model_factory.evaluation.metrics import paired_delta, pass_at_1, promotion_decision
from model_factory.training.tasks import hard_tasks

N, POOL, SIZE = 10_000, 600, 768


# ── held-out isolation ──────────────────────────────────────────────────


def test_a_day_window_never_touches_the_heldout_pool():
    for d in range(400):
        idx = day_window(date(2026, 1, 1).fromordinal(date(2026, 1, 1).toordinal() + d), N, POOL, SIZE)
        assert min(idx) >= POOL and max(idx) < N
        assert len(set(idx)) == SIZE


def test_consecutive_days_see_different_problems():
    d1 = set(day_window(date(2026, 10, 13), N, POOL, SIZE))
    d2 = set(day_window(date(2026, 10, 14), N, POOL, SIZE))
    assert not d1 & d2


def test_the_window_accepts_the_injected_datetime():
    assert day_window(datetime(2026, 10, 13), N, POOL, SIZE) == day_window(date(2026, 10, 13), N, POOL, SIZE)


def test_no_rows_left_for_training_is_an_error():
    with pytest.raises(ValueError):
        day_window(date(2026, 10, 13), 100, 100, 10)


def test_question_key_ignores_whitespace_and_case():
    assert question_key("Sum  two\nnumbers.") == question_key("sum two numbers.")


def test_find_leaks_flags_heldout_questions():
    held = ["Reverse a string.", "Sum a list."]
    assert find_leaks(["sum a list.", "Sort a list.", "Reverse  a string."], held) == [0, 2]
    assert find_leaks(["Sort a list."], held) == []


def test_normalize_rows_excludes_heldout_and_dedups():
    tests = "from solution import f\n" + "".join(f"def test_{i}():\n    assert f()\n" for i in range(3))
    raw = [
        {"question": "A", "solution": "def f(): return 1", "test": tests, "question_id": "a"},
        {"question": "a", "solution": "def f(): return 1", "test": tests, "question_id": "dup"},
        {"question": "Held", "solution": "def f(): return 1", "test": tests, "question_id": "h"},
        {"question": "B", "solution": "", "test": tests},
    ]
    rows, dropped = normalize_rows(raw, 3, exclude={question_key("held")})
    assert [r["task_id"] for r in rows] == ["a"]
    assert dropped == {"dup": 1, "few_tests": 0, "schema": 1, "heldout": 1}


def test_fingerprint_is_order_independent():
    assert heldout_fingerprint(["b", "a"]) == heldout_fingerprint(["a", "b"])
    assert heldout_fingerprint(["a"]) != heldout_fingerprint(["a", "b"])


# ── the curriculum ──────────────────────────────────────────────────────


def _tasks(ids):
    return pd.DataFrame({"task_id": ids, "question": [f"q-{i}" for i in ids]})


def test_hard_tasks_are_the_ones_the_policy_kept_failing_hardest_first():
    df = _tasks(["a", "b", "c", "d"])
    stats = {"a": [8, 8], "b": [0, 8], "c": [3, 8], "d": [5, 8]}
    hard = hard_tasks(df, stats)
    assert list(hard["task_id"]) == ["b", "c"]
    assert list(hard["rollout_pass_rate"]) == [0.0, 3 / 8]


def test_unseen_tasks_are_not_called_hard():
    assert hard_tasks(_tasks(["a"]), {}).empty


def test_seeds_come_from_champion_failures_first():
    hard = _tasks(["h1", "h2"])
    pool = _tasks(["p1", "p2", "p3", "p4"])
    seeds = pick_seeds(hard, pool, 4)
    assert list(seeds["origin"]) == ["champion-failure", "champion-failure", "random", "random"]
    assert list(seeds["task_id"][:2]) == ["h1", "h2"]


def test_day_one_seeds_are_random():
    seeds = pick_seeds(None, _tasks(["p1", "p2", "p3"]), 2)
    assert set(seeds["origin"]) == {"random"} and len(seeds) == 2


def test_more_failures_than_seed_slots_takes_the_hardest():
    seeds = pick_seeds(_tasks(["h1", "h2", "h3"]), _tasks(["p1"]), 2)
    assert list(seeds["task_id"]) == ["h1", "h2"]


# ── the comparison and the decision ─────────────────────────────────────


def test_paired_delta_counts_wins_and_losses():
    a = [True, True, False, True]
    b = [True, False, True, False]
    out = paired_delta(a, b)
    assert out["delta"] == pytest.approx(pass_at_1(a) - pass_at_1(b))
    assert (out["wins"], out["losses"]) == (2, 1)
    assert out["ci95"][0] <= out["delta"] <= out["ci95"][1]


def test_identical_models_have_a_zero_width_interval():
    a = [True, False] * 50
    assert paired_delta(a, a)["ci95"] == [0.0, 0.0]


def test_paired_delta_rejects_unpaired_inputs():
    with pytest.raises(ValueError):
        paired_delta([True], [True, False])


def test_release_keeps_the_champion_below_the_margin():
    assert promotion_decision(0.005, 0.01, approved=True)[0] == "champion"


def test_release_keeps_the_champion_when_a_human_says_no():
    assert promotion_decision(0.2, 0.01, approved=False)[0] == "champion"


def test_release_promotes_a_candidate_that_clears_the_margin():
    assert promotion_decision(0.01, 0.01, approved=True)[0] == "candidate"
