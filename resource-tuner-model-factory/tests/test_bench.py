"""RightSize Bench: the properties the leaderboard's honesty rests on.

The load-bearing tests:
- `test_features_cannot_see_the_answer` -- the information boundary. No
  arm may read the generator's knobs, the family, or anything measured on
  the run being decided.
- `test_oracle_has_zero_regret` / `test_underprovisioning_costs_retries` --
  the cost model ranks the way the product would.
- `test_splits_are_by_task` -- no test task's code is ever trained on.
- `test_cross_module_shapes` -- the station's call chain runs end to end
  outside flyte (these only fail on the cluster otherwise; see CLAUDE.md).
"""

from __future__ import annotations

import json
import math
from random import Random

import numpy as np
import pytest

from resource_tuner.bench import arms, dataset, features, grid, prompting, scoring
from resource_tuner.bench.workloads import (
    FAMILIES,
    HELDOUT_FAMILIES,
    describe_input,
    make_task,
    render_harness,
    render_task,
)


# ── fixtures ────────────────────────────────────────────────────────────


def _fake_measure(row: dict, rng: Random) -> dict:
    """A measurement with a plausible, input-dependent peak (no execution)."""
    inputs = json.loads(row["inputs_json"])
    nbytes = {}
    for s in inputs:
        if s["fmt"] == "npy":
            nbytes[s["name"]] = int(math.prod(s["shape"]) * (1 if s["dtype"] == "uint8" else 8))
        else:
            nbytes[s["name"]] = int(s["rows"] * len(s["columns"]) * 8)
    total = sum(nbytes.values())
    knobs = json.loads(row["knobs_json"])
    mult = 0.2 if any(v in ("streaming", "stream", "mmap_batched") for v in knobs.values()) else 3.0
    return {"ok": True, "censored": False, "input_bytes": nbytes, "peak_rss_mib": 150 + mult * total / 2**20 * rng.uniform(0.9, 1.1),
            "avg_cores": rng.choice([1.0, 1.0, 2.0]), "wall_s": 3.0, "t_peak_frac": 0.5, "prep_s": 1.0}


@pytest.fixture(scope="module")
def corpus():
    rng = Random(0)
    rows = dataset.generate_runs(8, seed=1)
    return dataset.attach_history([dataset.attach_measurement(r, _fake_measure(r, rng)) for r in rows])


@pytest.fixture(scope="module")
def splits(corpus):
    out = {s: [] for s in dataset.SPLITS}
    for r in corpus:
        out[r["split"]].append(r)
    return out


# ── grid ────────────────────────────────────────────────────────────────


def test_grid_is_increasing_and_formats_round_trip():
    assert list(grid.MEMORY_GRID_MIB) == sorted(set(grid.MEMORY_GRID_MIB))
    for m in grid.MEMORY_GRID_MIB:
        assert grid.parse_memory(grid.format_memory(m)) == m


def test_half_power_of_two_grid_bounds_the_waste_floor():
    ratios = [b / a for a, b in zip(grid.MEMORY_GRID_MIB, grid.MEMORY_GRID_MIB[1:])]
    assert max(ratios) <= 1.51


def test_label_is_the_cheapest_cell_that_fits():
    for peak in (50, 300, 1000, 5000, 15000):
        i = grid.label_memory_index(peak)
        assert grid.MEMORY_GRID_MIB[i] >= grid.required_memory_mib(peak)
        assert i == 0 or grid.MEMORY_GRID_MIB[i - 1] < grid.required_memory_mib(peak)


# ── workloads ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_every_family_renders_code_that_compiles(family):
    for seed in range(6):
        t = make_task(family, seed)
        compile(render_task(t), "task", "exec")
        compile(render_harness(t), "harness", "exec")
        # the text an arm reads is the text that runs
        for line in t.body.splitlines():
            assert line in render_task(t) and line in render_harness(t)


def test_the_policy_view_never_states_the_input_size():
    # sizes arrive as input metadata at submit time, never as literals in code
    for family in FAMILIES:
        t = make_task(family, 3)
        code = render_task(t)
        for s in t.inputs:
            if s["fmt"] != "npy":
                assert f"{s['rows']:_}" not in code and str(s["rows"]) not in code


def test_csv_and_jsonl_expose_no_row_count():
    spec = {"name": "x", "fmt": "csv", "rows": 123_456, "columns": [{"name": "a", "kind": "f64"}]}
    assert "123" not in describe_input(spec, 1000)
    spec["fmt"] = "jsonl"
    assert "123" not in describe_input(spec, 1000)
    spec["fmt"] = "parquet"
    assert "123,456" in describe_input(spec, 1000)


# ── dataset ─────────────────────────────────────────────────────────────


def test_splits_are_by_task(corpus):
    split_of: dict[str, set] = {}
    for r in corpus:
        split_of.setdefault(r["task_id"], set()).add(r["split"])
    assert all(len(s) == 1 for s in split_of.values())


def test_heldout_families_are_only_ood(corpus):
    for r in corpus:
        assert (r["split"] == "ood") == (r["family"] in HELDOUT_FAMILIES)


def test_history_is_only_earlier_runs(corpus):
    for r in corpus:
        hist = json.loads(r["history_json"])
        assert len(hist) == r["n_history"] <= r["run_idx"]
    assert any(r["n_history"] == 0 for r in corpus) and any(r["n_history"] == 2 for r in corpus)


# ── the information boundary ────────────────────────────────────────────


def test_features_cannot_see_the_answer(corpus):
    r = dict(corpus[5])
    base = features.row_features(r)
    poisoned = dict(r, knobs_json='{"mode": "streaming"}', family="other", peak_rss_mib=1e9,
                    required_mib=1e9, label_mem_idx=0, avg_cores=99.0, memory_driver="code")
    assert features.row_features(poisoned) == base


def test_prompt_cannot_see_the_answer(corpus):
    r = dict(corpus[5])
    base = prompting.user_prompt(r)
    poisoned = dict(r, knobs_json='{"mode": "streaming"}', family="zzz", peak_rss_mib=123456.0,
                    required_mib=123456.0, label_mem_idx=0)
    assert prompting.user_prompt(poisoned) == base
    assert "123456" not in base


# ── scoring ─────────────────────────────────────────────────────────────


def test_oracle_has_zero_regret(splits):
    rows = splits["test"]
    preds = {r["run_id"]: {"mem_idx": r["label_mem_idx"], "cpu_idx": r["label_cpu_idx"]} for r in rows}
    s = scoring.score(rows, preds)
    assert s["mem_regret"] == pytest.approx(0.0, abs=1e-12) and s["oom_rate"] == 0
    assert s["regret"] == pytest.approx(0.0, abs=1e-12) and s["cpu_exact"] == 1.0


def test_underprovisioning_costs_retries():
    need = 3000.0
    fit = scoring.simulate_run(grid.mem_index_covering(need), 1, need, 1.0, 0.5)
    under = scoring.simulate_run(grid.mem_index_covering(need) - 2, 1, need, 1.0, 0.5)
    over = scoring.simulate_run(grid.mem_index_covering(need) + 2, 1, need, 1.0, 0.5)
    assert under.oom_first and under.attempts >= 2 and not under.failed
    assert fit.cost < over.cost and fit.cost < under.cost
    assert fit.gib_hours < over.gib_hours and fit.gib_hours < under.gib_hours


def test_bayes_decode_of_a_certain_distribution_is_that_cell():
    for j in (0, 4, 9, 14):
        p = np.zeros(scoring.N_MEM)
        p[j] = 1.0
        assert scoring.bayes_mem_index(p) == j


def test_bayes_decode_hedges_up_under_uncertainty():
    p = np.zeros(scoring.N_MEM)
    p[6], p[7] = 0.5, 0.5
    assert scoring.bayes_mem_index(p) == 7  # an OOM costs more than one cell of waste


def test_ece_is_zero_when_calibrated():
    assert scoring.expected_calibration_error([0.0, 0.0, 1.0, 1.0], [False, False, True, True]) == 0.0


# ── arms ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("arm", sorted(arms.ARMS))
def test_every_cpu_arm_fits_and_predicts(arm, splits):
    fit, predict = arms.ARMS[arm]
    state = fit(splits["train"], splits["val"])
    preds = predict(state, splits["test"])
    assert set(preds) == {r["run_id"] for r in splits["test"]}
    s = scoring.score(splits["test"], preds)
    assert s["n"] == len(splits["test"]) and s["mem_regret"] >= -1e-9


def test_quantiles_to_probs_is_a_distribution():
    levels = np.array(arms.DIST_QUANTILES)
    qs = np.linspace(9, 12, len(levels))
    p = arms.quantiles_to_cell_probs(qs, levels)
    assert p.shape == (scoring.N_MEM,) and p.sum() == pytest.approx(1.0) and (p > 0).all()


# ── prompting ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("text,expected", [
    ('{"memory": "3Gi", "cpu": 1}', ("3Gi", 1)),
    ('sure: {"memory": "1.5Gi", "cpu": 2}', ("1.5Gi", 2)),
    ('{"memory": "1000Mi", "cpu": "500m"}', ("1Gi", 0.5)),
])
def test_parse_answer_snaps_up(text, expected):
    m, c = prompting.parse_answer(text)
    assert grid.MEMORY_GRID_MIB[m] == grid.parse_memory(expected[0])
    assert grid.CPU_GRID[c] == expected[1]


def test_parse_answer_rejects_junk():
    assert prompting.parse_answer("I think 4 gigs") is None


def test_target_text_round_trips():
    for m in range(scoring.N_MEM):
        for c in range(scoring.N_CPU):
            assert prompting.parse_answer(prompting.target_text(m, c)) == (m, c)


# ── cross-module call shapes ────────────────────────────────────────────


def test_cross_module_shapes(splits, corpus):
    from resource_tuner.bench import stations

    pred_rows, meta = stations._fit_cpu_arm("gbt-dist", splits)
    import pandas as pd

    by_variant = stations._preds_by_variant(pd.DataFrame(pred_rows))
    card = scoring.score_slices([r for r in corpus if dataset.usable(r)], by_variant["gbt-dist"])
    assert "split=test" in card and card["split=test"]["n"] > 0
    assert "ms_per_decision" in meta


def test_factory_graph():
    from resource_tuner.bench.factory import (
        ARTIFACT_ARM,
        ARTIFACT_LEADERBOARD,
        PARTITION_ARM,
        PARTITION_RELEASE,
        build_graph,
    )

    g = build_graph()
    assert g.validate() == []
    assert set(g._nodes[ARTIFACT_ARM].partitions) == {PARTITION_RELEASE, PARTITION_ARM}
    assert set(g._nodes[ARTIFACT_LEADERBOARD].partitions) == {PARTITION_RELEASE}


def test_pointer_offsets_land_on_their_options(corpus):
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer

    from resource_tuner.bench.llm_arms import _prompt_ids, option_offsets

    try:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    except Exception:  # noqa: BLE001 -- offline
        pytest.skip("tokenizer not available")
    mem, cpu = option_offsets(tok)
    for r in corpus[:20]:
        ids = _prompt_ids(tok, r, 4096)
        n = len(ids)
        for off, m in zip(mem, grid.MEMORY_GRID_MIB):
            assert tok.decode([ids[n - 1 - off]]) in grid.format_memory(m)
        for off, c in zip(cpu, grid.CPU_GRID):
            assert tok.decode([ids[n - 1 - off]]) in f"{c:g}"


def test_no_lazy_package_imports():
    """The code bundler ships only statically-imported modules; a relative
    import inside a function body imports fine locally and fails on the
    cluster (pilot-board-3 lost both zero-shot arms to `llm_client`)."""
    import ast
    import pathlib

    import resource_tuner.bench as pkg

    offenders = []
    for path in pathlib.Path(pkg.__file__).parent.glob("*.py"):
        tree = ast.parse(path.read_text())
        for fn in ast.walk(tree):
            if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for node in ast.walk(fn):
                    if isinstance(node, ast.ImportFrom) and node.level > 0:
                        offenders.append(f"{path.name}:{node.lineno} from {'.' * node.level}{node.module or ''}")
    assert not offenders, offenders
