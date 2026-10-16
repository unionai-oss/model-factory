"""The decision-model arm: cost matrix, training, and the decision frontier.

The headline test is `test_higher_under_penalty_reduces_under_provisioning`:
it is the claim the whole arm rests on — that `under_penalty` is a real knob
on the OOM/waste tradeoff and not just a number in a log.
"""

import json
import random
import re

import pytest

from resource_tuner.policy.action_space import DEFAULT_SPACE, get_action_space
from resource_tuner.training.decision_model import (
    ARMS,
    DEFAULT_ARMS,
    DecisionArm,
    cost_matrix,
    featurize,
    feature_names,
    fit,
    get_arm,
    gpu_cost_matrix,
    labels_from_records,
    load,
    save,
)

torch = pytest.importorskip("torch")


def make_records(n: int = 240, seed: int = 0) -> list[dict]:
    """Synthetic corpus rows whose footprint is a learnable function of params.

    Memory scales with `rows`, CPU with `workers`, and a third of the rows are
    GPU workloads — enough structure that a working model beats chance and a
    broken one does not.
    """
    rng = random.Random(seed)
    out = []
    for i in range(n):
        rows = rng.choice([1_000, 10_000, 100_000, 1_000_000])
        workers = rng.choice([1, 2, 4, 8])
        gpu = i % 3 == 0
        peak = rows * 0.002 + 180
        out.append(
            {
                "task_id": f"t{i}",
                "family": "ml_training" if gpu else "etl",
                "source_code": "import torch\nmodel.backward()" if gpu else "import pandas as pd\ndf.groupby('k')",
                "harness_code": "",
                "input_profile": f"{rows} rows of parquet",
                "params_json": json.dumps({"rows": rows, "workers": workers}),
                "generator": "template",
                "prior_json": "",
                "history_json": "",
                "true_peak_memory_mib": peak,
                "true_cpu_cores": float(workers),
                "true_gpu_mem_mib": 6000.0 if gpu else 0.0,
                "duration_s": 60,
                "split": "train",
            }
        )
    return out


# ── the cost matrix ─────────────────────────────────────────────────────


def test_the_right_choice_costs_nothing():
    c = cost_matrix([1, 2, 4, 8], under_penalty=10.0)
    for i in range(4):
        assert c[i][i] == 0.0


def test_under_provisioning_costs_the_flat_penalty():
    c = cost_matrix([1, 2, 4, 8], under_penalty=10.0)
    # Every cell below the diagonal is the penalty, regardless of distance:
    # the task is already dead at one bucket under.
    assert c[3][0] == 10.0
    assert c[3][2] == 10.0
    assert c[1][0] == 10.0


def test_over_provisioning_costs_the_extra_resource_paid_for():
    c = cost_matrix([1, 2, 4, 8], under_penalty=10.0)
    # Choosing 2x the needed bucket costs 1.0 (one extra multiple).
    assert c[0][1] == pytest.approx(1.0)
    assert c[0][2] == pytest.approx(3.0)
    assert c[1][2] == pytest.approx(1.0)


def test_under_provisioning_is_dearer_than_a_one_step_overshoot():
    # The whole asymmetry, in one assertion.
    c = cost_matrix(DEFAULT_SPACE.memory_step_ratios(), under_penalty=12.0)
    n = len(c)
    for t in range(1, n - 1):
        assert c[t][t - 1] > c[t][t + 1]


def test_gpu_cost_matrix_has_the_same_shape_and_asymmetry():
    g = gpu_cost_matrix(4, under_penalty=9.0)
    assert g[2][2] == 0.0
    assert g[2][0] == 9.0  # CPU-only for a GPU task = under-provision
    assert g[0][3] == 3.0  # biggest card for a CPU task = waste


# ── features ────────────────────────────────────────────────────────────


def test_features_are_stable_in_order_and_width():
    records = make_records(20)
    names = feature_names(records)
    matrix = featurize(records, names)
    assert len({len(row) for row in matrix}) == 1
    assert len(matrix[0]) == len(names)
    # Sorted, so a checkpoint's input layout is reproducible across processes.
    assert names == sorted(names)


def test_features_are_squashed_and_finite():
    records = make_records(30)
    names = feature_names(records)
    for row in featurize(records, names):
        for value in row:
            assert value == value  # not NaN
            assert abs(value) < 1e3  # log1p-squashed, not raw byte counts


def test_labels_cover_the_ground_truth():
    space = DEFAULT_SPACE
    records = make_records(50)
    for record, (mem_i, cpu_i, gpu_i) in zip(records, labels_from_records(records, space)):
        assert space.memory_grid_mib[mem_i] >= record["true_peak_memory_mib"]
        assert space.cpu_grid[cpu_i] >= record["true_cpu_cores"]


# ── arms ────────────────────────────────────────────────────────────────


def test_the_registered_suite_spans_the_frontier():
    penalties = [ARMS[a].under_penalty for a in DEFAULT_ARMS]
    assert penalties == sorted(penalties)
    assert len(set(penalties)) == len(penalties), "arms must differ in asymmetry"


def test_every_default_arm_is_registered_and_has_a_valid_space():
    for name in DEFAULT_ARMS:
        arm = get_arm(name)
        assert get_action_space(arm.action_space) is arm.space
    with pytest.raises(ValueError, match="unknown decision arm"):
        get_arm("nope")


# ── training ────────────────────────────────────────────────────────────


def test_training_runs_and_the_loss_goes_down():
    arm = DecisionArm(name="t", epochs=12, batch_size=64, seed=3)
    records = make_records(240)
    model = fit(records, arm, val_records=make_records(80, seed=99))
    assert len(model.history) == 12
    assert model.history[-1]["train_loss"] < model.history[0]["train_loss"]


def test_predictions_are_decodable_proposals():
    arm = DecisionArm(name="t", epochs=6, seed=5)
    records = make_records(160)
    model = fit(records, arm)
    proposals = model.predict(records[:10])
    assert len(proposals) == 10
    for p in proposals:
        kwargs = p.to_kwargs()
        assert "cpu" in kwargs and "memory" in kwargs
        assert p.memory_mib in arm.space.memory_grid_mib
        assert p.cpu in arm.space.cpu_grid


def test_predict_on_nothing_returns_nothing():
    model = fit(make_records(80), DecisionArm(name="t", epochs=2))
    assert model.predict([]) == []


def test_fit_rejects_an_empty_corpus():
    with pytest.raises(ValueError, match="no records"):
        fit([], DecisionArm(name="t", epochs=1))


def test_higher_under_penalty_reduces_under_provisioning():
    """The claim the arm exists to test: the asymmetry knob moves behaviour.

    A paranoid arm must under-provision memory no more often than a
    near-symmetric one on the same data and seed. Compared with `<=` rather
    than `<` because on an easy synthetic corpus a balanced arm can already
    reach zero under-provisioning, and a strict inequality would then fail for
    the right reason.
    """
    train = make_records(400, seed=1)
    val = make_records(200, seed=2)
    common = dict(epochs=25, batch_size=64, seed=11)
    balanced = fit(train, DecisionArm(name="b", under_penalty=1.0, **common), val_records=val)
    paranoid = fit(train, DecisionArm(name="p", under_penalty=60.0, **common), val_records=val)

    b_under = balanced.history[-1]["val_memory_under"]
    p_under = paranoid.history[-1]["val_memory_under"]
    assert p_under <= b_under + 1e-9, f"paranoid={p_under:.3f} balanced={b_under:.3f}"


# ── persistence ─────────────────────────────────────────────────────────


def test_save_load_round_trips_weights_grid_and_features(tmp_path):
    arm = DecisionArm(name="t", action_space="coarse", under_penalty=7.5, epochs=5, seed=2)
    records = make_records(120)
    model = fit(records, arm)
    out = str(tmp_path / "model")
    save(model, out, extra_manifest={"n_train": len(records)})

    back = load(out)
    assert back.arm.name == "t"
    assert back.arm.under_penalty == 7.5
    # The grid is part of the model's identity: decoded with the wrong one,
    # every proposal would be wrong.
    assert back.space.name == "coarse"
    assert back.feature_names == model.feature_names
    # Same inputs, same decisions.
    assert [p.to_kwargs() for p in back.predict(records[:8])] == [
        p.to_kwargs() for p in model.predict(records[:8])
    ]


def test_manifest_carries_the_grid_and_the_feature_layout(tmp_path):
    model = fit(make_records(80), DecisionArm(name="t", epochs=3))
    out = str(tmp_path / "m")
    save(model, out)
    with open(f"{out}/manifest.json") as f:
        manifest = json.load(f)
    assert manifest["action_space"]["memory_grid_mib"]
    assert manifest["feature_names"]
    assert manifest["arm"]["under_penalty"] == model.arm.under_penalty


# ── corpus reading: the station's memory safety ─────────────────────────


def _write_corpus(path, n_train=500, n_eval=120):
    """A parquet shaped like the real corpus, including the big code columns."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = []
    for split, n in (("train", n_train), ("heldout", n_eval)):
        for i in range(n):
            rows.append(
                {
                    "task_id": f"{split}-{i}",
                    "family": "etl",
                    # Deliberately bulky: this is what must NOT be read twice.
                    "source_code": "import pandas as pd\n" + ("# pad\n" * 50),
                    "harness_code": "x" * 2000,
                    "input_profile": "1000 rows of parquet",
                    "params_json": json.dumps({"rows": 1000, "workers": 2}),
                    "generator": "template",
                    "prior_json": "",
                    "history_json": "",
                    "true_peak_memory_mib": 700.0,
                    "true_cpu_cores": 2.0,
                    "true_gpu_mem_mib": 0.0,
                    "duration_s": 30,
                    "split": split,
                }
            )
    pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=256)
    return str(path)


def test_read_corpus_splits_and_respects_both_caps(tmp_path):
    from resource_tuner.training.decision_station import read_corpus

    path = _write_corpus(tmp_path / "c.parquet", n_train=500, n_eval=120)
    train, heldout = read_corpus(path, max_train=100, max_eval=25)
    assert len(train) == 100
    assert len(heldout) == 25
    assert {r["split"] for r in train} == {"train"}
    assert {r["split"] for r in heldout} == {"heldout"}


def test_read_corpus_projects_away_unused_columns(tmp_path):
    # `harness_code` is a second full copy of every workload's source; reading
    # it would double the station's memory for nothing.
    from resource_tuner.training.decision_station import read_corpus

    path = _write_corpus(tmp_path / "c.parquet", n_train=50, n_eval=10)
    train, _ = read_corpus(path, max_train=10, max_eval=5)
    assert "harness_code" not in train[0]
    assert "source_code" in train[0]  # the features DO read this one


def test_read_corpus_can_skip_the_train_split_entirely(tmp_path):
    # Scoring needs only heldout, so asking for 0 train rows must not load them.
    from resource_tuner.training.decision_station import read_corpus

    path = _write_corpus(tmp_path / "c.parquet", n_train=400, n_eval=30)
    train, heldout = read_corpus(path, max_train=0, max_eval=30)
    assert train == []
    assert len(heldout) == 30


def test_read_corpus_returns_what_exists_when_under_the_cap(tmp_path):
    from resource_tuner.training.decision_station import read_corpus

    path = _write_corpus(tmp_path / "c.parquet", n_train=20, n_eval=5)
    train, heldout = read_corpus(path, max_train=1000, max_eval=1000)
    assert len(train) == 20 and len(heldout) == 5


def test_records_from_the_reader_are_trainable(tmp_path):
    # The reader's row dicts must satisfy the feature extractor as-is.
    from resource_tuner.training.decision_station import read_corpus

    path = _write_corpus(tmp_path / "c.parquet", n_train=200, n_eval=60)
    train, heldout = read_corpus(path, max_train=200, max_eval=60)
    model = fit(train, DecisionArm(name="t", epochs=3, seed=4), val_records=heldout)
    assert model.predict(heldout[:5])


def test_to_tensor_matches_the_list_featurizer(tmp_path):
    # The memory-safe path must produce the same numbers as the simple one.
    from resource_tuner.training.decision_model import to_tensor

    records = make_records(40)
    names = feature_names(records)
    tensor = to_tensor(records, names)
    listed = featurize(records, names)
    assert tensor.shape == (len(records), len(names))
    for i in range(len(records)):
        for j in range(len(names)):
            assert float(tensor[i, j]) == pytest.approx(listed[i][j], abs=1e-6)


# ── report API ──────────────────────────────────────────────────────────


def test_the_station_only_calls_reporter_methods_that_exist():
    """Guards the bug that failed run rt-decision-r2 on all three arms.

    The station's reports are built with `Reporter`, whose API is `kv` /
    `h` / `table` / `p` — not `stats`, and `table` takes no `title=`. Those
    calls are inside async flyte tasks, so a typo in one surfaces only as a
    RuntimeUserError on the cluster, after the model has already trained.
    Checking the surface statically costs nothing and fails locally instead.
    """
    import inspect

    from resource_tuner.shared.reporting import Reporter
    from resource_tuner.training import decision_station

    source = inspect.getsource(decision_station)
    called = set(re.findall(r"\brep\.(\w+)\(", source))
    assert called, "no Reporter calls found — did the station stop reporting?"
    for name in sorted(called):
        assert hasattr(Reporter, name), f"Reporter has no method {name!r}"

    # `table` is positional-only here (headers, rows); a stray title= kwarg
    # is exactly what broke rt-decision-r2's sibling call.
    params = set(inspect.signature(Reporter.table).parameters)
    assert "title" not in params


def test_the_scoring_composition_runs_outside_flyte(tmp_path):
    """Exercises exactly what `score_decision_model` does, minus the task shell.

    Guards the bug that failed run rt-decision-r3: `baseline_proposal` is a
    LOOKUP into a per-family baseline that `fit_family_baseline` has to build
    from the train split first, not a function of a single record. Calls
    inside an async flyte task only fail on the cluster, after every model in
    the suite has already trained, so the composition is pinned here.
    """
    from resource_tuner.training.baseline import baseline_proposal, fit_family_baseline
    from resource_tuner.training.decision_station import _simulate, read_corpus

    path = _write_corpus(tmp_path / "c.parquet", n_train=300, n_eval=80)
    train, heldout = read_corpus(path, max_train=300, max_eval=80)

    model = fit(train, DecisionArm(name="t", epochs=4, seed=6), val_records=heldout)
    tuned = _simulate(heldout, model.predict(heldout))

    baselines = fit_family_baseline(train)
    base = _simulate(heldout, [baseline_proposal(baselines, r["family"]) for r in heldout])

    for row in (tuned, base):
        assert 0.0 <= row["fit_rate"] <= 1.0
        assert row["fit_rate"] + row["oom_rate"] == pytest.approx(1.0)
        assert row["cost_per_task_hr"] is not None and row["cost_per_task_hr"] > 0


def test_fit_and_oom_rates_are_complementary():
    # An episode either fit or OOMed; a third outcome would silently drop
    # episodes out of both numbers on the scorecard.
    from resource_tuner.training.decision_station import _simulate
    from resource_tuner.policy.actions import Proposal

    records = make_records(40)
    # Deliberately far too small: everything should OOM.
    tiny = [Proposal(cpu=0.5, memory_mib=128) for _ in records]
    row = _simulate(records, tiny)
    assert row["oom_rate"] == 1.0 and row["fit_rate"] == 0.0

    # Deliberately huge: everything should fit.
    huge = [Proposal(cpu=16, memory_mib=65536, gpu=1, gpu_type="L40S") for _ in records]
    row = _simulate(records, huge)
    assert row["fit_rate"] == 1.0 and row["oom_rate"] == 0.0
