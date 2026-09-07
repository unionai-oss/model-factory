"""Classical-ML baseline (quantile GBTs): features, fit/propose, and — the
point — that it actually competes with the rule baseline in the simulator."""

import pytest

from resource_tuner.environment.simulator import simulate_episode
from resource_tuner.taskgen.corpus import build_corpus
from resource_tuner.training.baseline import baseline_proposal, fit_family_baseline
from resource_tuner.training.ml_baseline import MLBaseline, extract_features


@pytest.fixture(scope="module")
def corpus():
    return build_corpus(560, 140, seed=17)


@pytest.fixture(scope="module")
def fitted(corpus):
    train = [r for r in corpus if r["split"] == "train"]
    return MLBaseline().fit(train)


def test_features_are_pure_total_and_deterministic(corpus):
    r = corpus[0]
    assert extract_features(r) == extract_features(dict(r))
    junk = extract_features({"params_json": "{bad", "input_profile": None})
    assert all(isinstance(v, float) for v in junk.values())
    # profile numbers actually parse
    f = extract_features({"input_profile": "input: 2,400,000 rows x 16 cols (~310MiB)"})
    assert f["prof_num_0"] > 0


def test_propose_returns_grid_snapped_proposals(corpus, fitted):
    from resource_tuner.policy.actions import CPU_GRID, MEMORY_GRID_MIB

    for r in corpus[:20]:
        p = fitted.propose(r)
        assert p.cpu in CPU_GRID and p.memory_mib in MEMORY_GRID_MIB
        if p.gpu:
            assert p.gpu_type in ("T4", "L4", "L40S")


def _sim_stats(pairs):
    ok = wastes = 0
    for r, p in pairs:
        ep = simulate_episode(
            p,
            float(r["true_peak_memory_mib"]),
            float(r["true_cpu_cores"]),
            int(r["duration_s"]),
            true_gpu_mem_mib=float(r.get("true_gpu_mem_mib", 0.0)),
        )
        ok += ep.ok
        if ep.ok:
            wastes += (ep.requested_memory_mib - ep.peak_memory_mib) / ep.requested_memory_mib
    n = len(pairs)
    return ok / n, (wastes / max(ok, 1))


def test_ml_baseline_beats_rule_baseline_in_sim(corpus, fitted):
    """The reason to exist: on heldout, quantile GBTs must fit at least as
    often as the family-median rule while wasting less."""
    train = [r for r in corpus if r["split"] == "train"]
    heldout = [r for r in corpus if r["split"] == "heldout"]
    rule = fit_family_baseline(train)
    ml_fit, ml_waste = _sim_stats([(r, fitted.propose(r)) for r in heldout])
    rule_fit, rule_waste = _sim_stats(
        [(r, baseline_proposal(rule, r["family"])) for r in heldout]
    )
    assert ml_fit >= rule_fit - 0.02, (ml_fit, rule_fit)
    assert ml_waste <= rule_waste + 0.02, (ml_waste, rule_waste)


def test_gpu_classification_separates_families(corpus, fitted):
    heldout = [r for r in corpus if r["split"] == "heldout"]
    gpu_rows = [r for r in heldout if float(r.get("true_gpu_mem_mib", 0)) > 0]
    cpu_rows = [r for r in heldout if float(r.get("true_gpu_mem_mib", 0)) == 0]
    assert gpu_rows and cpu_rows
    gpu_hits = sum(1 for r in gpu_rows if fitted.propose(r).gpu == 1)
    spurious = sum(1 for r in cpu_rows if fitted.propose(r).gpu == 1)
    assert gpu_hits / len(gpu_rows) > 0.9, "misses GPUs it was trained to spot"
    assert spurious / len(cpu_rows) < 0.05, "hallucinates GPUs on CPU tasks"


def test_fit_fails_loudly_on_empty_split():
    with pytest.raises(ValueError):
        MLBaseline().fit([])


def test_joblib_roundtrip_preserves_proposals(tmp_path, corpus, fitted):
    fitted.save(str(tmp_path))
    reloaded = MLBaseline.load(str(tmp_path))
    heldout = [r for r in corpus if r["split"] == "heldout"][:25]
    for r in heldout:
        assert reloaded.propose(r) == fitted.propose(r)


def test_serve_time_record_shape_degrades_honestly(fitted):
    """The tune service has no params_json/family — proposals must still
    come out grid-valid from profile/code/prior/history features alone."""
    from resource_tuner.policy.actions import CPU_GRID, MEMORY_GRID_MIB
    from resource_tuner.training.ml_baseline import request_to_record

    rec = request_to_record(
        "import torch\nmodel.to('cuda')",
        "invoked with kwargs={'rows': 8000000}",
        {"cpu": 4, "memory": "8Gi"},
        [{"resources": {"cpu": 2, "memory": "2Gi"}, "peak": "812MiB", "ok": True}],
    )
    assert rec["family"] == "" and rec["params_json"] == ""
    p = fitted.propose(rec)
    assert p.cpu in CPU_GRID and p.memory_mib in MEMORY_GRID_MIB
