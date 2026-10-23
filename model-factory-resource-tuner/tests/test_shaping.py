"""Reward shaping: forms, composable knobs, GPU semantics, and pricing."""

import math

import pytest

from resource_tuner import pricing
from resource_tuner.environment.simulator import simulate_episode
from resource_tuner.policy.actions import Proposal
from resource_tuner.rewards.shaping import (
    SHAPES,
    RewardShape,
    apply_group_tiebreak,
    axis_waste,
    episode_dollars_per_hr,
    get_shape,
    score_shaped,
)


def _ep(cpu=4, mem=4096, gpu=0, gpu_type=None, peak_mem=500, peak_cpu=1, gpu_mem=0.0):
    return simulate_episode(
        Proposal(cpu=cpu, memory_mib=mem, gpu=gpu, gpu_type=gpu_type),
        true_peak_memory_mib=peak_mem,
        true_cpu_cores=peak_cpu,
        duration_s=60,
        true_gpu_mem_mib=gpu_mem,
    )


# ── pricing ─────────────────────────────────────────────────────────────


def test_pricing_orders_the_gpu_ladder():
    t4 = pricing.GPU_DOLLARS_PER_HR["T4"]
    l4 = pricing.GPU_DOLLARS_PER_HR["L4"]
    l40s = pricing.GPU_DOLLARS_PER_HR["L40S"]
    assert t4 < l4 < l40s


def test_kwargs_pricing_matches_direct():
    direct = pricing.dollars_per_hr(2, 4096, "T4", 1)
    via_kwargs = pricing.kwargs_dollars_per_hr({"cpu": 2, "memory": "4Gi", "gpu": "T4:1"})
    assert via_kwargs == pytest.approx(direct)
    assert pricing.kwargs_dollars_per_hr({"cpu": "500m", "memory": "1Gi"}) == pytest.approx(
        0.5 * pricing.CPU_DOLLARS_PER_CORE_HR + pricing.MEM_DOLLARS_PER_GIB_HR
    )
    assert pricing.kwargs_dollars_per_hr({}) is None


def test_cheapest_gpu_walks_the_ladder():
    assert pricing.cheapest_gpu_for(8_000) == "T4"
    assert pricing.cheapest_gpu_for(20_000) == "L4"
    assert pricing.cheapest_gpu_for(40_000) == "L40S"
    assert pricing.cheapest_gpu_for(60_000) is None  # nothing fits


# ── waste forms (the mutually exclusive judgment) ───────────────────────


def test_log_ratio_distinguishes_where_linear_saturates():
    # 4Gi vs 1Gi for a ~200MiB peak: nearly identical under linear,
    # clearly separated under log_ratio — the round-5 plateau, fixed.
    lin_4g = axis_waste("linear", 4096, 200, None)
    lin_1g = axis_waste("linear", 1024, 200, None)
    log_4g = axis_waste("log_ratio", 4096, 200, None)
    log_1g = axis_waste("log_ratio", 1024, 200, None)
    assert lin_4g - lin_1g < 0.15
    assert log_4g - log_1g > 0.3


def test_curvatures_bracket_linear():
    frac_point = axis_waste("linear", 1000, 500, None)  # 0.5
    assert axis_waste("quadratic", 1000, 500, None) == pytest.approx(frac_point**2)
    assert axis_waste("sqrt", 1000, 500, None) == pytest.approx(math.sqrt(frac_point))


def test_bucket_form_counts_grid_steps():
    from resource_tuner.policy.actions import MEMORY_GRID_MIB

    span = len(MEMORY_GRID_MIB) - 1
    one_step = axis_waste("bucket", 2048, 1000, MEMORY_GRID_MIB)  # 1024 would fit
    three_steps = axis_waste("bucket", 8192, 1000, MEMORY_GRID_MIB)
    assert one_step == pytest.approx(1 / span)
    assert three_steps == pytest.approx(3 / span)
    assert axis_waste("bucket", 1024, 1000, MEMORY_GRID_MIB) == 0.0


def test_right_sized_is_free_under_every_form():
    for form in ("linear", "quadratic", "sqrt", "log_ratio", "bucket"):
        assert axis_waste(form, 512, 512, None) == 0.0


# ── composable knobs ────────────────────────────────────────────────────


def test_headroom_band_moves_the_optimum_off_the_knife_edge():
    shape_band = RewardShape(name="t", headroom=(1.1, 1.4))
    shape_min = RewardShape(name="t2")
    inside_band = _ep(cpu=1, mem=1024, peak_mem=800)  # 1.28x peak — inside (1.1, 1.4)
    knife = _ep(cpu=1, mem=1024, peak_mem=960)  # fits (960*1.05=1008), but 1.07x < band lo
    assert score_shaped(shape_band, [inside_band]).components["waste"] == 0.0
    assert score_shaped(shape_band, [knife]).components["knife_edge"] < 0.0
    # without the band, the knife-edge request scores BETTER — the failure
    # mode the band exists to fix
    assert score_shaped(shape_min, [knife]).total > score_shaped(shape_min, [inside_band]).total


def test_cost_weighting_makes_an_idle_gpu_dominate():
    cheap_axis = _ep(cpu=4, mem=1024, peak_mem=900, peak_cpu=1)  # some cpu waste
    idle_gpu = _ep(cpu=1, mem=1024, gpu=1, gpu_type="L40S", peak_mem=900, peak_cpu=1)
    shape = RewardShape(name="t", cost_weighted=True)
    assert (
        score_shaped(shape, [idle_gpu]).components["waste"]
        < score_shaped(shape, [cheap_axis]).components["waste"]
    )


def test_baseline_relative_rewards_costing_less():
    shape = RewardShape(name="t", baseline_relative=True)
    ep = _ep(cpu=1, mem=1024, peak_mem=800)
    base_cost = pricing.dollars_per_hr(4, 8192)
    with_savings = score_shaped(shape, [ep], baseline_cost_per_hr=base_cost)
    assert with_savings.components["savings"] > 0
    pricier = _ep(cpu=16, mem=65536, peak_mem=800)
    assert score_shaped(shape, [pricier], baseline_cost_per_hr=base_cost).components[
        "savings"
    ] < 0


def test_anneal_ramps_waste_weight():
    shape = RewardShape(name="t", w_waste=0.4, w_waste_final=1.0)
    assert shape.waste_weight(0.0) == pytest.approx(0.4)
    assert shape.waste_weight(0.5) == pytest.approx(0.7)
    assert shape.waste_weight(1.0) == pytest.approx(1.0)
    assert RewardShape(name="t2").waste_weight(0.5) == pytest.approx(0.5)


def test_group_tiebreak_pays_the_cheapest_survivor():
    shape = RewardShape(name="t", group_tiebreak=0.2)
    totals = [1.0, 1.0, 1.0, 0.0]
    oks = [True, True, True, False]
    costs = [0.5, 0.1, 0.3, None]
    out = apply_group_tiebreak(shape, totals, oks, costs, group_size=4)
    assert out[1] == pytest.approx(1.2)  # cheapest successful wins the bonus
    assert out[0] == 1.0 and out[3] == 0.0
    # ragged batch → no-op, never a crash
    assert apply_group_tiebreak(shape, [1.0] * 3, [True] * 3, [1, 2, 3], 4) == [1.0] * 3


def test_robustness_averaging_penalizes_knife_edges():
    from random import Random

    shape = RewardShape(name="t")
    proposal = Proposal(cpu=2, memory_mib=1024)
    rng = Random(7)
    eps = [
        simulate_episode(proposal, 970, 1, 60, rng=rng) for _ in range(50)
    ]  # 1024/970 = 1.056 sits inside the (0.02, 0.12) jitter band
    bd = score_shaped(shape, eps)
    assert 0.1 < bd.components["success"] < 0.9  # sometimes fits, sometimes OOMs


# ── GPU episode semantics ───────────────────────────────────────────────


def test_gpu_missing_fails_and_is_penalized_separately():
    ep = _ep(gpu=0, gpu_mem=8000)
    assert not ep.ok and ep.gpu_missing and not ep.oom
    bd = score_shaped(RewardShape(name="t"), [ep])
    assert bd.components["gpu_missing"] < 0 and bd.components["oom"] == 0


def test_undersized_gpu_ooms():
    ep = _ep(gpu=1, gpu_type="T4", gpu_mem=20_000)  # needs L4
    assert not ep.ok and ep.oom and not ep.gpu_missing


def test_right_gpu_succeeds_and_spurious_gpu_is_pure_waste():
    fits = _ep(gpu=1, gpu_type="L4", gpu_mem=20_000)
    assert fits.ok
    spurious = _ep(gpu=1, gpu_type="T4", gpu_mem=0.0)
    assert spurious.ok  # succeeds — but…
    shape = RewardShape(name="t", cost_weighted=True)
    assert (
        score_shaped(shape, [spurious]).components["waste"]
        < score_shaped(shape, [_ep()]).components["waste"]
    )


def test_episode_dollars_include_the_gpu():
    with_gpu = episode_dollars_per_hr(_ep(gpu=1, gpu_type="T4", gpu_mem=8000))
    without = episode_dollars_per_hr(_ep())
    assert with_gpu == pytest.approx(without + pricing.GPU_DOLLARS_PER_HR["T4"])


# ── the named arms ──────────────────────────────────────────────────────


def test_named_shapes_resolve_and_env_override_applies(monkeypatch):
    assert {"c-linear", "c-log", "c-bucket", "c-cost"} <= set(SHAPES)
    monkeypatch.setenv("RT_REWARD_SHAPE", '{"w_waste_final": 1.5}')
    assert get_shape("c-log").w_waste_final == 1.5
    monkeypatch.setenv("RT_REWARD_SHAPE", '{"nope": 1}')
    with pytest.raises(ValueError):
        get_shape("c-log")


def test_every_named_shape_prefers_lean_success_over_padded_success():
    lean = _ep(cpu=1, mem=1024, peak_mem=700, peak_cpu=1)
    padded = _ep(cpu=8, mem=65536, peak_mem=700, peak_cpu=1)
    base_cost = pricing.dollars_per_hr(4, 8192)
    for shape in SHAPES.values():
        lean_score = score_shaped(shape, [lean], baseline_cost_per_hr=base_cost).total
        padded_score = score_shaped(shape, [padded], baseline_cost_per_hr=base_cost).total
        assert lean_score > padded_score, shape.name


def test_every_named_shape_prefers_generous_success_over_oom():
    ok = _ep(cpu=4, mem=8192, peak_mem=700)
    boom = _ep(cpu=4, mem=512, peak_mem=700)
    for shape in SHAPES.values():
        assert (
            score_shaped(shape, [ok]).total > score_shaped(shape, [boom]).total
        ), shape.name
