"""Training report rendering (pure HTML builder, no trainer needed)."""

from resource_tuner.config import get_profile
from resource_tuner.training.grpo import _report_html

PROFILE = get_profile("smoke")


def test_empty_history_renders_placeholder():
    assert "waiting" in _report_html(PROFILE, [])
    assert "waiting" in _report_html(PROFILE, [{"loss": 1.0}])  # no reward yet


def test_reward_curve_and_stats_render():
    rows = [{"reward": 0.1, "loss": 2.0}, {"reward": 0.4, "loss": 1.5}, {"reward": 0.9}]
    html = _report_html(PROFILE, rows)
    assert "polyline" in html and "points=" in html
    assert "mean reward 0.900" in html and "(start 0.100)" in html
    assert PROFILE.base_model in html
    assert "step 3/" in html


def test_flat_rewards_do_not_divide_by_zero():
    html = _report_html(PROFILE, [{"reward": 0.5}, {"reward": 0.5}])
    assert "polyline" in html
