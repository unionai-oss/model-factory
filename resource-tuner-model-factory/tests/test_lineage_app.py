"""Lineage app helpers: metric extraction, link guards, page template."""

from resource_tuner.config import APP_DOMAIN, APP_PROJECT
from resource_tuner.lineage_app import (
    _PAGE_TEMPLATE,
    APP_NAME,
    _metrics_of,
    _run_url,
    lineage_app_env,
)


def test_metrics_of_extracts_the_dashboard_fields():
    m = _metrics_of(
        {
            "schema_validity": 1.0,
            "success_rate": 0.9,
            "baseline_success_rate": 0.75,
            "median_overprovision_pct": 83.0,
            "baseline_median_overprovision_pct": 28.0,
            "auto_gate_passed": False,
            "base_model": "Qwen/Qwen3-1.7B",
            "cluster_episodes": ["dropped"],
        }
    )
    assert m["success_rate"] == 0.9 and "cluster_episodes" not in m


def test_run_url_refuses_display_strings():
    assert _run_url("p", "d", "run x/y (attempt 1)") == ""
    assert _run_url("p", "d", "") == ""


def test_subdomain_is_app_project_domain():
    assert (
        lineage_app_env.domain.subdomain == f"{APP_NAME}-{APP_PROJECT}-{APP_DOMAIN}"
    )


def test_page_template_is_dark_and_has_the_graph_stack():
    # Union dark tokens present, and the zero-build graph stack is wired.
    for token in ("#4d65ff", "#e69812", "#F43B3E", "--bg: #0b0b0d"):
        assert token in _PAGE_TEMPLATE
    for lib in ("@xyflow/react", "@dagrejs/dagre", "esm.sh/react"):
        assert lib in _PAGE_TEMPLATE
    assert "__BOOT_JSON__" in _PAGE_TEMPLATE and "__FALLBACK__" in _PAGE_TEMPLATE
    assert "EvalBadges" in _PAGE_TEMPLATE  # the added eval-metric UI layer
