"""Lineage app helpers: metric extraction, link guards, page template."""

from resource_tuner.config import APP_DOMAIN, APP_PROJECT
from resource_tuner.lineage_app import (
    _PAGE_TEMPLATE,
    APP_NAME,
    _metrics_of,
    _run_url,
    build_version_edges,
    lineage_app_env,
)


def _v(station: str, name: str, run: str, url: str) -> dict:
    return {"id": f"{station}::{name}", "version": name, "source": run, "url": url}


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


# ── version-level lineage ───────────────────────────────────────────────
# The station graph says "corpus feeds checkpoint" as a contract. These
# edges say "THIS corpus fed THAT checkpoint", which is what lets the
# Versions view show one version's actual ancestors and descendants.
CONTRACT = [("corpus", "ckpt"), ("ckpt", "report")]


def test_same_run_co_production_links_versions():
    """One run produced a corpus AND a checkpoint, and the contract says
    corpus feeds checkpoint — that run is the link. This is the
    `tuner_pipeline` case, where every station shares a run."""
    edges = build_version_edges(
        {
            "corpus": [_v("corpus", "r1/a", "r1", "s3://c1")],
            "ckpt": [_v("ckpt", "r1/b", "r1", "s3://k1")],
            "report": [],
        },
        CONTRACT,
    )
    assert edges == [{"source": "corpus::r1/a", "target": "ckpt::r1/b", "kind": "run"}]


def test_declared_provenance_links_across_runs():
    """A trigger-fired training run shares no run name with the run that
    published its corpus, so co-production finds nothing. What the tasks
    RECORDED does: the report names the checkpoint, the checkpoint's
    manifest names the corpus."""
    edges = build_version_edges(
        {
            "corpus": [_v("corpus", "r1/a", "r1", "s3://c1")],
            "ckpt": [_v("ckpt", "r2/a", "r2", "s3://k1")],
            "report": [_v("report", "r3/a", "r3", "s3://e1")],
        },
        CONTRACT,
        declared={"report::r3/a": ["s3://k1"], "ckpt::r2/a": ["s3://c1"]},
    )
    pairs = {(e["source"], e["target"]) for e in edges}
    assert pairs == {
        ("corpus::r1/a", "ckpt::r2/a"),
        ("ckpt::r2/a", "report::r3/a"),
    }
    assert all(e["kind"] == "declared" for e in edges)


def test_declared_beats_the_run_inference_for_the_same_pair():
    """When both signals agree, the edge is labelled by the stronger one —
    the UI distinguishes 'the report names this checkpoint' from 'one run
    happened to make both'."""
    edges = build_version_edges(
        {
            "corpus": [_v("corpus", "r1/a", "r1", "s3://c1")],
            "ckpt": [_v("ckpt", "r1/b", "r1", "s3://k1")],
        },
        CONTRACT,
        declared={"ckpt::r1/b": ["s3://c1"]},
    )
    assert len(edges) == 1 and edges[0]["kind"] == "declared"


def test_nothing_is_inferred_from_recency():
    """The tempting heuristic — 'the corpus published most recently before
    this checkpoint' — is plausible and wrong the moment anyone trains on
    an older corpus. Unrecorded lineage must stay absent, not guessed."""
    edges = build_version_edges(
        {
            "corpus": [
                _v("corpus", "r1/a", "r1", "s3://c1"),
                _v("corpus", "r0/a", "r0", "s3://c0"),
            ],
            "ckpt": [_v("ckpt", "r2/a", "r2", "s3://k1")],
        },
        CONTRACT,
    )
    assert edges == []


def test_declared_paths_that_name_no_known_version_are_dropped():
    """A checkpoint trained on a corpus that has aged out of the version
    list must not produce a dangling edge to a node the graph lacks."""
    edges = build_version_edges(
        {"corpus": [], "ckpt": [_v("ckpt", "r2/a", "r2", "s3://k1")]},
        CONTRACT,
        declared={"ckpt::r2/a": ["s3://long-gone", ""]},
    )
    assert edges == []


def test_only_contract_pairs_are_linked():
    """Two artifacts produced by one run are not lineage unless the
    contract says one feeds the other."""
    edges = build_version_edges(
        {
            "corpus": [_v("corpus", "r1/a", "r1", "s3://c1")],
            "report": [_v("report", "r1/c", "r1", "s3://e1")],
        },
        CONTRACT,  # corpus→ckpt→report; corpus→report is NOT a contract
    )
    assert edges == []


def test_page_has_the_per_station_version_selector_and_focus_traversal():
    for marker in ("StationRail", "rail-select", "All versions", "focusSet",
                   "focusOf", "version_edges"):
        assert marker in _PAGE_TEMPLATE


def test_version_pickers_stay_out_of_the_react_flow_viewport():
    """React Flow stamps inline pointer-events:none on a node that is
    neither selectable nor draggable, so a control inside one is unclickable
    — clicks land on the pane. And at fit-zoom (~0.3x) it would render about
    6px tall anyway. Both were real: the in-canvas <select> shipped and
    could not be used. The pickers live in the rail, outside the zoomed
    viewport; selection INSIDE the graph is clicking a version card, whose
    nodes do receive pointer events."""
    rail_start = _PAGE_TEMPLATE.index("const StationRail")
    group_label = _PAGE_TEMPLATE.index("const GroupLabel")
    group_label_end = _PAGE_TEMPLATE.index("const EmptyNote")
    # The group label — a node inside the viewport — carries no form control.
    assert "<select" not in _PAGE_TEMPLATE[group_label:group_label_end]
    # The rail does.
    assert "<select" in _PAGE_TEMPLATE[rail_start:rail_start + 2000]
    # And the card is clickable as the in-graph picker.
    assert "pickable" in _PAGE_TEMPLATE
