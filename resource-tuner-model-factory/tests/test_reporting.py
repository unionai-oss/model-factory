"""Report builders (pure HTML) + flush safety."""

import asyncio

from resource_tuner.shared.reporting import GOOD, Reporter, esc, ok_pill, pill


def test_page_builds_with_all_section_types():
    rep = Reporter("Title", "sub")
    rep.kv({"a": 1}).h("Head").p("text").progress(2, 4, "steps").table(
        ["c1", "c2"], [[esc("<x>"), pill("ok", GOOD)]]
    )
    html = rep.html()
    assert "Title" in html and "sub" in html
    assert "&lt;x&gt;" in html  # cell content escaped by esc()
    assert "width:50%" in html  # progress bar
    assert html.count("<table") == 2  # kv + table


def test_reset_body_keeps_title():
    rep = Reporter("T")
    rep.p("gone")
    rep.reset_body().p("kept")
    html = rep.html()
    assert "kept" in html and "gone" not in html and "T" in html


def test_ok_pill_polarity():
    assert GOOD in ok_pill(True)
    assert GOOD not in ok_pill(False)


def test_flush_never_raises_outside_a_run_context():
    # flyte.report outside a task context must degrade to a print, not
    # kill the caller — observability never breaks the work.
    asyncio.run(Reporter("T").p("x").flush())
