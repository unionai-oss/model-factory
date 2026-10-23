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


def test_highlight_python_colors_and_escapes():
    from resource_tuner.shared.reporting import code_block, highlight_python

    html = highlight_python('def run():  # go\n    x = "a<b" + f"{n:,}"\n    return 42')
    assert '<span style="color:' in html
    assert "a&lt;b" in html and "<b" not in html.replace("<br", "")  # escaped
    assert ">def</span>" in html and "# go</span>" in html and ">42</span>" in html
    block = code_block("x = 1\n" * 4000, cap=100)
    assert "truncated at 100" in block
    assert "overflow-x:auto" in code_block("def f(): pass")


def test_markdown_block_renders_safely():
    from resource_tuner.shared.reporting import markdown_block

    md = (
        "## Hypothesis\n"
        "**Bigger adapters** close the *fit gap*.\n"
        "- arm: `r11-r64-mlp`\n"
        "- risk: <script>alert(1)</script>\n"
        "See [round 8](https://example.com/r8)."
    )
    html = markdown_block(md)
    assert "Hypothesis</div>" in html and "<strong>Bigger adapters</strong>" in html
    assert "<em>fit gap</em>" in html and "<li" in html
    assert 'href="https://example.com/r8"' in html
    assert "<script>" not in html and "&lt;script&gt;" in html  # escaped, always
    assert "x" * 4001 not in markdown_block("x" * 9000)  # capped at 4000
