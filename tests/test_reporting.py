from model_factory.reporting import line_chart, page, pill, stats_row, table


def test_page_escapes_title():
    html = page("<script>", "<p>body</p>")
    assert "&lt;script&gt;" in html
    assert "<p>body</p>" in html


def test_table_escapes_and_truncates():
    html = table(["a"], [[f"<r{i}>"] for i in range(60)], max_rows=50)
    assert "&lt;r0&gt;" in html
    assert "10 more rows" in html


def test_line_chart_handles_empty_and_flat():
    assert "no data" in line_chart({"x": []})
    assert "<svg" in line_chart({"x": [1.0, 1.0, 1.0]})


def test_stats_and_pill():
    assert "42" in stats_row({"tasks": 42})
    assert "pass" in pill(True)
    assert "fail" in pill(False)
