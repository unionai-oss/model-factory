"""HTML builders for flyte.report — the factory's observation windows.

Self-contained HTML (inline CSS, inline SVG charts): no external assets, so
reports render inside the Union console without network access.
"""

from __future__ import annotations

import html as _html

_STYLE = """
<style>
.mf { font-family: -apple-system, 'Segoe UI', Roboto, sans-serif; color: #1a1a2e; max-width: 1080px; }
.mf h2 { margin: 0.2em 0 0.4em; }
.mf table { border-collapse: collapse; margin: 0.6em 0; width: 100%; }
.mf th, .mf td { border: 1px solid #d8dbe2; padding: 6px 10px; text-align: left; font-size: 13px; vertical-align: top; }
.mf th { background: #eef0f5; }
.mf .stat { display: inline-block; background: #f5f6fa; border: 1px solid #d8dbe2; border-radius: 8px; padding: 10px 16px; margin: 4px 8px 4px 0; }
.mf .stat b { display: block; font-size: 20px; }
.mf .ok { color: #0a7a3d; } .mf .bad { color: #b00020; }
.mf pre { background: #f5f6fa; border: 1px solid #d8dbe2; border-radius: 6px; padding: 8px; font-size: 12px; overflow-x: auto; white-space: pre-wrap; }
.mf .pill { border-radius: 10px; padding: 1px 8px; font-size: 12px; }
.mf .pill.ok { background: #dcf5e7; } .mf .pill.bad { background: #fde3e3; }
</style>
"""


def esc(s: object) -> str:
    return _html.escape(str(s))


def page(title: str, body: str) -> str:
    return f"{_STYLE}<div class='mf'><h2>{esc(title)}</h2>{body}</div>"


def stats_row(stats: dict[str, object]) -> str:
    cells = "".join(
        f"<span class='stat'><b>{esc(v)}</b>{esc(k)}</span>" for k, v in stats.items()
    )
    return f"<div>{cells}</div>"


def table(headers: list[str], rows: list[list[object]], max_rows: int = 50) -> str:
    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = ""
    for row in rows[:max_rows]:
        body += "<tr>" + "".join(f"<td>{esc(c)}</td>" for c in row) + "</tr>"
    more = (
        f"<p><i>… {len(rows) - max_rows} more rows not shown</i></p>"
        if len(rows) > max_rows
        else ""
    )
    return f"<table><tr>{head}</tr>{body}</table>{more}"


def pill(ok: bool, text_ok: str = "pass", text_bad: str = "fail") -> str:
    cls = "ok" if ok else "bad"
    return f"<span class='pill {cls}'>{text_ok if ok else text_bad}</span>"


def line_chart(
    series: dict[str, list[float]],
    width: int = 640,
    height: int = 220,
    title: str = "",
) -> str:
    """Inline-SVG line chart; series maps label -> y values (shared x = index)."""
    colors = ["#2f6fed", "#0a7a3d", "#c26b0a", "#8536c9", "#b00020"]
    all_vals = [v for ys in series.values() for v in ys if v == v]  # drop NaN
    if not all_vals:
        return "<p><i>no data yet</i></p>"
    lo, hi = min(all_vals), max(all_vals)
    if hi == lo:
        hi = lo + 1.0
    pad, w, h = 30, width, height
    n = max(len(ys) for ys in series.values())

    def sx(i: int) -> float:
        return pad + (w - 2 * pad) * (i / max(n - 1, 1))

    def sy(v: float) -> float:
        return h - pad - (h - 2 * pad) * ((v - lo) / (hi - lo))

    paths, legend = "", ""
    for idx, (label, ys) in enumerate(series.items()):
        c = colors[idx % len(colors)]
        pts = " ".join(f"{sx(i):.1f},{sy(v):.1f}" for i, v in enumerate(ys) if v == v)
        paths += f"<polyline fill='none' stroke='{c}' stroke-width='2' points='{pts}'/>"
        legend += (
            f"<tspan x='{pad + idx * 150}' fill='{c}'>&#9632; {esc(label)}</tspan>"
        )
    axis = (
        f"<line x1='{pad}' y1='{h-pad}' x2='{w-pad}' y2='{h-pad}' stroke='#999'/>"
        f"<line x1='{pad}' y1='{pad}' x2='{pad}' y2='{h-pad}' stroke='#999'/>"
        f"<text x='{pad-4}' y='{sy(hi)+4}' text-anchor='end' font-size='11'>{hi:.2f}</text>"
        f"<text x='{pad-4}' y='{sy(lo)+4}' text-anchor='end' font-size='11'>{lo:.2f}</text>"
    )
    t = f"<text x='{pad}' y='16' font-size='13' font-weight='bold'>{esc(title)}</text>" if title else ""
    return (
        f"<svg width='{w}' height='{h}' xmlns='http://www.w3.org/2000/svg'>"
        f"{t}{axis}{paths}<text y='{h-8}' font-size='12'>{legend}</text></svg>"
    )
