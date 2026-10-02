"""Minimal HTML helpers for `flyte.report` pages.

Every station writes a report, because the reports are what make a suite run
reviewable: the data card shows what the model was trained on, the scorecard
shows where it fails, and the champion page shows why one model won. Kept
dependency-free and escaping-by-default — report bodies carry model output,
which is untrusted text.
"""

from __future__ import annotations

from html import escape
from typing import Any, Iterable, Mapping, Sequence

_CSS = """
body { font-family: ui-sans-serif, system-ui, -apple-system, sans-serif;
       margin: 0; padding: 24px; line-height: 1.5; color: #111; background: #fff; }
h1 { font-size: 20px; margin: 0 0 16px; }
h3 { font-size: 15px; margin: 24px 0 8px; }
h4 { font-size: 13px; margin: 16px 0 4px; color: #444; }
table { border-collapse: collapse; width: 100%; font-size: 13px; margin: 8px 0; }
th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid #e5e5e5;
         vertical-align: top; }
th { background: #fafafa; font-weight: 600; }
pre { background: #f6f6f6; padding: 10px; border-radius: 6px; overflow-x: auto;
      font-size: 12px; white-space: pre-wrap; }
.stats { display: flex; flex-wrap: wrap; gap: 10px; margin: 12px 0; }
.stat { border: 1px solid #e5e5e5; border-radius: 8px; padding: 8px 14px; min-width: 110px; }
.stat .k { font-size: 11px; text-transform: uppercase; letter-spacing: .04em; color: #666; }
.stat .v { font-size: 17px; font-weight: 600; }
.pill { display: inline-block; padding: 1px 8px; border-radius: 10px;
        font-size: 11px; font-weight: 600; }
.ok { background: #e6f4ea; color: #137333; }
.bad { background: #fce8e6; color: #c5221f; }
.win { background: #e8f0fe; color: #1a56c4; }
"""


def esc(value: Any) -> str:
    return escape(str(value), quote=False)


def page(title: str, body: str) -> str:
    return f"<style>{_CSS}</style><h1>{esc(title)}</h1>{body}"


def stats_row(stats: Mapping[str, Any]) -> str:
    cells = "".join(
        f'<div class="stat"><div class="k">{esc(k)}</div><div class="v">{esc(v)}</div></div>'
        for k, v in stats.items()
    )
    return f'<div class="stats">{cells}</div>'


def table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = ""
    for row in rows:
        # Cells are pre-escaped only when they came from pill()/raw markup
        # helpers in this module; everything else is escaped here.
        body += "<tr>" + "".join(f"<td>{_cell(c)}</td>" for c in row) + "</tr>"
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _cell(value: Any) -> str:
    if isinstance(value, _Raw):
        return str(value)
    return esc(value)


class _Raw(str):
    """Markup that has already been escaped by a helper below."""


def pill(ok: bool, true_label: str = "pass", false_label: str = "fail") -> _Raw:
    cls = "ok" if ok else "bad"
    return _Raw(f'<span class="pill {cls}">{esc(true_label if ok else false_label)}</span>')


def win_pill(label: str = "champion") -> _Raw:
    return _Raw(f'<span class="pill win">{esc(label)}</span>')


def pct(value: float) -> str:
    return f"{value:.1%}"


def bar(value: float, width: int = 18) -> str:
    """A text bar for a 0..1 value — readable in a table cell without a chart."""
    value = max(0.0, min(1.0, float(value)))
    filled = round(value * width)
    return "█" * filled + "·" * (width - filled)
