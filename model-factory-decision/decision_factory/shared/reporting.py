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


# ── the frontier chart ──────────────────────────────────────────────────
#: Categorical slots of the reference palette, in fixed order. Color = torso;
#: line style = readout, so identity is never color alone.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]
READOUT_STYLE = {
    "pointer": {"dash": "", "label": "pointer head (P(allow) threshold)"},
    "logprob": {"dash": "6 4", "label": "SFT token log-probs (threshold)"},
    "json": {"dash": None, "label": "SFT JSON (one point)"},
    "base_zero_shot": {"dash": "2 3", "label": "untuned base, option likelihood"},
}


def frontier_svg(series: list[dict], x_max: float | None = None, width: int = 640, height: int = 380) -> _Raw:
    """Autonomy (y) vs unsafe-allow rate (x), one series per model readout.

    ``series``: [{"name", "color_key", "readout", "points": [{"threshold",
    "autonomy", "unsafe_allow_rate"}], "operating": point-or-None}]. A curve
    for readouts with real probabilities; a single hollow dot for JSON.
    """
    pad_l, pad_r, pad_t, pad_b = 56, 16, 16, 44
    w, h = width - pad_l - pad_r, height - pad_t - pad_b
    xs = [p["unsafe_allow_rate"] for s in series for p in s["points"]]
    x_max = x_max or max(0.05, min(1.0, (max(xs) if xs else 0.1) * 1.05))

    def X(v):
        return pad_l + min(v, x_max) / x_max * w

    def Y(v):
        return pad_t + (1 - v) * h

    keys: list[str] = []
    for s in series:
        if s["color_key"] not in keys:
            keys.append(s["color_key"])
    color = {k: SERIES_COLORS[i % len(SERIES_COLORS)] for i, k in enumerate(keys)}
    out = [f'<svg viewBox="0 0 {width} {height}" width="100%" style="max-width:{width}px;font:11px system-ui" '
           'role="img" aria-label="Autonomy versus unsafe-allow rate">']
    for i in range(6):  # recessive grid + tick labels
        gy = i / 5
        out.append(f'<line x1="{pad_l}" x2="{pad_l + w}" y1="{Y(gy):.1f}" y2="{Y(gy):.1f}" stroke="#e5e5e5"/>'
                   f'<text x="{pad_l - 6}" y="{Y(gy) + 4:.1f}" text-anchor="end" fill="#666">{gy:.0%}</text>')
        gx = x_max * i / 5
        out.append(f'<text x="{X(gx):.1f}" y="{pad_t + h + 16}" text-anchor="middle" fill="#666">{gx:.1%}</text>')
    out.append(f'<text x="{pad_l + w / 2}" y="{height - 6}" text-anchor="middle" fill="#333">'
               'unsafe-allow rate (share of ask/deny calls auto-allowed)</text>')
    out.append(f'<text transform="translate(14,{pad_t + h / 2}) rotate(-90)" text-anchor="middle" fill="#333">'
               'autonomy (share of safe calls auto-allowed)</text>')
    for s in series:
        c = color[s["color_key"]]
        dash = READOUT_STYLE.get(s["readout"], {"dash": ""})["dash"]
        pts = sorted(s["points"], key=lambda p: (p["unsafe_allow_rate"], p["autonomy"]))
        if dash is not None and len(pts) > 1:
            path = " ".join(f'{X(p["unsafe_allow_rate"]):.1f},{Y(p["autonomy"]):.1f}' for p in pts)
            out.append(f'<polyline points="{path}" fill="none" stroke="{c}" stroke-width="2" stroke-dasharray="{dash}"/>')
        dots = pts[:1] if dash is None else ([s["operating"]] if s.get("operating") else [])
        for p in dots:
            fill = "#fff" if dash is None else c
            out.append(
                f'<circle cx="{X(p["unsafe_allow_rate"]):.1f}" cy="{Y(p["autonomy"]):.1f}" r="5" fill="{fill}" '
                f'stroke="{c}" stroke-width="2"><title>{esc(s["name"])}: autonomy {p["autonomy"]:.1%}, '
                f'unsafe-allow {p["unsafe_allow_rate"]:.2%}, t={p.get("threshold", "-")}</title></circle>')
    out.append("</svg>")
    legend = "".join(
        f'<span style="margin-right:14px;white-space:nowrap"><svg width="26" height="10"><line x1="0" x2="26" y1="5" '
        f'y2="5" stroke="{color[s["color_key"]]}" stroke-width="2" '
        f'stroke-dasharray="{READOUT_STYLE.get(s["readout"], {}).get("dash") or ""}"/></svg> {esc(s["name"])}</span>'
        for s in series
    )
    note = ("<div style='font-size:11px;color:#666'>Filled dot: gate threshold tuned on the val split for "
            "&le;1% unsafe-allow. Hollow dot: a JSON answer, which has no threshold to tune. Up and to the "
            "left is better.</div>")
    return _Raw("".join(out) + f"<div style='font-size:12px;margin:6px 0'>{legend}</div>" + note)
