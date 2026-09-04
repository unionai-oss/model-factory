"""Flyte report helpers: one consistent live page per task.

Every factory task carries `report=True` and streams its progress here —
the run page should answer "what is this step doing right now" without
log spelunking. Builders are pure string functions (unit-testable);
`flush()` is the only I/O.

Pattern per task:

    rep = Reporter("Corpus build")
    rep.kv({"profile": ..., "seed": ...})
    await rep.flush()          # early: page exists while the work runs
    ...work...
    rep.table(...); await rep.flush()
"""

from __future__ import annotations

import html as _html

_STYLE = (
    "font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
    "background:#0b0b0d;color:#f2f2f3;padding:16px 20px;border-radius:12px;"
)
_TABLE = "border-collapse:collapse;margin:10px 0;font-size:13px;"
_TH = (
    "text-align:left;padding:4px 10px;border-bottom:1px solid #2a2a31;"
    "color:#9a9aa4;font-size:11px;text-transform:uppercase;letter-spacing:.06em;"
)
_TD = "padding:4px 10px;border-bottom:1px solid #1b1b21;color:#c9c9cf;"
GOOD, WARN, BAD, MUTED = "#35c48d", "#e69812", "#F43B3E", "#9a9aa4"


def esc(v: object) -> str:
    return _html.escape(str(v))


def pill(text: str, color: str = MUTED) -> str:
    return (
        f'<span style="border:1px solid {color};color:{color};border-radius:999px;'
        f'padding:0 8px;font-size:11px;white-space:nowrap">{esc(text)}</span>'
    )


def ok_pill(ok: bool, yes: str = "ok", no: str = "failed") -> str:
    return pill(yes, GOOD) if ok else pill(no, BAD)


class Reporter:
    """Accumulates sections; each flush() replaces the task's report."""

    def __init__(self, title: str, subtitle: str = ""):
        self.title = title
        self.subtitle = subtitle
        self._sections: list[str] = []

    # ---- builders (append a section, return self for chaining) ----
    def raw(self, html_fragment: str) -> "Reporter":
        self._sections.append(html_fragment)
        return self

    def h(self, text: str) -> "Reporter":
        return self.raw(
            f'<h3 style="margin:14px 0 4px;font-size:13px;color:#9a9aa4;'
            f'text-transform:uppercase;letter-spacing:.07em">{esc(text)}</h3>'
        )

    def p(self, text: str, color: str = "#c9c9cf") -> "Reporter":
        return self.raw(f'<p style="margin:6px 0;color:{color};font-size:13px">{esc(text)}</p>')

    def kv(self, items: dict) -> "Reporter":
        rows = "".join(
            f'<tr><td style="{_TD}color:#9a9aa4">{esc(k)}</td>'
            f'<td style="{_TD}">{esc(v)}</td></tr>'
            for k, v in items.items()
        )
        return self.raw(f'<table style="{_TABLE}">{rows}</table>')

    def table(self, headers: list[str], rows: list[list[str]]) -> "Reporter":
        """Cells are treated as HTML (pass through esc()/pill() yourself)."""
        head = "".join(f'<th style="{_TH}">{esc(h)}</th>' for h in headers)
        body = "".join(
            "<tr>" + "".join(f'<td style="{_TD}">{c}</td>' for c in r) + "</tr>"
            for r in rows
        )
        return self.raw(
            f'<table style="{_TABLE}"><thead><tr>{head}</tr></thead>'
            f"<tbody>{body}</tbody></table>"
        )

    def progress(self, done: int, total: int, label: str = "") -> "Reporter":
        f = 0 if total <= 0 else min(done / total, 1.0)
        return self.raw(
            f'<div style="margin:8px 0;font-size:12px;color:#9a9aa4">'
            f"{esc(label)} {done}/{total}"
            f'<div style="background:#1b1b21;border-radius:6px;height:8px;margin-top:4px">'
            f'<div style="background:#4d65ff;width:{f * 100:.0f}%;height:8px;'
            f'border-radius:6px"></div></div></div>'
        )

    # ---- render + flush ----
    def html(self) -> str:
        sub = (
            f'<p style="margin:2px 0 0;color:#9a9aa4;font-size:12px">{esc(self.subtitle)}</p>'
            if self.subtitle
            else ""
        )
        return (
            f'<div style="{_STYLE}">'
            f'<h2 style="margin:0;font-size:16px">{esc(self.title)}</h2>{sub}'
            + "".join(self._sections)
            + "</div>"
        )

    def reset_body(self) -> "Reporter":
        """Drop sections (title stays) — for pages rebuilt every flush."""
        self._sections = []
        return self

    async def flush(self) -> None:
        """Replace the task report; never let reporting kill the task."""
        try:
            import flyte.report

            await flyte.report.replace.aio(self.html(), do_flush=True)
        except Exception as e:  # noqa: BLE001 — observability must not break work
            print(f"[reporting] flush failed: {e}")
