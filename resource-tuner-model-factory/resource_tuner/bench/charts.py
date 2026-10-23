"""Inline-SVG charts for the bench reports (no JS, Union palette)."""

from __future__ import annotations

import math

from ..shared.reporting import esc

ACCENT, ORANGE, RED, GREEN, MUTED = "#4d65ff", "#e69812", "#F43B3E", "#35c48d", "#9a9aa4"
_BG = "background:#0e0e12;border:1px solid #222228;border-radius:10px"


def hbars(items: list[tuple[str, float, str]], fmt: str = "{:.1%}", width: int = 760, title: str = "") -> str:
    """Horizontal bars: [(label, value, color)]. Negative values extend left of zero."""
    if not items:
        return ""
    lo = min(0.0, min(v for _, v, _ in items))
    hi = max(0.0, max(v for _, v, _ in items)) or 1.0
    span = hi - lo or 1.0
    row_h, left, right = 26, 190, 90
    h = row_h * len(items) + 30
    x = lambda v: left + (width - left - right) * (v - lo) / span  # noqa: E731
    parts = [f'<svg viewBox="0 0 {width} {h}" style="width:100%;max-width:{width}px;{_BG}">']
    if title:
        parts.append(f'<text x="10" y="16" font-size="11" fill="{MUTED}">{esc(title)}</text>')
    for i, (label, v, color) in enumerate(items):
        y = 24 + i * row_h
        x0, x1 = sorted((x(0), x(v)))
        parts.append(f'<text x="{left - 8}" y="{y + 13}" font-size="12" fill="#c9c9cf" text-anchor="end">{esc(label)}</text>')
        parts.append(f'<rect x="{x0:.1f}" y="{y + 2}" width="{max(x1 - x0, 1):.1f}" height="{row_h - 8}" rx="3" fill="{color}"/>')
        parts.append(f'<text x="{x1 + 6:.1f}" y="{y + 13}" font-size="11" fill="#c9c9cf">{esc(fmt.format(v))}</text>')
    parts.append(f'<line x1="{x(0):.1f}" y1="20" x2="{x(0):.1f}" y2="{h - 4}" stroke="#2e2e38"/>')
    parts.append("</svg>")
    return "".join(parts)


def scatter(points: list[tuple[str, float, float, str]], x_label: str, y_label: str,
            x_fmt: str = "{:.0%}", y_fmt: str = "{:.0%}", width: int = 760, height: int = 360,
            log_x: bool = False, log_y: bool = False) -> str:
    """Labelled scatter: [(label, x, y, color)] -- for frontier plots."""
    if not points:
        return ""
    tx = (lambda v: math.log10(max(v, 1e-6))) if log_x else (lambda v: v)
    ty = (lambda v: math.log10(max(v, 1e-6))) if log_y else (lambda v: v)
    xs, ys = [tx(p[1]) for p in points], [ty(p[2]) for p in points]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    padx, pady = (x1 - x0) * 0.08 or 0.1, (y1 - y0) * 0.1 or 0.1
    x0, x1, y0, y1 = x0 - padx, x1 + padx, y0 - pady, y1 + pady
    L, R, T, B = 60, 20, 20, 44
    px = lambda v: L + (width - L - R) * (v - x0) / (x1 - x0)  # noqa: E731
    py = lambda v: height - B - (height - T - B) * (v - y0) / (y1 - y0)  # noqa: E731
    parts = [f'<svg viewBox="0 0 {width} {height}" style="width:100%;max-width:{width}px;{_BG}">']
    for f in (0, 0.5, 1):
        gx, gy = x0 + (x1 - x0) * f, y0 + (y1 - y0) * f
        vx, vy = (10**gx if log_x else gx), (10**gy if log_y else gy)
        parts.append(f'<line x1="{px(gx):.0f}" y1="{T}" x2="{px(gx):.0f}" y2="{height - B}" stroke="#1d1d23"/>')
        parts.append(f'<line x1="{L}" y1="{py(gy):.0f}" x2="{width - R}" y2="{py(gy):.0f}" stroke="#1d1d23"/>')
        parts.append(f'<text x="{px(gx):.0f}" y="{height - B + 14}" font-size="10" fill="{MUTED}" text-anchor="middle">{esc(x_fmt.format(vx))}</text>')
        parts.append(f'<text x="{L - 6}" y="{py(gy) + 3:.0f}" font-size="10" fill="{MUTED}" text-anchor="end">{esc(y_fmt.format(vy))}</text>')
    parts.append(f'<text x="{(L + width - R) / 2:.0f}" y="{height - 8}" font-size="11" fill="{MUTED}" text-anchor="middle">{esc(x_label)}</text>')
    parts.append(f'<text x="14" y="{(T + height - B) / 2:.0f}" font-size="11" fill="{MUTED}" transform="rotate(-90 14 {(T + height - B) / 2:.0f})" text-anchor="middle">{esc(y_label)}</text>')
    for (label, _, _, color), vx, vy in zip(points, xs, ys):
        parts.append(f'<circle cx="{px(vx):.1f}" cy="{py(vy):.1f}" r="6" fill="{color}"/>')
        parts.append(f'<text x="{px(vx) + 9:.1f}" y="{py(vy) + 4:.1f}" font-size="11" fill="#c9c9cf">{esc(label)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def arm_color(variant: str) -> str:
    if variant.startswith("jev"):
        return ACCENT
    if variant.startswith(("sft", "zeroshot")):
        return ORANGE
    if variant.startswith("gbt"):
        return GREEN
    return MUTED


def frontier_chart(series: list[tuple[str, str, list[tuple[float, float]]]], x_label: str, y_label: str,
                   width: int = 760, height: int = 380) -> str:
    """Connected curves, one per arm: [(label, color, [(x, y), ...])]."""
    pts = [p for _, _, ps in series for p in ps]
    if not pts:
        return ""
    x0, x1 = 0.0, max(p[0] for p in pts) * 1.08 or 0.1
    y0, y1 = min(p[1] for p in pts) * 0.95, max(p[1] for p in pts) * 1.05
    L, R, T, B = 64, 150, 20, 44
    px = lambda v: L + (width - L - R) * (v - x0) / ((x1 - x0) or 1)  # noqa: E731
    py = lambda v: height - B - (height - T - B) * (v - y0) / ((y1 - y0) or 1)  # noqa: E731
    parts = [f'<svg viewBox="0 0 {width} {height}" style="width:100%;max-width:{width}px;{_BG}">']
    for f in (0, 0.25, 0.5, 0.75, 1):
        gx, gy = x0 + (x1 - x0) * f, y0 + (y1 - y0) * f
        parts.append(f'<line x1="{px(gx):.0f}" y1="{T}" x2="{px(gx):.0f}" y2="{height - B}" stroke="#1d1d23"/>')
        parts.append(f'<line x1="{L}" y1="{py(gy):.0f}" x2="{width - R}" y2="{py(gy):.0f}" stroke="#1d1d23"/>')
        parts.append(f'<text x="{px(gx):.0f}" y="{height - B + 14}" font-size="10" fill="{MUTED}" text-anchor="middle">{gx:.0%}</text>')
        parts.append(f'<text x="{L - 6}" y="{py(gy) + 3:.0f}" font-size="10" fill="{MUTED}" text-anchor="end">{gy:.0%}</text>')
    parts.append(f'<text x="{(L + width - R) / 2:.0f}" y="{height - 8}" font-size="11" fill="{MUTED}" text-anchor="middle">{esc(x_label)}</text>')
    parts.append(f'<text x="14" y="{(T + height - B) / 2:.0f}" font-size="11" fill="{MUTED}" transform="rotate(-90 14 {(T + height - B) / 2:.0f})" text-anchor="middle">{esc(y_label)}</text>')
    for i, (label, color, ps) in enumerate(series):
        ps = sorted(ps)
        poly = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in ps)
        parts.append(f'<polyline points="{poly}" fill="none" stroke="{color}" stroke-width="2" opacity="0.9"/>')
        for x, y in ps:
            parts.append(f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="3.5" fill="{color}"/>')
        ly = T + 14 + i * 16
        parts.append(f'<rect x="{width - R + 12}" y="{ly - 9}" width="10" height="10" fill="{color}"/>')
        parts.append(f'<text x="{width - R + 28}" y="{ly}" font-size="11" fill="#c9c9cf">{esc(label)}</text>')
    parts.append("</svg>")
    return "".join(parts)


PALETTE = ("#4d65ff", "#e69812", "#35c48d", "#F43B3E", "#b98cff", "#4dd0e1", "#ffd54f", "#9a9aa4", "#ff8a65", "#81c784")
