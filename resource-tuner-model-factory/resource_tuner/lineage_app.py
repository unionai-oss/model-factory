"""Resource-tuner lineage dashboard — the factory's "Podium".

A scale-to-zero FastAPI app that reads the cluster's artifact registry and
renders (a) every version of every factory artifact with its producing run,
and (b) aggregate eval performance ACROSS tuner checkpoints: success rate
vs the rule-based baseline, median overprovisioning, and schema validity,
one point per tuner-eval-report version. Charts are server-rendered inline
SVG — no CDN, so the page degrades to nothing worse than itself.

Deploy:  uv run flyte --config .flyte/config.yaml deploy app.py lineage_app_env
"""

from __future__ import annotations

import html as _html
import json

import flyte
import flyte.app
import flyte.io
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from flyte.app.extras import FastAPIAppEnvironment

from .config import cluster_env_vars, cpu_resources
from .contracts import (
    ARTIFACT_EVAL_REPORT,
    ARTIFACT_SYNTHETIC,
    ARTIFACT_TASK_CORPUS,
    ARTIFACT_TUNER_CHECKPOINT,
)
from .shared import assets
from .shared.images import driver_image

STATIONS = [
    (ARTIFACT_TASK_CORPUS, "Task corpus"),
    (ARTIFACT_SYNTHETIC, "Synthetic corpus"),
    (ARTIFACT_TUNER_CHECKPOINT, "Tuner checkpoint"),
    (ARTIFACT_EVAL_REPORT, "Eval report"),
]

app = FastAPI(title="resource-tuner lineage")

# Eval-report payloads are immutable per URI; cache downloads for the
# lifetime of the replica.
_report_cache: dict[str, dict] = {}


async def _load_report(uri: str) -> dict | None:
    if uri in _report_cache:
        return _report_cache[uri]
    try:
        local = await flyte.io.File.from_existing_remote(uri).download()
        with open(local) as f:
            report = json.load(f)
    except Exception as e:  # noqa: BLE001 — one bad report must not blank the page
        report = {"error": f"{type(e).__name__}: {e}"}
    _report_cache[uri] = report
    return report


async def _state() -> dict:
    out: dict = {"artifacts": {}, "reports": []}
    for name, label in STATIONS:
        try:
            versions = await assets.list_versions(name, limit=50)
        except Exception as e:  # noqa: BLE001
            out["artifacts"][name] = {"label": label, "error": str(e), "versions": []}
            continue
        out["artifacts"][name] = {
            "label": label,
            "versions": [
                {"path": v.path, "run": v.run_name, "created_at": v.created_at}
                for v in versions
            ],
        }
    # Oldest-first so chart x-axis reads left→right in time.
    for v in reversed(out["artifacts"].get(ARTIFACT_EVAL_REPORT, {}).get("versions", [])):
        report = await _load_report(v["path"])
        if report and "error" not in report:
            out["reports"].append(
                {
                    "run": v["run"],
                    "created_at": v["created_at"],
                    "schema_validity": report.get("schema_validity"),
                    "success_rate": report.get("success_rate"),
                    "baseline_success_rate": report.get("baseline_success_rate"),
                    "median_overprovision_pct": report.get("median_overprovision_pct"),
                    "baseline_median_overprovision_pct": report.get(
                        "baseline_median_overprovision_pct"
                    ),
                    "auto_gate_passed": report.get("auto_gate_passed"),
                    "base_model": report.get("base_model"),
                }
            )
    return out


def _svg_lines(series: list[tuple[str, str, list[float | None]]], y_max: float,
               y_fmt: str = "{:.0%}") -> str:
    """Multi-series line chart. series = [(label, color, values)]."""
    n = max((len(v) for _, _, v in series), default=0)
    if n == 0:
        return "<p class='dim'>no eval reports yet</p>"
    w, h, pad = 640, 200, 34
    x = lambda i: pad + (w - 2 * pad) * (i / max(n - 1, 1))  # noqa: E731
    y = lambda v: h - pad - (h - 2 * pad) * (min(v, y_max) / y_max)  # noqa: E731
    parts = [f'<svg viewBox="0 0 {w} {h}" style="max-width:{w}px;background:#fafafa;border:1px solid #ddd">']
    for frac in (0.0, 0.5, 1.0):
        gy = h - pad - (h - 2 * pad) * frac
        parts.append(
            f'<line x1="{pad}" y1="{gy}" x2="{w - pad}" y2="{gy}" stroke="#e5e5e5"/>'
            f'<text x="4" y="{gy + 4}" font-size="10" fill="#999">'
            f"{y_fmt.format(y_max * frac)}</text>"
        )
    legend_x = pad
    for label, color, values in series:
        pts = " ".join(
            f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(values) if v is not None
        )
        if pts:
            parts.append(
                f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2"/>'
            )
            for i, v in enumerate(values):
                if v is not None:
                    parts.append(f'<circle cx="{x(i):.1f}" cy="{y(v):.1f}" r="3" fill="{color}"/>')
        parts.append(
            f'<text x="{legend_x}" y="14" font-size="11" fill="{color}">— {label}</text>'
        )
        legend_x += 9 * len(label) + 40
    parts.append("</svg>")
    return "".join(parts)


@app.get("/api/state")
async def api_state() -> JSONResponse:
    return JSONResponse(await _state())


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    state = await _state()
    reports = state["reports"]

    def col(key):
        return [r.get(key) for r in reports]

    success_chart = _svg_lines(
        [
            ("policy success", "#2a7d46", col("success_rate")),
            ("baseline success", "#888888", col("baseline_success_rate")),
            ("schema validity", "#4a7dbd", col("schema_validity")),
        ],
        y_max=1.0,
    )
    waste_chart = _svg_lines(
        [
            ("policy waste %", "#b4552d", col("median_overprovision_pct")),
            ("baseline waste %", "#888888", col("baseline_median_overprovision_pct")),
        ],
        y_max=100.0,
        y_fmt="{:.0f}%",
    )

    rows = "".join(
        "<tr>"
        f"<td>{i + 1}</td>"
        f"<td>{_html.escape(str(r['run']))}</td>"
        f"<td>{_html.escape(str(r['base_model'] or ''))}</td>"
        f"<td>{(r['schema_validity'] or 0):.0%}</td>"
        f"<td>{(r['success_rate'] or 0):.0%} / {(r['baseline_success_rate'] or 0):.0%}</td>"
        f"<td>{'-' if r['median_overprovision_pct'] is None else f'{r['median_overprovision_pct']:.0f}%'}"
        f" / {'-' if r['baseline_median_overprovision_pct'] is None else f'{r['baseline_median_overprovision_pct']:.0f}%'}</td>"
        f"<td>{'PASS' if r['auto_gate_passed'] else 'fail'}</td>"
        "</tr>"
        for i, r in enumerate(reports)
    )

    stations = "".join(
        f"<tr><td>{_html.escape(info['label'])}</td><td>{_html.escape(name)}</td>"
        f"<td>{len(info['versions'])}</td>"
        f"<td>{_html.escape(info['versions'][0]['run'] if info['versions'] else '-')}</td></tr>"
        for name, info in state["artifacts"].items()
    )

    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>resource-tuner lineage</title>
<style>
 body {{ font: 14px system-ui, sans-serif; margin: 2rem; color: #222; }}
 table {{ border-collapse: collapse; margin: 0.5rem 0 1.5rem; }}
 td, th {{ border: 1px solid #ccc; padding: 4px 10px; }}
 h2 {{ margin-top: 2rem; }} .dim {{ color: #888; }}
</style></head><body>
<h1>resource-tuner model factory</h1>
<p class="dim">artifact-mediated stations: corpus → GRPO train → eval; refresh for live state</p>
<h2>Eval performance across checkpoints</h2>
{success_chart}
{waste_chart}
<h2>Eval reports (oldest → newest)</h2>
<table><tr><th>#</th><th>producing run</th><th>base model</th><th>validity</th>
<th>success (policy/baseline)</th><th>waste (policy/baseline)</th><th>gate</th></tr>{rows}</table>
<h2>Artifacts</h2>
<table><tr><th>station</th><th>artifact</th><th>versions</th><th>latest run</th></tr>{stations}</table>
</body></html>"""


lineage_app_env = FastAPIAppEnvironment(
    name="rt-lineage",
    app=app,
    image=driver_image.with_pip_packages("fastapi", "uvicorn"),
    resources=cpu_resources(),
    scaling=flyte.app.Scaling(replicas=(0, 1), scaledown_after=900),
    requires_auth=False,
    env_vars=cluster_env_vars(),
)
