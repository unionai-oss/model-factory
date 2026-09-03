"""Resource-tuner lineage dashboard — the factory's "Podium".

A scale-to-zero FastAPI app rendering the artifact + run lineage as an
interactive graph (React Flow + dagre, zero-build from esm.sh — ported
from basic-model-factory's lineage app) in Union's dark design language,
plus the resource-tuner's own eval layer:

- checkpoint version cards carry eval badges (success vs baseline, waste,
  gate) linked via the report's `checkpoint_path`;
- an Eval panel charts schema validity, success rate and median waste
  across every tuner-eval-report version.

Server-rendered markup stays visible if the CDN is unreachable, so the
page degrades to tables + charts rather than a blank canvas.

Deploy:  uv run flyte --config .flyte/config.yaml deploy app.py lineage_app_env
Served at https://rt-lineage-{project}-{domain}.apps.<cluster> (pinned
subdomain, not a random one).
"""

from __future__ import annotations

import html as _html
import json
import os
from datetime import datetime, timezone

import flyte
import flyte.app
import flyte.io
import flyte.remote
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from flyte.app.extras import FastAPIAppEnvironment

from .config import APP_DOMAIN, APP_PROJECT, cluster_env_vars, cpu_resources
from .contracts import (
    ARTIFACT_EVAL_REPORT,
    ARTIFACT_SYNTHETIC,
    ARTIFACT_TASK_CORPUS,
    ARTIFACT_TUNER_CHECKPOINT,
)
from .shared import assets
from .shared.images import driver_image

# Stations → graph nodes, grouped by factory layer (drives accent colors).
STATIONS: list[dict[str, str]] = [
    {"artifact": ARTIFACT_SYNTHETIC, "label": "Synthetic corpus", "team": "data"},
    {"artifact": ARTIFACT_TASK_CORPUS, "label": "Task corpus", "team": "data"},
    {"artifact": ARTIFACT_TUNER_CHECKPOINT, "label": "Tuner checkpoint", "team": "training"},
    {"artifact": ARTIFACT_EVAL_REPORT, "label": "Eval report", "team": "eval"},
]
TEAMS: dict[str, str] = {"data": "Data", "training": "Training", "eval": "Evaluation"}
CONTRACT_EDGES: list[tuple[str, str]] = [
    (ARTIFACT_SYNTHETIC, ARTIFACT_TASK_CORPUS),
    (ARTIFACT_TASK_CORPUS, ARTIFACT_TUNER_CHECKPOINT),
    (ARTIFACT_TUNER_CHECKPOINT, ARTIFACT_EVAL_REPORT),
]
PRODUCER_TASKS: dict[str, str] = {
    "build_task_corpus": ARTIFACT_TASK_CORPUS,
    "synthetic_data_release": ARTIFACT_SYNTHETIC,
    "train_tuner": ARTIFACT_TUNER_CHECKPOINT,
    "eval_tuner": ARTIFACT_EVAL_REPORT,
}
REFRESH_INTERVAL_S = 20

app = FastAPI(title="resource-tuner lineage")


def _esc(s: object) -> str:
    return _html.escape(str(s))


def _run_url(project: str, domain: str, run_name: str) -> str:
    if not run_name or any(c in run_name for c in " /()"):
        return ""
    try:
        from flyte._initialize import get_client

        return get_client().console.run_url(project=project, domain=domain, run_name=run_name)
    except Exception:
        return ""


def _run_fields(run) -> dict:
    phase = getattr(run, "phase", None)
    out = {
        "task": "",
        "task_short": "",
        "phase": str(getattr(phase, "value", phase) or ""),
        "created_at": "",
        "triggered": False,
    }
    try:
        meta = run.pb2.action.metadata
        out["task"] = meta.task.id.name or ""
        out["task_short"] = meta.task.short_name or meta.funtion_name or ""
        out["triggered"] = "ARTIFACT_TRIGGER" in str(getattr(meta, "source", ""))
    except Exception:
        pass
    try:
        started = run.pb2.action.status.start_time
        if started.seconds:
            out["created_at"] = datetime.fromtimestamp(
                started.seconds, timezone.utc
            ).isoformat()
    except Exception:
        pass
    return out


# Eval-report payloads are immutable per URI; cache for the replica's life.
_report_cache: dict[str, dict] = {}


async def _load_report(uri: str) -> dict:
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


def _metrics_of(report: dict) -> dict:
    return {
        "schema_validity": report.get("schema_validity"),
        "success_rate": report.get("success_rate"),
        "baseline_success_rate": report.get("baseline_success_rate"),
        "median_overprovision_pct": report.get("median_overprovision_pct"),
        "baseline_median_overprovision_pct": report.get("baseline_median_overprovision_pct"),
        "auto_gate_passed": report.get("auto_gate_passed"),
        "base_model": report.get("base_model"),
    }


async def _collect() -> dict:
    project = os.environ.get("RT_PROJECT", APP_PROJECT)
    domain = os.environ.get("RT_DOMAIN", APP_DOMAIN)
    data: dict = {
        "stations": [],
        "edges": [{"source": s, "target": t} for s, t in CONTRACT_EDGES],
        "teams": TEAMS,
        "runs": [],
        "reports": [],
        "project": project,
        "domain": domain,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    versions_by_station: dict[str, list] = {}
    for station in STATIONS:
        name = station["artifact"]
        try:
            found = await assets.list_versions(name, limit=25)
        except Exception as e:  # noqa: BLE001
            found = []
            data.setdefault("errors", []).append(f"{name}: {e}")
        versions_by_station[name] = [
            {
                "version": f"{v.run_name}/{v.action_name}"
                if v.action_name
                else v.path.rsplit("/", 2)[-2],
                "url": v.path,
                "source": v.run_name,
                "run_url": _run_url(project, domain, v.run_name),
                "created_at": v.created_at,
                "eval": None,
            }
            for v in found
        ]

    # Attach eval metrics: to each report version, and to the checkpoint
    # version it scored (linked by checkpoint_path in the report payload).
    ckpt_by_path = {v["url"]: v for v in versions_by_station.get(ARTIFACT_TUNER_CHECKPOINT, [])}
    for v in versions_by_station.get(ARTIFACT_EVAL_REPORT, []):
        report = await _load_report(v["url"])
        if "error" in report:
            continue
        metrics = _metrics_of(report)
        v["eval"] = metrics
        target = ckpt_by_path.get(report.get("checkpoint_path", ""))
        if target is not None and target["eval"] is None:
            target["eval"] = metrics
        data["reports"].append(
            {"run": v["source"], "created_at": v["created_at"], **metrics}
        )
    data["reports"].reverse()  # oldest → newest for the charts

    for station in STATIONS:
        data["stations"].append(
            {
                "artifact": station["artifact"],
                "label": station["label"],
                "team": station["team"],
                "team_label": TEAMS[station["team"]],
                "versions": versions_by_station[station["artifact"]],
            }
        )

    produced_by: dict[str, set[str]] = {}
    for name, versions in versions_by_station.items():
        for v in versions:
            if v["source"]:
                produced_by.setdefault(v["source"], set()).add(name)
    try:
        async for r in flyte.remote.Run.listall.aio(
            limit=25, project=project, domain=domain, sort_by=("created_at", "desc")
        ):
            fields = _run_fields(r)
            stations = set(produced_by.get(r.name, ()))
            short = fields["task_short"] or fields["task"].rsplit(".", 1)[-1]
            if short in PRODUCER_TASKS:
                stations.add(PRODUCER_TASKS[short])
            data["runs"].append(
                {
                    "name": r.name,
                    "url": getattr(r, "url", "") or "",
                    "stations": sorted(stations),
                    **fields,
                }
            )
    except Exception as e:  # noqa: BLE001
        data.setdefault("errors", []).append(f"runs: {e}")
    return data


@app.get("/api/lineage")
@app.get("/api/state")  # kept for callers of the pre-graph dashboard
async def api_lineage() -> JSONResponse:
    return JSONResponse(await _collect())


def _fallback_html(d: dict) -> str:
    blocks = ""
    for st in d["stations"]:
        rows = (
            "".join(
                f"<li><code>{_esc(v['version'][:28])}</code></li>" for v in st["versions"][:6]
            )
            or "<li class='fb-empty'>no versions yet</li>"
        )
        blocks += (
            f"<div class='fb-station'><b>{_esc(st['label'])}</b>"
            f"<div class='fb-artifact'>{_esc(st['artifact'])}</div><ul>{rows}</ul></div>"
        )
    return (
        "<div class='fallback'><p>Rendering the interactive graph… if it never appears the "
        "CDN is unreachable — the <a href='/api/lineage'>JSON API</a> has everything.</p>"
        f"<div class='fb-grid'>{blocks}</div></div>"
    )


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    d = await _collect()
    boot = json.dumps({"data": d, "refreshMs": REFRESH_INTERVAL_S * 1000}).replace("</", "<\\/")
    return _PAGE_TEMPLATE.replace("__FALLBACK__", _fallback_html(d)).replace(
        "__BOOT_JSON__", boot
    )


# --------------------------------------------------------------------- page
# React Flow (@xyflow/react) + dagre, zero-build from esm.sh, in Union's
# dark palette (near-black surfaces, indigo #4d65ff primary, amber #e69812,
# red #F43B3E — tokens sampled from union.ai; swap for the Figma variables
# when exported). Plain (non-f) string: __TOKEN__ placeholders keep braces
# literal.

_PAGE_TEMPLATE = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Resource Tuner — lineage</title>
<link rel="stylesheet" href="https://unpkg.com/@xyflow/react@12.8.2/dist/style.css"/>
<style>
  :root {
    /* Union design language, dark */
    --bg: #0b0b0d; --panel: #131316; --card: #1a1a1f; --card-border: #2a2a31;
    --line: #222228; --line-soft: #1b1b21;
    --text: #f2f2f3; --muted: #9a9aa4; --dim: #5f5f6a;
    --primary: #4d65ff; --primary-soft: rgba(77,101,255,0.16);
    --amber: #e69812; --red: #F43B3E; --green: #35c48d;
    --data: #4d65ff; --training: #e69812; --eval: #35c48d;
  }
  * { box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Inter", sans-serif;
    margin: 0; background: var(--bg); color: var(--text);
  }
  code, .mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
  a { color: #8b9bff; }

  header { padding: 1.2rem 2rem 0.6rem; }
  .header-row { display: flex; align-items: center; gap: 10px; margin-bottom: 0.35rem; }
  .logo {
    width: 22px; height: 22px; border-radius: 6px; flex: none;
    background: linear-gradient(135deg, var(--primary), #2c3cb4);
    display: flex; align-items: center; justify-content: center;
    color: #fff; font-size: 12px; font-weight: 700;
  }
  h1 { font-size: 1.12rem; margin: 0; font-weight: 600; letter-spacing: -0.01em; }
  .sub { color: var(--muted); font-size: 0.8rem; margin: 0; }
  .sub code { background: #1b1b21; padding: 0 0.3rem; border-radius: 4px; color: #cfcfd6; }
  .spacer { margin-left: auto; }
  .chip {
    font-size: 0.75rem; color: var(--muted); text-decoration: none;
    border: 1px solid var(--card-border); border-radius: 8px; padding: 5px 12px;
    background: transparent; font-family: inherit; cursor: pointer;
    transition: color .15s, border-color .15s;
  }
  .chip:hover { color: var(--text); border-color: #3d3d47; }

  main { padding: 0.8rem 2rem 0.6rem; display: flex; flex-direction: column; gap: 0.9rem; }
  .top { display: flex; gap: 0.9rem; height: 54vh; min-height: 400px; }

  .sidebar {
    width: 252px; flex: none; background: var(--panel); border: 1px solid var(--line);
    border-radius: 12px; padding: 0.7rem; display: flex; flex-direction: column;
    gap: 2px; overflow-y: auto;
  }
  .side-tabs {
    display: flex; gap: 2px; margin: 0.1rem 0.15rem 0.55rem;
    background: #0e0e12; border: 1px solid var(--line); border-radius: 8px; padding: 3px;
  }
  .side-tab {
    flex: 1; border: 0; background: transparent; color: var(--muted);
    font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.06em;
    padding: 5px 0; border-radius: 6px; cursor: pointer; font-family: inherit;
  }
  .side-tab:hover { color: var(--text); }
  .side-tab.active { background: var(--primary-soft); color: #b9c3ff; }
  .side-group {
    font-size: 0.63rem; text-transform: uppercase; letter-spacing: 0.08em;
    color: var(--dim); padding: 0.65rem 0.6rem 0.25rem;
  }
  .side-item {
    display: flex; align-items: center; gap: 8px; padding: 7px 10px;
    border-radius: 8px; cursor: pointer; border: 1px solid transparent; user-select: none;
  }
  .side-item:hover { background: #1b1b21; }
  .side-item.active { background: #202027; border-color: #32323b; }
  .side-item .dot { width: 7px; height: 7px; border-radius: 999px; flex: none; }
  .dot.ok { background: var(--green); }
  .dot.bad { background: var(--red); }
  .dot.running { background: var(--primary); box-shadow: 0 0 0 0 rgba(77,101,255,0.6); animation: pulse 1.8s infinite; }
  .dot.idle { background: #43434e; }
  @keyframes pulse { to { box-shadow: 0 0 0 7px rgba(77,101,255,0); } }
  .side-col { display: flex; flex-direction: column; min-width: 0; gap: 1px; }
  .side-task { font-size: 11.5px; font-weight: 600; color: var(--text); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .side-name { font-size: 10px; color: var(--muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .side-count { margin-left: auto; flex: none; font-size: 9.5px; color: var(--muted); background: #24242c; border-radius: 999px; padding: 1px 7px; }
  .side-note { margin-top: auto; padding: 0.7rem 0.5rem 0.2rem; font-size: 0.68rem; color: var(--dim); line-height: 1.5; }

  .canvas {
    flex: 1; border: 1px solid var(--line); border-radius: 12px;
    background: #0e0e12; overflow: hidden; position: relative; min-width: 0;
  }
  .react-flow__controls { box-shadow: none; border: 1px solid var(--card-border); border-radius: 8px; overflow: hidden; }
  .react-flow__controls-button { background: var(--card); border-bottom: 1px solid var(--card-border); fill: var(--muted); }
  .react-flow__controls-button:hover { background: #24242c; }
  .react-flow__edge-path { stroke: #43434e; }
  .react-flow__edge-textbg { fill: #0e0e12; }
  .react-flow__edge-text { fill: var(--muted); font-size: 10px; }
  .react-flow__attribution { background: transparent; color: #4b4b55; }

  .st-card {
    width: 248px; background: var(--card); border: 1px solid var(--card-border);
    border-radius: 12px; padding: 11px 13px; cursor: pointer;
    transition: border-color .15s, box-shadow .15s; border-left: 3px solid var(--accent);
  }
  .st-card:hover { border-color: #3d3d47; }
  .st-card.selected { box-shadow: 0 0 0 1.5px var(--accent); border-color: var(--accent); }
  .st-card .row1 { display: flex; align-items: center; gap: 8px; }
  .st-card .icon { color: var(--accent); flex: none; display: flex; }
  .st-card .label { font-size: 12.5px; font-weight: 600; color: var(--text); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .st-card .count { margin-left: auto; flex: none; font-size: 9.5px; color: var(--muted); background: #24242c; border-radius: 999px; padding: 1px 7px; }
  .st-card .row2 { display: flex; align-items: center; gap: 8px; margin-top: 6px; }
  .st-card .artifact { font-size: 10.5px; color: var(--muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .st-card .latest {
    margin-top: 7px; padding-top: 7px; border-top: 1px solid #24242c;
    font-size: 10px; color: var(--dim); display: flex; align-items: center; gap: 6px;
    white-space: nowrap;
  }
  .st-card .latest b { color: var(--muted); font-weight: 500; }
  .st-card .latest span { overflow: hidden; text-overflow: ellipsis; }
  .st-card .latest .iconbtn { margin-left: auto; color: #8b9bff; }

  .v-card {
    width: 232px; background: #16161b; border: 1px solid #282830;
    border-radius: 9px; padding: 8px 10px; cursor: pointer; transition: border-color .15s;
  }
  .v-card:hover { border-color: var(--accent); }
  .v-card .vid { font-size: 10.5px; color: var(--text); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .v-card .vrow { display: flex; align-items: center; gap: 6px; margin-top: 4px; }
  .v-card .vrun { font-size: 9.5px; color: var(--dim); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .v-card .evalrow { display: flex; gap: 4px; margin-top: 6px; flex-wrap: wrap; }
  .badge {
    font-size: 9px; padding: 1.5px 6px; border-radius: 5px; white-space: nowrap;
    border: 1px solid transparent;
  }
  .badge.good { background: rgba(53,196,141,0.13); color: #5fd8a9; border-color: rgba(53,196,141,0.3); }
  .badge.warn { background: rgba(230,152,18,0.13); color: #ecb257; border-color: rgba(230,152,18,0.3); }
  .badge.bad { background: rgba(244,59,62,0.12); color: #f9797b; border-color: rgba(244,59,62,0.3); }
  .badge.neutral { background: #22222a; color: var(--muted); }
  .group-label { font-size: 10.5px; letter-spacing: 0.02em; color: var(--muted); display: flex; align-items: center; gap: 6px; white-space: nowrap; }
  .group-label .swatch { width: 7px; height: 7px; border-radius: 2px; flex: none; }

  .pill { flex: none; font-size: 9.5px; padding: 1px 7px; border-radius: 999px; text-transform: capitalize; white-space: nowrap; }
  .pill.right { margin-left: auto; }
  .pill.succeeded { background: rgba(53,196,141,0.15); color: #5fd8a9; }
  .pill.failed, .pill.aborted, .pill.timed_out { background: rgba(244,59,62,0.14); color: #f9797b; }
  .pill.running, .pill.initializing { background: var(--primary-soft); color: #9daaff; }
  .pill.queued, .pill.unknown { background: rgba(148,148,160,0.15); color: #a1a1ab; }
  .pill.dark-run { background: rgba(230,152,18,0.14); color: #ecb257; text-transform: none; }

  .toolbar {
    display: flex; align-items: center; gap: 0.55rem; flex-wrap: wrap;
    background: var(--panel); border: 1px solid var(--line); border-radius: 12px;
    padding: 0.5rem 0.7rem;
  }
  .search {
    display: flex; align-items: center; gap: 7px; flex: 1; min-width: 220px; max-width: 460px;
    background: #0e0e12; border: 1px solid #24242c; border-radius: 9px; padding: 5px 10px;
    transition: border-color .15s;
  }
  .search:focus-within { border-color: var(--primary); }
  .search svg { color: var(--dim); flex: none; }
  .search input { flex: 1; min-width: 0; background: transparent; border: 0; outline: none; color: var(--text); font-family: inherit; font-size: 0.8rem; }
  .search input::placeholder { color: var(--dim); }
  .search kbd { flex: none; font-family: inherit; font-size: 0.62rem; color: var(--dim); border: 1px solid #2a2a33; border-radius: 4px; padding: 0 4px; }
  .search .clear { flex: none; border: 0; background: transparent; color: var(--muted); cursor: pointer; font-size: 0.85rem; padding: 0 2px; }
  .search .clear:hover { color: var(--text); }
  .hits { font-size: 0.7rem; color: var(--dim); flex: none; white-space: nowrap; }
  .team-chips { display: flex; gap: 4px; flex-wrap: wrap; }
  .team-chip {
    display: flex; align-items: center; gap: 6px; font-size: 0.68rem; color: var(--muted);
    border: 1px solid var(--card-border); border-radius: 999px; padding: 4px 10px;
    background: transparent; cursor: pointer; font-family: inherit; white-space: nowrap;
    transition: color .15s, border-color .15s, background .15s;
  }
  .team-chip .swatch { width: 7px; height: 7px; border-radius: 999px; flex: none; opacity: 0.35; }
  .team-chip.on { color: var(--text); border-color: #3d3d47; background: #1b1b21; }
  .team-chip.on .swatch { opacity: 1; }
  .team-chip:hover { border-color: #47474f; }
  .toggle { font-size: 0.68rem; color: var(--muted); border: 1px solid var(--card-border); border-radius: 999px; padding: 4px 11px; background: transparent; cursor: pointer; font-family: inherit; white-space: nowrap; }
  .toggle.on { color: #b9c3ff; border-color: var(--primary); background: var(--primary-soft); }
  .toggle:hover { border-color: #47474f; }

  .dimmed { opacity: 0.22; }
  .ringed { box-shadow: 0 0 0 1.5px var(--accent); border-color: var(--accent) !important; }
  .iconbtn { flex: none; border: 0; background: transparent; color: var(--muted); cursor: pointer; padding: 1px 3px; border-radius: 5px; font-size: 11px; line-height: 1; font-family: inherit; }
  .iconbtn:hover { background: #2a2a33; color: var(--text); }

  .bottom { display: flex; gap: 0.9rem; align-items: stretch; }
  .panel { flex: 1; min-width: 0; background: var(--panel); border: 1px solid var(--line); border-radius: 12px; display: flex; flex-direction: column; overflow: hidden; }
  .panel h2 {
    margin: 0; padding: 0.6rem 0.9rem; font-size: 0.7rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.07em; color: var(--muted);
    border-bottom: 1px solid var(--line); display: flex; align-items: center; gap: 8px;
  }
  .panel h2 .sel { color: var(--text); text-transform: none; letter-spacing: 0; font-weight: 600; }
  .panel h2 .spacer { margin-left: auto; }
  .panel h2 .team-chip { padding: 2px 8px; font-size: 0.62rem; text-transform: none; letter-spacing: 0; }
  tr.ringed td { background: #202027; box-shadow: inset 2px 0 0 var(--primary); }
  .side-item.dimmed { opacity: 0.3; }
  .panel-body { overflow: auto; max-height: 240px; }
  table { border-collapse: collapse; width: 100%; }
  th { position: sticky; top: 0; background: var(--panel); text-align: left; font-size: 0.63rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--dim); font-weight: 600; padding: 6px 12px; border-bottom: 1px solid var(--line); }
  td { padding: 6px 12px; font-size: 11.5px; border-bottom: 1px solid var(--line-soft); color: var(--muted); }
  tbody tr:hover td { background: #18181e; }
  td.primary { color: var(--text); }
  td .link { cursor: pointer; color: #8b9bff; }
  .empty-row { padding: 1.1rem 0.9rem; color: var(--dim); font-size: 11.5px; font-style: italic; }

  .charts { display: flex; gap: 1.2rem; padding: 0.8rem 0.9rem; flex-wrap: wrap; }
  .chart { flex: 1; min-width: 300px; }
  .chart .cap { font-size: 0.68rem; color: var(--dim); margin-bottom: 4px; }
  .chart svg { width: 100%; height: auto; background: #0e0e12; border: 1px solid var(--line); border-radius: 10px; }

  .errors { display: flex; flex-direction: column; }
  .err { color: #f9797b; font-size: 11.5px; background: rgba(244,59,62,0.08); border: 1px solid rgba(244,59,62,0.25); border-radius: 8px; padding: 6px 10px; margin: 0 0 6px; }
  footer { padding: 0.6rem 2rem 2rem; font-size: 0.75rem; color: var(--dim); }

  .fallback { flex: 1; overflow: auto; padding: 1.2rem 1.4rem; color: var(--muted); font-size: 0.8rem; }
  .fb-grid { display: flex; flex-wrap: wrap; gap: 0.7rem; margin-top: 0.9rem; }
  .fb-station { background: var(--card); border: 1px solid var(--card-border); border-radius: 10px; padding: 10px 13px; min-width: 210px; }
  .fb-station b { color: var(--text); font-size: 12.5px; }
  .fb-artifact { font-size: 10.5px; color: var(--dim); margin: 2px 0 7px; }
  .fb-station ul { list-style: none; padding: 0; margin: 0; }
  .fb-station li { font-size: 11px; padding: 2px 0; }
  .fb-empty { color: var(--dim); font-style: italic; }
  .toast { position: fixed; bottom: 1.5rem; left: 50%; transform: translateX(-50%); background: #24242c; border: 1px solid #383842; color: var(--text); font-size: 0.78rem; padding: 7px 14px; border-radius: 999px; z-index: 50; }
</style>
</head>
<body>
<header>
  <div class="header-row">
    <span class="logo">U</span>
    <h1>Resource Tuner — factory lineage</h1>
    <span class="spacer"></span>
    <a class="chip" href="/api/lineage">JSON API ↗</a>
  </div>
  <p class="sub">Artifact flow across tuner stations — corpus → GRPO training → eval; checkpoint cards carry their eval verdicts.</p>
</header>
<main id="root">__FALLBACK__</main>
<footer>Read-only view of the control plane · dark runs are trigger-fired (OnArtifact) · palette follows Union's design tokens.</footer>

<script type="importmap">
{
  "imports": {
    "react": "https://esm.sh/react@18.3.1",
    "react-dom/client": "https://esm.sh/react-dom@18.3.1/client?deps=react@18.3.1",
    "@xyflow/react": "https://esm.sh/@xyflow/react@12.8.2?deps=react@18.3.1,react-dom@18.3.1",
    "@dagrejs/dagre": "https://esm.sh/@dagrejs/dagre@1.1.4",
    "htm": "https://esm.sh/htm@3.1.1"
  }
}
</script>
<script id="lineage-data" type="application/json">__BOOT_JSON__</script>
<script type="module">
try {
  const React = (await import("react")).default;
  const { createRoot } = await import("react-dom/client");
  const { ReactFlow, Background, BackgroundVariant, Controls, Handle, Position, MarkerType } =
    await import("@xyflow/react");
  const dagre = (await import("@dagrejs/dagre")).default;
  const htm = (await import("htm")).default;
  const html = htm.bind(React.createElement);

  const BOOT = JSON.parse(document.getElementById("lineage-data").textContent);
  const TEAM_COLOR = { data: "#4d65ff", training: "#e69812", eval: "#35c48d" };
  const ACTIVE_PHASES = ["running", "initializing", "queued"];
  const FAILED_PHASES = ["failed", "aborted", "timedout", "timed_out"];

  const refreshMs = (() => {
    const raw = new URLSearchParams(location.search).get("refresh");
    if (raw === null) return BOOT.refreshMs;
    const s = Number(raw);
    return Number.isFinite(s) && s > 0 ? Math.max(2, s) * 1000 : 0;
  })();

  const shortId = (v) => {
    const parts = String(v || "").split("/").filter(Boolean);
    return parts.length > 1 ? parts.slice(-2).join("/") : (parts[0] || "?");
  };
  const phaseOf = (p) => String(p || "unknown").toLowerCase().replace(/[^a-z_]/g, "") || "unknown";
  const relTime = (iso) => {
    if (!iso) return "";
    const t = Date.parse(iso);
    if (!Number.isFinite(t)) return "";
    const s = Math.max(0, (Date.now() - t) / 1000);
    if (s < 60) return Math.round(s) + "s ago";
    if (s < 3600) return Math.round(s / 60) + "m ago";
    if (s < 86400) return Math.round(s / 3600) + "h ago";
    return Math.round(s / 86400) + "d ago";
  };
  const openUrl = (url) => url && window.open(url, "_blank", "noopener");
  const pct = (x) => (x == null ? "–" : Math.round(x * 100) + "%");
  const pctRaw = (x) => (x == null ? "–" : Math.round(x) + "%");

  const stationHay = (s) =>
    [s.label, s.artifact, s.team_label, ...s.versions.map((v) => v.version + " " + v.source)].join(" ").toLowerCase();
  const versionHay = (v) => [v.version, v.source, v.url].join(" ").toLowerCase();
  const runHay = (r) => [r.name, r.task, r.task_short, r.phase, ...(r.stations || [])].join(" ").toLowerCase();

  function connectedSet(edges, root) {
    const up = {}, down = {};
    edges.forEach((e) => {
      (down[e.source] = down[e.source] || []).push(e.target);
      (up[e.target] = up[e.target] || []).push(e.source);
    });
    const out = new Set([root]);
    const walk = (m, id) => (m[id] || []).forEach((n) => { if (!out.has(n)) { out.add(n); walk(m, n); } });
    walk(up, root); walk(down, root);
    return out;
  }

  const StationIcon = () => html`
    <svg class="icon" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor"
         stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <path d="M2 20h20V9l-6 4V9l-6 4V5H4l-2 15Z"></path>
    </svg>`;
  const SearchIcon = () => html`
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor"
         stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <circle cx="11" cy="11" r="7"></circle><path d="m20 20-3.5-3.5"></path>
    </svg>`;

  // Eval badges: the added UI layer — success vs baseline, waste, gate.
  const EvalBadges = ({ ev }) => {
    if (!ev) return null;
    const beatsSuccess = ev.success_rate != null && ev.baseline_success_rate != null &&
      ev.success_rate >= ev.baseline_success_rate;
    const beatsWaste = ev.median_overprovision_pct != null && ev.baseline_median_overprovision_pct != null &&
      ev.median_overprovision_pct <= ev.baseline_median_overprovision_pct;
    return html`
      <div class="evalrow">
        <span class="badge ${ev.auto_gate_passed ? "good" : "bad"}">${ev.auto_gate_passed ? "GATE PASS" : "gate fail"}</span>
        <span class="badge ${beatsSuccess ? "good" : "warn"}" title="success vs baseline">
          fit ${pct(ev.success_rate)} vs ${pct(ev.baseline_success_rate)}</span>
        <span class="badge ${beatsWaste ? "good" : "warn"}" title="median overprovision vs baseline">
          waste ${pctRaw(ev.median_overprovision_pct)} vs ${pctRaw(ev.baseline_median_overprovision_pct)}</span>
        <span class="badge neutral" title="schema validity">valid ${pct(ev.schema_validity)}</span>
      </div>`;
  };

  const StationNode = ({ data }) => html`
    <div class="st-card ${data.selected ? "selected" : ""} ${data.dimmed ? "dimmed" : ""} ${data.ringed ? "ringed" : ""}"
         style=${{ "--accent": data.color }}
         title=${data.artifact + " · " + data.teamLabel}
         onClick=${() => data.onSelect(data.artifact)}>
      <${Handle} type="target" position=${Position.Left} style=${{ opacity: 0 }} />
      <div class="row1">
        <${StationIcon} />
        <span class="label">${data.label}</span>
        <span class="count">${data.versionCount}</span>
      </div>
      <div class="row2">
        <span class="artifact">${data.artifact}</span>
        ${data.activeRuns
          ? html`<span class="pill right running">${data.activeRuns} running</span>`
          : html`<span class="pill right ${data.versionCount ? "succeeded" : "unknown"}">
              ${data.versionCount ? "ready" : "no versions"}</span>`}
      </div>
      <div class="latest">
        ${data.latest
          ? html`<span>latest <b>${shortId(data.latest.version)}</b></span>
              <button class="iconbtn" title=${"Open producing run " + data.latest.source}
                      onClick=${(e) => { e.stopPropagation(); openUrl(data.latest.run_url); }}>run ↗</button>`
          : html`<span>awaiting first publish</span>`}
      </div>
      ${data.latest && data.latest.eval ? html`<${EvalBadges} ev=${data.latest.eval} />` : null}
      <${Handle} type="source" position=${Position.Right} style=${{ opacity: 0 }} />
    </div>`;

  const VersionNode = ({ data }) => html`
    <div class="v-card ${data.dimmed ? "dimmed" : ""} ${data.ringed ? "ringed" : ""}"
         style=${{ "--accent": data.color }}
         title=${(data.run_url ? "Open run " + data.source + "\n" : "") + data.url}
         onClick=${() => (data.run_url ? openUrl(data.run_url) : data.onCopy(data.url))}>
      <${Handle} type="target" position=${Position.Left} style=${{ opacity: 0 }} />
      <div class="vid">${shortId(data.version)}</div>
      <div class="vrow">
        <span class="vrun">${data.source || "unknown run"}</span>
        <button class="iconbtn" title="Copy object-store path"
                onClick=${(e) => { e.stopPropagation(); data.onCopy(data.url); }}>⧉</button>
      </div>
      <${EvalBadges} ev=${data.eval} />
      <${Handle} type="source" position=${Position.Right} style=${{ opacity: 0 }} />
    </div>`;

  const GroupLabel = ({ data }) => html`
    <div class="group-label"><span class="swatch" style=${{ background: data.color }}></span>${data.label}</div>`;

  const nodeTypes = { station: StationNode, version: VersionNode, groupLabel: GroupLabel };

  const ST_W = 248, ST_H = 148;
  const V_W = 232, V_H = 76, V_GAP = 10;
  const PAD_X = 22, PAD_TOP = 40, PAD_BOTTOM = 18;

  const runsByStation = (data) => {
    const m = {};
    (data.runs || []).forEach((r) => (r.stations || []).forEach((a) => (m[a] = m[a] || []).push(r)));
    return m;
  };

  function layoutStations(data, byStation, ctx) {
    const g = new dagre.graphlib.Graph();
    g.setGraph({ rankdir: "LR", nodesep: 34, ranksep: 96 });
    g.setDefaultEdgeLabel(() => ({}));
    data.stations.forEach((s) => g.setNode(s.artifact, { width: ST_W, height: ST_H }));
    data.edges.forEach((e) => g.setEdge(e.source, e.target));
    dagre.layout(g);

    const active = {};
    Object.entries(byStation).forEach(([k, runs]) => {
      active[k] = runs.filter((r) => ACTIVE_PHASES.includes(phaseOf(r.phase))).length;
    });
    const nodes = data.stations.map((s) => {
      const p = g.node(s.artifact);
      return {
        id: s.artifact, type: "station",
        position: { x: p.x - ST_W / 2, y: p.y - ST_H / 2 },
        data: {
          ...s, color: TEAM_COLOR[s.team] || "#8a8a94", teamLabel: s.team_label,
          versionCount: s.versions.length, latest: s.versions[0] || null,
          activeRuns: active[s.artifact] || 0,
          selected: ctx.selected === s.artifact,
          dimmed: ctx.dimStation(s), ringed: ctx.ringStation(s), onSelect: ctx.onSelect,
        },
      };
    });
    const edges = data.edges.map((e, i) => ({
      id: "se" + i, source: e.source, target: e.target, type: "smoothstep",
      animated: (active[e.target] || 0) > 0, label: e.source,
      style: ctx.dimEdge(e.source, e.target) ? { opacity: 0.18 } : undefined,
      markerEnd: { type: MarkerType.ArrowClosed, color: "#43434e", width: 16, height: 16 },
    }));
    return { nodes, edges };
  }

  function layoutVersions(data, ctx) {
    const shown = data.stations.filter((s) => s.versions.length).map((s) => ({
      ...s, versions: s.versions.slice(0, 8),
    }));
    if (!shown.length) return { nodes: [], edges: [] };
    const hOf = (v) => (v.eval ? V_H + 22 : V_H);
    const sizeOf = (s) => ({
      width: V_W + 2 * PAD_X,
      height: PAD_TOP + s.versions.reduce((a, v) => a + hOf(v) + V_GAP, -V_GAP) + PAD_BOTTOM,
    });
    const g = new dagre.graphlib.Graph();
    g.setGraph({ rankdir: "LR", nodesep: 44, ranksep: 82 });
    g.setDefaultEdgeLabel(() => ({}));
    shown.forEach((s) => g.setNode(s.artifact, sizeOf(s)));
    const visible = new Set(shown.map((s) => s.artifact));
    data.edges.filter((e) => visible.has(e.source) && visible.has(e.target))
      .forEach((e) => g.setEdge(e.source, e.target));
    dagre.layout(g);

    const nodes = [], idOf = {}, byRun = {}, dimmed = {};
    shown.forEach((s) => {
      const color = TEAM_COLOR[s.team] || "#8a8a94";
      const size = sizeOf(s), p = g.node(s.artifact);
      const gid = "g:" + s.artifact;
      nodes.push({
        id: gid, type: "group",
        position: { x: p.x - size.width / 2, y: p.y - size.height / 2 },
        data: {}, draggable: false, selectable: false,
        style: {
          width: size.width, height: size.height,
          background: ctx.selected === s.artifact ? "rgba(77,101,255,0.05)" : "rgba(255,255,255,0.015)",
          border: "1px dashed " + (ctx.selected === s.artifact ? color : "#2a2a33"),
          borderRadius: 14,
        },
      });
      nodes.push({
        id: gid + ":label", type: "groupLabel", parentId: gid,
        position: { x: 14, y: 13 }, data: { label: s.label, color },
        draggable: false, selectable: false,
      });
      idOf[s.artifact] = [];
      let y = PAD_TOP;
      s.versions.forEach((v, i) => {
        const vid = s.artifact + "::" + i;
        idOf[s.artifact].push(vid);
        if (v.source) byRun[s.artifact + "|" + v.source] = vid;
        dimmed[vid] = ctx.dimVersion(s, v);
        nodes.push({
          id: vid, type: "version", parentId: gid, extent: "parent",
          position: { x: PAD_X, y },
          data: {
            ...v, color, station: s.artifact, onCopy: ctx.onCopy,
            dimmed: dimmed[vid], ringed: ctx.ringVersion(s, v),
          },
        });
        y += hOf(v) + V_GAP;
      });
    });

    const edges = [];
    const fade = (a, b) => (dimmed[a] || dimmed[b] ? 0.15 : 1);
    data.edges.filter((e) => visible.has(e.source) && visible.has(e.target)).forEach((e, i) => {
      const target = shown.find((s) => s.artifact === e.target);
      let observed = 0;
      target.versions.forEach((v, vi) => {
        const up = v.source && byRun[e.source + "|" + v.source];
        if (!up) return;
        observed += 1;
        const down = e.target + "::" + vi;
        edges.push({
          id: "ve" + i + "-" + vi, source: up, target: down, type: "smoothstep",
          label: v.source, style: { opacity: fade(up, down) },
          markerEnd: { type: MarkerType.ArrowClosed, color: "#43434e", width: 14, height: 14 },
        });
      });
      if (!observed) {
        const up = idOf[e.source][0], down = idOf[e.target][0];
        edges.push({
          id: "ce" + i, source: up, target: down, type: "smoothstep", label: "contract",
          style: { strokeDasharray: "5 4", stroke: "#38383f", opacity: fade(up, down) },
          markerEnd: { type: MarkerType.ArrowClosed, color: "#38383f", width: 14, height: 14 },
        });
      }
    });
    return { nodes, edges };
  }

  // ------------------------------------------------- eval aggregate charts
  const LineChart = ({ series, yMax, yFmt }) => {
    const n = Math.max(...series.map(([, , v]) => v.length), 0);
    if (!n) return html`<div class="empty-row">no eval reports yet</div>`;
    const W = 620, H = 190, P = 34;
    const x = (i) => P + (W - 2 * P) * (n === 1 ? 0.5 : i / (n - 1));
    const y = (v) => H - P - (H - 2 * P) * Math.min(v, yMax) / yMax;
    return html`
      <svg viewBox="0 0 ${W} ${H}">
        ${[0, 0.5, 1].map((f) => html`
          <g key=${f}>
            <line x1=${P} y1=${H - P - (H - 2 * P) * f} x2=${W - P} y2=${H - P - (H - 2 * P) * f}
                  stroke="#222228"/>
            <text x="4" y=${H - P - (H - 2 * P) * f + 4} font-size="10" fill="#5f5f6a">${yFmt(yMax * f)}</text>
          </g>`)}
        ${series.map(([label, color, values], si) => html`
          <g key=${label}>
            <polyline fill="none" stroke=${color} stroke-width="2"
              points=${values.map((v, i) => (v == null ? null : x(i) + "," + y(v))).filter(Boolean).join(" ")} />
            ${values.map((v, i) => (v == null ? null :
              html`<circle key=${i} cx=${x(i)} cy=${y(v)} r="3" fill=${color} />`))}
            <text x=${P + si * 150} y="14" font-size="11" fill=${color}>— ${label}</text>
          </g>`)}
      </svg>`;
  };

  const EvalPanel = ({ reports }) => {
    const col = (k) => reports.map((r) => r[k]);
    return html`
      <section class="panel">
        <h2>Eval performance across checkpoints <span class="side-count">${reports.length}</span></h2>
        <div class="charts">
          <div class="chart">
            <div class="cap">success rate & schema validity (policy vs baseline)</div>
            <${LineChart} yMax=${1} yFmt=${(v) => Math.round(v * 100) + "%"}
              series=${[
                ["policy fit", "#35c48d", col("success_rate")],
                ["baseline fit", "#9a9aa4", col("baseline_success_rate")],
                ["validity", "#4d65ff", col("schema_validity")],
              ]} />
          </div>
          <div class="chart">
            <div class="cap">median overprovision % (lower is better)</div>
            <${LineChart} yMax=${100} yFmt=${(v) => Math.round(v) + "%"}
              series=${[
                ["policy waste", "#e69812", col("median_overprovision_pct")],
                ["baseline waste", "#9a9aa4", col("baseline_median_overprovision_pct")],
              ]} />
          </div>
        </div>
      </section>`;
  };

  const Toolbar = ({ query, setQuery, hits, teams, toggleTeam, data, trace, setTrace,
                     highlightRun, setHighlightRun, onReset, searchRef }) => html`
    <div class="toolbar">
      <label class="search">
        <${SearchIcon} />
        <input ref=${searchRef} value=${query} placeholder="Search artifacts, versions, runs…"
               onInput=${(e) => setQuery(e.target.value)} />
        ${query
          ? html`<button class="clear" title="Clear (Esc)" onClick=${() => setQuery("")}>✕</button>`
          : html`<kbd>/</kbd>`}
      </label>
      ${query ? html`<span class="hits">${hits} match${hits === 1 ? "" : "es"}</span>` : null}
      <div class="team-chips">
        ${Object.entries(data.teams || {}).map(([team, label]) => html`
          <button key=${team} class="team-chip ${teams.has(team) ? "on" : ""}"
                  onClick=${() => toggleTeam(team)}>
            <span class="swatch" style=${{ background: TEAM_COLOR[team] }}></span>${label}
          </button>`)}
      </div>
      <button class="toggle ${trace ? "on" : ""}" title="Dim stations not up/downstream of the selected one"
              onClick=${() => setTrace(!trace)}>Trace lineage</button>
      ${highlightRun
        ? html`<button class="toggle on" onClick=${() => setHighlightRun(null)}>run ${highlightRun} ✕</button>`
        : null}
      <button class="toggle" onClick=${onReset}>Reset</button>
    </div>`;

  const Sidebar = ({ data, view, setView, selected, onSelect, byStation, statusLine, matchStation }) => {
    const teams = Object.entries(data.teams || {});
    return html`
      <aside class="sidebar">
        <div class="side-tabs">
          <button class="side-tab ${view === "stations" ? "active" : ""}" onClick=${() => setView("stations")}>Stations</button>
          <button class="side-tab ${view === "versions" ? "active" : ""}" onClick=${() => setView("versions")}>Versions</button>
        </div>
        ${teams.map(([team, teamLabel]) => {
          const members = data.stations.filter((s) => s.team === team);
          if (!members.length) return null;
          return html`
            <div key=${team}>
              <div class="side-group">${teamLabel}</div>
              ${members.map((s) => {
                const runs = byStation[s.artifact] || [];
                const running = runs.some((r) => ACTIVE_PHASES.includes(phaseOf(r.phase)));
                const failed = !running && runs.length && FAILED_PHASES.includes(phaseOf(runs[0].phase));
                const status = running ? "running" : failed ? "bad" : s.versions.length ? "ok" : "idle";
                return html`
                  <div key=${s.artifact} title=${s.artifact}
                       class="side-item ${selected === s.artifact ? "active" : ""} ${matchStation(s) ? "" : "dimmed"}"
                       onClick=${() => onSelect(s.artifact)}>
                    <span class="dot ${status}"></span>
                    <span class="side-col">
                      <span class="side-task">${s.label}</span>
                      <span class="side-name mono">${s.artifact}</span>
                    </span>
                    <span class="side-count">${s.versions.length}</span>
                  </div>`;
              })}
            </div>`;
        })}
        <div class="side-note">${statusLine}</div>
      </aside>`;
  };

  const VersionsPanel = ({ station, rows, onCopy, highlightRun, setHighlightRun }) => html`
    <section class="panel">
      <h2>Versions <span class="sel">${station ? station.label : ""}</span>
        ${station ? html`<span class="side-count">${rows.length}</span>` : null}</h2>
      <div class="panel-body">
        ${rows.length
          ? html`<table>
              <thead><tr><th>version</th><th>produced by</th><th>eval</th><th>path</th><th></th></tr></thead>
              <tbody>
                ${rows.map((v, i) => html`
                  <tr key=${i} class=${highlightRun && v.source === highlightRun ? "ringed" : ""}>
                    <td class="primary mono">${shortId(v.version)}</td>
                    <td class="mono link" title="Highlight everything this run produced"
                        onClick=${() => setHighlightRun(v.source === highlightRun ? null : v.source)}>
                      ${v.source || "—"}</td>
                    <td>${v.eval
                      ? html`<span class="badge ${v.eval.auto_gate_passed ? "good" : "bad"}">
                          ${v.eval.auto_gate_passed ? "PASS" : "fail"}</span>
                          ${" fit " + pct(v.eval.success_rate) + " · waste " + pctRaw(v.eval.median_overprovision_pct)}`
                      : "—"}</td>
                    <td class="mono link" title=${"Copy " + v.url} onClick=${() => onCopy(v.url)}>
                      ${v.url ? v.url.slice(0, 36) + (v.url.length > 36 ? "…" : "") : "—"}</td>
                    <td>${v.run_url
                      ? html`<button class="iconbtn" onClick=${() => openUrl(v.run_url)}>↗</button>` : null}</td>
                  </tr>`)}
              </tbody>
            </table>`
          : html`<div class="empty-row">
              ${station ? "No versions match the current filters." : "Select a station to see its versions."}
            </div>`}
      </div>
    </section>`;

  const PHASE_FILTERS = [["all", "All"], ["active", "Running"], ["succeeded", "Succeeded"], ["failed", "Failed"]];

  const RunsPanel = ({ rows, stationLabels, phaseFilter, setPhaseFilter, highlightRun, setHighlightRun, onSelect }) => html`
    <section class="panel">
      <h2>Recent runs <span class="side-count">${rows.length}</span>
        <span class="spacer"></span>
        <span class="team-chips">
          ${PHASE_FILTERS.map(([key, label]) => html`
            <button key=${key} class="team-chip ${phaseFilter === key ? "on" : ""}"
                    onClick=${() => setPhaseFilter(key)}>${label}</button>`)}
        </span>
      </h2>
      <div class="panel-body">
        ${rows.length
          ? html`<table>
              <thead><tr><th>run</th><th>task</th><th>feeds</th><th>phase</th><th>started</th><th></th></tr></thead>
              <tbody>
                ${rows.map((r) => html`
                  <tr key=${r.name} class=${highlightRun === r.name ? "ringed" : ""}>
                    <td class="primary mono link"
                        onClick=${() => setHighlightRun(highlightRun === r.name ? null : r.name)}>
                      ${r.name} ${r.triggered ? html`<span class="pill dark-run">dark</span>` : null}</td>
                    <td title=${r.task}>${r.task_short || r.task || "—"}</td>
                    <td>${(r.stations || []).length
                      ? (r.stations || []).map((a, i) => html`
                          <span key=${a}>${i ? ", " : ""}<span class="link"
                            onClick=${() => onSelect(a)}>${stationLabels[a] || a}</span></span>`)
                      : "—"}</td>
                    <td><span class="pill ${phaseOf(r.phase)}">${r.phase || "unknown"}</span></td>
                    <td>${relTime(r.created_at) || "—"}</td>
                    <td>${r.url
                      ? html`<button class="iconbtn" onClick=${() => openUrl(r.url)}>↗</button>` : null}</td>
                  </tr>`)}
              </tbody>
            </table>`
          : html`<div class="empty-row">No runs match the current filters.</div>`}
      </div>
    </section>`;

  const firstWithVersions = (d) => {
    const s = d.stations.find((x) => x.versions.length) || d.stations[0];
    return s ? s.artifact : null;
  };
  const initialParams = new URLSearchParams(location.search);
  const allTeams = (d) => new Set(Object.keys(d.teams || {}));

  const App = () => {
    const [data, setData] = React.useState(BOOT.data);
    const [view, setView] = React.useState(initialParams.get("view") === "versions" ? "versions" : "stations");
    const [selected, setSelected] = React.useState(() => initialParams.get("sel") || firstWithVersions(BOOT.data));
    const [query, setQuery] = React.useState(() => initialParams.get("q") || "");
    const [teams, setTeams] = React.useState(() => {
      const raw = initialParams.get("teams");
      return raw ? new Set(raw.split(",").filter(Boolean)) : allTeams(BOOT.data);
    });
    const [trace, setTrace] = React.useState(initialParams.get("trace") === "1");
    const [highlightRun, setHighlightRun] = React.useState(() => initialParams.get("run") || null);
    const [phaseFilter, setPhaseFilter] = React.useState("all");
    const [updatedAt, setUpdatedAt] = React.useState(null);
    const [toast, setToast] = React.useState("");
    const searchRef = React.useRef(null);

    const onCopy = React.useCallback((text) => {
      if (!text) return;
      navigator.clipboard?.writeText(text).then(
        () => setToast("Copied path to clipboard"), () => setToast(text));
    }, []);
    React.useEffect(() => {
      if (!toast) return;
      const id = setTimeout(() => setToast(""), 1800);
      return () => clearTimeout(id);
    }, [toast]);

    React.useEffect(() => {
      const onKey = (e) => {
        const typing = /^(INPUT|TEXTAREA)$/.test(document.activeElement?.tagName || "");
        if (e.key === "/" && !typing) { e.preventDefault(); searchRef.current?.focus(); }
        else if (e.key === "Escape") {
          if (query) setQuery("");
          else if (highlightRun) setHighlightRun(null);
          document.activeElement?.blur?.();
        }
      };
      window.addEventListener("keydown", onKey);
      return () => window.removeEventListener("keydown", onKey);
    }, [query, highlightRun]);

    React.useEffect(() => {
      if (!refreshMs) return;
      let inFlight = false;
      const id = setInterval(async () => {
        if (inFlight) return;
        inFlight = true;
        try {
          const resp = await fetch("/api/lineage", { cache: "no-store" });
          if (resp.ok) { setData(await resp.json()); setUpdatedAt(new Date()); }
        } catch (e) { console.warn("lineage refresh failed", e); }
        finally { inFlight = false; }
      }, refreshMs);
      return () => clearInterval(id);
    }, []);

    React.useEffect(() => {
      const p = new URLSearchParams(location.search);
      const set = (k, v) => (v ? p.set(k, v) : p.delete(k));
      set("view", view === "stations" ? "" : view);
      set("sel", selected || "");
      set("q", query);
      set("run", highlightRun || "");
      set("trace", trace ? "1" : "");
      const all = allTeams(data);
      set("teams", teams.size === all.size ? "" : [...teams].sort().join(","));
      const qs = p.toString();
      history.replaceState(null, "", location.pathname + (qs ? "?" + qs : ""));
    }, [view, selected, query, highlightRun, trace, teams, data]);

    const byStation = React.useMemo(() => runsByStation(data), [data]);
    const onSelect = React.useCallback((artifact) => setSelected(artifact), []);
    const toggleTeam = React.useCallback((team) => setTeams((prev) => {
      const next = new Set(prev);
      next.has(team) ? next.delete(team) : next.add(team);
      return next;
    }), []);
    const onReset = React.useCallback(() => {
      setQuery(""); setHighlightRun(null); setTrace(false); setTeams(allTeams(data));
      setPhaseFilter("all");
    }, [data]);

    const q = query.trim().toLowerCase();
    const traceSet = React.useMemo(
      () => (trace && selected ? connectedSet(data.edges, selected) : null), [trace, selected, data]);
    const runStations = React.useMemo(() => {
      if (!highlightRun) return null;
      const r = (data.runs || []).find((x) => x.name === highlightRun);
      return new Set(r ? r.stations || [] : []);
    }, [highlightRun, data]);

    const matchStation = React.useCallback((s) =>
      teams.has(s.team) && (!q || stationHay(s).includes(q)), [teams, q]);
    const dimStation = React.useCallback((s) => {
      if (!matchStation(s)) return true;
      if (traceSet && !traceSet.has(s.artifact)) return true;
      if (runStations && !runStations.has(s.artifact)) return true;
      return false;
    }, [matchStation, traceSet, runStations]);
    const ringStation = React.useCallback((s) =>
      !!(runStations && runStations.has(s.artifact)), [runStations]);
    const matchVersion = React.useCallback((s, v) =>
      teams.has(s.team) && (!q || stationHay(s).includes(q) || versionHay(v).includes(q)), [teams, q]);
    const dimVersion = React.useCallback((s, v) => {
      if (!matchVersion(s, v)) return true;
      if (traceSet && !traceSet.has(s.artifact)) return true;
      if (highlightRun && v.source !== highlightRun) return true;
      return false;
    }, [matchVersion, traceSet, highlightRun]);
    const ringVersion = React.useCallback((s, v) =>
      !!(highlightRun && v.source === highlightRun), [highlightRun]);

    const ctx = {
      selected, onSelect, onCopy, dimStation, ringStation, dimVersion, ringVersion,
      dimEdge: (a, b) => {
        const find = (id) => data.stations.find((s) => s.artifact === id);
        const sa = find(a), sb = find(b);
        return !sa || !sb || dimStation(sa) || dimStation(sb);
      },
    };
    const { nodes, edges } = React.useMemo(
      () => (view === "stations" ? layoutStations(data, byStation, ctx) : layoutVersions(data, ctx)),
      [data, view, selected, byStation, q, teams, traceSet, highlightRun]);

    const hits = React.useMemo(() => {
      if (!q) return 0;
      let n = data.stations.filter((s) => stationHay(s).includes(q)).length;
      data.stations.forEach((s) => { n += s.versions.filter((v) => versionHay(v).includes(q)).length; });
      n += (data.runs || []).filter((r) => runHay(r).includes(q)).length;
      return n;
    }, [q, data]);

    const stationLabels = React.useMemo(
      () => Object.fromEntries(data.stations.map((s) => [s.artifact, s.label])), [data]);
    const station = data.stations.find((s) => s.artifact === selected) || null;
    const versionRows = station
      ? station.versions.filter((v) =>
          (!q || versionHay(v).includes(q) || stationHay(station).includes(q)) &&
          (!highlightRun || v.source === highlightRun))
      : [];
    const runRows = (data.runs || []).filter((r) => {
      if (q && !runHay(r).includes(q)) return false;
      const ph = phaseOf(r.phase);
      if (phaseFilter === "active" && !ACTIVE_PHASES.includes(ph)) return false;
      if (phaseFilter === "succeeded" && ph !== "succeeded") return false;
      if (phaseFilter === "failed" && !FAILED_PHASES.includes(ph)) return false;
      if (!(r.stations || []).length) return teams.size === allTeams(data).size;
      return (r.stations || []).some((a) => {
        const s = data.stations.find((x) => x.artifact === a);
        return s && teams.has(s.team);
      });
    });

    const snapshot = relTime(data.fetched_at);
    const statusLine =
      data.project + "/" + data.domain +
      (snapshot ? " · snapshot " + snapshot : "") + " · " +
      (refreshMs
        ? "auto-refresh " + refreshMs / 1000 + "s" +
          (updatedAt ? " (last " + updatedAt.toLocaleTimeString() + ")" : "")
        : "auto-refresh off — add ?refresh=20");

    return html`
      <${React.Fragment}>
        <${Toolbar} query=${query} setQuery=${setQuery} hits=${hits} teams=${teams}
            toggleTeam=${toggleTeam} data=${data} trace=${trace} setTrace=${setTrace}
            highlightRun=${highlightRun} setHighlightRun=${setHighlightRun}
            onReset=${onReset} searchRef=${searchRef} />
        <div class="top">
          <${Sidebar} data=${data} view=${view} setView=${setView} selected=${selected}
              onSelect=${onSelect} byStation=${byStation} statusLine=${statusLine}
              matchStation=${matchStation} />
          <div class="canvas">
            <${ReactFlow} key=${view} nodes=${nodes} edges=${edges} nodeTypes=${nodeTypes}
                fitView fitViewOptions=${{ padding: 0.14, maxZoom: 1 }}
                minZoom=${0.2} proOptions=${{ hideAttribution: true }}
                nodesConnectable=${false} colorMode="dark">
              <${Background} variant=${BackgroundVariant.Dots} gap=${22} size=${1.4} color="#222228" />
              <${Controls} showInteractive=${false} />
            <//>
          </div>
        </div>
        <${EvalPanel} reports=${data.reports || []} />
        <div class="bottom">
          <${VersionsPanel} station=${station} rows=${versionRows} onCopy=${onCopy}
              highlightRun=${highlightRun} setHighlightRun=${setHighlightRun} />
          <${RunsPanel} rows=${runRows} stationLabels=${stationLabels} phaseFilter=${phaseFilter}
              setPhaseFilter=${setPhaseFilter} highlightRun=${highlightRun}
              setHighlightRun=${setHighlightRun} onSelect=${onSelect} />
        </div>
        ${(data.errors || []).length
          ? html`<div class="errors">${data.errors.map((e, i) => html`<p key=${i} class="err">${e}</p>`)}</div>`
          : null}
        ${toast ? html`<div class="toast">${toast}</div>` : null}
      <//>`;
  };

  createRoot(document.getElementById("root")).render(html`<${App} />`);
} catch (err) {
  console.error(err);
}
</script>
</body>
</html>
"""

APP_NAME = "rt-lineage"

lineage_app_env = FastAPIAppEnvironment(
    name=APP_NAME,
    app=app,
    image=driver_image.with_pip_packages("fastapi", "uvicorn"),
    resources=cpu_resources(),
    scaling=flyte.app.Scaling(replicas=(0, 1), scaledown_after=900),
    requires_auth=False,
    # Pinned {app_name}-{project}-{domain} subdomain — stable across
    # redeploys, unlike the default randomly-generated one.
    domain=flyte.app.Domain(subdomain=f"{APP_NAME}-{APP_PROJECT}-{APP_DOMAIN}"),
    env_vars=cluster_env_vars(),
    description="Resource-tuner lineage: artifact graph + eval metrics across checkpoints",
)
