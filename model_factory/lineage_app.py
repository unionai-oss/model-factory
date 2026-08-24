"""Factory lineage app — the POC's "Podium": one page showing the whole
factory in action.

A FastAPI AppEnvironment (scale-to-zero) that reads the cluster's own APIs
(`flyte.remote`) and renders the artifact graph: every version of every
factory artifact, the run that produced it, recent runs of the factory
stations, and paused HITL gates awaiting a signal.

The page is a React Flow + dagre canvas loaded zero-build from esm.sh (same
approach as the sensor plugin's run-lineage dashboard). Server-rendered
markup inside ``#app`` stays visible if the CDN is unreachable, so the page
degrades to a plain table rather than a blank canvas.

Deploy:  uv run flyte --config ~/.flyte/config-model-factory.yaml deploy app.py lineage_app_env
Local:   uv run flyte serve --local app.py lineage_app_env
"""

from __future__ import annotations

import html as _html
import json
import os
from datetime import datetime, timezone

import flyte
import flyte.app
from flyte.app.extras import FastAPIAppEnvironment
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

from .config import (
    APP_DOMAIN,
    APP_ORG,
    APP_PROJECT,
    REQUIRE_APP_AUTH,
    cluster_env_vars,
)
from .contracts import (
    ARTIFACT_CHECKPOINT,
    ARTIFACT_EVAL_REPORT,
    ARTIFACT_INFERENCE_ENDPOINT,
    ARTIFACT_PROMOTED,
    ARTIFACT_RL_DATASET,
    ARTIFACT_SYNTHETIC,
)
from .shared import assets
from .shared.images import cpu_image

# Factory stations → graph nodes. Each artifact is one team's published
# contract (contracts.py), so the station's owner is part of its identity.
STATIONS: list[dict[str, str]] = [
    {"artifact": ARTIFACT_SYNTHETIC, "label": "Synthetic tasks", "team": "data-eng"},
    {"artifact": ARTIFACT_RL_DATASET, "label": "RL task dataset", "team": "data-eng"},
    {"artifact": ARTIFACT_CHECKPOINT, "label": "Policy checkpoint", "team": "training"},
    {"artifact": ARTIFACT_EVAL_REPORT, "label": "Eval report", "team": "eval"},
    {"artifact": ARTIFACT_PROMOTED, "label": "Promoted model", "team": "eval"},
    {"artifact": ARTIFACT_INFERENCE_ENDPOINT, "label": "Inference endpoint", "team": "inference"},
]

TEAMS: dict[str, str] = {
    "data-eng": "Data engineering",
    "training": "Model training",
    "eval": "Model eval",
    "inference": "Inference",
}

# The consumes/publishes table from contracts.py, as a DAG. Note this is not
# a straight line: `policy-checkpoint` fans out to eval and inference.
CONTRACT_EDGES: list[tuple[str, str]] = [
    (ARTIFACT_SYNTHETIC, ARTIFACT_RL_DATASET),
    (ARTIFACT_RL_DATASET, ARTIFACT_CHECKPOINT),
    (ARTIFACT_CHECKPOINT, ARTIFACT_EVAL_REPORT),
    (ARTIFACT_EVAL_REPORT, ARTIFACT_PROMOTED),
    (ARTIFACT_CHECKPOINT, ARTIFACT_INFERENCE_ENDPOINT),
]

REFRESH_INTERVAL_S = 20

app = FastAPI(title="Model Factory Lineage")


@app.exception_handler(Exception)
async def _traceback_handler(request, exc):
    """Surface handler tracebacks in the response — this is an internal
    read-only debugging app; opacity costs more than exposure here."""
    import traceback

    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return JSONResponse({"error": str(exc), "traceback": tb.splitlines()[-15:]}, status_code=500)


def _esc(s: object) -> str:
    return _html.escape(str(s))


def _station_for_task(task_name: str) -> str:
    """Map a run's task name back to the station it feeds (assets.PRODUCERS)."""
    if not task_name:
        return ""
    for artifact, producers in assets.PRODUCERS.items():
        if any(task_name.endswith(p) or p.endswith(task_name) for p in producers):
            return artifact
    return ""


def _run_url(project: str, domain: str, run_name: str) -> str:
    """Console URL for a run — pure string building, no API call.

    `Run.url` only exists on runs we already fetched; asset versions name
    their producing run without carrying one, so build it directly.
    """
    if not run_name:
        return ""
    try:
        from flyte._initialize import get_client

        return get_client().console.run_url(project=project, domain=domain, run_name=run_name)
    except Exception:
        return ""


def _run_fields(run) -> dict:
    """Task name, phase and start time for a run.

    `Run` exposes neither `task_name` nor `created_at` — both live on the
    `pb2.action` message (`metadata.task.id.name`, `status.start_time`), and
    `phase` is an `ActionPhase` enum whose `str()` is "ActionPhase.SUCCEEDED".
    pb2 is not a stable API surface, so every hop is guarded.
    """
    phase = getattr(run, "phase", None)
    out = {
        "task": "",
        "task_short": "",
        "phase": str(getattr(phase, "value", phase) or ""),
        "created_at": "",
    }
    try:
        meta = run.pb2.action.metadata
        out["task"] = meta.task.id.name or ""
        out["task_short"] = meta.task.short_name or meta.funtion_name or ""
    except Exception:
        pass
    try:
        started = run.pb2.action.status.start_time
        if started.seconds:
            out["created_at"] = datetime.fromtimestamp(started.seconds, timezone.utc).isoformat()
    except Exception:
        pass
    return out


async def _collect() -> dict:
    """Pull asset versions + recent runs from the control plane.

    Uses model_factory.assets: artifact registry when the backend serves it,
    otherwise run-scan fallback (see assets.py).
    """
    project = os.environ.get("MF_PROJECT", "model-factory")
    domain = os.environ.get("MF_DOMAIN", "development")
    data: dict = {
        "stations": [],
        "edges": [{"source": s, "target": t} for s, t in CONTRACT_EDGES],
        "teams": TEAMS,
        "runs": [],
        "project": project,
        "domain": domain,
        "run_url_base": _run_url(project, domain, "RUN").removesuffix("RUN"),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    produced_by: dict[str, set[str]] = {}
    for station in STATIONS:
        name = station["artifact"]
        try:
            versions = [
                {
                    "version": f"{v.run_name}/{v.action_name}" if v.action_name else v.path.rsplit("/", 2)[-2],
                    "url": v.path,
                    "kind": v.via,
                    "source": v.run_name,
                    "run_url": _run_url(project, domain, v.run_name),
                }
                for v in await assets.list_versions(name, project=project, domain=domain, limit=8)
            ]
        except Exception as e:
            versions = []
            data.setdefault("errors", []).append(f"{name}: {e}")
        for v in versions:
            if v["source"]:
                produced_by.setdefault(v["source"], set()).add(name)
        data["stations"].append(
            {
                "artifact": name,
                "label": station["label"],
                "team": station["team"],
                "team_label": TEAMS[station["team"]],
                "versions": versions,
            }
        )
    try:
        async for r in flyte.remote.Run.listall.aio(
            limit=25, project=project, domain=domain, sort_by=("created_at", "desc")
        ):
            fields = _run_fields(r)
            # A run feeds a station either because it published one of its
            # versions (historical, from the scan above) or because its root
            # task is that station's producer (live, before anything lands).
            stations = set(produced_by.get(r.name, ()))
            live = _station_for_task(fields["task"])
            if live:
                stations.add(live)
            data["runs"].append(
                {
                    "name": r.name,
                    "url": getattr(r, "url", "") or "",
                    "stations": sorted(stations),
                    **fields,
                }
            )
    except Exception as e:
        data.setdefault("errors", []).append(f"runs: {e}")
    return data


@app.get("/api/lineage")
async def api_lineage() -> JSONResponse:
    return JSONResponse(await _collect())


def _fallback_html(d: dict) -> str:
    """Server-rendered summary shown until React mounts (and forever if the
    CDN is blocked). Keeps the page useful without any JS."""
    blocks = ""
    for st in d["stations"]:
        rows = "".join(
            f"<li><code>{_esc(v['version'][:28])}</code>"
            f"<span class='fb-kind'>{_esc(v['kind'])}</span></li>"
            for v in st["versions"]
        ) or "<li class='fb-empty'>no versions yet</li>"
        blocks += (
            f"<div class='fb-station'><b>{_esc(st['label'])}</b>"
            f"<div class='fb-artifact'>{_esc(st['artifact'])} · {_esc(st['team_label'])}</div>"
            f"<ul>{rows}</ul></div>"
        )
    return (
        "<div class='top'><div class='fallback'><p>Rendering the interactive graph… if it never "
        "appears the CDN is unreachable — the <a href='/api/lineage'>JSON API</a> "
        "has everything below.</p>"
        f"<div class='fb-grid'>{blocks}</div></div></div>"
    )


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    d = await _collect()
    boot = json.dumps({"data": d, "refreshMs": REFRESH_INTERVAL_S * 1000}).replace("</", "<\\/")
    return (
        _PAGE_TEMPLATE.replace("__FALLBACK__", _fallback_html(d))
        .replace("__BOOT_JSON__", boot)
    )


# --------------------------------------------------------------------- page
#
# React Flow (@xyflow/react) + dagre, zero-build from esm.sh. Plain (non-f)
# string with __TOKEN__ placeholders so the CSS/JS braces stay literal.

_PAGE_TEMPLATE = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Model Factory — lineage</title>
<link rel="stylesheet" href="https://unpkg.com/@xyflow/react@12.8.2/dist/style.css"/>
<style>
  :root {
    --bg: #0a0a0f; --panel: #131318; --card: #1c1c22; --card-border: #33333c;
    --text: #e7e7ea; --muted: #8a8a94; --dim: #5c5c66;
    --data-eng: #38bdf8; --training: #a78bfa; --eval: #f59e0b; --inference: #34d399;
  }
  * { box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    margin: 0; background: var(--bg); color: var(--text);
  }
  code, .mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
  a { color: #93c5fd; }

  header { padding: 1.2rem 2rem 0.6rem; }
  .header-row { display: flex; align-items: center; gap: 10px; margin-bottom: 0.35rem; }
  h1 { font-size: 1.15rem; margin: 0; font-weight: 600; }
  .logo { font-size: 1.15rem; }
  .sub { color: var(--muted); font-size: 0.8rem; margin: 0; }
  .sub code { background: #1e1e25; padding: 0 0.3rem; border-radius: 4px; color: #cfcfd6; }
  .spacer { margin-left: auto; }
  .chip {
    font-size: 0.75rem; color: var(--muted); text-decoration: none;
    border: 1px solid #2a2a33; border-radius: 8px; padding: 5px 12px;
    background: transparent; font-family: inherit; cursor: pointer;
    transition: color 0.15s ease, border-color 0.15s ease;
  }
  .chip:hover { color: var(--text); border-color: #4d4d59; }

  main { padding: 0.8rem 2rem 0.6rem; display: flex; flex-direction: column; gap: 0.9rem; }
  .top { display: flex; gap: 0.9rem; height: 60vh; min-height: 420px; }

  .sidebar {
    width: 252px; flex: none; background: var(--panel); border: 1px solid #23232b;
    border-radius: 12px; padding: 0.7rem; display: flex; flex-direction: column;
    gap: 2px; overflow-y: auto;
  }
  .side-tabs {
    display: flex; gap: 2px; margin: 0.1rem 0.15rem 0.55rem;
    background: #0e0e13; border: 1px solid #23232b; border-radius: 8px; padding: 3px;
  }
  .side-tab {
    flex: 1; border: 0; background: transparent; color: var(--muted);
    font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.06em;
    padding: 5px 0; border-radius: 6px; cursor: pointer; font-family: inherit;
  }
  .side-tab:hover { color: var(--text); }
  .side-tab.active { background: #26262e; color: var(--text); }
  .side-group {
    font-size: 0.63rem; text-transform: uppercase; letter-spacing: 0.08em;
    color: var(--dim); padding: 0.65rem 0.6rem 0.25rem;
  }
  .side-item {
    display: flex; align-items: center; gap: 8px; padding: 7px 10px;
    border-radius: 8px; cursor: pointer; border: 1px solid transparent; user-select: none;
  }
  .side-item:hover { background: #1d1d24; }
  .side-item.active { background: #22222b; border-color: #34343e; }
  .side-item .dot { width: 7px; height: 7px; border-radius: 999px; flex: none; }
  .dot.ok { background: #4ade80; }
  .dot.bad { background: #f87171; }
  .dot.running { background: #60a5fa; box-shadow: 0 0 0 0 rgba(96,165,250,0.6); animation: pulse 1.8s infinite; }
  .dot.idle { background: #4a4a55; }
  @keyframes pulse {
    to { box-shadow: 0 0 0 7px rgba(96,165,250,0); }
  }
  .side-col { display: flex; flex-direction: column; min-width: 0; gap: 1px; }
  .side-task {
    font-size: 11.5px; font-weight: 600; color: var(--text);
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  }
  .side-name { font-size: 10px; color: var(--muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .side-count {
    margin-left: auto; flex: none; font-size: 9.5px; color: var(--muted);
    background: #26262e; border-radius: 999px; padding: 1px 7px;
  }
  .side-note { margin-top: auto; padding: 0.7rem 0.5rem 0.2rem; font-size: 0.68rem; color: var(--dim); line-height: 1.5; }

  .canvas {
    flex: 1; border: 1px solid #23232b; border-radius: 12px;
    background: #101015; overflow: hidden; position: relative; min-width: 0;
  }
  .react-flow__controls { box-shadow: none; border: 1px solid #2a2a33; border-radius: 8px; overflow: hidden; }
  .react-flow__controls-button { background: var(--card); border-bottom: 1px solid #2a2a33; fill: var(--muted); }
  .react-flow__controls-button:hover { background: #26262e; }
  .react-flow__edge-path { stroke: #4a4a55; }
  .react-flow__edge-textbg { fill: #101015; }
  .react-flow__edge-text { fill: var(--muted); font-size: 10px; }
  .react-flow__attribution { background: transparent; color: #55555f; }

  /* station + version cards */
  .st-card {
    width: 248px; background: var(--card); border: 1px solid var(--card-border);
    border-radius: 12px; padding: 11px 13px; cursor: pointer;
    transition: border-color 0.15s ease, box-shadow 0.15s ease;
    border-left: 3px solid var(--accent);
  }
  .st-card:hover { border-color: #4d4d59; }
  .st-card.selected { box-shadow: 0 0 0 1.5px var(--accent); border-color: var(--accent); }
  .st-card .row1 { display: flex; align-items: center; gap: 8px; }
  .st-card .icon { color: var(--accent); flex: none; display: flex; }
  .st-card .label {
    font-size: 12.5px; font-weight: 600; color: var(--text);
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  }
  .st-card .count {
    margin-left: auto; flex: none; font-size: 9.5px; color: var(--muted);
    background: #26262e; border-radius: 999px; padding: 1px 7px;
  }
  .st-card .row2 { display: flex; align-items: center; gap: 8px; margin-top: 6px; }
  .st-card .artifact {
    font-size: 10.5px; color: var(--muted); white-space: nowrap;
    overflow: hidden; text-overflow: ellipsis;
  }
  .st-card .latest {
    margin-top: 7px; padding-top: 7px; border-top: 1px solid #26262e;
    font-size: 10px; color: var(--dim); white-space: nowrap;
    overflow: hidden; text-overflow: ellipsis;
  }
  .st-card .latest b { color: var(--muted); font-weight: 500; }
  .st-card .latest { display: flex; align-items: center; gap: 6px; }
  .st-card .latest span { overflow: hidden; text-overflow: ellipsis; }
  .st-card .latest .iconbtn { margin-left: auto; color: #93c5fd; }

  .v-card {
    width: 216px; background: #16161c; border: 1px solid #2c2c35;
    border-radius: 9px; padding: 8px 10px; cursor: pointer;
    transition: border-color 0.15s ease;
  }
  .v-card:hover { border-color: var(--accent); }
  .v-card .vid {
    font-size: 10.5px; color: var(--text); white-space: nowrap;
    overflow: hidden; text-overflow: ellipsis;
  }
  .v-card .vrow { display: flex; align-items: center; gap: 6px; margin-top: 4px; }
  .v-card .vrun { font-size: 9.5px; color: var(--dim); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .group-label {
    font-size: 10.5px; letter-spacing: 0.02em; color: var(--muted);
    display: flex; align-items: center; gap: 6px; white-space: nowrap;
  }
  .group-label .swatch { width: 7px; height: 7px; border-radius: 2px; flex: none; }

  .pill {
    flex: none; font-size: 9.5px; padding: 1px 7px; border-radius: 999px;
    text-transform: capitalize; white-space: nowrap;
  }
  .pill.right { margin-left: auto; }
  .pill.succeeded { background: rgba(22,163,74,0.15); color: #4ade80; }
  .pill.failed, .pill.aborted, .pill.timed_out { background: rgba(220,38,38,0.15); color: #f87171; }
  .pill.running, .pill.initializing { background: rgba(2,132,199,0.18); color: #60a5fa; }
  .pill.queued, .pill.unknown { background: rgba(148,148,160,0.15); color: #a1a1ab; }
  .pill.artifact-api { background: rgba(167,139,250,0.15); color: #c4b5fd; text-transform: none; }
  .pill.run-scan { background: rgba(148,148,160,0.15); color: #a1a1ab; text-transform: none; }

  /* toolbar: search + filters */
  .toolbar {
    display: flex; align-items: center; gap: 0.55rem; flex-wrap: wrap;
    background: var(--panel); border: 1px solid #23232b; border-radius: 12px;
    padding: 0.5rem 0.7rem;
  }
  .search {
    display: flex; align-items: center; gap: 7px; flex: 1; min-width: 220px; max-width: 460px;
    background: #0e0e13; border: 1px solid #26262e; border-radius: 9px; padding: 5px 10px;
    transition: border-color 0.15s ease;
  }
  .search:focus-within { border-color: #4d4d59; }
  .search svg { color: var(--dim); flex: none; }
  .search input {
    flex: 1; min-width: 0; background: transparent; border: 0; outline: none;
    color: var(--text); font-family: inherit; font-size: 0.8rem;
  }
  .search input::placeholder { color: var(--dim); }
  .search kbd {
    flex: none; font-family: inherit; font-size: 0.62rem; color: var(--dim);
    border: 1px solid #2c2c35; border-radius: 4px; padding: 0 4px;
  }
  .search .clear { flex: none; border: 0; background: transparent; color: var(--muted); cursor: pointer; font-size: 0.85rem; padding: 0 2px; }
  .search .clear:hover { color: var(--text); }
  .hits { font-size: 0.7rem; color: var(--dim); flex: none; white-space: nowrap; }
  .team-chips { display: flex; gap: 4px; flex-wrap: wrap; }
  .team-chip {
    display: flex; align-items: center; gap: 6px; font-size: 0.68rem; color: var(--muted);
    border: 1px solid #2a2a33; border-radius: 999px; padding: 4px 10px;
    background: transparent; cursor: pointer; font-family: inherit; white-space: nowrap;
    transition: color 0.15s ease, border-color 0.15s ease, background 0.15s ease;
  }
  .team-chip .swatch { width: 7px; height: 7px; border-radius: 999px; flex: none; opacity: 0.35; }
  .team-chip.on { color: var(--text); border-color: #43434f; background: #1d1d24; }
  .team-chip.on .swatch { opacity: 1; }
  .team-chip:hover { border-color: #4d4d59; }
  .toggle {
    font-size: 0.68rem; color: var(--muted); border: 1px solid #2a2a33; border-radius: 999px;
    padding: 4px 11px; background: transparent; cursor: pointer; font-family: inherit; white-space: nowrap;
  }
  .toggle.on { color: var(--text); border-color: #43434f; background: #1d1d24; }
  .toggle:hover { border-color: #4d4d59; }

  /* dim / highlight states shared by both views */
  .dimmed { opacity: 0.22; }
  .ringed { box-shadow: 0 0 0 1.5px var(--accent); border-color: var(--accent) !important; }
  .iconbtn {
    flex: none; border: 0; background: transparent; color: var(--muted); cursor: pointer;
    padding: 1px 3px; border-radius: 5px; font-size: 11px; line-height: 1; font-family: inherit;
  }
  .iconbtn:hover { background: #2c2c35; color: var(--text); }

  /* bottom panels */
  .bottom { display: flex; gap: 0.9rem; align-items: stretch; }
  .panel {
    flex: 1; min-width: 0; background: var(--panel); border: 1px solid #23232b;
    border-radius: 12px; display: flex; flex-direction: column; overflow: hidden;
  }
  .panel h2 {
    margin: 0; padding: 0.6rem 0.9rem; font-size: 0.7rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.07em; color: var(--muted);
    border-bottom: 1px solid #23232b; display: flex; align-items: center; gap: 8px;
  }
  .panel h2 .sel { color: var(--text); text-transform: none; letter-spacing: 0; font-weight: 600; }
  .panel h2 .spacer { margin-left: auto; }
  .panel h2 .team-chip { padding: 2px 8px; font-size: 0.62rem; text-transform: none; letter-spacing: 0; }
  tr.ringed td { background: #22222b; box-shadow: inset 2px 0 0 var(--eval, #f59e0b); }
  .side-item.dimmed { opacity: 0.3; }
  .panel-body { overflow: auto; max-height: 240px; }
  table { border-collapse: collapse; width: 100%; }
  th {
    position: sticky; top: 0; background: var(--panel); text-align: left;
    font-size: 0.63rem; text-transform: uppercase; letter-spacing: 0.06em;
    color: var(--dim); font-weight: 600; padding: 6px 12px; border-bottom: 1px solid #23232b;
  }
  td { padding: 6px 12px; font-size: 11.5px; border-bottom: 1px solid #1c1c23; color: var(--muted); }
  tbody tr:hover td { background: #1a1a21; }
  td.primary { color: var(--text); }
  td .link { cursor: pointer; color: #93c5fd; }
  .empty-row { padding: 1.1rem 0.9rem; color: var(--dim); font-size: 11.5px; font-style: italic; }

  .errors { display: flex; flex-direction: column; }
  .err {
    color: #f87171; font-size: 11.5px; background: rgba(220,38,38,0.08);
    border: 1px solid rgba(220,38,38,0.25); border-radius: 8px;
    padding: 6px 10px; margin: 0 0 6px;
  }
  footer { padding: 0.6rem 2rem 2rem; font-size: 0.75rem; color: var(--dim); }

  /* server-rendered fallback (replaced on mount) */
  .fallback { flex: 1; overflow: auto; padding: 1.2rem 1.4rem; color: var(--muted); font-size: 0.8rem; }
  .fb-grid { display: flex; flex-wrap: wrap; gap: 0.7rem; margin-top: 0.9rem; }
  .fb-station { background: var(--card); border: 1px solid var(--card-border); border-radius: 10px; padding: 10px 13px; min-width: 210px; }
  .fb-station b { color: var(--text); font-size: 12.5px; }
  .fb-artifact { font-size: 10.5px; color: var(--dim); margin: 2px 0 7px; }
  .fb-station ul { list-style: none; padding: 0; margin: 0; }
  .fb-station li { font-size: 11px; padding: 2px 0; display: flex; gap: 6px; align-items: center; }
  .fb-kind { font-size: 9.5px; color: var(--dim); }
  .fb-empty { color: var(--dim); font-style: italic; }
  .toast {
    position: fixed; bottom: 1.5rem; left: 50%; transform: translateX(-50%);
    background: #26262e; border: 1px solid #3a3a45; color: var(--text);
    font-size: 0.78rem; padding: 7px 14px; border-radius: 999px; z-index: 50;
  }
</style>
</head>
<body>
<header>
  <div class="header-row">
    <span class="logo">🏭</span>
    <h1>Model Factory — global lineage</h1>
    <span class="spacer"></span>
    <a class="chip" href="/api/lineage">JSON API ↗</a>
  </div>
  <p class="sub" id="subline">Artifact flow across factory stations — each edge is an inter-team contract from <code>contracts.py</code>.</p>
</header>
<main id="root">__FALLBACK__</main>
<footer>Read-only view of the control plane · versions resolve via the artifact registry, falling back to a run scan.</footer>

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
// If anything below throws, the server-rendered fallback in #root stays visible.
try {
  const React = (await import("react")).default;
  const { createRoot } = await import("react-dom/client");
  const { ReactFlow, Background, BackgroundVariant, Controls, Handle, Position, MarkerType } =
    await import("@xyflow/react");
  const dagre = (await import("@dagrejs/dagre")).default;
  const htm = (await import("htm")).default;
  const html = htm.bind(React.createElement);

  const BOOT = JSON.parse(document.getElementById("lineage-data").textContent);
  const TEAM_COLOR = {
    "data-eng": "#38bdf8", training: "#a78bfa", eval: "#f59e0b", inference: "#34d399",
  };
  const ACTIVE_PHASES = ["running", "initializing", "queued"];
  const FAILED_PHASES = ["failed", "aborted", "timedout", "timed_out"];

  // ?refresh=<seconds> overrides the server default; 0 disables polling.
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

  // Everything a station or version can be matched against by the search box.
  const stationHay = (s) =>
    [s.label, s.artifact, s.team_label,
     ...s.versions.map((v) => v.version + " " + v.source)].join(" ").toLowerCase();
  const versionHay = (v) => [v.version, v.source, v.url, v.kind].join(" ").toLowerCase();
  const runHay = (r) =>
    [r.name, r.task, r.task_short, r.phase, ...(r.stations || [])].join(" ").toLowerCase();

  // Ancestors + descendants of a station in the contract DAG (lineage trace).
  function connectedSet(edges, root) {
    const up = {}, down = {};
    edges.forEach((e) => {
      (down[e.source] = down[e.source] || []).push(e.target);
      (up[e.target] = up[e.target] || []).push(e.source);
    });
    const out = new Set([root]);
    const walk = (m, id) =>
      (m[id] || []).forEach((n) => { if (!out.has(n)) { out.add(n); walk(m, n); } });
    walk(up, root);
    walk(down, root);
    return out;
  }

  // ---------------------------------------------------------------- icons
  const StationIcon = () => html`
    <svg class="icon" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor"
         stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <path d="M2 20h20V9l-6 4V9l-6 4V5H4l-2 15Z"></path>
    </svg>`;
  const VersionIcon = () => html`
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor"
         stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <ellipse cx="12" cy="5" rx="9" ry="3"></ellipse>
      <path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"></path>
      <path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"></path>
    </svg>`;
  const SearchIcon = () => html`
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor"
         stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <circle cx="11" cy="11" r="7"></circle><path d="m20 20-3.5-3.5"></path>
    </svg>`;

  // ---------------------------------------------------------------- nodes
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
      <${Handle} type="source" position=${Position.Right} style=${{ opacity: 0 }} />
    </div>`;

  // Clicking the card opens the run that produced this version — the copy
  // button stops propagation so it doesn't also navigate.
  const VersionNode = ({ data }) => html`
    <div class="v-card ${data.dimmed ? "dimmed" : ""} ${data.ringed ? "ringed" : ""}"
         style=${{ "--accent": data.color }}
         title=${(data.run_url ? "Open run " + data.source + "\n" : "") + data.url}
         onClick=${() => (data.run_url ? openUrl(data.run_url) : data.onCopy(data.url))}>
      <${Handle} type="target" position=${Position.Left} style=${{ opacity: 0 }} />
      <div class="vid">${shortId(data.version)}</div>
      <div class="vrow">
        <span style=${{ color: data.color, display: "flex" }}><${VersionIcon} /></span>
        <span class="vrun">${data.source || "unknown run"}</span>
        <button class="iconbtn" title="Copy object-store path"
                onClick=${(e) => { e.stopPropagation(); data.onCopy(data.url); }}>⧉</button>
        <span class="pill ${data.kind}">${data.kind}</span>
      </div>
      <${Handle} type="source" position=${Position.Right} style=${{ opacity: 0 }} />
    </div>`;

  const GroupLabel = ({ data }) => html`
    <div class="group-label">
      <span class="swatch" style=${{ background: data.color }}></span>${data.label}
    </div>`;

  const nodeTypes = { station: StationNode, version: VersionNode, groupLabel: GroupLabel };

  // --------------------------------------------------------------- layout
  const ST_W = 248, ST_H = 116;
  const V_W = 216, V_H = 54, V_GAP = 10;
  const PAD_X = 22, PAD_TOP = 40, PAD_BOTTOM = 18;

  const runsByStation = (data) => {
    const m = {};
    (data.runs || []).forEach((r) => {
      (r.stations || []).forEach((a) => (m[a] = m[a] || []).push(r));
    });
    return m;
  };

  // Stations view: the contract DAG, one card per station.
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
        id: s.artifact,
        type: "station",
        position: { x: p.x - ST_W / 2, y: p.y - ST_H / 2 },
        data: {
          ...s,
          color: TEAM_COLOR[s.team] || "#8a8a94",
          teamLabel: s.team_label,
          versionCount: s.versions.length,
          latest: s.versions[0] || null,
          activeRuns: active[s.artifact] || 0,
          selected: ctx.selected === s.artifact,
          dimmed: ctx.dimStation(s),
          ringed: ctx.ringStation(s),
          onSelect: ctx.onSelect,
        },
      };
    });
    const edges = data.edges.map((e, i) => ({
      id: "se" + i,
      source: e.source,
      target: e.target,
      type: "smoothstep",
      animated: (active[e.target] || 0) > 0,
      label: e.source,
      style: ctx.dimEdge(e.source, e.target) ? { opacity: 0.18 } : undefined,
      markerEnd: { type: MarkerType.ArrowClosed, color: "#4a4a55", width: 16, height: 16 },
    }));
    return { nodes, edges };
  }

  // Versions view: a group box per station holding its versions. Solid edges
  // are observed provenance (same producing run); dashed edges are the
  // declared contract, drawn only when nothing was observed for that hop.
  function layoutVersions(data, ctx) {
    const shown = data.stations.filter((s) => s.versions.length);
    if (!shown.length) return { nodes: [], edges: [] };
    const sizeOf = (s) => ({
      width: V_W + 2 * PAD_X,
      height: PAD_TOP + s.versions.length * V_H + (s.versions.length - 1) * V_GAP + PAD_BOTTOM,
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
        id: gid,
        type: "group",
        position: { x: p.x - size.width / 2, y: p.y - size.height / 2 },
        data: {},
        draggable: false,
        selectable: false,
        style: {
          width: size.width, height: size.height,
          background: ctx.selected === s.artifact ? "rgba(255,255,255,0.045)" : "rgba(255,255,255,0.015)",
          border: "1px dashed " + (ctx.selected === s.artifact ? color : "#2c2c35"),
          borderRadius: 14,
        },
      });
      nodes.push({
        id: gid + ":label",
        type: "groupLabel",
        parentId: gid,
        position: { x: 14, y: 13 },
        data: { label: s.label, color },
        draggable: false,
        selectable: false,
      });
      idOf[s.artifact] = [];
      s.versions.forEach((v, i) => {
        const vid = s.artifact + "::" + i;
        idOf[s.artifact].push(vid);
        if (v.source) byRun[s.artifact + "|" + v.source] = vid;
        dimmed[vid] = ctx.dimVersion(s, v);
        nodes.push({
          id: vid,
          type: "version",
          parentId: gid,
          extent: "parent",
          position: { x: PAD_X, y: PAD_TOP + i * (V_H + V_GAP) },
          data: {
            ...v, color, station: s.artifact,
            onCopy: ctx.onCopy, onSelect: ctx.onSelect,
            dimmed: dimmed[vid], ringed: ctx.ringVersion(s, v),
          },
        });
      });
    });

    const edges = [];
    const fade = (a, b) => (dimmed[a] || dimmed[b] ? 0.15 : 1);
    data.edges.filter((e) => visible.has(e.source) && visible.has(e.target)).forEach((e, i) => {
      const target = data.stations.find((s) => s.artifact === e.target);
      let observed = 0;
      target.versions.forEach((v, vi) => {
        const up = v.source && byRun[e.source + "|" + v.source];
        if (!up) return;
        observed += 1;
        const down = e.target + "::" + vi;
        edges.push({
          id: "ve" + i + "-" + vi,
          source: up,
          target: down,
          type: "smoothstep",
          label: v.source,
          style: { opacity: fade(up, down) },
          markerEnd: { type: MarkerType.ArrowClosed, color: "#4a4a55", width: 14, height: 14 },
        });
      });
      if (!observed) {
        const up = idOf[e.source][0], down = idOf[e.target][0];
        edges.push({
          id: "ce" + i,
          source: up,
          target: down,
          type: "smoothstep",
          label: "contract",
          style: { strokeDasharray: "5 4", stroke: "#3a3a45", opacity: fade(up, down) },
          markerEnd: { type: MarkerType.ArrowClosed, color: "#3a3a45", width: 14, height: 14 },
        });
      }
    });
    return { nodes, edges };
  }

  // -------------------------------------------------------------- toolbar
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
                  title=${(teams.has(team) ? "Hide " : "Show ") + label}
                  onClick=${() => toggleTeam(team)}>
            <span class="swatch" style=${{ background: TEAM_COLOR[team] }}></span>${label}
          </button>`)}
      </div>
      <button class="toggle ${trace ? "on" : ""}" title="Dim stations not up/downstream of the selected one"
              onClick=${() => setTrace(!trace)}>Trace lineage</button>
      ${highlightRun
        ? html`<button class="toggle on" title="Clear run highlight"
                       onClick=${() => setHighlightRun(null)}>run ${highlightRun} ✕</button>`
        : null}
      <button class="toggle" onClick=${onReset}>Reset</button>
    </div>`;

  // -------------------------------------------------------------- sidebar
  const Sidebar = ({ data, view, setView, selected, onSelect, byStation, statusLine, matchStation }) => {
    const teams = Object.entries(data.teams || {});
    return html`
      <aside class="sidebar">
        <div class="side-tabs">
          <button class="side-tab ${view === "stations" ? "active" : ""}"
                  onClick=${() => setView("stations")}>Stations</button>
          <button class="side-tab ${view === "versions" ? "active" : ""}"
                  onClick=${() => setView("versions")}>Versions</button>
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

  // --------------------------------------------------------------- panels
  const VersionsPanel = ({ station, rows, onCopy, highlightRun, setHighlightRun }) => html`
    <section class="panel">
      <h2>Versions <span class="sel">${station ? station.label : ""}</span>
        ${station ? html`<span class="side-count">${rows.length}</span>` : null}</h2>
      <div class="panel-body">
        ${rows.length
          ? html`<table>
              <thead><tr><th>version</th><th>produced by</th><th>via</th><th>path</th><th></th></tr></thead>
              <tbody>
                ${rows.map((v, i) => html`
                  <tr key=${i} class=${highlightRun && v.source === highlightRun ? "ringed" : ""}>
                    <td class="primary mono">${shortId(v.version)}</td>
                    <td class="mono link" title="Highlight everything this run produced"
                        onClick=${() => setHighlightRun(v.source === highlightRun ? null : v.source)}>
                      ${v.source || "—"}</td>
                    <td><span class="pill ${v.kind}">${v.kind}</span></td>
                    <td class="mono link" title=${"Copy " + v.url} onClick=${() => onCopy(v.url)}>
                      ${v.url ? v.url.slice(0, 40) + (v.url.length > 40 ? "…" : "") : "—"}
                    </td>
                    <td>${v.run_url
                      ? html`<button class="iconbtn" title=${"Open run " + v.source}
                                     onClick=${() => openUrl(v.run_url)}>↗</button>`
                      : null}</td>
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
                    <td class="primary mono link" title="Highlight the artifacts this run produced"
                        onClick=${() => setHighlightRun(highlightRun === r.name ? null : r.name)}>
                      ${r.name}</td>
                    <td title=${r.task}>${r.task_short || r.task || "—"}</td>
                    <td>${(r.stations || []).length
                      ? (r.stations || []).map((a, i) => html`
                          <span key=${a}>${i ? ", " : ""}<span class="link"
                            onClick=${() => onSelect(a)}>${stationLabels[a] || a}</span></span>`)
                      : "—"}</td>
                    <td><span class="pill ${phaseOf(r.phase)}">${r.phase || "unknown"}</span></td>
                    <td>${relTime(r.created_at) || "—"}</td>
                    <td>${r.url
                      ? html`<button class="iconbtn" title="Open run in console"
                                     onClick=${() => openUrl(r.url)}>↗</button>`
                      : null}</td>
                  </tr>`)}
              </tbody>
            </table>`
          : html`<div class="empty-row">No runs match the current filters.</div>`}
      </div>
    </section>`;

  // ------------------------------------------------------------------ app
  const firstWithVersions = (d) => {
    const s = d.stations.find((x) => x.versions.length) || d.stations[0];
    return s ? s.artifact : null;
  };
  const initialParams = new URLSearchParams(location.search);
  const allTeams = (d) => new Set(Object.keys(d.teams || {}));

  const App = () => {
    const [data, setData] = React.useState(BOOT.data);
    const [view, setView] = React.useState(
      initialParams.get("view") === "versions" ? "versions" : "stations");
    const [selected, setSelected] = React.useState(
      () => initialParams.get("sel") || firstWithVersions(BOOT.data));
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
        () => setToast("Copied path to clipboard"),
        () => setToast(text)
      );
    }, []);
    React.useEffect(() => {
      if (!toast) return;
      const id = setTimeout(() => setToast(""), 1800);
      return () => clearTimeout(id);
    }, [toast]);

    // "/" focuses search, Escape clears search then selection.
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
        } catch (e) {
          console.warn("lineage refresh failed", e);
        } finally {
          inFlight = false;
        }
      }, refreshMs);
      return () => clearInterval(id);
    }, []);

    // Keep the URL shareable: view, selection, search, filters, highlight.
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

    // One dim/ring model drives the graph, the sidebar and the panels.
    const q = query.trim().toLowerCase();
    const traceSet = React.useMemo(
      () => (trace && selected ? connectedSet(data.edges, selected) : null),
      [trace, selected, data]);
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
      teams.has(s.team) && (!q || stationHay(s).includes(q) || versionHay(v).includes(q)),
      [teams, q]);
    const dimVersion = React.useCallback((s, v) => {
      if (!matchVersion(s, v)) return true;
      if (traceSet && !traceSet.has(s.artifact)) return true;
      if (highlightRun && v.source !== highlightRun) return true;
      return false;
    }, [matchVersion, traceSet, highlightRun]);
    const ringVersion = React.useCallback((s, v) =>
      !!(highlightRun && v.source === highlightRun), [highlightRun]);

    const ctx = {
      selected, onSelect, onCopy,
      dimStation, ringStation, dimVersion, ringVersion,
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
      () => Object.fromEntries(data.stations.map((s) => [s.artifact, s.label])),
      [data]);
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

    // Remount the canvas when the view flips so fitView re-runs; data
    // refreshes update node positions in place.
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
              <${Background} variant=${BackgroundVariant.Dots} gap=${22} size=${1.4} color="#26262e" />
              <${Controls} showInteractive=${false} />
            <//>
          </div>
        </div>
        <div class="bottom">
          <${VersionsPanel} station=${station} rows=${versionRows} onCopy=${onCopy}
              highlightRun=${highlightRun} setHighlightRun=${setHighlightRun} />
          <${RunsPanel} rows=${runRows} stationLabels=${stationLabels} phaseFilter=${phaseFilter}
              setPhaseFilter=${setPhaseFilter} highlightRun=${highlightRun}
              setHighlightRun=${setHighlightRun} onSelect=${onSelect} />
        </div>
        ${(data.errors || []).length
          ? html`<div class="errors">
              ${data.errors.map((e, i) => html`<p key=${i} class="err">${e}</p>`)}
            </div>`
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


lineage_app_env = FastAPIAppEnvironment(
    name="mf-lineage",
    app=app,
    image=cpu_image.with_pip_packages("fastapi", "uvicorn"),
    resources=flyte.Resources(cpu=1, memory="1Gi"),
    scaling=flyte.app.Scaling(replicas=(0, 1), scaledown_after=600),
    requires_auth=REQUIRE_APP_AUTH,
    env_vars={
        **cluster_env_vars(),
        "MF_ORG": APP_ORG,
        "MF_PROJECT": APP_PROJECT,
        "MF_DOMAIN": APP_DOMAIN,
    },
    description="Model factory lineage: artifact versions + runs across all stations",
)


@lineage_app_env.on_startup
async def _init_remote() -> None:
    """Auth against the control plane from inside the cluster.

    Must never raise: a failed lifespan hook would 500 every request. Any
    auth problem surfaces per-request in the JSON "errors" list instead.
    """
    try:
        await flyte.init_in_cluster.aio(
            org=os.environ.get("MF_ORG") or None,
            project=os.environ.get("MF_PROJECT") or None,
            domain=os.environ.get("MF_DOMAIN") or None,
        )
        return
    except Exception:
        pass
    try:
        await flyte.init_in_cluster.aio()
    except Exception:
        pass
