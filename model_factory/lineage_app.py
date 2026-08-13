"""Factory lineage app — the POC's "Podium": one page showing the whole
factory in action.

A FastAPI AppEnvironment (scale-to-zero) that reads the cluster's own APIs
(`flyte.remote`) and renders the artifact graph: every version of every
factory artifact, the run that produced it, recent runs of the factory
stations, and paused HITL gates awaiting a signal.

Deploy:  flyte deploy src/model_factory/lineage_app.py lineage_app_env
Local:   uv run flyte serve --local src/model_factory/lineage_app.py lineage_app_env
"""

from __future__ import annotations

import html as _html
import os
from datetime import datetime, timezone

import flyte
import flyte.app
from flyte.app.extras import FastAPIAppEnvironment
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

from .config import (
    ARTIFACT_CHECKPOINT,
    ARTIFACT_EVAL_REPORT,
    ARTIFACT_PROMOTED,
    ARTIFACT_RL_DATASET,
    ARTIFACT_SYNTHETIC,
)
from .envs import cpu_image

# Factory stations in pipeline order → rendered as graph columns.
STATIONS: list[tuple[str, str]] = [
    (ARTIFACT_RL_DATASET, "Data"),
    (ARTIFACT_SYNTHETIC, "Synthetic"),
    (ARTIFACT_CHECKPOINT, "Training"),
    (ARTIFACT_EVAL_REPORT, "Evaluation"),
    (ARTIFACT_PROMOTED, "Promoted"),
]

app = FastAPI(title="Model Factory Lineage")


def _esc(s: object) -> str:
    return _html.escape(str(s))


def _collect() -> dict:
    """Pull artifact versions + recent runs from the control plane."""
    data: dict = {"stations": [], "runs": [], "fetched_at": datetime.now(timezone.utc).isoformat()}
    for name, label in STATIONS:
        versions = []
        try:
            for a in flyte.remote.Artifact.listall(name=name, limit=10):
                versions.append(
                    {
                        "version": a.version,
                        "url": a.url,
                        "kind": a.kind,
                        "source": str(getattr(a, "source", "") or ""),
                    }
                )
        except Exception as e:  # artifact name may not exist yet
            versions = []
            data.setdefault("errors", []).append(f"{name}: {e}")
        data["stations"].append({"artifact": name, "label": label, "versions": versions})
    try:
        for r in flyte.remote.Run.listall(
            limit=25,
            project=os.environ.get("MF_PROJECT", "model-factory"),
            domain=os.environ.get("MF_DOMAIN", "development"),
        ):
            details = r
            data["runs"].append(
                {
                    "name": r.name,
                    "task": getattr(r, "task_name", "") or "",
                    "phase": str(getattr(details, "phase", "") or ""),
                    "url": getattr(r, "url", "") or "",
                }
            )
    except Exception as e:
        data.setdefault("errors", []).append(f"runs: {e}")
    return data


@app.get("/api/lineage")
def api_lineage() -> JSONResponse:
    return JSONResponse(_collect())


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    d = _collect()
    cols = ""
    for st in d["stations"]:
        rows = "".join(
            f"<li title='{_esc(v['source'])}'><code>{_esc(v['version'][:12])}</code>"
            f" <span class='k'>{_esc(v['kind'])}</span></li>"
            for v in st["versions"]
        ) or "<li class='empty'>no versions yet</li>"
        cols += (
            f"<div class='station'><h3>{_esc(st['label'])}</h3>"
            f"<div class='aname'>{_esc(st['artifact'])}</div><ul>{rows}</ul></div>"
            "<div class='arrow'>→</div>"
        )
    cols = cols.rsplit("<div class='arrow'>", 1)[0]  # drop trailing arrow

    runs = "".join(
        f"<tr><td><a href='{_esc(r['url'])}'>{_esc(r['name'])}</a></td>"
        f"<td>{_esc(r['task'])}</td><td class='{ _esc(r['phase']).lower() }'>{_esc(r['phase'])}</td></tr>"
        for r in d["runs"]
    )
    errors = "".join(f"<p class='err'>{_esc(e)}</p>" for e in d.get("errors", []))
    return f"""<!doctype html><html><head><title>Model Factory Lineage</title>
<style>
body {{ font-family: -apple-system, 'Segoe UI', sans-serif; margin: 2rem; color: #1a1a2e; background: #fafbfc; }}
.flow {{ display: flex; align-items: flex-start; gap: 8px; flex-wrap: wrap; }}
.station {{ background: white; border: 1px solid #d8dbe2; border-radius: 10px; padding: 12px 16px; min-width: 180px; }}
.station h3 {{ margin: 0 0 2px; }} .aname {{ font-size: 11px; color: #667; margin-bottom: 8px; }}
.station ul {{ list-style: none; padding: 0; margin: 0; }} .station li {{ padding: 2px 0; font-size: 13px; }}
.k {{ font-size: 10px; background: #eef0f5; border-radius: 8px; padding: 1px 6px; }}
.empty {{ color: #99a; font-style: italic; }}
.arrow {{ align-self: center; font-size: 22px; color: #99a; }}
table {{ border-collapse: collapse; margin-top: 1rem; }} td, th {{ border: 1px solid #d8dbe2; padding: 5px 10px; font-size: 13px; }}
.succeeded {{ color: #0a7a3d; }} .failed {{ color: #b00020; }} .running {{ color: #c26b0a; }}
.err {{ color: #b00020; font-size: 12px; }}
</style></head><body>
<h1>🏭 Model Factory — global lineage</h1>
<p>Artifact flow across factory stations (latest 10 versions each). Fetched {_esc(d['fetched_at'])}.</p>
<div class='flow'>{cols}</div>
<h2>Recent runs</h2>
<table><tr><th>run</th><th>task</th><th>phase</th></tr>{runs}</table>
{errors}
<p><a href='/api/lineage'>JSON API</a></p>
</body></html>"""


lineage_app_env = FastAPIAppEnvironment(
    name="mf-lineage",
    app=app,
    image=cpu_image.with_pip_packages("fastapi", "uvicorn"),
    resources=flyte.Resources(cpu=1, memory="1Gi"),
    scaling=flyte.app.Scaling(replicas=(0, 1), scaledown_after=600),
    requires_auth=False,
    env_vars={"MF_ORG": "demo", "MF_PROJECT": "model-factory", "MF_DOMAIN": "development"},
    description="Model factory lineage: artifact versions + runs across all stations",
)


@lineage_app_env.on_startup
async def _init_remote() -> None:
    """Auth against the control plane from inside the cluster."""
    try:
        await flyte.init_in_cluster.aio(
            org=os.environ.get("MF_ORG") or None,
            project=os.environ.get("MF_PROJECT") or None,
            domain=os.environ.get("MF_DOMAIN") or None,
        )
    except Exception:
        flyte.init_from_config()
