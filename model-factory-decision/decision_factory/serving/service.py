"""The permission gate: serves the champion the factory bound to it.

The model arrives as an app *parameter* bound to `permission-champion`, so the
app never looks a model up itself:

    factory.serve("permission-gate").using(permission_app_env, model=champion)

Endpoints
    GET  /health   -> what is loaded, and which model won
    GET  /champion -> the champion's manifest (slug, metrics, leaderboard, gate threshold)
    POST /decide   -> {observation} -> decision, P(option), the gated decision,
                      the oracle's verdict, agreement, latency and tokens
    POST /oracle   -> {observation} -> the oracle's verdict and which rule fired

`/decide` answers twice. ``decision`` is the model's argmax; ``gated`` applies
the champion's tuned threshold ("auto-allow only if P(allow) >= t, else ask"),
which is how you would actually wire a decision model into an agent.

Deploy: flyte deploy app.py permission_app_env
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import traceback

import flyte
import flyte.app
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from flyte.app.extras import FastAPIAppEnvironment

from ..config import APP_DOMAIN, APP_PROJECT, REQUIRE_APP_AUTH, env_vars, serving_resources
from ..contracts import ARTIFACT_CHAMPION, ENDPOINT_APP
from ..harness.policy import DECISIONS, Observation, decide
from ..harness.prompt import build_chat
from ..harness.guard import evidence_disagrees
from ..harness.scoring import apply_temperature, gate, temper
from ..heads import modeling
from ..shared.images import serving_image

#: The app parameter the factory binds the champion artifact to.
MODEL_PARAM = "model"

app = FastAPI(title="Permission gate")

_state: dict = {"dm": None, "manifest": None, "loading": None, "load_error": None}
_load_lock = asyncio.Lock()
# asyncio holds only weak refs to tasks; keep the preload alive.
_bg_tasks: set = set()


def bound_model_path() -> str | None:
    """Local path of the champion the factory bound to this app, if any.
    `get_parameter` returns the already-downloaded directory."""
    try:
        value = flyte.app.get_parameter(MODEL_PARAM)
    except Exception:
        return None
    if value and os.path.exists(value):
        return os.path.abspath(value)
    return None


def _load_sync(model_dir: str):
    import torch

    # A container sees every core on the node but is only granted a few; torch
    # defaulting to the node's core count oversubscribes them badly.
    torch.set_num_threads(max(1, int(os.environ.get("DF_SERVING_CPU", "2"))))
    return modeling.DecisionModel(model_dir)


async def _ensure_loaded() -> None:
    async with _load_lock:
        if _state["dm"] is None:
            path = bound_model_path()
            if not path:
                raise RuntimeError(
                    f"no champion bound to this app: parameter {MODEL_PARAM!r} is unset. Materialize the "
                    f"{ENDPOINT_APP} endpoint so the factory binds a {ARTIFACT_CHAMPION} version to it.")
            dm = await asyncio.to_thread(_load_sync, path)
            _state.update({"dm": dm, "manifest": dm.manifest})


def observation_from_body(body: dict) -> Observation:
    """An Observation from a request body, defaulting the optional fields."""
    return Observation.from_dict(body.get("observation", body))


def decide_with(dm: "modeling.DecisionModel", obs: Observation, threshold: float | None,
                temperature: float = 1.0, guard: bool = True) -> dict:
    """The model's answer for one observation, plus the gated answer.

    ``gated`` is how you would wire the model into an agent: temper the
    probabilities, auto-allow only above the threshold (both tuned on held-out
    validation families), and never auto-allow when the call's evidence
    fields disagree (`harness/guard.py`). Pure, so a local test pins it."""
    t0 = time.perf_counter()
    out = dm.decide([build_chat(obs)], prefix_cache=True)[0]
    latency_ms = (time.perf_counter() - t0) * 1000
    t = 0.5 if threshold is None else float(threshold)
    tempered = (temper([out["logits"]], float(temperature)) if "logits" in out
                else apply_temperature([out["probs"]], float(temperature)))[0]
    why = evidence_disagrees(obs) if guard else None
    gated = gate([tempered], t)[0]
    if why and gated == "allow":
        gated = "ask_user"
    return {
        "decision": out["decision"],
        "probabilities": dict(zip(DECISIONS, [round(p, 4) for p in out["probs"]])),
        "gated": {"decision": gated, "threshold": t, "temperature": float(temperature),
                  "tempered_probabilities": dict(zip(DECISIONS, [round(p, 4) for p in tempered])),
                  "evidence_disagreement": why},
        "parsed": out.get("parsed", True),
        "raw": out.get("raw"),
        "latency_ms": round(latency_ms, 1),
        "input_tokens": out["input_tokens"],
        "output_tokens": out["output_tokens"],
        # tokens actually run through the torso (the shared prefix is cached)
        "computed_tokens": out.get("computed_tokens", out["input_tokens"]),
    }


@app.get("/health")
async def health() -> JSONResponse:
    m = _state["manifest"] or {}
    return JSONResponse({
        "loaded": _state["dm"] is not None, "loading": _state["loading"], "load_error": _state["load_error"],
        "bound_model": bound_model_path(), "champion": m.get("model_slug"), "head": m.get("head"),
        "base_model": m.get("base_model"), "gate_threshold": m.get("gate_threshold"),
        "serving_gate": m.get("serving_gate"),
        "test_accuracy": (m.get("test_metrics") or {}).get("accuracy"),
    })


@app.get("/champion")
async def champion() -> JSONResponse:
    if _state["manifest"] is None:
        path = bound_model_path()
        if path and os.path.exists(os.path.join(path, modeling.MANIFEST)):
            with open(os.path.join(path, modeling.MANIFEST)) as f:
                return JSONResponse(json.load(f))
        return JSONResponse({"error": "no champion bound to this app"}, status_code=503)
    return JSONResponse(_state["manifest"])


@app.post("/decide")
async def decide_endpoint(body: dict) -> JSONResponse:
    """The served model's decision, next to the oracle's, so a caller sees the
    model being right or wrong without running an eval."""
    try:
        await _ensure_loaded()
        obs = observation_from_body(body)
        manifest = _state["manifest"] or {}
        sg = manifest.get("serving_gate") or {}
        threshold = body.get("threshold", sg.get("threshold", manifest.get("gate_threshold")))
        temperature = body.get("temperature", sg.get("temperature", 1.0))
        result = await asyncio.to_thread(decide_with, _state["dm"], obs, threshold, temperature,
                                         bool(body.get("guard", sg.get("guard", True))))
        verdict = decide(obs)
        result.update({
            "oracle": {"decision": verdict.decision, "rule": verdict.rule, "reason": verdict.reason},
            "agrees_with_oracle": result["decision"] == verdict.decision,
            "champion": manifest.get("model_slug"),
        })
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e), "traceback": traceback.format_exc().splitlines()[-8:]}, status_code=500)


@app.post("/oracle")
async def oracle_endpoint(body: dict) -> JSONResponse:
    """What the policy says. No model involved — the ground truth, served."""
    try:
        v = decide(observation_from_body(body))
        return JSONResponse({"decision": v.decision, "rule": v.rule, "reason": v.reason})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


permission_app_env = FastAPIAppEnvironment(
    name=ENDPOINT_APP,
    app=app,
    image=serving_image,
    resources=serving_resources(),
    scaling=flyte.app.Scaling(replicas=(0, 1), scaledown_after=900),
    requires_auth=REQUIRE_APP_AUTH,
    env_vars={**env_vars(), "DF_PROJECT": APP_PROJECT, "DF_DOMAIN": APP_DOMAIN},
    parameters=[
        # Rebound to the version each materialization resolves. On a cold
        # registry a plain `flyte deploy` of this env FAILS to materialize the
        # artifact — expected: that deploy exists to build the image.
        # No `mount=`: get_parameter already returns the downloaded path.
        flyte.app.Parameter(name=MODEL_PARAM, value=flyte.app.ArtifactValue(name=ARTIFACT_CHAMPION)),
    ],
    description="Coding-agent permission gate: serves the champion decision model",
)


@permission_app_env.on_startup
async def _init() -> None:
    """Init control-plane access and warm the champion; never raise (a raising
    on_startup 500s every request — failures surface via /health)."""
    try:
        await flyte.init_in_cluster.aio(project=os.environ.get("DF_PROJECT") or None,
                                        domain=os.environ.get("DF_DOMAIN") or None)
    except Exception:
        try:
            await flyte.init_in_cluster.aio()
        except Exception:
            pass
    bound = bound_model_path()
    if not bound:
        return

    async def _preload() -> None:
        try:
            _state["loading"] = bound
            await _ensure_loaded()
        except Exception as e:
            _state["load_error"] = f"preload {type(e).__name__}: {e}"
        finally:
            _state["loading"] = None

    task = asyncio.create_task(_preload())
    _bg_tasks.add(task)
    task.add_done_callback(_bg_tasks.discard)
