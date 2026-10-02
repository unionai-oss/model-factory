"""The decision API: serves the champion the factory bound to it.

The model arrives as an app *parameter* bound to the `decision-champion`
artifact, so the app never looks up a model itself:

    factory.serve("decision-api").using(decision_app_env, model=champion)

A materialization deploys this app with the exact champion version it
resolved, and Flyte downloads the Dir before the container starts.

Endpoints
    GET  /health   -> what is loaded, and which model won
    GET  /champion -> the champion's manifest (slug, accuracy, suite it beat)
    POST /decide   -> {observation} -> the tool call the model chooses
    POST /oracle   -> {observation} -> what the oracle policy says (ground
                      truth, for comparing the served model against the rule
                      it was trained on, live)

Deploy: flyte deploy app.py decision_app_env
"""

from __future__ import annotations

import asyncio
import json
import os
import traceback

import flyte
import flyte.app
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from flyte.app.extras import FastAPIAppEnvironment

from ..config import (
    APP_DOMAIN,
    APP_PROJECT,
    REQUIRE_APP_AUTH,
    env_vars,
    get_profile,
    serving_resources,
)
from ..contracts import ARTIFACT_CHAMPION, ENDPOINT_APP
from ..harness.policy import Observation, build_chat, decide
from ..harness.scoring import parse_call
from ..shared.images import serving_image

#: The app parameter the factory binds the champion artifact to.
MODEL_PARAM = "model"

app = FastAPI(title="Decision API")

_state: dict = {
    "model": None,
    "tok": None,
    "manifest": None,
    "dir": None,
    "loading": None,
    "load_error": None,
}
_load_lock = asyncio.Lock()
# Strong refs to in-flight background loads: asyncio only holds weak
# references to tasks, so an unreferenced task can be garbage-collected
# mid-load and the load would vanish silently.
_bg_tasks: set = set()


def bound_model_path() -> str | None:
    """Local path of the champion the factory bound to this app, if any.

    `get_parameter` returns a path relative to the app's working directory
    (Flyte has already downloaded the artifact), so this is a local directory,
    not an object-store URI. None when the app was deployed without a binding.
    """
    try:
        value = flyte.app.get_parameter(MODEL_PARAM)
    except Exception:
        return None
    if value and os.path.exists(value):
        return os.path.abspath(value)
    return None


def _load_sync(model_dir: str):
    """Blocking weight load. Always called via ``asyncio.to_thread``.

    transformers and peft are blocking, and a multi-minute load on the event
    loop makes the app stop answering /health — which is exactly what callers
    poll to track the load.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    with open(os.path.join(model_dir, "manifest.json")) as f:
        manifest = json.load(f)
    base = manifest["base_model"]

    tok = AutoTokenizer.from_pretrained(model_dir, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base,
        dtype=torch.bfloat16,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
    )
    model = PeftModel.from_pretrained(model, model_dir)
    model.eval()
    return model, tok, manifest


async def _load(model_dir: str) -> None:
    model, tok, manifest = await asyncio.to_thread(_load_sync, model_dir)
    _state.update({"model": model, "tok": tok, "manifest": manifest, "dir": model_dir})


async def _ensure_loaded() -> None:
    async with _load_lock:
        if _state["model"] is None:
            path = bound_model_path()
            if not path:
                raise RuntimeError(
                    f"no champion bound to this app: parameter {MODEL_PARAM!r} is unset. "
                    f"Materialize the {ENDPOINT_APP} endpoint so the factory binds a "
                    f"{ARTIFACT_CHAMPION} version to it."
                )
            await _load(path)


def _generate_sync(chats: list[list[dict]], max_new_tokens: int) -> list[str]:
    import torch

    model, tok = _state["model"], _state["tok"]
    rendered = [
        tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True) for c in chats
    ]
    inputs = tok(rendered, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(
        model.device
    )
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
    return tok.batch_decode(out[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)


@app.get("/health")
async def health() -> JSONResponse:
    manifest = _state["manifest"] or {}
    return JSONResponse(
        {
            "loaded": _state["model"] is not None,
            "loading": _state["loading"],
            "load_error": _state["load_error"],
            "bound_model": bound_model_path(),
            "champion": manifest.get("model_slug"),
            "base_model": manifest.get("base_model"),
            "exact_accuracy": manifest.get("exact_accuracy"),
        }
    )


@app.get("/champion")
async def champion() -> JSONResponse:
    """The champion's manifest: which model won, and what it beat."""
    if _state["manifest"] is None:
        path = bound_model_path()
        if path and os.path.exists(os.path.join(path, "manifest.json")):
            with open(os.path.join(path, "manifest.json")) as f:
                return JSONResponse(json.load(f))
        return JSONResponse({"error": "no champion bound to this app"}, status_code=503)
    return JSONResponse(_state["manifest"])


def _observation(body: dict) -> Observation:
    """An Observation from a request body, defaulting the optional fields."""
    obs = body.get("observation", body)
    return Observation(
        customer_message=str(obs.get("customer_message", "")),
        intent=str(obs.get("intent", "chitchat")),
        order_id=obs.get("order_id"),
        order_looked_up=bool(obs.get("order_looked_up", False)),
        order_status=obs.get("order_status"),
        order_total_cents=obs.get("order_total_cents"),
        days_since_delivery=obs.get("days_since_delivery"),
        tracking_number=obs.get("tracking_number"),
    )


@app.post("/decide")
async def decide_endpoint(body: dict) -> JSONResponse:
    """The served model's decision for one observation.

    Also returns the oracle's answer and whether they agree, so a caller can
    see the model being right or wrong without running a separate eval.
    """
    try:
        await _ensure_loaded()
        obs = _observation(body)
        max_new_tokens = int(body.get("max_new_tokens", get_profile("smoke").max_new_tokens))

        # Generation is blocking; off the loop it goes so /health stays
        # answerable while a request is in flight.
        raw = (await asyncio.to_thread(_generate_sync, [build_chat(obs)], max_new_tokens))[0]
        call = parse_call(raw)
        oracle = decide(obs)
        return JSONResponse(
            {
                "decision": {"tool": call.tool, "args": dict(call.args)} if call else None,
                "parsed": call is not None,
                "raw": raw[:500],
                "oracle": {"tool": oracle.tool, "args": dict(oracle.args)},
                "agrees_with_oracle": bool(call and call.tool == oracle.tool),
                "champion": (_state["manifest"] or {}).get("model_slug"),
            }
        )
    except Exception as e:
        return JSONResponse(
            {"error": str(e), "traceback": traceback.format_exc().splitlines()[-8:]},
            status_code=500,
        )


@app.post("/oracle")
async def oracle_endpoint(body: dict) -> JSONResponse:
    """What the policy says. No model involved — the ground truth, served."""
    try:
        obs = _observation(body)
        call = decide(obs)
        from ..harness.policy import why

        return JSONResponse(
            {"tool": call.tool, "args": dict(call.args), "rationale": why(obs)}
        )
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


decision_app_env = FastAPIAppEnvironment(
    name=ENDPOINT_APP,
    app=app,
    image=serving_image,
    resources=serving_resources(),
    scaling=flyte.app.Scaling(replicas=(0, 1), scaledown_after=900),
    requires_auth=REQUIRE_APP_AUTH,
    env_vars={**env_vars(), "DF_PROJECT": APP_PROJECT, "DF_DOMAIN": APP_DOMAIN},
    parameters=[
        # Rebound to the version each materialization resolves. `version=None`
        # means a plain `flyte deploy` of this env picks the newest champion —
        # but on a cold registry (no champion yet) that deploy FAILS to
        # materialize the artifact. That is expected: the first deploy exists
        # to build the image, and `factory deploy` does not build app images.
        #
        # Deliberately no `mount=`: a mount path outside the app's writable
        # working directory crash-loops the container, and `get_parameter`
        # already hands back the downloaded path.
        flyte.app.Parameter(
            name=MODEL_PARAM,
            value=flyte.app.ArtifactValue(name=ARTIFACT_CHAMPION),
        )
    ],
    description="Serves the champion decision model chosen from the suite",
)


@decision_app_env.on_startup
async def _init() -> None:
    """Init control-plane access and warm the champion; never raise.

    A raising on_startup 500s every request, so everything here is guarded and
    failures surface through /health's `load_error` instead.
    """
    try:
        await flyte.init_in_cluster.aio(
            project=os.environ.get("DF_PROJECT") or None,
            domain=os.environ.get("DF_DOMAIN") or None,
        )
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
            async with _load_lock:
                if _state["model"] is None:
                    await _load(bound)
        except Exception as e:
            _state["load_error"] = f"preload {type(e).__name__}: {e}"
        finally:
            _state["loading"] = None

    task = asyncio.create_task(_preload())
    _bg_tasks.add(task)
    task.add_done_callback(_bg_tasks.discard)
