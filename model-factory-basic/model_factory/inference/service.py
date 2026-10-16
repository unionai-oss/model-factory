"""The inference serving app: serves the model the factory bound to it.

The model comes in as an app *parameter* bound to the `promoted-model`
artifact: `factory.serve("mf-inference").using(inference_app_env,
model=promoted)` deploys this app with the exact artifact version the
materialization resolved, and Flyte downloads it before the app starts. Rolling
back is a materialization too: `--version mf-promoted-model=<version>`.

Endpoints:
- GET  /health    -> {loaded, loading, error, base_model, adapter, generation}
- POST /generate  -> {"chats": [[{role,content},...]], "use_adapter": bool,
                      "max_new_tokens": int}
                     ``use_adapter: false`` answers with the base model, from
                     the same weights, for side-by-side comparisons.

Deploy: flyte deploy team_inference.py inference_app_env
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

from ..config import REQUIRE_APP_AUTH, cluster_env_vars, inference_resources
from ..contracts import ARTIFACT_PROMOTED, ENDPOINT_APP
from ..shared.images import gpu_image

_GEN_BATCH = 8

#: Name of the app parameter the factory binds the model artifact to.
MODEL_PARAM = "model"

app = FastAPI(title="Model Factory Inference Service")

_state: dict = {"model": None, "tok": None, "manifest": None, "loading": False, "error": None}
# Strong refs to in-flight background tasks: asyncio only holds weak
# references, so an unreferenced task can be garbage-collected mid-load.
_bg_tasks: set = set()


def bound_model_path() -> str | None:
    """Local path of the artifact the factory bound to this app, if any.

    `flyte.app.get_parameter` returns a path relative to the app's working
    directory (Flyte has already downloaded the artifact). Deliberately no
    `mount=` on the parameter: a mount outside the app's writable working
    directory crash-loops the container.
    """
    try:
        value = flyte.app.get_parameter(MODEL_PARAM)
    except Exception:
        return None
    if value and os.path.exists(value):
        return os.path.abspath(value)
    return None


def _load_sync(local: str):
    """Base weights (+ the LoRA adapter, when the release has one) onto the GPU."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    with open(os.path.join(local, "manifest.json")) as f:
        manifest = json.load(f)
    base_model = manifest["base_model"]
    tok = AutoTokenizer.from_pretrained(base_model, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
    )
    if manifest.get("adapter", True):
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, local)
    model.eval()
    return model, tok, manifest


def _generate_sync(chats: list, use_adapter: bool, max_new_tokens: int) -> list[str]:
    """Blocking batched greedy generation — always called via ``asyncio.to_thread``."""
    import contextlib

    import torch

    model, tok = _state["model"], _state["tok"]
    has_adapter = bool(_state["manifest"].get("adapter", True))
    outs: list[str] = []
    for i in range(0, len(chats), _GEN_BATCH):
        rendered = [
            tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True) for c in chats[i : i + _GEN_BATCH]
        ]
        inputs = tok(rendered, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(model.device)
        ctx = model.disable_adapter() if (has_adapter and not use_adapter) else contextlib.nullcontext()
        with torch.no_grad(), ctx:
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        outs.extend(tok.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
    return outs


@app.get("/health")
async def health() -> JSONResponse:
    m = _state["manifest"] or {}
    return JSONResponse(
        {
            "loaded": _state["model"] is not None,
            "loading": _state["loading"],
            "error": _state["error"],
            "base_model": m.get("base_model"),
            "adapter": m.get("adapter"),
            "generation": m.get("generation"),
            "bound_model": bound_model_path(),
        }
    )


@app.post("/generate")
async def generate(body: dict) -> JSONResponse:
    if _state["model"] is None:
        return JSONResponse({"error": "model not loaded yet; poll /health"}, status_code=503)
    try:
        outs = await asyncio.to_thread(
            _generate_sync,
            body["chats"],
            bool(body.get("use_adapter", True)),
            int(body.get("max_new_tokens", 512)),
        )
        return JSONResponse({"completions": outs, "generation": _state["manifest"].get("generation")})
    except Exception as e:
        return JSONResponse({"error": str(e), "traceback": traceback.format_exc().splitlines()[-8:]}, status_code=500)


inference_app_env = FastAPIAppEnvironment(
    name=ENDPOINT_APP,
    app=app,
    image=gpu_image.with_pip_packages("fastapi", "uvicorn"),
    resources=inference_resources(),
    scaling=flyte.app.Scaling(replicas=(0, 1), scaledown_after=900),
    requires_auth=REQUIRE_APP_AUTH,
    env_vars=cluster_env_vars(),
    parameters=[
        # The factory rebinds this to the version it resolved on every
        # materialization.
        flyte.app.Parameter(name=MODEL_PARAM, value=flyte.app.ArtifactValue(name=ARTIFACT_PROMOTED)),
    ],
    description="Serves the factory's daily release (adapter toggleable per request)",
)


@inference_app_env.on_startup
async def _init() -> None:
    """Load the bound model in the background; never raise (would 500 every request)."""
    bound = bound_model_path()
    if not bound:
        _state["error"] = "no model bound; materialize mf-inference through the factory"
        return

    async def _preload() -> None:
        _state["loading"] = True
        try:
            model, tok, manifest = await asyncio.to_thread(_load_sync, bound)
            _state.update(model=model, tok=tok, manifest=manifest)
        except Exception as e:
            _state["error"] = f"load {type(e).__name__}: {e}"
        finally:
            _state["loading"] = False

    task = asyncio.create_task(_preload())
    _bg_tasks.add(task)
    task.add_done_callback(_bg_tasks.discard)
