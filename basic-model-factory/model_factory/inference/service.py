"""The inference serving app: loads policy checkpoints, serves generations.

Endpoints:
- GET  /health    → {loaded, base_model, checkpoint_path, loading, reload_error}
- POST /reload    → kicks off the (re)load in the background and returns
                    immediately; poll /health until loaded. Returning inline
                    would make the request outlast the Knative activator
                    timeout (504) on large checkpoints.
                    {"checkpoint_path": "<s3 dir>"} | {} (= latest artifact)
- POST /generate  → {"chats": [[{role,content},...]], "use_adapter": bool,
                     "max_new_tokens": int, "checkpoint_path": str|null}
                    Reloads first if checkpoint_path differs from loaded.

Deploy: flyte deploy team_inference.py inference_app_env
"""

from __future__ import annotations

import asyncio
import os
import traceback

import flyte
import flyte.app
import flyte.io
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from flyte.app.extras import FastAPIAppEnvironment

from . import APP_NAME
from ..config import (
    APP_DOMAIN,
    APP_ORG,
    APP_PROJECT,
    cluster_env_vars,
    inference_resources,
    REQUIRE_APP_AUTH,
)
from ..contracts import ARTIFACT_CHECKPOINT
from ..shared import assets
from ..shared.images import gpu_image

_GEN_BATCH = 8

app = FastAPI(title="Model Factory Inference Service")

_state: dict = {
    "model": None,
    "tok": None,
    "base_model": None,
    "checkpoint_path": None,
    "loading": None,  # checkpoint path currently being loaded (or None)
    "reload_error": None,
}
_load_lock = asyncio.Lock()
# Strong refs to in-flight background reloads: asyncio only holds weak
# references to tasks, so an unreferenced task can be garbage-collected
# mid-load and the reload would vanish silently.
_bg_tasks: set = set()


async def _resolve_latest_checkpoint() -> str:
    v = await assets.latest(
        ARTIFACT_CHECKPOINT,
        project=os.environ.get("MF_PROJECT", "model-factory"),
        domain=os.environ.get("MF_DOMAIN", "development"),
    )
    return v.path


def _load_weights_sync(base_model: str, local: str):
    """Blocking part of a checkpoint load: base weights + adapter onto the GPU.

    Kept synchronous and OFF the event loop (see ``_load``) — transformers and
    peft are blocking, and a multi-minute load on the loop makes the app stop
    answering /health, which is exactly what callers poll to track the load.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base_model, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
    )
    model = PeftModel.from_pretrained(model, local)
    model.eval()
    return model, tok


async def _load(checkpoint_path: str) -> dict:
    """Download a checkpoint Dir and (re)load base + adapter.

    The blocking weight load runs in a worker thread so the event loop keeps
    serving /health while it happens; otherwise the caller polling /health
    sees nothing but timeouts for the whole load and gives up.
    """
    import json

    import torch

    local = await flyte.io.Dir.from_existing_remote(checkpoint_path).download()
    with open(os.path.join(local, "manifest.json")) as f:
        manifest = json.load(f)
    base_model = manifest["base_model"]

    # Set to None rather than popping: /health and /generate read these keys
    # throughout the load, and a missing key 500s them.
    old = _state["model"]
    _state["model"] = None
    _state["checkpoint_path"] = None
    if old is not None:
        del old
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    model, tok = await asyncio.to_thread(_load_weights_sync, base_model, local)
    _state.update(
        {"model": model, "tok": tok, "base_model": base_model, "checkpoint_path": checkpoint_path}
    )
    return {"base_model": base_model, "checkpoint_path": checkpoint_path}


async def _ensure_loaded(checkpoint_path: str | None) -> None:
    async with _load_lock:
        if checkpoint_path and checkpoint_path != _state["checkpoint_path"]:
            await _load(checkpoint_path)
        elif _state["model"] is None:
            await _load(checkpoint_path or await _resolve_latest_checkpoint())


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse(
        {
            "loaded": _state["model"] is not None,
            "base_model": _state["base_model"],
            "checkpoint_path": _state["checkpoint_path"],
            "loading": _state["loading"],
            "reload_error": _state["reload_error"],
        }
    )


@app.post("/reload")
async def reload(body: dict | None = None) -> JSONResponse:
    """Kick off a checkpoint (re)load in the background and return immediately.

    Loading a checkpoint (S3 download + base model + adapter onto GPU) takes
    longer than the Knative activator's request timeout, so doing it inline
    yields a 504 for the caller even when the load succeeds. Callers should
    poll /health until `loaded` and `checkpoint_path` match.
    """
    body = body or {}
    try:
        path = body.get("checkpoint_path") or await _resolve_latest_checkpoint()
    except Exception as e:
        return JSONResponse(
            {"ok": False, "error": f"could not resolve checkpoint: {e}"}, status_code=500
        )

    # Already serving this checkpoint — nothing to do.
    if _state["model"] is not None and _state["checkpoint_path"] == path:
        return JSONResponse(
            {"ok": True, "base_model": _state["base_model"], "checkpoint_path": path}
        )
    # A load of this exact checkpoint is already in flight.
    if _state["loading"] == path:
        return JSONResponse({"ok": True, "loading": True, "checkpoint_path": path})

    _state["reload_error"] = None
    _state["loading"] = path

    async def _job() -> None:
        try:
            async with _load_lock:
                await _load(path)
        except Exception as e:
            _state["reload_error"] = f"{type(e).__name__}: {e}\n" + "\n".join(
                traceback.format_exc().splitlines()[-8:]
            )
        finally:
            _state["loading"] = None

    task = asyncio.create_task(_job())
    _bg_tasks.add(task)
    task.add_done_callback(_bg_tasks.discard)
    return JSONResponse({"ok": True, "loading": True, "checkpoint_path": path})


def _generate_sync(
    chats: list,
    use_adapter: bool,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
) -> list[str]:
    """Blocking batched generation — always called via ``asyncio.to_thread``."""
    import torch

    model, tok = _state["model"], _state["tok"]
    outs: list[str] = []
    for i in range(0, len(chats), _GEN_BATCH):
        chunk = chats[i : i + _GEN_BATCH]
        rendered = [
            tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True) for c in chunk
        ]
        inputs = tok(
            rendered, return_tensors="pt", padding=True, truncation=True, max_length=2048
        ).to(model.device)
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
        if do_sample:
            gen_kwargs["temperature"] = temperature
        with torch.no_grad():
            if use_adapter:
                out = model.generate(**inputs, **gen_kwargs)
            else:
                with model.disable_adapter():
                    out = model.generate(**inputs, **gen_kwargs)
        outs.extend(
            tok.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        )
    return outs


@app.post("/generate")
async def generate(body: dict) -> JSONResponse:
    """Batched greedy/sampled generation; adapter can be toggled per request
    so eval can compare candidate (adapter on) vs base (adapter off) against
    the exact same weights service."""
    try:
        await _ensure_loaded(body.get("checkpoint_path"))
        chats: list = body["chats"]
        use_adapter: bool = bool(body.get("use_adapter", True))
        max_new_tokens: int = int(body.get("max_new_tokens", 512))
        do_sample: bool = bool(body.get("do_sample", False))
        temperature: float = float(body.get("temperature", 1.0))

        # Generation is blocking and can run for minutes; off the loop it goes,
        # so /health stays answerable while a batch is in flight.
        outs: list[str] = await asyncio.to_thread(
            _generate_sync, chats, use_adapter, max_new_tokens, do_sample, temperature
        )
        return JSONResponse(
            {"completions": outs, "checkpoint_path": _state["checkpoint_path"], "use_adapter": use_adapter}
        )
    except Exception as e:
        return JSONResponse(
            {"error": str(e), "traceback": traceback.format_exc().splitlines()[-8:]},
            status_code=500,
        )


inference_app_env = FastAPIAppEnvironment(
    name=APP_NAME,
    app=app,
    image=gpu_image.with_pip_packages("fastapi", "uvicorn"),
    resources=inference_resources(),
    scaling=flyte.app.Scaling(replicas=(0, 1), scaledown_after=900),
    requires_auth=REQUIRE_APP_AUTH,
    env_vars={
        **cluster_env_vars(),
        "MF_ORG": APP_ORG,
        "MF_PROJECT": APP_PROJECT,
        "MF_DOMAIN": APP_DOMAIN,
    },
    description="Serves the latest policy-checkpoint for rollouts and evals (adapter toggleable)",
)


@inference_app_env.on_startup
async def _init() -> None:
    """Init control-plane access; never raise (would 500 every request)."""
    try:
        await flyte.init_in_cluster.aio(
            org=os.environ.get("MF_ORG") or None,
            project=os.environ.get("MF_PROJECT") or None,
            domain=os.environ.get("MF_DOMAIN") or None,
        )
    except Exception:
        try:
            await flyte.init_in_cluster.aio()
        except Exception:
            pass
