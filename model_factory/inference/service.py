"""The inference serving app: loads policy checkpoints, serves generations.

Endpoints:
- GET  /health    → {loaded, base_model, checkpoint_path}
- POST /reload    → {"checkpoint_path": "<s3 dir>"} | {} (= latest artifact)
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
from ..contracts import ARTIFACT_CHECKPOINT
from ..shared import assets
from ..shared.images import gpu_image

_GEN_BATCH = 8

app = FastAPI(title="Model Factory Inference Service")

_state: dict = {"model": None, "tok": None, "base_model": None, "checkpoint_path": None}
_load_lock = asyncio.Lock()


async def _resolve_latest_checkpoint() -> str:
    v = await assets.latest(
        ARTIFACT_CHECKPOINT,
        project=os.environ.get("MF_PROJECT", "model-factory"),
        domain=os.environ.get("MF_DOMAIN", "development"),
    )
    return v.path


async def _load(checkpoint_path: str) -> dict:
    """Download a checkpoint Dir and (re)load base + adapter."""
    import json

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    local = await flyte.io.Dir.from_existing_remote(checkpoint_path).download()
    with open(os.path.join(local, "manifest.json")) as f:
        manifest = json.load(f)
    base_model = manifest["base_model"]

    old = _state.pop("model", None)
    if old is not None:
        del old
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    tok = AutoTokenizer.from_pretrained(base_model, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
    )
    model = PeftModel.from_pretrained(model, local)
    model.eval()
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
        }
    )


@app.post("/reload")
async def reload(body: dict | None = None) -> JSONResponse:
    body = body or {}
    try:
        path = body.get("checkpoint_path") or await _resolve_latest_checkpoint()
        async with _load_lock:
            info = await _load(path)
        return JSONResponse({"ok": True, **info})
    except Exception as e:
        return JSONResponse(
            {"ok": False, "error": str(e), "traceback": traceback.format_exc().splitlines()[-8:]},
            status_code=500,
        )


@app.post("/generate")
async def generate(body: dict) -> JSONResponse:
    """Batched greedy/sampled generation; adapter can be toggled per request
    so eval can compare candidate (adapter on) vs base (adapter off) against
    the exact same weights service."""
    import torch

    try:
        await _ensure_loaded(body.get("checkpoint_path"))
        chats: list = body["chats"]
        use_adapter: bool = bool(body.get("use_adapter", True))
        max_new_tokens: int = int(body.get("max_new_tokens", 512))
        do_sample: bool = bool(body.get("do_sample", False))
        temperature: float = float(body.get("temperature", 1.0))

        model, tok = _state["model"], _state["tok"]
        outs: list[str] = []
        for i in range(0, len(chats), _GEN_BATCH):
            chunk = chats[i : i + _GEN_BATCH]
            rendered = [
                tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True)
                for c in chunk
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
    resources=flyte.Resources(cpu=6, memory="24Gi", gpu="L4:1", disk="100Gi", shm="auto"),
    scaling=flyte.app.Scaling(replicas=(0, 1), scaledown_after=900),
    requires_auth=False,
    env_vars={"MF_ORG": "demo", "MF_PROJECT": "model-factory", "MF_DOMAIN": "development"},
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
