"""Inference ops: react to new checkpoints by refreshing the serving app.

The OnArtifact trigger on `policy-checkpoint` makes weight rollout to the
inference service automatic; the published `inference-endpoint` artifact
tells consumers (eval, rollout workers) what is being served and where.
"""

from __future__ import annotations

import asyncio

import flyte
import flyte.io

from ..contracts import (
    ARTIFACT_CHECKPOINT,
    ARTIFACT_INFERENCE_ENDPOINT,
    InferenceEndpoint,
    publish,
)
from ..config import cluster_env_vars, cpu_resources
from ..shared import inference_client
from ..shared.images import cpu_image
from . import APP_NAME

inference_ops_env = flyte.TaskEnvironment(
    name="inference-ops",
    resources=cpu_resources(),
    env_vars=cluster_env_vars(),
    image=cpu_image,
)

_refresh_trigger = flyte.Trigger(
    name="serve-new-checkpoint",
    automation=flyte.OnArtifact(name=ARTIFACT_CHECKPOINT),
    inputs={"checkpoint": flyte.TriggeredArtifact},
    description="New policy-checkpoint → load weights into the inference service",
    auto_activate=False,
)


@inference_ops_env.task(triggers=[_refresh_trigger], timeout=flyte.Timeout(max_runtime=1800), produces_artifacts=True)
async def refresh_inference_service(checkpoint: flyte.io.Dir) -> flyte.io.File:
    """Point the serving app at a new checkpoint and verify it generates."""
    url = inference_client.resolve_endpoint(APP_NAME)

    # The client is blocking (it polls /health on a sleep loop). Run it in a
    # worker thread — blocking this task's event loop stalls the Flyte
    # runtime's own coroutines for the whole reload.
    # The deadline stays comfortably under the task timeout below, so a reload
    # that never converges surfaces as a real error rather than a bare
    # TIMED_OUT with no explanation.
    info = await asyncio.to_thread(
        inference_client.reload_checkpoint,
        url,
        checkpoint_path=checkpoint.path,
        deadline_s=1500,
    )

    # Smoke generation proves the weights actually serve.
    completions = await asyncio.to_thread(
        inference_client.generate,
        url,
        [[{"role": "user", "content": "Write a python function that adds two numbers."}]],
        use_adapter=True,
        max_new_tokens=64,
    )
    if not completions or not completions[0].strip():
        raise flyte.errors.NonRecoverableError("inference service returned empty generation")

    endpoint = InferenceEndpoint(
        url=url,
        base_model=str(info.get("base_model", "")),
        checkpoint_path=checkpoint.path,
        checkpoint_run="",
    )
    out = "/tmp/inference_endpoint.json"
    with open(out, "w") as f:
        f.write(endpoint.to_json())
    f_out = await flyte.io.File.from_local(out)
    return publish(
        f_out,
        ARTIFACT_INFERENCE_ENDPOINT,
        description=f"Serving {endpoint.base_model} + adapter from {checkpoint.path.rsplit('/', 1)[-1]} at {url}",
    )
