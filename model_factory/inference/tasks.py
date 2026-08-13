"""Inference ops: react to new checkpoints by refreshing the serving app.

The OnArtifact trigger on `policy-checkpoint` makes weight rollout to the
inference service automatic; the published `inference-endpoint` artifact
tells consumers (eval, rollout workers) what is being served and where.
"""

from __future__ import annotations

import flyte
import flyte.io

from ..contracts import (
    ARTIFACT_CHECKPOINT,
    ARTIFACT_INFERENCE_ENDPOINT,
    InferenceEndpoint,
    publish,
)
from ..shared import inference_client
from ..shared.images import cpu_image
from .service import APP_NAME

inference_ops_env = flyte.TaskEnvironment(
    name="inference-ops",
    resources=flyte.Resources(cpu=1, memory="2Gi"),
    image=cpu_image,
)

_refresh_trigger = flyte.Trigger(
    name="serve-new-checkpoint",
    automation=flyte.OnArtifact(name=ARTIFACT_CHECKPOINT),
    inputs={"checkpoint": flyte.TriggeredArtifact},
    description="New policy-checkpoint → load weights into the inference service",
    auto_activate=False,
)


@inference_ops_env.task(triggers=[_refresh_trigger], timeout=flyte.Timeout(max_runtime=1800))
async def refresh_inference_service(checkpoint: flyte.io.Dir) -> flyte.io.File:
    """Point the serving app at a new checkpoint and verify it generates."""
    import json

    url = inference_client.resolve_endpoint(APP_NAME)
    info = inference_client.reload_checkpoint(url, checkpoint_path=checkpoint.path)

    # Smoke generation proves the weights actually serve.
    completions = inference_client.generate(
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
