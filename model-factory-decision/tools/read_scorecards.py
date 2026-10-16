"""Read scorecards from inside the cluster, for the research log.

A laptop without object-store credentials cannot download artifact blobs, so
this runs a tiny CPU task that does, and returns a compact JSON summary as a
string output (readable via the run's outputs). Pass the blob URIs from
`flyte.remote.Artifact.listall(name="permission-scorecard")`.

    uv run flyte --config .flyte/config.yaml run tools/read_scorecards.py read_scorecards \
        --uris '["s3://.../scorecard.json", ...]'
"""

from __future__ import annotations

import json

import flyte
import flyte.io

from decision_factory.shared.images import cpu_image

env = flyte.TaskEnvironment(name="df-tools", image=cpu_image, resources=flyte.Resources(cpu=1, memory="1Gi"))

KEEP = ("accuracy", "unsafe_allow_rate", "severe_unsafe_rate", "unnecessary_ask_rate", "brier", "ece",
        "expected_cost", "gate", "val", "latency", "input_tokens", "output_tokens", "parse_rate",
        "slices", "unseen_command", "confusion", "frontier", "reliability")


@env.task
async def read_scorecards(uris: list[str]) -> str:
    out = []
    for uri in uris:
        with open(await flyte.io.File(path=uri).download()) as f:
            c = json.load(f)
        out.append({
            "model": c["model_slug"], "torso": c["torso"], "head": c["head"], "primary": c["primary"],
            "params_m": c.get("params_m"), "head_params": c.get("head_params"),
            "trainable_params": c.get("trainable_params"), "train_seconds": c.get("train_seconds"),
            "readouts": {k: {m: v.get(m) for m in KEEP} for k, v in c["readouts"].items()},
            "reference": {k: ({m: v.get(m) for m in KEEP} if isinstance(v, dict) else v)
                          for k, v in c["reference"].items()},
            "errors": c.get("errors", [])[:25],
        })
    return json.dumps(out)
