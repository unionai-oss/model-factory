"""Estimation-context → chat prompt.

The context mirrors the PRD's §8 estimation context: task code and input
profile always; prior and history when available (the prototype's corpus is
cold-start, so those default empty). The output contract is a single JSON
object — the exact kwargs to `flyte.Resources`.
"""

from __future__ import annotations

import json


def parse_context_fields(
    prior_json: str | None, history_json: str | None
) -> tuple[dict | None, list | None]:
    """Corpus/task-arg JSON strings → render_messages kwargs. Empty or
    malformed fields degrade to cold-start (None), never to a crash."""
    prior = history = None
    if prior_json:
        try:
            parsed = json.loads(prior_json)
            prior = parsed if isinstance(parsed, dict) and parsed else None
        except (json.JSONDecodeError, TypeError):
            pass
    if history_json:
        try:
            parsed = json.loads(history_json)
            history = parsed if isinstance(parsed, list) and parsed else None
        except (json.JSONDecodeError, TypeError):
            pass
    return prior, history

SYSTEM_PROMPT = """\
You size compute requests for Flyte tasks. Given a task's source code and \
input profile, respond with ONLY a JSON object of flyte.Resources kwargs:

{"cpu": <cores>, "memory": "<size>"}

Rules:
- memory: a Kubernetes quantity string like "512Mi" or "4Gi". Enough that \
the task cannot run out of memory, but not so much that most of it sits idle.
- cpu: integer cores, or a millicore string like "500m". Match the task's \
real parallelism; extra cores sit idle.
- add "gpu": "<TYPE>:<count>" ONLY if the code clearly uses a GPU (moves \
models/tensors to CUDA). TYPE must be one of T4 (16GiB VRAM, cheapest), \
L4 (24GiB), L40S (48GiB, priciest) — pick the cheapest whose VRAM fits \
the model plus activations. CPU-only code must not request a gpu.
- no prose, no code fences, no keys other than cpu, memory, gpu."""

USER_TEMPLATE = """\
Task source:
```python
{source_code}
```

Input profile: {input_profile}
{extra}
Respond with the flyte.Resources kwargs JSON only."""


def render_messages(
    source_code: str,
    input_profile: str,
    prior: dict | None = None,
    history: list[dict] | None = None,
    no_think: bool = True,
) -> list[dict]:
    """Chat messages for one estimation context.

    `no_think` appends Qwen3's `/no_think` soft switch: without it the
    model spends the whole completion budget inside a `<think>` block and
    never reaches the JSON — observed as clipped_ratio=1.0 and all-zero
    rewards on the first smoke run. Belt-and-braces with the trainer's
    `chat_template_kwargs={"enable_thinking": False}`.
    """
    extra = ""
    if prior:
        extra += f"Author-declared prior: {prior}\n"
    if history:
        lines = "\n".join(
            f"- requested {h.get('resources')} peak {h.get('peak')} success={h.get('ok')}"
            for h in history
        )
        extra += f"Recent runs:\n{lines}\n"
    user = USER_TEMPLATE.format(
        source_code=source_code.rstrip(), input_profile=input_profile, extra=extra
    )
    if no_think:
        user += " /no_think"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
