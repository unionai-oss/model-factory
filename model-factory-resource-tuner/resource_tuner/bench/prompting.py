"""The one prompt every LLM-backed arm reads.

Zero-shot, fine-tuned generative and decision-head arms all see this exact
text, so any difference between them is the model and the output head,
never the information. It carries what `features.py` allows and nothing
more: the task source, this run's input metadata, and the task's history.
"""

from __future__ import annotations

import json
import re

from . import grid

SYSTEM = (
    "You size Flyte tasks before they run. Given a task's source code, the metadata of "
    "this run's inputs, and what earlier runs of the same task used, choose the cheapest "
    "memory and CPU request that will not run out of memory. Memory is a hard limit: "
    "below the peak the pod is OOM-killed and retried with double the memory. CPU is "
    "soft: too little only slows the run."
)

_SCHEMA = re.compile(r", schema=\[.*?\]")
#: Bound on prompt size. Code is never cut (it is what has to be read);
#: very wide schemas are, from the middle.
MAX_DESC_CHARS = 1600


def _short_inputs(desc: str) -> str:
    return "; ".join(_SCHEMA.sub("", line) for line in desc.splitlines())


def _clip(text: str, n: int) -> str:
    return text if len(text) <= n else text[: n // 2] + " ... " + text[-n // 2 :]


def format_history(history: list[dict]) -> str:
    if not history:
        return "none -- this is the first run of this task"
    lines = []
    for h in history:
        peak = h["peak_rss_mib"] + grid.FLYTE_RUNTIME_OVERHEAD_MIB
        lines.append(f"- inputs {_short_inputs(h['inputs_desc'])} -> peak memory {peak:,.0f} MiB, "
                     f"{h['avg_cores']:.1f} cores")
    return "\n".join(lines)


def options_text() -> str:
    mem = ", ".join(grid.format_memory(m) for m in grid.MEMORY_GRID_MIB)
    cpu = ", ".join(f"{c:g}" for c in grid.CPU_GRID)
    return f"memory: {mem}\ncpu: {cpu}"


def user_prompt(row: dict) -> str:
    history = json.loads(row.get("history_json") or "[]")
    return (
        f"### Task code\n```python\n{row['task_code'].strip()}\n```\n\n"
        f"### This run's inputs\n{_clip(row['inputs_desc'], MAX_DESC_CHARS)}\n\n"
        f"### Earlier runs of this task (peak includes the Flyte runtime)\n{format_history(history)}\n\n"
        f"### Options\n{options_text()}\n\n"
        'Answer with JSON only: {"memory": "<one memory option>", "cpu": <one cpu option>}'
    )


def messages(row: dict) -> list[dict]:
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_prompt(row)}]


def target_text(mem_idx: int, cpu_idx: int) -> str:
    cpu = grid.CPU_GRID[cpu_idx]
    return json.dumps({"memory": grid.format_memory(grid.MEMORY_GRID_MIB[mem_idx]), "cpu": cpu if cpu % 1 else int(cpu)})


_JSON = re.compile(r"\{[^{}]*\}")


def parse_answer(text: str) -> tuple[int, int] | None:
    """Model text -> (mem_idx, cpu_idx), snapping UP onto the grid; None if
    there is no usable answer. Snapping up is the generous reading: a
    generative arm is never penalized for a value between two cells."""
    for m in reversed(_JSON.findall(text or "")):
        try:
            obj = json.loads(m)
            mem = grid.parse_memory(obj["memory"])
            cpu = float(str(obj.get("cpu", 1)).rstrip("m") or 1)
            if str(obj.get("cpu", "")).endswith("m"):
                cpu /= 1000
        except (ValueError, KeyError, TypeError, json.JSONDecodeError):
            continue
        if mem <= 0:
            continue
        return grid.mem_index_covering(mem), grid.cpu_index_covering(cpu)
    return None
