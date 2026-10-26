"""The one prompt every LLM arm of the combined bench reads.

Same layout as the three benches' prompts (code, this run's inputs, the
task's history, the options), with all three decisions. Nothing says which
bench a task came from: whether it needs a GPU is read from the code.
"""

from __future__ import annotations

import json
import re

from ..bench.grid import FLYTE_RUNTIME_OVERHEAD_MIB, parse_memory
from ..bench.prompting import MAX_DESC_CHARS, _clip, _short_inputs
from ..bench.spec import snap_up
from ..bench_gpu.grid import DEVICES
from . import grid

_DEVICES = ", ".join(f"{n} ({g:g} GiB, ${p:.2f}/h)" for n, g, p, _s in DEVICES)

SYSTEM = (
    "You size Flyte tasks before they run. Given a task's source code, the metadata of this run's "
    "inputs, and what earlier runs of the same task used, choose the cheapest request that runs it: "
    "memory, CPU cores, and a GPU or none. Memory is a hard limit: below the peak the pod is "
    "OOM-killed and retried with double. CPU below the busiest 15 seconds' use throttles the task and "
    "it finishes late. Thread pools the code does not size get one thread. gpu is the peak GPU memory "
    f"to provision, or none for a task that does not use CUDA; it lands on the cheapest GPU that holds "
    f"it -- {_DEVICES} -- and a GPU pod comes with its host memory and cores. A GPU task given no GPU "
    "fails at start; a GPU given to a CPU task is paid for the whole run."
)

KEYS = ("memory", "cpu", "gpu")
OPTIONS = (grid.MEM_OPTIONS, grid.CPU_OPTIONS, grid.GPU_OPTIONS)


def format_history(history: list[dict]) -> str:
    if not history:
        return "none -- this is the first run of this task"
    lines = []
    for h in history:
        parts = []
        if "peak_mem_mib" in h:
            parts.append(f"peak memory {h['peak_mem_mib'] + FLYTE_RUNTIME_OVERHEAD_MIB:,.0f} MiB")
        if "peak_cores" in h:
            parts.append(f"peak {h['peak_cores']:.2f} cores (busiest 15 s)")
        if "mean_cores" in h:
            parts.append(f"mean {h['mean_cores']:.2f} cores")
        if "peak_gpu_mib" in h:
            parts.append(f"peak GPU memory {h['peak_gpu_mib'] / 1024:.2f} GiB, "
                         f"GPU busy {h.get('mean_util', 0):.0f}%")
        lines.append(f"- inputs {_short_inputs(h['inputs_desc'])} -> {', '.join(parts)}, {h.get('wall_s', 0):.0f} s")
    return "\n".join(lines)


def options_text() -> str:
    return "\n".join(f"{k}: {', '.join(o)}" for k, o in zip(KEYS, OPTIONS))


ANSWER = 'Answer with JSON only: {"memory": "<one memory option>", "cpu": "<one cpu option>", "gpu": "<one gpu option>"}'


def user_prompt(row: dict) -> str:
    history = json.loads(row.get("history_json") or "[]")
    return (
        f"### Task code\n```python\n{row['task_code'].strip()}\n```\n\n"
        f"### This run's inputs\n{_clip(row['inputs_desc'], MAX_DESC_CHARS)}\n\n"
        f"### Earlier runs of this task (memory peaks include the Flyte runtime)\n{format_history(history)}\n\n"
        f"### Options\n{options_text()}\n\n{ANSWER}"
    )


def messages(row: dict) -> list[dict]:
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_prompt(row)}]


_JSON = re.compile(r"\{[^{}]*\}")
_GIB = re.compile(r"(\d+(?:\.\d+)?)\s*([MG])?i?B?", re.IGNORECASE)


def _mem(v) -> int | None:
    try:
        mib = parse_memory(v)
    except (TypeError, ValueError):
        return None
    return grid.mem_cell(mib) if mib > 0 else None


def _gpu(v) -> int | None:
    s = str(v).strip().lower()
    if s in ("none", "no", "0", "", "null", "cpu", "false"):
        return grid.NO_GPU
    m = _GIB.search(s)
    if not m:
        return None
    gib = float(m.group(1)) / (1024 if (m.group(2) or "").upper() == "M" else 1)
    return 1 + snap_up(gib, list(grid.GPU_MEM_GRID_GIB))


def parse_answer(text: str) -> tuple[int, int, int] | None:
    """The last JSON object with a usable memory answer -> (mem, cpu, gpu),
    snapping up onto each grid. A missing cpu is 1 core; a missing gpu, none."""
    for m in reversed(_JSON.findall(text or "")):
        try:
            obj = json.loads(m)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict) or "memory" not in obj:
            continue
        mi = _mem(obj["memory"])
        if mi is None:
            continue
        ci = snap_up(obj.get("cpu", 1), list(grid.CPU_GRID))
        gi = _gpu(obj.get("gpu", "none"))
        return mi, grid.cpu_cell(1) if ci is None else ci, grid.NO_GPU if gi is None else gi
    return None


def target_text(m: int, c: int, g: int) -> str:
    return json.dumps({"memory": grid.MEM_OPTIONS[m], "cpu": grid.CPU_OPTIONS[c], "gpu": grid.GPU_OPTIONS[g]})


class Spec:
    """What the shared LLM helpers read off a bench (``_prompt_ids`` calls
    ``spec.messages``); the combined bench has three decisions, not two."""

    name = "all"
    title = "RightSize Bench"

    @staticmethod
    def messages(row: dict) -> list[dict]:
        return messages(row)


SPEC = Spec()
