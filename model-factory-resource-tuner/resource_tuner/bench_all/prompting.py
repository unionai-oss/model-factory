"""The one prompt every LLM arm of the combined bench reads.

Same layout as the three benches' prompts (code, this run's inputs, the
task's history, the options), with all three decisions. Nothing says which
bench a task came from: whether it needs a GPU is read from the code.

Since round 22 a run is a Flyte task or its Kubernetes twin (``k8s``): a
plain ``job.py`` and its Job manifest, whose history is the container's own
peaks. The prompt ends the same way on both, so the pointer head's option
offsets (counted from the end) are one set.
"""

from __future__ import annotations

import json
import re

from ..bench.grid import parse_memory
from ..bench.prompting import MAX_DESC_CHARS, _clip, _short_inputs
from ..bench.spec import snap_up, with_hint
from ..bench_gpu.grid import DEVICES
from . import grid, k8s

_DEVICES = ", ".join(f"{d.name} ({d.capacity_gib:g} GiB, ${d.price:.2f}/h)" for d in DEVICES)

SYSTEM = (
    "You size batch jobs before they run: Flyte tasks, or Kubernetes Jobs given as a Python module and its "
    "manifest. Given the code, the metadata of this run's inputs, and what earlier runs of the same job "
    "used, choose the cheapest request that runs it: memory, CPU cores, and a GPU or none. Memory is a hard "
    "limit: below the peak the container is OOM-killed and retried with double. A Flyte task's memory "
    "includes the Flyte runtime; a Kubernetes container's includes whatever the job loads to fetch its "
    "inputs, and files written to a memory-backed emptyDir (medium: Memory) are charged to it. "
    "CPU below the busiest 15 seconds' use throttles the task and "
    "it finishes late. Thread pools the code does not size get one thread. gpu is the peak GPU memory "
    f"to provision, or none for a task that does not use CUDA; it lands on the cheapest machine that holds "
    f"it -- {_DEVICES} -- and a GPU pod comes with its host memory and cores. A multi-GPU machine only "
    "helps code that spreads its model over every visible GPU. A GPU task given no GPU "
    "fails at start; a GPU given to a CPU task is paid for the whole run."
)

KEYS = ("memory", "cpu", "gpu")
OPTIONS = (grid.MEM_OPTIONS, grid.CPU_OPTIONS, grid.GPU_OPTIONS)


def format_history(history: list[dict], platform: str = "flyte") -> str:
    if not history:
        return "none -- this is the first run of this task"
    # Flyte's history stores the workload's peak (the runtime is added here);
    # a K8s twin's is the container's (``k8s.container_peak_mib``)
    overhead = k8s.FLYTE_RUNTIME_MIB if platform == "flyte" else 0.0
    lines = []
    for h in history:
        parts = []
        if "peak_mem_mib" in h:
            parts.append(f"peak memory {h['peak_mem_mib'] + overhead:,.0f} MiB")
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
    platform = row.get("platform") or "flyte"
    if platform == "k8s":
        head = (f"### Job code (job.py)\n```python\n{row['task_code'].strip()}\n```\n\n"
                f"### Job manifest\n```yaml\n{row['manifest'].strip()}\n```\n\n")
        hist = "### Earlier runs of this job (container memory peaks, from the cluster's metrics)"
    else:
        head = f"### Task code\n```python\n{row['task_code'].strip()}\n```\n\n"
        hist = "### Earlier runs of this task (memory peaks include the Flyte runtime)"
    return (
        f"{head}### This run's inputs\n{_clip(row['inputs_desc'], MAX_DESC_CHARS)}\n\n"
        f"{hist}\n{format_history(history, platform)}\n\n"
        f"### Options\n{options_text()}\n\n{ANSWER}"
    )


def messages(row: dict) -> list[dict]:
    user = with_hint(user_prompt(row), row.get("hint") or "")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def hint_text(probs: tuple, pick: tuple[int, int, int], top: int = 2) -> str:
    """The trees' joint estimate (``arms.fit_gbt_dist``), as a stacked arm's prompt carries it."""
    import numpy as np

    pm, pc, pg = (np.asarray(x, dtype=float) / max(float(np.sum(x)), 1e-12) for x in probs)

    def lead(p, opts):
        return ", ".join(f"{opts[i]} {p[i]:.0%}" for i in np.argsort(-p)[:top])

    gpu = (f"P(no GPU) {pg[0]:.0%}" + (f", else {lead(pg[1:] / max(pg[1:].sum(), 1e-12), grid.GPU_OPTIONS[1:])}"
                                        if pg[0] < 0.99 else ""))
    m, c, g = pick
    return ("Gradient-boosted trees fit on other tasks' measured runs (they read the input sizes, the API names "
            f"and numbers in the code, and this task's earlier runs) estimate memory {lead(pm, grid.MEM_OPTIONS)}; "
            f"cpu {lead(pc, grid.CPU_OPTIONS)}; gpu {gpu}. Their pick: memory {grid.MEM_OPTIONS[m]}, "
            f"cpu {grid.CPU_OPTIONS[c]}, gpu {grid.GPU_OPTIONS[g]}.")


_JSON = re.compile(r"\{[^{}]*\}")
_GIB = re.compile(r"(\d+(?:\.\d+)?)\s*([MG])?i?B?", re.IGNORECASE)


_QTY = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(Gi|Mi|Ki|GiB|MiB|KiB|GB|MB|G|M|K)?\s*$", re.IGNORECASE)
_TO_MIB = {"gi": 1024.0, "gib": 1024.0, "g": 1e9 / 2**20, "gb": 1e9 / 2**20, "mi": 1.0, "mib": 1.0,
           "m": 1e6 / 2**20, "mb": 1e6 / 2**20, "ki": 1 / 1024, "kib": 1 / 1024, "k": 1e3 / 2**20}


def _mem(v) -> int | None:
    """'2Gi', '2 GiB', '2GB', '768Mi', 3072 -> a memory cell (a bare number is MiB)."""
    try:
        mib = parse_memory(v)
    except (TypeError, ValueError):
        m = _QTY.match(str(v))
        if not m:
            return None
        mib = float(m.group(1)) * _TO_MIB.get((m.group(2) or "mi").lower(), 1.0)
    return grid.mem_cell(mib) if mib > 0 else None


def _cpu(v) -> int | None:
    """'2', 2.5, '500m' (Kubernetes millicores), '2 cores' -> the smallest CPU cell >= it."""
    s = str(v).strip().lower()
    m = re.match(r"^(\d+(?:\.\d+)?)\s*m$", s)
    return snap_up(float(m.group(1)) / 1000 if m else v, list(grid.CPU_GRID))


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
        ci = _cpu(obj.get("cpu", 1))
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
