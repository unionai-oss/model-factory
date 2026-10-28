"""RightSizeGPU Bench as a ``BenchSpec``: peak GPU memory + mean GPU utilization."""

from __future__ import annotations

import json
import re

from ..bench.prompting import MAX_DESC_CHARS, _clip, _short_inputs
from ..bench.spec import BenchSpec, Decision, answer_instruction, options_block, snap_up
from . import features, grid, scoring

_DEVICES = ", ".join(f"{d.name} ({d.capacity_gib:g} GiB{'' if d.n_gpus == 1 else f' as {d.n_gpus} x {d.per_gpu_gib:g}'}, "
                     f"${d.price:.2f}/h)" for d in grid.DEVICES)

SYSTEM = (
    "You size GPU tasks before they run. Given a PyTorch task's source code, the metadata of this "
    "run's inputs, and what earlier runs of the same task used, predict two things the platform's "
    "GPU monitoring will measure. gpu_memory is the peak GPU memory the task holds, summed over the "
    "GPUs it uses (CUDA context and PyTorch's caching allocator included); the task runs on the "
    f"cheapest machine whose GPU memory covers your answer -- {_DEVICES} -- and if it needs more it "
    "dies with a CUDA out-of-memory error and is retried on the next machine up. A multi-GPU machine "
    "only helps code that spreads its model over every visible GPU; other code uses one of its GPUs. "
    "gpu_util is the share of time a GPU kernel is running, averaged over the run and the GPUs it uses."
)

_MEM_OPTIONS = tuple(grid.format_gib(g) for g in grid.GPU_MEM_GRID_GIB)
_UNIT = re.compile(r"(-?\d+(?:\.\d+)?)\s*([MG])?i?B?", re.IGNORECASE)


def _parse_gib(value: object) -> int | None:
    m = _UNIT.search(str(value))
    if not m:
        return None
    x = float(m.group(1)) / (1024 if (m.group(2) or "").upper() == "M" else 1)
    return snap_up(x, list(grid.GPU_MEM_GRID_GIB))


def _parse_util(value: object) -> int | None:
    m = re.search(r"\d+(?:\.\d+)?", str(value))
    return grid.util_index(float(m.group(0))) if m else None


def _history(history: list[dict]) -> str:
    if not history:
        return "none -- this is the first run of this task"
    return "\n".join(f"- inputs {_short_inputs(h['inputs_desc'])} -> peak GPU memory {h['peak_gpu_mib'] / 1024:.2f} GiB, "
                     f"mean GPU utilization {h['mean_util']:.0f}% (on {h['measured_on'] or 'a T4'}), {h['wall_s']:.0f} s"
                     for h in history)


def user_prompt(row: dict) -> str:
    history = json.loads(row.get("history_json") or "[]")
    return (
        f"### Task code\n```python\n{row['task_code'].strip()}\n```\n\n"
        f"### This run's inputs\n{_clip(row['inputs_desc'], MAX_DESC_CHARS)}\n\n"
        f"### Earlier runs of this task\n{_history(history)}\n\n"
        f"### Options\n{options_block(GPU.a, GPU.b)}\n\n{answer_instruction(GPU.a, GPU.b)}"
    )


GPU = BenchSpec(
    name="gpu",
    title="RightSizeGPU Bench",
    system=SYSTEM,
    a=Decision("gpu_memory", _MEM_OPTIONS, "label_gmem_idx", "gmem_idx", grid.mem_index_covering(8), _parse_gib),
    b=Decision("gpu_util", grid.UTIL_BUCKETS, "label_util_idx", "util_idx", 5, _parse_util),
    user_prompt=user_prompt,
    decode=scoring.decode_index,
    headline=lambda rows, preds: 1.0 - scoring.score(rows, preds)["mem_within_margin"],
    row_features=features.row_features,
    feature_names=features.feature_names,
    probs_key="gmem_probs",
)
