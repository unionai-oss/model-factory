"""RightSizeCPU Bench as a ``BenchSpec``: peak cores + mean cores."""

from __future__ import annotations

import json

from ..bench.prompting import MAX_DESC_CHARS, _clip, _short_inputs
from ..bench.spec import BenchSpec, Decision, answer_instruction, options_block
from . import features, grid, scoring

SYSTEM = (
    "You size Flyte tasks before they run. Given a task's source code, the metadata of this run's "
    "inputs, and what earlier runs of the same task used, predict the task's CPU usage as the "
    "platform's monitoring will measure it. peak_cpu is the most cores the task uses in any "
    "15-second window; mean_cpu is its average over the whole run. Thread pools the code does not "
    "size get one thread (BLAS and OpenMP are pinned to 1 unless the code sets them). A CPU limit "
    "below the peak throttles the task; cores above it sit idle."
)

_OPTIONS = tuple(grid.format_cores(c) for c in grid.CPU_GRID)


def _history(history: list[dict]) -> str:
    if not history:
        return "none -- this is the first run of this task"
    return "\n".join(f"- inputs {_short_inputs(h['inputs_desc'])} -> peak {h['peak_cores']:.2f} cores "
                     f"(busiest 15 s), mean {h['mean_cores']:.2f} cores, {h['wall_s']:.0f} s" for h in history)


def user_prompt(row: dict) -> str:
    history = json.loads(row.get("history_json") or "[]")
    return (
        f"### Task code\n```python\n{row['task_code'].strip()}\n```\n\n"
        f"### This run's inputs\n{_clip(row['inputs_desc'], MAX_DESC_CHARS)}\n\n"
        f"### Earlier runs of this task\n{_history(history)}\n\n"
        f"### Options\n{options_block(CPU.a, CPU.b)}\n\n{answer_instruction(CPU.a, CPU.b)}"
    )


CPU = BenchSpec(
    name="cpu",
    title="RightSizeCPU Bench",
    system=SYSTEM,
    a=Decision("peak_cpu", _OPTIONS, "label_peak_idx", "peak_idx", grid.index_covering(1)),
    b=Decision("mean_cpu", _OPTIONS, "label_mean_idx", "mean_idx", grid.index_covering(1)),
    user_prompt=user_prompt,
    decode=scoring.decode_index,
    headline=lambda rows, preds: 1.0 - scoring.score(rows, preds)["within_margin"],
    row_features=features.row_features,
    feature_names=features.feature_names,
    probs_key="peak_probs",
)
