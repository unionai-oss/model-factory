"""RightSizeMem Bench as a ``BenchSpec``: memory request + CPU request.

The prompt, the answer format and the expected-cost decoder are the ones
rounds 17's leaderboards were computed with (``prompting``, ``scoring``);
this module only names them so the shared LLM code can read them.
"""

from __future__ import annotations

from . import features, grid, prompting, scoring
from .spec import BenchSpec, Decision


def _parse_memory(value: object) -> int | None:
    try:
        mib = grid.parse_memory(value)
    except (TypeError, ValueError):
        return None
    return grid.mem_index_covering(mib) if mib > 0 else None


def _parse_cpu(value: object) -> int | None:
    s = str(value).strip()
    try:
        cores = float(s.rstrip("m")) / (1000 if s.endswith("m") else 1)
    except ValueError:
        return None
    return grid.cpu_index_covering(cores)


MEM = BenchSpec(
    name="mem",
    title="RightSizeMem Bench",
    system=prompting.SYSTEM,
    a=Decision("memory", tuple(grid.format_memory(m) for m in grid.MEMORY_GRID_MIB), "label_mem_idx", "mem_idx",
               grid.mem_index_covering(4096), _parse_memory),
    b=Decision("cpu", tuple(f"{c:g}" for c in grid.CPU_GRID), "label_cpu_idx", "cpu_idx",
               grid.cpu_index_covering(1), _parse_cpu),
    user_prompt=prompting.user_prompt,
    decode=scoring.bayes_mem_index,
    headline=lambda rows, preds: scoring.score(rows, preds)["mem_regret"],
    row_features=features.row_features,
    feature_names=features.feature_names,
    probs_key="mem_probs",
    target_text=prompting.target_text,
    parse_answer=prompting.parse_answer,
)
