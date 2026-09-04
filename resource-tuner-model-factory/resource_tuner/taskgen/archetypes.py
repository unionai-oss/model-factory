"""Archetype-scale synthetic generation: 10⁵ pipelines from 10² teacher calls.

Teacher-per-task doesn't scale (one 27B llama.cpp replica ≈ a minute per
generation → 100k tasks ≈ two months), so scale comes from factoring the
work:

    teacher writes an ARCHETYPE: novel workload code parameterized by a
    PARAMS dict, with declared numeric ranges and the param that
    dominates memory
      → AST safety screen (same rules as single-task synthetic)
      → ORACLE CALIBRATION: the harness runs the archetype at K sampled
        parameter points and measures real peak RSS / avg CPU
      → a per-archetype linear fit (peak vs memory-param) labels
        INSTANTIATIONS: variants sampled across the param ranges

Labels stay measurement-anchored (the teacher's own guess is never used);
interpolation error is bounded by the calibration fit and visible in the
corpus (`label_source` = measured | fitted).
"""

from __future__ import annotations

import ast
import functools
import json
import math
import re
from random import Random

from .synthetic import RejectedTask, validate_task_code

ARCHETYPE_PROMPT = """\
You generate realistic, PARAMETERIZED Python workloads for benchmarking a
workflow system's resource estimation. Write ONE self-contained module that:

1. starts with a literal dict of numeric knobs:  PARAMS = {{...}}
2. defines `def run() -> dict:` doing realistic {family_hint} work, whose
   memory footprint and runtime are driven by PARAMS (construct synthetic
   in-memory data — no files, no network),
3. holds a roughly steady memory footprint for at least {duration_s}
   seconds via a `time.monotonic()` deadline loop,
4. returns a small dict of result stats,
5. imports only from: {allowed}. Keep the module under 70 lines.

Be creative: unusual-but-plausible data shapes, mixed libraries, realistic
variable names — this corpus trains a model to read arbitrary pipeline code.

Respond with ONLY this JSON (no code fences, keep it compact):
{{"description": "<one line>",
 "param_ranges": {{"<param>": [<lo>, <hi>], ...}},
 "memory_param": "<the PARAMS key that most drives peak memory>",
 "code": "<the module source, \\n-escaped>"}}"""


class Archetype:
    def __init__(self, description: str, code: str, param_ranges: dict, memory_param: str):
        self.description = description
        self.code = code
        self.param_ranges = param_ranges
        self.memory_param = memory_param


def parse_archetype_response(text: str) -> Archetype:
    """Teacher completion → validated Archetype. Raises RejectedTask."""
    text = re.sub(r"<think>.*?(?:</think>|$)", "", text, flags=re.DOTALL)
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise RejectedTask("no JSON object in teacher response")
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        raise RejectedTask(f"bad JSON: {e}")
    code = obj.get("code")
    ranges = obj.get("param_ranges")
    mem = obj.get("memory_param")
    if not code or not isinstance(code, str):
        raise RejectedTask("missing code")
    if not isinstance(ranges, dict) or not ranges:
        raise RejectedTask("missing param_ranges")
    for k, v in ranges.items():
        if (
            not isinstance(v, (list, tuple))
            or len(v) != 2
            or not all(isinstance(x, (int, float)) for x in v)
            or v[0] > v[1]
            or v[0] <= 0
        ):
            raise RejectedTask(f"bad range for {k!r}: {v!r}")
    if mem not in ranges:
        raise RejectedTask(f"memory_param {mem!r} not in param_ranges")
    validate_task_code(code)
    _find_params_span(code, frozenset(ranges))  # raises if PARAMS malformed
    return Archetype(str(obj.get("description") or "archetype"), code, ranges, mem)


@functools.lru_cache(maxsize=1024)  # instantiate() calls this ~10^5 times
def _find_params_span(code: str, required: frozenset[str]) -> tuple[int, int, dict]:
    """Locate the module-level `PARAMS = {...}` assignment.

    Returns (start_line, end_line) 1-based inclusive and the literal dict.
    """
    tree = ast.parse(code)
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "PARAMS"
        ):
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, SyntaxError):
                raise RejectedTask("PARAMS is not a literal dict")
            if not isinstance(value, dict):
                raise RejectedTask("PARAMS is not a dict")
            missing = required - set(value)
            if missing:
                raise RejectedTask(f"PARAMS missing declared params {sorted(missing)}")
            return node.lineno, node.end_lineno or node.lineno, value
    raise RejectedTask("no module-level PARAMS = {...}")


def instantiate(archetype: Archetype, values: dict) -> str:
    """Rewrite the PARAMS assignment with sampled values."""
    start, end, base = _find_params_span(archetype.code, frozenset(archetype.param_ranges))
    merged = {**base, **values}
    lines = archetype.code.splitlines()
    lines[start - 1 : end] = [f"PARAMS = {merged!r}"]
    return "\n".join(lines) + "\n"


def sample_params(archetype: Archetype, rng: Random) -> dict:
    """One variant's params: memory_param log-uniform (footprints span
    orders of magnitude), the rest uniform. Integer bounds stay integer."""
    out: dict = {}
    for name, (lo, hi) in archetype.param_ranges.items():
        if name == archetype.memory_param and lo > 0:
            v = math.exp(rng.uniform(math.log(lo), math.log(hi)))
        else:
            v = rng.uniform(lo, hi)
        if isinstance(lo, int) and isinstance(hi, int):
            v = max(int(round(v)), lo)
        out[name] = v
    return out


def calibration_points(archetype: Archetype, k: int = 3) -> list[dict]:
    """K param sets spanning the memory range (log-spaced); other params
    at their midpoints, so the fit isolates the memory driver."""
    lo, hi = archetype.param_ranges[archetype.memory_param]
    mids = {
        n: (int(round((a + b) / 2)) if isinstance(a, int) and isinstance(b, int) else (a + b) / 2)
        for n, (a, b) in archetype.param_ranges.items()
    }
    points = []
    for i in range(k):
        f = i / max(k - 1, 1)
        v = math.exp(math.log(lo) + f * (math.log(hi) - math.log(lo)))
        if isinstance(lo, int) and isinstance(hi, int):
            v = max(int(round(v)), 1)
        points.append({**mids, archetype.memory_param: v})
    return points


class FootprintFit:
    """peak_mib ≈ a + b·memory_param (least squares over calibration).

    Linear in the declared driver is deliberately simple: it labels
    interpolations honestly enough for RL (sim jitter already randomizes
    the boundary), and its residuals are visible via label_source.
    """

    def __init__(self, points: list[tuple[float, dict]]):
        # points: [(measured_peak_mib, param_values)]
        self.points = points

    def predict(self, memory_value: float, memory_param: str) -> float:
        xs = [p[1][memory_param] for p in self.points]
        ys = [p[0] for p in self.points]
        n = len(xs)
        if n == 1:
            return ys[0]
        mx, my = sum(xs) / n, sum(ys) / n
        denom = sum((x - mx) ** 2 for x in xs)
        b = 0.0 if denom == 0 else sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom
        a = my - b * mx
        lo, hi = min(ys), max(ys)
        # Never extrapolate below the smallest measurement's floor.
        return max(a + b * memory_value, min(lo * 0.5, lo - 64), 32.0)
