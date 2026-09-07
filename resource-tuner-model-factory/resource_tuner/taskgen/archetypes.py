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
   data in memory{io_clause}; no network),
3. sustains its PEAK memory phase for at least {duration_s} seconds via a
   `time.monotonic()` deadline loop (multi-phase workloads are welcome —
   only the peak phase must hold), and FINISHES well under 10 minutes at
   the top of every declared range,
4. returns a small dict of result stats,
5. imports only from: {allowed}. Keep the module under 90 lines.
   NEVER import os, sys, pathlib, or shutil — for temporary files use
   `tempfile.TemporaryDirectory()` / `tempfile.NamedTemporaryFile()` and
   their own cleanup; a module importing os is DISCARDED.
{gpu_clause}
Size the declared ranges so peak memory spans roughly 150 MiB at the range
lows to AT MOST 8 GiB at the range highs — the calibration pods have 12 GiB
and anything above that is discarded.

Scenario to ground the workload (be faithful to it):
{scenario}

Realism over cleverness: variable names, data shapes, and steps a real
pipeline in that domain would have. Do NOT resemble these already-written
archetypes:
{avoid}

Respond with ONLY this JSON (no code fences, keep it compact):
{{"description": "<one line naming the domain and what drives its size>",
 "param_ranges": {{"<param>": [<lo>, <hi>], ...}},
 "memory_param": "<the PARAMS key that most drives peak memory>",
 "code": "<the module source, \\n-escaped>"}}"""

IO_CLAUSE = " or stage it through a tempfile between phases"
GPU_CLAUSE = """\
6. This is a GPU workload: move the model/tensors to CUDA behind a
   `torch.cuda.is_available()` guard (fall back to CPU with smaller sizes
   so the module always runs), use fp16 on CUDA, and make VRAM demand
   scale with the memory_param.
"""


def render_archetype_prompt(
    family_hint: str,
    scenario: str,
    avoid: list[str],
    allowed: str,
    duration_s: int = 60,
    gpu: bool = False,
) -> str:
    avoid_block = (
        "\n".join(f"- {a[:110]}" for a in avoid[-8:]) if avoid else "- (none yet)"
    )
    return ARCHETYPE_PROMPT.format(
        family_hint=family_hint,
        scenario=scenario,
        avoid=avoid_block,
        allowed=allowed,
        duration_s=duration_s,
        io_clause=IO_CLAUSE,
        gpu_clause=GPU_CLAUSE if gpu else "",
    )


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


def calibration_points(
    archetype: Archetype, k: int = 3, rng: Random | None = None, n_random: int = 0
) -> list[dict]:
    """K param sets spanning the memory range (log-spaced, other params at
    midpoints — isolates the memory driver) plus `n_random` FULL random
    draws (round-12: pinning everything at midpoints hid the footprint
    contribution of the non-memory params the sampler varies freely)."""
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
    if rng is not None:
        points += [sample_params(archetype, rng) for _ in range(n_random)]
    return points


def variant_profile(description: str, values: dict) -> str:
    """Per-variant input profile: the archetype's description PLUS the
    sampled scale. Round-12 fix — every variant used to share one static
    profile string, which trained the policy to IGNORE the input profile
    (the exact input-scale sensitivity serving needs)."""
    rendered = ", ".join(
        f"{k}={v:,}" if isinstance(v, int) else f"{k}={v:.3g}"
        for k, v in sorted(values.items())
    )
    return f"{description} — this invocation: {rendered}"


class ParamFit:
    """y ≈ exp(a + b·log(x)) — power-law fit of any measured quantity
    against the declared memory driver.

    Round-12 upgrade from linear: allocation curves are polynomial in
    their driver (rows×cols, n²…), and a log-log linear fit captures any
    single-exponent power law exactly where the old linear fit had
    structural bias at the range ends. Floors keep extrapolation sane.
    """

    def __init__(self, points: list[tuple[float, dict]], param: str, floor: float = 0.0):
        # points: [(measured_value, param_values)]
        self.points = [(y, p) for y, p in points if y > 0 and p.get(param, 0) > 0]
        self.param = param
        self.floor = floor

    def predict(self, value: float) -> float:
        ys = [y for y, _ in self.points]
        if not ys:
            return self.floor
        if len(ys) == 1 or value <= 0:
            return max(ys[0], self.floor)
        xs = [math.log(p[self.param]) for _, p in self.points]
        ls = [math.log(y) for y in ys]
        n = len(xs)
        mx, my = sum(xs) / n, sum(ls) / n
        denom = sum((x - mx) ** 2 for x in xs)
        b = 0.0 if denom == 0 else sum((x - mx) * (l - my) for x, l in zip(xs, ls)) / denom
        a = my - b * mx
        pred = math.exp(a + b * math.log(value))
        lo = min(ys)
        return max(pred, min(lo * 0.5, lo - 64), self.floor)


class FootprintFit:
    """Back-compat wrapper: the old (points).predict(value, param) shape,
    now backed by the power-law ParamFit with the 32MiB floor."""

    def __init__(self, points: list[tuple[float, dict]]):
        self.points = points

    def predict(self, memory_value: float, memory_param: str) -> float:
        return ParamFit(self.points, memory_param, floor=32.0).predict(memory_value)


def holdout_error(
    fit: "ParamFit", holdout: list[tuple[float, dict]]
) -> float | None:
    """Median relative |fitted − measured| / measured over holdout pods —
    the number that gates whether an archetype's labels are shippable."""
    errs = [
        abs(fit.predict(p[fit.param]) - y) / y
        for y, p in holdout
        if y > 0 and p.get(fit.param, 0) > 0
    ]
    errs.sort()
    return errs[len(errs) // 2] if errs else None
