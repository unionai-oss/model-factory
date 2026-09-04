"""Teacher-generated synthetic tasks, verified by the execution oracle.

The template families (templates.py) are in-distribution but narrow. The
teacher LLM widens the corpus with NOVEL workloads — but a teacher's guess
about its own code's footprint would be exactly the labeling bias this
project exists to remove. So the labels never come from the teacher:

    teacher writes code → AST safety screen → harness pod EXECUTES it
    with generous resources and measures peak RSS + avg CPU (the oracle)
    → only tasks that ran clean, in-bounds, become corpus records.

Mirrors basic-model-factory's oracle-verified synthetic station: only
verifiable samples survive curation.
"""

from __future__ import annotations

import ast
import json
import re
import textwrap

ALLOWED_IMPORTS = {
    "numpy", "pandas", "sklearn", "torch", "time", "math", "random",
    "itertools", "collections", "functools", "json", "string", "statistics",
}
FORBIDDEN_NAMES = {
    "open", "exec", "eval", "compile", "__import__", "input", "breakpoint",
}
FORBIDDEN_MODULES = {
    "os", "sys", "subprocess", "socket", "shutil", "pathlib", "requests",
    "urllib", "http", "multiprocessing", "ctypes", "pickle", "importlib",
}

GENERATION_PROMPT = """\
You generate realistic Python workloads for benchmarking a workflow \
system's resource estimation. Write ONE self-contained Python module that:

1. defines `def run() -> dict:` doing realistic {family_hint} work \
(construct synthetic in-memory data — no files, no network),
2. holds a significant, roughly steady memory footprint for at least \
{duration_s} seconds by repeating its core work in a timed loop \
(`time.monotonic()` deadline pattern),
3. returns a small dict of result stats,
4. imports only from: {allowed},
5. targets roughly {target_mib} MiB of peak memory (choose data sizes \
accordingly) and about {target_cores} CPU core(s).

Also provide a one-line human description of the workload's input profile.

Respond with ONLY this JSON (no code fences):
{{"description": "<one line>", "code": "<the module source, \\n-escaped>"}}"""

FAMILY_HINTS = [
    ("data_engineering", "tabular ETL (pandas joins/groupbys/window ops)"),
    ("data_science", "statistical model fitting (sklearn)"),
    ("ml_training", "small neural-net training loop (torch, CPU)"),
    ("batch_inference", "vectorized scoring/embedding math (numpy)"),
    ("etl", "record parsing and aggregation (stdlib)"),
]


class RejectedTask(ValueError):
    """Why a teacher sample failed the safety/shape screen."""


def parse_teacher_response(text: str) -> tuple[str, str]:
    """Teacher completion → (description, code). Raises RejectedTask."""
    text = re.sub(r"<think>.*?(?:</think>|$)", "", text, flags=re.DOTALL)
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise RejectedTask("no JSON object in teacher response")
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        raise RejectedTask(f"bad JSON: {e}")
    code, desc = obj.get("code"), obj.get("description")
    if not code or not isinstance(code, str):
        raise RejectedTask("missing code")
    return (str(desc or "teacher-generated workload"), textwrap.dedent(code))


def validate_task_code(code: str) -> None:
    """AST screen: parseable, defines run(), imports only from the
    allowlist, touches no filesystem/network/process surface. Raises
    RejectedTask. This is a curation filter for OUR OWN teacher's output
    running in an isolated pod — not a general sandbox."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise RejectedTask(f"syntax error: {e}")
    has_run = any(
        isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "run"
        for n in tree.body
    )
    if not has_run:
        raise RejectedTask("no top-level def run()")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = (
                [a.name for a in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
            )
            for name in names:
                root = name.split(".")[0]
                if root in FORBIDDEN_MODULES:
                    raise RejectedTask(f"forbidden import {root!r}")
                if root not in ALLOWED_IMPORTS:
                    raise RejectedTask(f"import {root!r} not in allowlist")
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            raise RejectedTask(f"forbidden builtin {node.id!r}")
        if isinstance(node, ast.Attribute) and node.attr in ("system", "popen", "fork"):
            raise RejectedTask(f"forbidden attribute .{node.attr}")


def curate_measurement(measured: dict, min_mib: float = 96, max_mib: float = 12288,
                       min_s: float = 30, max_s: float = 400) -> str | None:
    """Oracle result → rejection reason, or None if the sample survives.

    Bounds keep the corpus schedulable (max) and the metrics pipeline
    honest (min duration: attempts under ~30s have no pod-metric samples).
    """
    if not measured.get("ok"):
        return f"execution failed: {measured.get('error', '')[:200]}"
    if not (min_mib <= measured["peak_rss_mib"] <= max_mib):
        return f"peak {measured['peak_rss_mib']:.0f}MiB out of bounds"
    if not (min_s <= measured["duration_s"] <= max_s):
        return f"duration {measured['duration_s']:.0f}s out of bounds"
    return None


def synthetic_record(
    task_id: str, family: str, code: str, description: str, measured: dict
) -> dict:
    """Oracle-labeled corpus row (same schema as template records — the
    policy cannot tell them apart, which is the point)."""
    return {
        "task_id": task_id,
        "family": family,
        "source_code": code,  # policy sees the plain module: cold-start UC2
        "harness_code": code,
        "input_profile": description,
        "params_json": json.dumps({"synthetic": True}),
        "true_peak_memory_mib": float(measured["peak_rss_mib"]),
        "true_cpu_cores": max(1.0, round(float(measured.get("cpu_avg_cores", 1.0)), 1)),
        "true_gpu_mem_mib": 0.0,  # oracle pods are CPU-only
        "duration_s": int(measured["duration_s"]),
        "split": "train",
    }
