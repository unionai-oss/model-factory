"""Data station: ingest seed tasks, curate, oracle-verify, publish artifact.

Canonical task schema (parquet columns):

- ``task_id``              unique id
- ``question``             natural-language problem statement
- ``function_declaration`` required solution signature (graders import it)
- ``tests``                hidden pytest suite (``from solution import ...``)
- ``reference_solution``   oracle solution (must pass its own tests)
- ``difficulty``           easy/medium/hard (as labeled upstream)
- ``n_tests``              number of test functions
- ``source``               seed dataset name or "synthetic"
- ``split``                "train" | "heldout"

Curation gates (all automated — the human gate is the approval condition in
the pipeline): schema mapping, dedup, min test count, reference solution
passes its own tests in the sandbox (execution as the oracle).
"""

from __future__ import annotations

import asyncio
import hashlib
import json

import flyte
import flyte.io
import flyte.report

from . import reporting
from .config import SEED_DATASET, ARTIFACT_RL_DATASET, get_profile
from .envs import cpu_env
from .rewards import count_test_functions
from .sandbox import run_solution_against_tests

_ORACLE_CONCURRENCY = 8


def _extract_declaration(test_info: object) -> str | None:
    """KodCode's test_info carries the required function signature."""
    try:
        if isinstance(test_info, str):
            test_info = json.loads(test_info.replace("'", '"'))
        if isinstance(test_info, (list, tuple)) and test_info:
            decl = test_info[0].get("function_declaration")
            return str(decl) if decl else None
    except Exception:
        pass
    return None


async def _oracle_verify(rows: list[dict]) -> list[bool]:
    """Reference solution must pass its own tests. Runs in worker threads."""
    sem = asyncio.Semaphore(_ORACLE_CONCURRENCY)

    async def check(row: dict) -> bool:
        async with sem:
            result = await asyncio.to_thread(
                run_solution_against_tests, row["reference_solution"], row["tests"]
            )
            return result.passed

    return list(await asyncio.gather(*(check(r) for r in rows)))


@cpu_env.task(report=True, cache="auto")
async def ingest_and_curate(profile_name: str = "smoke") -> flyte.io.File:
    """Pull the seed dataset, curate it, and emit a *candidate* dataset file.

    The output is deliberately NOT an artifact: only after the human data
    validation gate approves does `publish_dataset` mint the artifact version
    that triggers training.
    """
    import pandas as pd
    from datasets import load_dataset

    profile = get_profile(profile_name)
    want = profile.train_tasks + profile.eval_tasks
    # Over-fetch to survive filtering losses.
    fetch = min(want * 3, 10000)

    ds = load_dataset(SEED_DATASET, split=f"train[:{fetch}]")
    raw = ds.to_list()

    rows, seen, dropped = [], set(), {"dup": 0, "few_tests": 0, "schema": 0, "oracle": 0}
    for r in raw:
        question = (r.get("question") or "").strip()
        solution = (r.get("solution") or "").strip()
        tests = (r.get("test") or "").strip()
        if not (question and solution and tests):
            dropped["schema"] += 1
            continue
        key = hashlib.sha1(question.encode()).hexdigest()
        if key in seen:
            dropped["dup"] += 1
            continue
        seen.add(key)
        n_tests = count_test_functions(tests)
        if n_tests < profile.min_test_functions:
            dropped["few_tests"] += 1
            continue
        rows.append(
            {
                "task_id": str(r.get("question_id") or key[:12]),
                "question": question,
                "function_declaration": _extract_declaration(r.get("test_info")) or "",
                "tests": tests,
                "reference_solution": solution,
                "difficulty": str(r.get("gpt_difficulty") or "unknown"),
                "n_tests": n_tests,
                "source": SEED_DATASET,
            }
        )
        if len(rows) >= want * 2:
            break

    # Execution as the oracle: drop tasks whose reference solution fails.
    verdicts = await _oracle_verify(rows)
    verified = [r for r, ok in zip(rows, verdicts) if ok]
    dropped["oracle"] = len(rows) - len(verified)
    verified = verified[:want]

    for i, r in enumerate(verified):
        r["split"] = "heldout" if i < profile.eval_tasks else "train"

    df = pd.DataFrame(verified)
    out = "/tmp/rl_tasks_candidate.parquet"
    df.to_parquet(out, index=False)

    # --- data card report (bottleneck 1: what the human gate inspects) ---
    n_train = int((df["split"] == "train").sum())
    n_heldout = int((df["split"] == "heldout").sum())
    body = reporting.stats_row(
        {
            "curated tasks": len(df),
            "train": n_train,
            "heldout": n_heldout,
            "dropped (dup)": dropped["dup"],
            "dropped (<min tests)": dropped["few_tests"],
            "dropped (oracle fail)": dropped["oracle"],
            "dropped (schema)": dropped["schema"],
        }
    )
    body += "<h3>Difficulty mix</h3>" + reporting.table(
        ["difficulty", "count"],
        [[k, int(v)] for k, v in df["difficulty"].value_counts().items()],
    )
    sample = df.sample(min(8, len(df)), random_state=7)
    body += "<h3>Random samples (inspect before approving)</h3>" + reporting.table(
        ["task_id", "difficulty", "n_tests", "question", "tests (head)"],
        [
            [
                r.task_id,
                r.difficulty,
                r.n_tests,
                r.question[:300],
                r.tests[:300],
            ]
            for r in sample.itertuples()
        ],
    )
    await flyte.report.replace.aio(reporting.page("Data card: curated RL tasks", body))
    await flyte.report.flush.aio()

    if len(df) == 0:
        raise flyte.errors.NonRecoverableError("curation produced zero tasks")
    return await flyte.io.File.from_local(out)


@cpu_env.task
async def publish_dataset(dataset: flyte.io.File, note: str = "") -> flyte.io.File:
    """Mint an approved dataset as a versioned `rl-tasks-dataset` artifact.

    New versions of this artifact are what kick training off via the
    OnArtifact trigger (see pipeline.py).
    """
    import flyte.artifacts as artifacts

    local = await dataset.download()
    f = await flyte.io.File.from_local(local)
    meta = artifacts.Metadata(
        name=ARTIFACT_RL_DATASET,
        description=f"Curated + human-approved RL coding tasks. {note}".strip(),
        kind="data",
    )
    return artifacts.new(f, meta)


@cpu_env.task
async def merge_datasets(base: flyte.io.File, extra: flyte.io.File) -> flyte.io.File:
    """Merge synthetic tasks into the current dataset (dedup by question)."""
    import pandas as pd

    df_a = pd.read_parquet(await base.download())
    df_b = pd.read_parquet(await extra.download())
    merged = (
        pd.concat([df_a, df_b], ignore_index=True)
        .drop_duplicates(subset=["question"], keep="first")
        .reset_index(drop=True)
    )
    out = "/tmp/rl_tasks_merged.parquet"
    merged.to_parquet(out, index=False)
    return await flyte.io.File.from_local(out)
