"""Data station: ingest seed tasks, curate, oracle-verify, assemble a release.

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

Curation gates (all automated): schema mapping, dedup, min test count,
reference solution passes its own tests in the sandbox (execution as the
oracle). The one HUMAN gate lives in ``assemble_dataset`` — the station that
the factory publishes `rl-tasks-dataset` from, so nothing downstream is built
from data a person has not signed off on.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import datetime

import flyte
import flyte.io
import flyte.report

from ..config import SEED_DATASET, get_profile
from ..shared import reporting
from ..shared.gates import gate
from ..shared.rewards import count_test_functions
from ..shared.sandbox import run_solution_against_tests
from .envs import de_cpu_env

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


@de_cpu_env.task(report=True, cache="auto")
async def ingest_and_curate(
    profile_name: str = "smoke", date: datetime | None = None
) -> flyte.io.File:
    """Pull the seed dataset, curate it, and emit the curated seed-task file.

    ``date`` is the release partition being built. A factory injects it
    automatically because the parameter is named like the ``Daily``
    dimension — and it arrives as a ``datetime``, not a string.
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


@de_cpu_env.task(report=True)
async def assemble_dataset(
    seed: flyte.io.File, synthetic: flyte.io.File, auto_approve: bool = False
) -> flyte.io.File:
    """Merge seed + synthetic tasks, then hold the HUMAN data-validation gate.

    This is the station the factory publishes `rl-tasks-dataset` from, so the
    gate here is what keeps unreviewed data from reaching training: the build
    does not finish — and therefore no dataset version exists — until a person
    approves. ``auto_approve=True`` switches the light off for smoke runs.
    """
    import pandas as pd

    df_seed = pd.read_parquet(await seed.download())
    df_syn = pd.read_parquet(await synthetic.download())
    merged = (
        pd.concat([df_seed, df_syn], ignore_index=True)
        .drop_duplicates(subset=["question"], keep="first")
        .reset_index(drop=True)
    )
    out = "/tmp/rl_tasks_release.parquet"
    merged.to_parquet(out, index=False)

    n_syn = int((merged["source"] == "synthetic").sum())
    body = reporting.stats_row(
        {
            "release tasks": len(merged),
            "from seed": len(merged) - n_syn,
            "synthetic": n_syn,
            "train": int((merged["split"] == "train").sum()),
            "heldout": int((merged["split"] == "heldout").sum()),
        }
    )
    body += "<h3>Random samples (inspect before approving)</h3>" + reporting.table(
        ["task_id", "source", "difficulty", "n_tests", "question"],
        [
            [r.task_id, r.source, r.difficulty, r.n_tests, r.question[:300]]
            for r in merged.sample(min(8, len(merged)), random_state=7).itertuples()
        ],
    )
    await flyte.report.replace.aio(reporting.page("Dataset release candidate", body))
    await flyte.report.flush.aio()

    approved = await gate(
        "approve-dataset",
        "## Data validation gate\n\n"
        f"Release candidate: **{len(merged)}** tasks ({n_syn} synthetic). "
        "Inspect the data card on this action and on `ingest_and_curate` "
        "(curation stats, difficulty mix, samples).\n\n"
        "**Approve this dataset for release?**",
        auto_approve,
    )
    if not approved:
        raise flyte.errors.NonRecoverableError("dataset rejected at data-validation gate")
    return await flyte.io.File.from_local(out)
