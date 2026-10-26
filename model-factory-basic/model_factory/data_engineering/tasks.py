"""Data station: the held-out split, each day's seed tasks, the release.

Canonical task schema (parquet columns, see ``contracts.DATASET_COLUMNS``):
``task_id, question, function_declaration, tests, reference_solution,
difficulty, n_tests, source, split``.

Where the data comes from, and why it cannot leak
--------------------------------------------------
The seed dataset (KodCode-Light-RL-10K) is cut in two by row index:

    rows [0, heldout_pool)        -> `mf-heldout-tasks`  (built once)
    rows [heldout_pool, N)        -> a window per day   -> `mf-seed-tasks[date]`

The held-out split is its own artifact with no `date` partition, so every day
is scored on the same version of it. Each day's training window rotates by
date through the rest of the dataset — a new day brings new problems — and is
additionally filtered against the held-out questions, and `assemble_dataset`
refuses to publish a release that still contains one. Training never receives
the held-out artifact at all.

Curation gates (all automated): schema mapping, dedup, min test count,
reference solution passes its own tests in the sandbox (execution as the
oracle). The one HUMAN gate lives in ``assemble_dataset``.
"""

from __future__ import annotations

import asyncio
import json
from datetime import date as date_type
from datetime import datetime

import flyte
import flyte.io
import flyte.report
import pandas as pd

from ..config import SEED_DATASET, FactoryProfile, get_profile
from ..contracts import DATASET_COLUMNS, find_leaks, heldout_fingerprint, question_key
from ..shared import reporting
from ..shared.gates import gate
from ..shared.rewards import count_test_functions
from ..shared.sandbox import run_solution_against_tests
from .envs import de_cpu_env

_ORACLE_CONCURRENCY = 8

#: Day 0 of the rotating training window. Any fixed date works; it only has to
#: never change, or every day's window moves.
WINDOW_EPOCH = date_type(2026, 1, 1)


def day_window(day: date_type | datetime, n_rows: int, heldout_pool: int, size: int) -> list[int]:
    """Row indices of one day's training window.

    Windows tile ``[heldout_pool, n_rows)`` in date order and wrap around, so
    consecutive days see different problems and no day ever sees a held-out
    row.
    """
    if isinstance(day, datetime):
        day = day.date()
    span = n_rows - heldout_pool
    if span <= 0:
        raise ValueError(f"no rows left for training: n_rows={n_rows}, heldout_pool={heldout_pool}")
    size = min(size, span)
    start = ((day - WINDOW_EPOCH).days * size) % span
    return [heldout_pool + (start + i) % span for i in range(size)]


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


def normalize_rows(raw: list[dict], min_tests: int, exclude: set[str] = frozenset()) -> tuple[list[dict], dict]:
    """Schema-map, dedup and test-count filter seed rows (no execution yet).

    ``exclude`` holds `question_key`s that must never appear (the held-out set).
    """
    rows, seen = [], set()
    dropped = {"dup": 0, "few_tests": 0, "schema": 0, "heldout": 0}
    for r in raw:
        question = (r.get("question") or "").strip()
        solution = (r.get("solution") or "").strip()
        tests = (r.get("test") or "").strip()
        if not (question and solution and tests):
            dropped["schema"] += 1
            continue
        key = question_key(question)
        if key in exclude:
            dropped["heldout"] += 1
            continue
        if key in seen:
            dropped["dup"] += 1
            continue
        seen.add(key)
        n_tests = count_test_functions(tests)
        if n_tests < min_tests:
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
    return rows, dropped


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


def _seed_dataset():
    from datasets import load_dataset

    return load_dataset(SEED_DATASET, split="train")


async def _curate(raw: list[dict], profile: FactoryProfile, want: int, split: str, exclude=frozenset()):
    rows, dropped = normalize_rows(raw, profile.min_test_functions, exclude)
    rows = rows[: want * 2]  # bound the oracle's work; over-fetch survives its losses
    verdicts = await _oracle_verify(rows)
    verified = [r for r, ok in zip(rows, verdicts) if ok]
    dropped["oracle"] = len(rows) - len(verified)
    verified = verified[:want]
    for r in verified:
        r["split"] = split
    return pd.DataFrame(verified, columns=DATASET_COLUMNS), dropped


async def _data_card(title: str, df: pd.DataFrame, stats: dict) -> None:
    body = reporting.stats_row(stats)
    if len(df):
        body += "<h3>Difficulty mix</h3>" + reporting.table(
            ["difficulty", "count"],
            [[k, int(v)] for k, v in df["difficulty"].value_counts().items()],
        )
        sample = df.sample(min(8, len(df)), random_state=7)
        body += "<h3>Random samples (inspect before approving)</h3>" + reporting.table(
            ["task_id", "difficulty", "n_tests", "question", "tests (head)"],
            [[r.task_id, r.difficulty, r.n_tests, r.question[:300], r.tests[:300]] for r in sample.itertuples()],
        )
    await flyte.report.replace.aio(reporting.page(title, body))
    await flyte.report.flush.aio()


@de_cpu_env.task(report=True, cache="auto")
async def curate_heldout(profile_name: str = "daily") -> flyte.io.File:
    """The fixed held-out split: the first rows of the seed dataset, curated.

    Built once — the factory declares it with no partitions — so every day is
    scored against the same version.
    """
    profile = get_profile(profile_name)
    raw = _seed_dataset().select(range(profile.heldout_pool)).to_list()
    df, dropped = await _curate(raw, profile, profile.eval_tasks, "heldout")
    if len(df) < profile.eval_tasks:
        raise flyte.errors.NonRecoverableError(
            f"held-out pool of {profile.heldout_pool} rows yielded {len(df)} < {profile.eval_tasks} tasks"
        )
    out = "/tmp/heldout_tasks.parquet"
    df.to_parquet(out, index=False)
    await _data_card(
        "Held-out split (fixed)",
        df,
        {"held-out tasks": len(df), "fingerprint": heldout_fingerprint(df["task_id"]), **{f"dropped ({k})": v for k, v in dropped.items()}},
    )
    return await flyte.io.File.from_local(out)


@de_cpu_env.task(report=True, cache="auto")
async def ingest_and_curate(
    heldout: flyte.io.File, profile_name: str = "daily", date: datetime | None = None
) -> flyte.io.File:
    """One day's curated, oracle-verified training tasks.

    ``date`` is the release partition being built. A factory injects it
    automatically because the parameter is named like the ``Daily``
    dimension — and it arrives as a ``datetime``, not a string. It picks the
    day's window of the seed dataset (see `day_window`).
    """
    profile = get_profile(profile_name)
    day = date or datetime(WINDOW_EPOCH.year, WINDOW_EPOCH.month, WINDOW_EPOCH.day)
    ds = _seed_dataset()
    indices = day_window(day, len(ds), profile.heldout_pool, profile.train_tasks * 3)

    held = pd.read_parquet(await heldout.download())
    exclude = {question_key(q) for q in held["question"]}
    df, dropped = await _curate(ds.select(indices).to_list(), profile, profile.train_tasks, "train", exclude)
    if len(df) == 0:
        raise flyte.errors.NonRecoverableError("curation produced zero tasks")

    out = "/tmp/seed_tasks.parquet"
    df.to_parquet(out, index=False)
    await _data_card(
        f"Seed tasks for {day:%Y-%m-%d}",
        df,
        {
            "curated tasks": len(df),
            "seed rows": f"{indices[0]}..{indices[-1]}",
            **{f"dropped ({k})": v for k, v in dropped.items()},
        },
    )
    return await flyte.io.File.from_local(out)


@de_cpu_env.task(report=True)
async def assemble_dataset(
    seed: flyte.io.File,
    synthetic: flyte.io.File,
    heldout: flyte.io.File,
    auto_approve: bool = False,
) -> flyte.io.File:
    """Merge seed + synthetic tasks, check for leaks, hold the HUMAN data gate.

    This is the station the factory publishes `rl-tasks-dataset` from, so the
    checks here are what keep held-out tasks and unreviewed data out of
    training: the build does not finish — and therefore no dataset version
    exists — until both pass.
    """
    df_seed = pd.read_parquet(await seed.download())
    df_syn = pd.read_parquet(await synthetic.download())
    held = pd.read_parquet(await heldout.download())
    merged = (
        pd.concat([df_seed, df_syn], ignore_index=True)
        .drop_duplicates(subset=["question"], keep="first")
        .reset_index(drop=True)
    )
    leaks = find_leaks(merged["question"], held["question"])
    if leaks:
        raise flyte.errors.NonRecoverableError(
            f"held-out leak: {len(leaks)} release task(s) are in the held-out split, "
            f"e.g. {merged.iloc[leaks[0]]['task_id']}"
        )
    out = "/tmp/rl_tasks_release.parquet"
    merged.to_parquet(out, index=False)

    n_syn = int((merged["source"] == "synthetic").sum())
    body = reporting.stats_row(
        {
            "release tasks": len(merged),
            "from seed": len(merged) - n_syn,
            "synthetic": n_syn,
            "held-out overlap": 0,
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
        f"Release candidate: **{len(merged)}** tasks ({n_syn} synthetic), no held-out "
        "overlap. Inspect the data cards on `ingest_and_curate` and "
        "`generate_synthetic_tasks`.\n\n**Approve this dataset for release?**",
        auto_approve,
    )
    if not approved:
        raise flyte.errors.NonRecoverableError("dataset rejected at data-validation gate")
    return await flyte.io.File.from_local(out)
