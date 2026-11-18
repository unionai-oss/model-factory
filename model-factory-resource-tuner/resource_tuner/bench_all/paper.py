"""Export a finished release for the paper's offline analysis (round 22).

The stations score every arm one way. A paper needs more than that: confidence
intervals that resample tasks (a run, its siblings and its K8s twin are not
independent), the pair metric split by whether the need moved, the cost
model's sensitivity to its penalties, and what left-truncation cuts from a
prompt. All of it is cheap once the laptop has every arm's request per run and
the rows' scoring columns. Those live in the cluster's object store, which the
laptop cannot read, so ``paper_export`` returns them inline: gzipped JSON,
base64.

``scoring`` runs unchanged on the exported rows (they keep every column it
reads), so the offline numbers reproduce the scorecards exactly.
"""

from __future__ import annotations

import base64
import gzip
import json
import re
from random import Random

import flyte
import flyte.io

from ..bench.common import read_rows
from . import dataset, prompting
from .envs import allbench_env

#: The columns ``scoring`` reads, and the ones the analysis groups by.
SCORE_COLS = ("run_id", "task_id", "twin_of", "split", "platform", "staging", "source", "family", "n_history",
              "need_mib", "t_peak_frac", "need_cores", "mean_cores", "cpu_series_json", "peak_gpu_gib", "sharded",
              "gpu_imbalance", "label_mem", "label_cpu", "label_gpu", "input_bytes_json", "wall_s")
#: Prompts kept whole, per (platform, source, staging), for tokenizing offline.
PROMPTS_PER_CELL = 60
#: Where a prompt's load-bearing lines are, as characters from its end (left
#: truncation keeps the end): a marker cut from a prompt is a marker the model never saw.
MARKERS = ("import boto3", "medium: Memory", "initContainers", "### This run's inputs", "### Job manifest",
           "### Earlier runs", "def ")


def _clean(v):
    if isinstance(v, float) and v != v:
        return None
    if hasattr(v, "item"):
        return v.item()
    return v


def export_rows(rows: list[dict]) -> list[dict]:
    return [{k: _clean(r.get(k)) for k in SCORE_COLS} for r in rows]


def prompt_stats(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Every evaluated prompt's length and marker offsets; a stratified sample whole."""
    stats, cells = [], {}
    for r in rows:
        msgs = prompting.messages(r)
        text = msgs[1]["content"]
        st = {"run_id": r["run_id"], "chars": len(text) + len(msgs[0]["content"]),
              "user_chars": len(text), "from_end": {}}
        for m in MARKERS:
            i = text.find(m)
            if i >= 0:
                st["from_end"][m] = len(text) - i
        stats.append(st)
        cells.setdefault((dataset.platform_of(r), r["source"], r.get("staging") or ""), []).append(r)
    sample = []
    for key in sorted(cells):
        mine = sorted(cells[key], key=lambda r: r["run_id"])
        Random(11).shuffle(mine)
        for r in mine[:PROMPTS_PER_CELL]:
            msgs = prompting.messages(r)
            sample.append({"run_id": r["run_id"], "system": msgs[0]["content"], "user": msgs[1]["content"]})
    return stats, sample


async def _arm_tables(d: flyte.io.Dir) -> tuple[dict, dict]:
    import os

    import pandas as pd

    local = await d.download()
    with open(os.path.join(local, "meta.json")) as f:
        meta = json.load(f)
    preds = pd.read_parquet(os.path.join(local, "predictions.parquet"))
    picks: dict[str, dict] = {}
    for r in preds.to_dict("records"):
        picks.setdefault(r["variant"], {})[r["run_id"]] = [int(r["mem_idx"]), int(r["cpu_idx"]), int(r["gpu_idx"]),
                                                           str(r.get("raw") or "")]
    return picks, meta


@allbench_env.task(report=False)
async def paper_export(corpus: flyte.io.File, arm_dirs: list[flyte.io.Dir], part: str = "picks") -> str:
    """``part``: ``rows`` (the evaluated rows' scoring columns), ``picks`` (every
    variant's request per evaluated run, metas, and the hosted arm's raw
    answers), or ``prompts`` (lengths, marker offsets, a sample whole)."""
    rows = [r for r in await read_rows(corpus) if dataset.usable(r) and r["split"] in ("val", "test", "ood")]
    if part == "rows":
        out: object = export_rows(rows)
    elif part == "prompts":
        stats, sample = prompt_stats(rows)
        out = {"stats": stats, "sample": sample}
    else:
        ids = {r["run_id"] for r in rows}
        picks, metas = {}, {}
        for d in arm_dirs:
            p, meta = await _arm_tables(d)
            for v, by_id in p.items():
                keep_raw = v.startswith("zeroshot")
                picks[v] = {rid: (x if keep_raw else x[:3]) for rid, x in by_id.items() if rid in ids}
                metas[v] = meta
        out = {"picks": picks, "metas": metas}
    return base64.b64encode(gzip.compress(json.dumps(out, default=float).encode())).decode()


@allbench_env.task(report=False)
async def paper_file(f: flyte.io.File) -> str:
    """A station's JSON output (the counterfactual's), inline: gzipped, base64."""
    path = await f.download()
    with open(path, "rb") as fh:
        return base64.b64encode(gzip.compress(fh.read())).decode()


_IMPORT = re.compile(r"^\s*(?:import|from)\s+([\w.]+)", re.M)


def code_audit(rows: list[dict]) -> dict:
    """Round 24: how much of test is code seen in training, and what each family imports.

    ``seen``: every usable test/ood Flyte run whose ``task_code`` is identical to
    some training task's (same corpus). ``families``: tasks and distinct programs
    per family and split. ``imports``: the modules each family's code imports, and
    for held-out families the ones no training family imports."""
    flyte_rows = [r for r in rows if dataset.usable(r) and dataset.platform_of(r) == "flyte"]
    train_code: dict[str, set[str]] = {}
    for r in flyte_rows:
        if r["split"] == "train":
            train_code.setdefault(r["source"], set()).add(r["task_code"])
    seen = {r["run_id"]: r["task_code"] in train_code.get(r["source"], set())
            for r in flyte_rows if r["split"] in ("val", "test", "ood")}
    fam: dict[str, dict] = {}
    imports: dict[str, set[str]] = {}
    for r in flyte_rows:
        k = f"{r['source']}:{r['family']}"
        f = fam.setdefault(k, {"split": set(), "tasks": set(), "programs": set()})
        f["split"].add(r["split"]), f["tasks"].add(r["task_id"]), f["programs"].add(r["task_code"])
        imports.setdefault(k, set()).update(_IMPORT.findall(r["task_code"]))
    train_fams = [k for k, f in fam.items() if "ood" not in f["split"]]
    train_mods = set().union(*(imports[k] for k in train_fams))
    train_top = {m.split(".")[0] for m in train_mods}
    return {
        "seen": seen,
        "families": {k: {"split": sorted(f["split"]), "tasks": len(f["tasks"]), "programs": len(f["programs"])}
                     for k, f in sorted(fam.items())},
        "imports": {k: sorted(v) for k, v in sorted(imports.items())},
        "new_modules": {k: sorted(imports[k] - train_mods) for k in fam if k not in train_fams},
        "new_packages": {k: sorted({m.split(".")[0] for m in imports[k]} - train_top) for k in fam if k not in train_fams},
    }


@allbench_env.task(report=False)
async def paper_audit(corpus: flyte.io.File) -> str:
    """``code_audit`` on a release's corpus, inline: gzipped JSON, base64."""
    rows = await read_rows(corpus)
    return base64.b64encode(gzip.compress(json.dumps(code_audit(rows)).encode())).decode()
