"""Station plumbing every RightSize bench shares: rows in and out of parquet,
an arm's predictions directory, and the hosted arms' stratified subsample.

Prediction tables are written with the bench's own column names
(``BenchSpec.a.pred_key`` etc.), so a RightSizeCPU predictions file says
``peak_idx`` and a RightSizeMem one ``mem_idx``.
"""

from __future__ import annotations

import json
import os
import tempfile
from random import Random

import flyte
import flyte.io
import pandas as pd

from . import dataset, hybrid
from .spec import BenchSpec


def write_parquet(rows: list[dict], name: str) -> str:
    path = os.path.join(tempfile.mkdtemp(prefix="rsb-"), name)
    pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]).to_parquet(path, index=False)
    return path


async def read_rows(f: flyte.io.File) -> list[dict]:
    return pd.read_parquet(await f.download()).to_dict("records")


def usable_by_split(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {s: [] for s in dataset.SPLITS}
    for r in rows:
        if dataset.usable(r):
            out[r["split"]].append(r)
    return out


def hosted_subsample(splits: dict[str, list[dict]], n: int) -> list[dict]:
    """Deterministic, family-stratified test+ood subsample (`n` per split)."""
    out = []
    for split in ("test", "ood"):
        by_fam: dict[str, list[dict]] = {}
        for r in sorted(splits[split], key=lambda r: r["run_id"]):
            by_fam.setdefault(r["family"], []).append(r)
        rng = Random(17)
        per = max(1, n // max(1, len(by_fam)))
        for fam in sorted(by_fam):
            out += rng.sample(by_fam[fam], min(per, len(by_fam[fam])))
    return out


def pred_rows(variant: str, preds: dict, rows: list[dict], spec: BenchSpec) -> list[dict]:
    split = {r["run_id"]: r["split"] for r in rows}
    a, b, pk = spec.a.pred_key, spec.b.pred_key, spec.probs_key
    return [
        {"variant": variant, "run_id": rid, "split": split.get(rid, ""), a: int(p[a]), b: int(p[b]),
         pk: json.dumps([round(float(x), 6) for x in p[pk]]) if p.get(pk) is not None else "",
         "raw": str(p.get("raw", ""))[:200]}
        for rid, p in preds.items()
    ]


def preds_by_variant(preds: pd.DataFrame, spec: BenchSpec) -> dict[str, dict[str, dict]]:
    a, b, pk = spec.a.pred_key, spec.b.pred_key, spec.probs_key
    out: dict[str, dict[str, dict]] = {}
    for r in preds.to_dict("records"):
        out.setdefault(r["variant"], {})[r["run_id"]] = {
            a: int(r[a]), b: int(r[b]), pk: json.loads(r[pk]) if r.get(pk) else None}
    return out


async def write_arm_dir(rows: list[dict], meta: dict) -> flyte.io.Dir:
    d = tempfile.mkdtemp(prefix="rsb-arm-")
    pd.DataFrame(rows).to_parquet(os.path.join(d, "predictions.parquet"), index=False)
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)
    return await flyte.io.Dir.from_local(d)


async def read_arm(d: flyte.io.Dir) -> tuple[pd.DataFrame, dict]:
    local = await d.download()
    preds = pd.read_parquet(os.path.join(local, "predictions.parquet"))
    with open(os.path.join(local, "meta.json")) as f:
        meta = json.load(f)
    return preds, meta


def tasks_for_release(release: str, sizes: dict[str, int]) -> int:
    """Tasks per family for a release, keyed by its name's prefix ("full-1" -> "full")."""
    kind = release.split("-", 1)[0]
    if kind not in sizes:
        raise ValueError(f"release {release!r}: prefix must be one of {sorted(sizes)}")
    return sizes[kind]


def profile_for_release(release: str) -> str:
    """The workload profile a release is generated with: ``big-*`` releases
    (RightSizeMem's scale) use the ``big`` profile, every other the default."""
    return "big" if release.split("-", 1)[0] == "big" else "default"


def shard(rows: list[dict], keep: tuple[str, ...], size: int, seed: int = 0) -> list[str]:
    """Rows -> JSON shards of `size`, shuffled so heavy families spread out."""
    order = list(range(len(rows)))
    Random(seed).shuffle(order)
    return [json.dumps([{k: rows[i][k] for k in keep} for i in order[s : s + size]])
            for s in range(0, len(order), size)]


def collect(results, measured: dict[str, dict]) -> int:
    """Fold shard results into `measured`; returns shards lost."""
    lost = 0
    for res in results:
        if isinstance(res, BaseException):
            lost += 1
            print(f"shard failed: {res}", flush=True)
            continue
        for m in json.loads(res):
            measured[m["run_id"]] = m
    return lost


async def hinted_corpus(rows: list[dict], fit, predict, spec: BenchSpec) -> tuple[flyte.io.File, dict]:
    """The corpus a stacked arm trains on: every usable row with the trees'
    prediction in a ``hint`` column (out-of-fold on train, ``hybrid.crossfit``)."""
    import time

    splits = usable_by_split(rows)
    eval_rows = [r for s in ("val", "test", "ood") for r in splits[s]]
    t0 = time.perf_counter()
    preds = hybrid.crossfit(splits["train"], splits["val"], eval_rows, fit, predict)
    hinted = hybrid.with_hints([r for s in dataset.SPLITS for r in splits[s]], preds, spec)
    meta = {"hint_anchor": hybrid.ANCHOR, "hint_folds": hybrid.FOLDS, "hinted_rows": len(preds),
            "crossfit_s": time.perf_counter() - t0, "hint_example": next((r["hint"] for r in hinted if r["hint"]), "")}
    return await flyte.io.File.from_local(write_parquet(hinted, "hinted.parquet")), meta


async def pooled_arm(rows: list[dict], arm_dirs: list, spec: BenchSpec) -> flyte.io.Dir:
    """Every distributional LLM variant of a release pooled with its trees (``hybrid.pool_variants``)."""
    variants: dict[str, dict[str, dict]] = {}
    for d in arm_dirs:
        preds, _meta = await read_arm(d)
        variants.update(preds_by_variant(preds, spec))
    if hybrid.ANCHOR not in variants:
        raise ValueError(f"pooled_arm: no {hybrid.ANCHOR!r} arm among the {len(arm_dirs)} built for this release")
    pooled, knobs = hybrid.pool_variants(rows, variants, spec)
    table = []
    for v, p in pooled.items():
        table += pred_rows(v, p, rows, spec)
    return await write_arm_dir(table, {"arm": "pool", "anchor": hybrid.ANCHOR, "members": hybrid.poolable(variants, spec),
                                       "knobs": knobs})
