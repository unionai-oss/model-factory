"""Trees x language models: the two hybrids every RightSize bench runs (round 21).

The benches' two strongest families fail differently. Quantile trees over
scheduler-visible features win in distribution and read numbers exactly,
but cannot read code semantics they have no feature for (big-1: gbt-dist
71% in distribution, 26% on unseen families). The LLM arms read the code
and transfer (45-51% on unseen families) but read numbers as tokens. Two
ways to combine them, both against the bench's own ``gbt-dist`` arm:

- **stacked** (``stack-<arm>``): the trees' prediction is written into the
  LLM's prompt (``hint_text``: their distribution's top cells and their
  pick), and the LLM arm -- SFT, decision head, any torso -- is trained
  and evaluated with it. Training rows carry *out-of-fold* predictions
  (``crossfit``: K folds by task), so the LLM learns how far to trust the
  trees on runs the trees did not fit, not on runs they memorized; val,
  test and OOD rows carry the predictions of trees fit on all of train.
- **pooled** (``pool-<variant>``): a logarithmic opinion pool of the two
  calibrated distributions over the same cells, p ~ p_trees^w . p_llm^(1-w),
  with w tuned on val (``pool_variants``). No retraining; every
  distributional LLM variant is pooled, and decoded like any other arm (the
  bench's decoder, and ``-argmax``).

Pure logic: no flyte, no torch.
"""

from __future__ import annotations

import zlib

import numpy as np

from .spec import BenchSpec

ANCHOR = "gbt-dist"
STACK_PREFIX = "stack-"
POOL_PREFIX = "pool-"
POOL_WEIGHTS = tuple(round(w, 2) for w in np.linspace(0, 1, 11))
FOLDS = 5


def fold_of(row: dict, k: int = FOLDS) -> int:
    """Folds by task: both runs of a task land in the same fold, so a run's
    out-of-fold prediction never saw its sibling."""
    return zlib.crc32(str(row["task_id"]).encode()) % k


def crossfit(train: list[dict], val: list[dict], eval_rows: list[dict], fit, predict, k: int = FOLDS) -> dict[str, dict]:
    """run_id -> the arm's prediction: out-of-fold for `train`, from a fit on
    all of train for `eval_rows`."""
    out: dict[str, dict] = {}
    for f in range(k):
        held = [r for r in train if fold_of(r, k) == f]
        rest = [r for r in train if fold_of(r, k) != f]
        if held and rest:
            out.update(predict(fit(rest, val), held))
    out.update(predict(fit(train, val), eval_rows))
    return out


def hint_text(pred: dict, spec: BenchSpec, top: int = 3) -> str:
    """The trees' prediction, as the prompt carries it."""
    a, b = spec.a, spec.b
    probs = pred.get(spec.probs_key)
    pick = a.options[int(pred[a.pred_key])]
    lead = ("Gradient-boosted trees fit on other tasks' measured runs (they read the input sizes, the API "
            "names and numbers in the code, and this task's earlier runs)")
    if probs is None:
        return f"{lead} predict {a.key} {pick} and {b.key} {b.options[int(pred[b.pred_key])]}."
    p = np.asarray(probs, dtype=float)
    p = p / p.sum()
    best = [int(i) for i in np.argsort(-p)[:top]]
    dist = ", ".join(f"{a.options[i]} {p[i]:.0%}" for i in best)
    return f"{lead} put {a.key} at {dist}; their pick: {pick}. Their {b.key}: {b.options[int(pred[b.pred_key])]}."


def with_hints(rows: list[dict], preds: dict[str, dict], spec: BenchSpec) -> list[dict]:
    """Rows with a ``hint`` column (rows without a prediction get none)."""
    return [{**r, "hint": hint_text(preds[r["run_id"]], spec) if r["run_id"] in preds else ""} for r in rows]


def is_stacked(arm: str) -> bool:
    return arm.startswith(STACK_PREFIX)


def base_arm(arm: str) -> str:
    return arm[len(STACK_PREFIX):] if is_stacked(arm) else arm


# ── pooling ─────────────────────────────────────────────────────────────


def log_pool(p_anchor, p_llm, w: float) -> np.ndarray:
    """p ~ p_anchor^w . p_llm^(1-w): both are calibrated distributions over the same cells."""
    p = np.exp(w * np.log(np.asarray(p_anchor, dtype=float) + 1e-9) + (1 - w) * np.log(np.asarray(p_llm, dtype=float) + 1e-9))
    return p / p.sum()


def _pooled(anchor: dict, member: dict, w: float, spec: BenchSpec, b_from_anchor: bool) -> dict[str, dict]:
    pk, a, b = spec.probs_key, spec.a.pred_key, spec.b.pred_key
    out = {}
    for rid, x in anchor.items():
        y = member.get(rid)
        if y is None:
            continue
        p = log_pool(x[pk], y[pk], w)
        out[rid] = {a: spec.decode(p), b: int((x if b_from_anchor else y)[b]), pk: p.tolist()}
    return out


def poolable(variants: dict[str, dict[str, dict]], spec: BenchSpec) -> list[str]:
    """LLM variants with a distribution on every run (argmax reads and earlier pools excluded)."""
    return sorted(v for v, p in variants.items()
                  if v != ANCHOR and not v.endswith("-argmax") and not v.startswith(POOL_PREFIX) and p
                  and all(x.get(spec.probs_key) is not None for x in p.values()))


def pool_variants(rows: list[dict], variants: dict[str, dict[str, dict]], spec: BenchSpec) -> tuple[dict, dict]:
    """Every poolable LLM variant pooled with the anchor, w tuned on val for the
    bench's headline. Returns ({variant: preds}, {variant: knobs})."""
    val = [r for r in rows if r["split"] == "val"]
    anchor = variants[ANCHOR]
    out, meta = {}, {}
    for v in poolable(variants, spec):
        member = variants[v]
        val_ids = [r for r in val if r["run_id"] in anchor and r["run_id"] in member]
        if not val_ids:
            continue
        # the secondary decision from whichever member gets it right more often on val
        lb = {r["run_id"]: spec.labels(r)[1] for r in val_ids}
        acc_a = np.mean([anchor[rid][spec.b.pred_key] == y for rid, y in lb.items()])
        acc_m = np.mean([member[rid][spec.b.pred_key] == y for rid, y in lb.items()])
        b_anchor = bool(acc_a > acc_m)
        sub_a = {r["run_id"]: anchor[r["run_id"]] for r in val_ids}
        scored = [(spec.headline(val_ids, _pooled(sub_a, member, w, spec, b_anchor)), -w, w) for w in POOL_WEIGHTS]
        best, _, w = min(scored)
        pooled = _pooled(anchor, member, w, spec, b_anchor)
        out[f"{POOL_PREFIX}{v}"] = pooled
        out[f"{POOL_PREFIX}{v}-argmax"] = {rid: {**p, spec.a.pred_key: int(np.argmax(p[spec.probs_key]))}
                                           for rid, p in pooled.items()}
        meta[f"{POOL_PREFIX}{v}"] = {"w_trees": w, "val_headline": float(best),
                                     "val_headline_by_w": {str(x): float(s) for s, _n, x in scored},
                                     "secondary_from": ANCHOR if b_anchor else v}
    return out, meta
