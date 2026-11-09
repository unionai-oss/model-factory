"""Two post-hoc arms over trained ones (round 22): a novelty gate and the
Flyte -> Kubernetes adapter. Neither retrains anything.

- **gate** (``gate-<signal>-q<q>``): the in-distribution champion on runs
  that look like the training runs, the robust arm on runs that do not.
  The two fail differently (round 21): the pooled trees read numbers
  exactly on families they have seen and are lost on new ones, where the
  RL-tuned decision model still reads the code. The novelty signal needs
  no family labels: distance to the nearest training runs in the trees'
  feature space, or the trees' and the decision model's disagreement
  (summed Jensen-Shannon divergence over the three axes). The threshold is
  a quantile of the signal on val, which holds only seen families: q = 0.95
  sends 5% of in-distribution runs to the robust arm, by construction, and
  every run less familiar than that.
- **adapter** (``<arm>-adapt``): a model trained on Flyte runs only sizes a
  K8s Job's memory by sizing its Flyte rendition -- the same function,
  inputs and history (``-adapt-prompt``: the Job's own prompt, read as Flyte
  units) -- and moving the distribution through the twin's accounting
  (``k8s``): no Flyte runtime, boto3's footprint, staged bytes on a
  memory-backed volume, the download in front of the CPU series (for a
  busiest-window CPU label; a mean-cores label is left as is). What the
  adapter reads is what a scheduler sees: the manifest's volume medium,
  the job's ``import boto3``, and the input sizes.

Pure logic: no flyte, no torch.
"""

from __future__ import annotations

import json

import numpy as np

from ..bench_cpu.grid import PLATFORM_WINDOW_S
from . import grid, k8s

GATE_PREFIX = "gate-"
#: The novelty signals a gate can route on (``signals``):
#:   knn          mean distance to the 10 nearest training runs in the trees'
#:                standardized feature space -- code tokens, input sizes, history
#:   knn-numeric  the same over input sizes and history only (no code tokens)
#:   js           the trees' and the decision model's disagreement
#:   knn-raw      knn as k8s-1 scaled it (constant-on-train features at sd 1e-9), for comparison
GATE_SIGNALS = ("knn", "knn-numeric", "js", "knn-raw")
#: val quantiles each gate is evaluated at; the headline gate is GATE_SIGNAL
#: at GATE_Q (5% of in-distribution runs to the robust arm).
GATE_QS = (0.5, 0.7, 0.8, 0.9, 0.95, 0.99)
GATE_SIGNAL = "knn"
GATE_Q = 0.95
KNN_K = 10
#: Below this a feature's train sd counts as zero (``novelty_knn``).
SD_FLOOR = 1e-6
#: Feature-name prefixes of the code-token features (``features.code_features``).
CODE_PREFIXES = ("m_", "c_", "g_", "cuda_", "k8s_")
ADAPT_SUFFIX = "-adapt"


# ── the novelty signal ──────────────────────────────────────────────────


def js(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence in nats (0 .. ln 2)."""
    p, q = (np.clip(np.asarray(x, dtype=float), 1e-12, None) for x in (p, q))
    p, q = p / p.sum(), q / q.sum()
    m = (p + q) / 2
    return float(0.5 * (p * np.log(p / m)).sum() + 0.5 * (q * np.log(q / m)).sum())


def disagreement(a: list[np.ndarray], b: list[np.ndarray]) -> float:
    """Summed over the three axes."""
    return sum(js(x, y) for x, y in zip(a, b))


def auroc(neg: list[float], pos: list[float]) -> float:
    """P(a positive scores above a negative), ties half."""
    neg, pos = np.asarray(neg, dtype=float), np.asarray(pos, dtype=float)
    if not len(neg) or not len(pos):
        return float("nan")
    allv = np.concatenate([neg, pos])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv))
    ranks[order] = np.arange(1, len(allv) + 1)
    for v in np.unique(allv):  # average ranks over ties
        tie = allv == v
        ranks[tie] = ranks[tie].mean()
    return float((ranks[len(neg):].sum() - len(pos) * (len(pos) + 1) / 2) / (len(neg) * len(pos)))


def novelty_knn(rows: list[dict], feats: dict[str, dict[str, float]], numeric_only: bool = False,
                k: int = KNN_K, legacy_scale: bool = False) -> dict[str, float]:
    """run_id -> mean distance to its k nearest training runs, features
    standardized on train. A feature constant on train (a library no training
    run imports) makes any run that has it as far as it gets."""
    from sklearn.neighbors import NearestNeighbors

    names = sorted(next(iter(feats.values())))
    if numeric_only:
        names = [n for n in names if not n.startswith(CODE_PREFIXES)]
    ids = [r["run_id"] for r in rows]
    X = np.asarray([[feats[i].get(n, 0.0) for n in names] for i in ids], dtype=float)
    tr = np.asarray([r["split"] == "train" for r in rows])
    mu, sd = X[tr].mean(0), X[tr].std(0)
    # A feature constant on train (a library no training run imports) keeps its
    # raw unit: one novel token is one unit away, like one sd of anything else.
    # Through k8s-1 it got sd = 1e-9, which put such a run ~1e9 away and made
    # the detector a token lookup (AUROC 0.9999) -- review, 2026-11-08.
    sd = sd + 1e-9 if legacy_scale else np.where(sd > SD_FLOOR, sd, 1.0)
    Z = (X - mu) / sd
    nn = NearestNeighbors(n_neighbors=k).fit(Z[tr])
    ev = ~tr
    dist, _ = nn.kneighbors(Z[ev])
    return dict(zip([i for i, e in zip(ids, ev) if e], dist.mean(1).tolist()))


def gate(rows: list[dict], signal: dict[str, float], inner: dict[str, dict], outer: dict[str, dict],
         q: float) -> tuple[dict[str, dict], float]:
    """inner's request where the signal is at most its val q-quantile, outer's elsewhere."""
    val = [signal[r["run_id"]] for r in rows if r["split"] == "val" and r["run_id"] in signal]
    tau = float(np.quantile(val, q))
    out = {}
    for rid, s in signal.items():
        if rid in inner and rid in outer:
            out[rid] = {**(outer[rid] if s > tau else inner[rid]), "gate_signal": s, "gated_out": bool(s > tau)}
    return out, tau


# ── the adapter ─────────────────────────────────────────────────────────


def staging_seen(row: dict) -> str:
    """The staging as a scheduler reads it: the manifest's volume, the job's imports."""
    tmpfs = "medium: Memory" in (row.get("manifest") or "")
    boto3 = "import boto3" in (row.get("task_code") or "")
    return f"{'boto3' if boto3 else 'init'}-{'tmpfs' if tmpfs else 'disk'}"


def _mem_reps() -> np.ndarray:
    """A representative need per memory cell: the geometric middle of its interval."""
    g = np.asarray(grid.MEMORY_GRID_MIB, dtype=float)
    lo = np.concatenate([[g[0] * 0.75], g[:-1]])
    return np.sqrt(lo * g)


def _cpu_reps() -> np.ndarray:
    g = np.asarray(grid.CPU_GRID, dtype=float)
    lo = np.concatenate([[g[0] * 0.75], g[:-1]])
    return np.sqrt(lo * g)


def _move(p: np.ndarray, targets: list[int], n: int) -> np.ndarray:
    out = np.zeros(n)
    np.add.at(out, targets, np.asarray(p, dtype=float))
    return out


def adapt_probs(probs: list[np.ndarray], row: dict) -> list[np.ndarray]:
    """A Flyte-unit distribution [P(mem), P(cpu), P(gpu)] for the K8s Job `row`."""
    pm, pc, pg = (np.asarray(x, dtype=float) for x in probs)
    st = staging_seen(row)
    sizes = {k: int(v) for k, v in json.loads(row.get("input_bytes_json") or "{}").items()}
    mem_to = [grid.mem_cell(k8s.twin_need_mib(n, sizes, st)) for n in _mem_reps()]
    cpu_to = list(range(grid.N_CPU))
    if k8s.is_boto3(st) and row.get("source") == "cpu":
        # a busiest-window label (RightSizeCPU's): the download's window is a floor
        # under it. A mean-cores label (RightSizeMem's) barely moves: left as is.
        w = min(1.0, k8s.download_s(sizes) / PLATFORM_WINDOW_S)
        cpu_to = [grid.cpu_cell(max(c, k8s.DL_CORES * w + c * (1 - w))) for c in _cpu_reps()]
    return [_move(pm, mem_to, grid.N_MEM), _move(pc, cpu_to, grid.N_CPU), pg]


#: Where an adapted twin's memory distribution comes from: its Flyte
#: rendition's (the model reads the workload in the format it was trained
#: on), or its own K8s prompt's, read as Flyte units. CPU and GPU always come
#: from the twin's own prompt (rt-k8sgate-2: the rendition's were worse).
ADAPT_SOURCES = {"": "rendition", "-prompt": "prompt"}


def adapted(rows: list[dict], probs: dict[str, list[np.ndarray]], memory_from: str = "rendition",
            ) -> dict[str, list[np.ndarray]]:
    """run_id -> distribution: Flyte rows as they are, each K8s twin adapted."""
    out = {}
    for r in rows:
        rid = r["run_id"]
        if r.get("platform") != "k8s":
            if rid in probs:
                out[rid] = probs[rid]
            continue
        own, rend = probs.get(rid), probs.get(r.get("twin_of"))
        src = rend if memory_from == "rendition" else own
        if own is None or src is None:
            continue
        pm, _pc, _pg = adapt_probs(src, r)
        # memory only: CPU and GPU are the model's own reading of the Job, unmoved
        # (through k8s-1 they went through adapt_probs too, which shifted a boto3
        # RightSizeCPU row's CPU by the download window -- review, 2026-11-08)
        out[rid] = [pm, np.asarray(own[1], dtype=float), np.asarray(own[2], dtype=float)]
    return out
