"""Offline analysis of an exported release (paper.paper_export): task-clustered
bootstrap CIs, paired K8s - Flyte gaps, the pair metric split by whether the
need moved, and the cost model's sensitivity to its penalties.

    uv run python analyze.py <dir> <tag>   ->  <dir>/<tag>-analysis.json

Round 24 (the third review): regret is measured against the cheapest request
under the cost model (``scoring.candidate_actions``), recomputed under every
penalty setting, instead of against the label cell; unseen-split intervals are
also clustered by family (7 families); each approach's read-out is chosen on
validation; every approach is scored on the zero-shot model's sample; results
are sliced by corpus and by whether the test task's code appears in training
(``<dir>/<run>-audit.json.gz``, from paper_audit, when present).
"""
import gzip
import json
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, "/Users/nielsbantilan/git/model-factory/model-factory-resource-tuner")
from resource_tuner.bench_all import scoring  # noqa: E402

D, TAG = sys.argv[1], sys.argv[2]
RUN = {"r2": "rsk-k8s-1-gate", "r1": "rsk-k8s-1", "k2s1": "rsk-k8s-2", "k2s2": "rsk-k8s-2-s2", "k2s3": "rsk-k8s-2-s3"}.get(TAG, TAG)
rows = json.load(gzip.open(f"{D}/{sys.argv[3] if len(sys.argv) > 3 else RUN}-rows.json.gz"))
picks = json.load(gzip.open(f"{D}/{RUN}-picks.json.gz"))["picks"]
by_id = {r["run_id"]: r for r in rows}
B = 2000
rng = np.random.default_rng(0)
#: (OOM penalty per retry, in oracle costs; delay penalty per hour late, in oracle hourly prices).
#: oom5 / oom20 stand for a raw Job (backoffLimit 0): an OOM is a failed Job someone resubmits.
SETTINGS = {"base": (1.0, 1.0), "oom0": (0.0, 1.0), "oom5": (5.0, 1.0), "oom20": (20.0, 1.0), "delay3": (1.0, 3.0)}

_raw: dict = {}


def raw(rid, m, c, g):
    k = (rid, m, c, g)
    o = _raw.get(k)
    if o is None:
        r = by_id[rid]
        x = scoring.raw_cost(r, m, c, g)
        o = _raw[k] = (x.cost, x.attempts, x.slowdown, x.failed, scoring.oracle_cost(r))
    return o


def _total(rid, pick, s):
    oom, delay = SETTINGS[s]
    c, att, slow, _f, base = raw(rid, *pick[:3])
    return c + oom * base * (att - 1) + delay * base * (slow - 1.0)


_cheap: dict = {}


def cheapest(rid, s="base"):
    """The lowest total cost over the oracle's candidate requests, under setting s
    (every setting at once per run, without filling the ``raw`` cache)."""
    v = _cheap.get(rid)
    if v is None:
        r = by_id[rid]
        base = scoring.oracle_cost(r)
        outs = [scoring.raw_cost(r, *a) for a in scoring.candidate_actions(r)]
        v = _cheap[rid] = {k: min(o.cost + oom * base * (o.attempts - 1) + delay * base * (o.slowdown - 1.0)
                                  for o in outs) for k, (oom, delay) in SETTINGS.items()}
    return v[s]


def cost(rid, pick, s="base"):
    """(this request's total cost, the cheapest request's) under setting s."""
    return _total(rid, pick, s), cheapest(rid, s)


def cluster(r):
    return r["task_id"].split("@")[0]


def family_cluster(r):
    return f"{r['source']}:{r['family']}"


AUDIT = f"{D}/{RUN}-audit.json.gz"
try:
    seen_code = json.load(gzip.open(AUDIT))["seen"]
except FileNotFoundError:
    seen_code = {}


def code_seen(r):
    """Whether this pod's task code (its measured run's) is identical to a training task's."""
    return seen_code.get(r.get("twin_of") or r["run_id"])


def boot(values, clusters):
    """mean and 95% CI of `values`, resampling clusters (Poisson bootstrap)."""
    values = np.asarray(values, dtype=float)
    keys, inv = np.unique(np.asarray(clusters), return_inverse=True)
    s = np.bincount(inv, weights=values, minlength=len(keys))
    n = np.bincount(inv, minlength=len(keys)).astype(float)
    W = rng.poisson(1.0, size=(B, len(keys)))
    est = (W @ s) / np.maximum(W @ n, 1e-9)
    return [float(values.mean()), float(np.quantile(est, 0.025)), float(np.quantile(est, 0.975))]


def boot_ratio(num, den, clusters):
    """sum(num)/sum(den) - 1, clustered: dollar regret."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    keys, inv = np.unique(np.asarray(clusters), return_inverse=True)
    a = np.bincount(inv, weights=num, minlength=len(keys))
    b = np.bincount(inv, weights=den, minlength=len(keys))
    W = rng.poisson(1.0, size=(B, len(keys)))
    est = (W @ a) / np.maximum(W @ b, 1e-9) - 1
    return [float(num.sum() / den.sum() - 1), float(np.quantile(est, 0.025)), float(np.quantile(est, 0.975))]


def subset(split, platform=None, source=None, staging=None, cpu_node=False):
    out = []
    for r in rows:
        if r["split"] != split:
            continue
        if platform and (r.get("platform") or "flyte") != platform:
            continue
        if source and r["source"] != source:
            continue
        if staging and r.get("staging") != staging:
            continue
        if cpu_node and r["source"] == "gpu":
            continue
        out.append(r)
    return out


def regret(variant, rs, s="base"):
    p = picks[variant]
    rs = [r for r in rs if r["run_id"] in p]
    if not rs:
        return None
    cs = [cost(r["run_id"], p[r["run_id"]], s) for r in rs]
    ratios = [c / max(b, 1e-9) for c, b in cs]
    cl = [cluster(r) for r in rs]
    lr = np.log(np.maximum(ratios, 1e-9))
    return {"n": len(rs), "regret": [x - 1 for x in boot(ratios, cl)],
            "regret_fam": [x - 1 for x in boot(ratios, [family_cluster(r) for r in rs])],
            "below_cheapest": float(np.mean(np.array(ratios) < 1 - 1e-9)),
            "geo_regret": [float(np.exp(x)) - 1 for x in boot(lr, cl)],
            "capped_regret": [x - 1 for x in boot(np.minimum(ratios, 20.0), cl)],
            "first_ok": float(np.mean([raw(r["run_id"], *p[r["run_id"]][:3])[1] == 1 for r in rs])),
            "dollar_regret": boot_ratio([c for c, _ in cs], [b for _, b in cs], cl),
            "cost_per_1k": 1000 * float(np.mean([c for c, _ in cs])),
            "oracle_per_1k": 1000 * float(np.mean([b for _, b in cs]))}


def paired_gap(variant, split, s="base", cpu_node=False):
    """K8s twin's ratio minus its Flyte run's, per pair, clustered by task."""
    p = picks[variant]
    d, cl = [], []
    for t in subset(split, "k8s", cpu_node=cpu_node):
        f = by_id.get(t["twin_of"])
        if f is None or t["run_id"] not in p or f["run_id"] not in p:
            continue
        ct, bt = cost(t["run_id"], p[t["run_id"]], s)
        cf, bf = cost(f["run_id"], p[f["run_id"]], s)
        d.append(ct / bt - cf / bf)
        cl.append(cluster(t))
    return {"n": len(d), "gap": boot(d, cl)} if d else None


def diff(a, b, rs, s="base"):
    """regret(a) - regret(b) on the runs both answered, clustered by task and by family."""
    pa, pb = picks[a], picks[b]
    rs = [r for r in rs if r["run_id"] in pa and r["run_id"] in pb]
    d = []
    for r in rs:
        ca, ba = cost(r["run_id"], pa[r["run_id"]], s)
        cb, bb = cost(r["run_id"], pb[r["run_id"]], s)
        d.append(ca / ba - cb / bb)
    if not d:
        return None
    return {"n": len(d), "diff": boot(d, [cluster(r) for r in rs]),
            "diff_fam": boot(d, [family_cluster(r) for r in rs])}


def pairs(variant, split):
    """The memory request across a run and its twin (CPU-node runs), split by
    whether the label cell moved. A platform-blind arm (the same request for
    both) scores exactly the unmoved share overall, and 0 on moved pairs."""
    p = picks[variant]
    rec = defaultdict(list)
    for t in subset(split, "k8s", cpu_node=True):
        f = by_id.get(t["twin_of"])
        if f is None or t["run_id"] not in p or f["run_id"] not in p:
            continue
        dl = int(t["label_mem"]) - int(f["label_mem"])
        dp = p[t["run_id"]][0] - p[f["run_id"]][0]
        c = cluster(t)
        rec["all"].append((float(np.sign(dl) == np.sign(dp)), c))
        rec["moved" if dl else "unmoved"].append((float(np.sign(dl) == np.sign(dp)), c))
        if dl:
            rec[f"moved_{'up' if dl > 0 else 'down'}"].append((float(np.sign(dl) == np.sign(dp)), c))
            rec["moved_exact"].append((float(dl == dp), c))
        rec["is_moved"].append((float(dl != 0), c))
    return {k: {"n": len(v), "rate": boot([x for x, _ in v], [c for _, c in v])} for k, v in rec.items()}


HEAD = [v for v in ("static", "rules", "rules-sized", "gbt-dist", "gbt-dist-flyte", "gbt-dist-k8s", "zeroshot-27b",
                    "zeroshot-27b-json", "jev-pointer", "jev-pointer-argmax", "jev-pointer-flyte", "jev-pointer-flyte-argmax",
                    "jev-pointer-k8s", "jev-pointer-k8s-argmax", "rl-pointer", "rl-pointer-argmax",
                    "pool-jev-pointer", "pool-jev-pointer-argmax", "pool-jev-pointer-flyte", "pool-jev-pointer-flyte-argmax",
                    "pool-jev-pointer-k8s", "pool-jev-pointer-k8s-argmax", "pool-rl-pointer", "pool-rl-pointer-argmax",
                    "jev-pointer-flyte-adapt", "pool-jev-pointer-flyte-adapt", "gate-rl-pointer", "gate-knn-q95",
                    "gate-knn-raw-q95", "gate-knn-numeric-q95", "gate-js-q50") if v in picks]

#: Each trained approach's two read-outs; ``selected`` keeps the one with the lower validation regret (all pods).
READOUTS = [v for v in ("jev-pointer", "jev-pointer-k8s", "jev-pointer-flyte", "rl-pointer", "pool-jev-pointer",
                        "pool-jev-pointer-k8s", "pool-jev-pointer-flyte", "pool-rl-pointer")
            if v in picks and f"{v}-argmax" in picks]
ZS = picks.get("zeroshot-27b", {})

out = {"variants": HEAD, "n_rows": len(rows), "settings": SETTINGS, "table": {}, "gap": {}, "pairs": {}, "diff": {},
       "sens": {}, "staging": {}, "cold": {}, "selected": {}, "zs_sample": {}, "corpus": {}, "code": {},
       "audit": bool(seen_code), "sens_diff": {}}
for v in HEAD:
    t = out["table"][v] = {}
    for split in ("val", "test", "ood"):
        for plat in (None, "k8s", "flyte"):
            x = regret(v, subset(split, plat))
            if x:
                t[f"{split}/{plat or 'all'}"] = x
            if plat:
                x = regret(v, subset(split, plat, cpu_node=True))
                if x:
                    t[f"{split}/{plat}-cpu"] = x
    out["gap"][v] = {s: paired_gap(v, s) for s in ("test", "ood")}
    out["gap"][v].update({f"{s}-cpu": paired_gap(v, s, cpu_node=True) for s in ("test", "ood")})
    out["pairs"][v] = {s: pairs(v, s) for s in ("test", "ood")}
    out["staging"][v] = {st: regret(v, subset("test", "k8s", staging=st, cpu_node=True))
                         for st in ("init-disk", "init-tmpfs", "boto3-disk", "boto3-tmpfs")}
    out["cold"][v] = {f"{split}/{plat}/{h}": (regret(v, [r for r in subset(split, plat) if bool(r["n_history"]) == (h == "warm")]) or {}).get("regret")
                      for split in ("test", "ood") for plat in ("k8s", "flyte") for h in ("cold", "warm")}
    out["sens"][v] = {s: {split: (regret(v, subset(split, "k8s"), s) or {}).get("regret") for split in ("test", "ood")}
                      for s in SETTINGS}
    out["zs_sample"][v] = {split: (regret(v, [r for r in subset(split, "k8s") if r["run_id"] in ZS]) or {}).get("regret")
                           for split in ("test", "ood")}
    out["corpus"][v] = {f"{split}/{src}": (regret(v, subset(split, "k8s", source=src)) or {}).get("regret")
                        for split in ("test", "ood") for src in ("mem", "cpu", "gpu")}
    if seen_code:
        out["code"][v] = {f"test/{k}": (regret(v, [r for r in subset("test", "k8s") if code_seen(r) is want]) or {}).get("regret")
                          for k, want in (("seen", True), ("novel", False))}
    print(v, {k: round(x["regret"][0], 4) for k, x in t.items() if k.endswith("/all")}, flush=True)

for a, b in (("gate-rl-pointer", "pool-rl-pointer-argmax"), ("gate-rl-pointer", "rl-pointer"),
             ("gate-rl-pointer", "gbt-dist"), ("pool-rl-pointer-argmax", "gbt-dist"), ("jev-pointer", "gbt-dist"),
             ("jev-pointer", "jev-pointer-flyte"), ("jev-pointer", "jev-pointer-k8s"), ("rl-pointer", "jev-pointer"),
             ("pool-rl-pointer-argmax", "pool-jev-pointer"), ("jev-pointer-flyte-adapt", "jev-pointer-flyte"),
             ("rules-sized", "rules"), ("gbt-dist", "gbt-dist-k8s"), ("jev-pointer-k8s", "jev-pointer-flyte"),
             ("gate-rl-pointer", "pool-jev-pointer"), ("rules-sized", "rl-pointer"),
             ("rules-sized", "rl-pointer-argmax"), ("rules-sized", "gate-rl-pointer"),
             ("pool-jev-pointer", "rl-pointer-argmax"), ("pool-jev-pointer", "gate-rl-pointer"),
             ("pool-jev-pointer-argmax", "pool-jev-pointer"), ("rl-pointer-argmax", "rl-pointer")):
    if a in picks and b in picks:
        out["diff"][f"{a} - {b}"] = {f"{split}/{plat}": diff(a, b, subset(split, None if plat == "all" else plat))
                                     for split in ("test", "ood") for plat in ("all", "k8s", "flyte")}

for v in READOUTS:
    dec, am = out["table"][v]["val/all"]["regret"][0], out["table"][f"{v}-argmax"]["val/all"]["regret"][0]
    out["selected"][v] = v if dec <= am else f"{v}-argmax"

if seen_code:
    t = subset("test", "k8s")
    out["code"]["_share_seen"] = float(np.mean([bool(code_seen(r)) for r in t]))
    out["code"]["_n"] = {"seen": sum(bool(code_seen(r)) for r in t), "novel": sum(code_seen(r) is False for r in t)}

#: every leaderboard approach against the best point estimate per column (Job pods), for bolding the
#: approaches the best cannot be separated from (paired, task-clustered; family-clustered on unseen too)
LEAD = [v for v in ("static", "rules", "rules-sized", "gbt-dist", "jev-pointer", "jev-pointer-argmax", "rl-pointer",
                    "rl-pointer-argmax", "pool-jev-pointer", "pool-jev-pointer-argmax", "pool-rl-pointer",
                    "pool-rl-pointer-argmax", "gate-rl-pointer") if v in picks]
out["vs_best"] = {}
for split in ("test", "ood"):
    for plat in ("k8s", "flyte"):
        key = f"{split}/{plat}"
        best_v = min(LEAD, key=lambda v: out["table"][v][key]["regret"][0])
        out["vs_best"][key] = {"best": best_v, **{v: diff(v, best_v, subset(split, plat)) for v in LEAD if v != best_v}}

ZS_PAIRS = [("zeroshot-27b", b) for b in ("rules", "rules-sized", "rl-pointer", "rl-pointer-argmax", "gbt-dist",
                                          "pool-jev-pointer")]
for a, b in ZS_PAIRS:
    if a in picks and b in picks:
        out["diff"][f"{a} - {b}"] = {f"{split}/k8s": diff(a, b, subset(split, "k8s")) for split in ("test", "ood")}

#: the retry-penalty claims, paired (Job pods)
SENS_PAIRS = [("rules-sized", b) for b in ("rl-pointer", "rl-pointer-argmax", "gate-rl-pointer", "pool-rl-pointer",
                                           "gbt-dist", "zeroshot-27b-json")]
SENS_PAIRS += [("pool-jev-pointer", b) for b in ("pool-rl-pointer-argmax", "rl-pointer-argmax", "gbt-dist")]
for a, b in SENS_PAIRS:
    if a in picks and b in picks:
        out["sens_diff"][f"{a} - {b}"] = {f"{s}/{split}": diff(a, b, subset(split, "k8s"), s)
                                          for s in SETTINGS for split in ("test", "ood")}

json.dump(out, open(f"{D}/{TAG}-analysis.json", "w"), indent=1)
print("raw evaluations", len(_raw))
