"""Write the paper's generated tables and figure data from the analysis files.

    python scripts/tables.py <analysis dir> <counterfactual dir>  ->  generated/*.tex, generated/*.dat

Inputs: ``k2s{1,2,3}-analysis.json`` (``analyze.py``) and
``rsk-k8s-2{,-s2,-s3}-counterfactual.json.gz`` (``paper_file`` on the
counterfactual station's output). Seed 1 gives the point estimates and
intervals; seeds 2 and 3 give the range across trainings.

Round 24: the leaderboard shows each approach with the read-out validation
chose (``selected``); every read-out is in ``tab_readouts``. Bold marks every
approach whose paired difference from the column's best has a 95% interval
that includes zero. ``num_main.tex`` defines the numbers the prose quotes.
"""
import gzip
import json
import os
import sys

import numpy as np

D, CF = sys.argv[1], sys.argv[2]
OUT = os.path.join(os.path.dirname(__file__), "..", "generated")
os.makedirs(OUT, exist_ok=True)
SEEDS = ("k2s1", "k2s2", "k2s3")
A = {s: json.load(open(f"{D}/{s}-analysis.json")) for s in SEEDS if os.path.exists(f"{D}/{s}-analysis.json")}
A1 = A["k2s1"]

NAME = {
    "static": "static", "rules": "rules", "rules-sized": "rules-sized",
    "gbt-dist": "trees", "gbt-dist-k8s": "trees, Job-only", "gbt-dist-flyte": "trees, Flyte-only",
    "zeroshot-27b": "zero-shot 27B", "zeroshot-27b-json": "zero-shot 27B, JSON",
    "jev-pointer": "decision", "jev-pointer-argmax": "decision, argmax",
    "jev-pointer-k8s": "decision, Job-only", "jev-pointer-k8s-argmax": "decision, Job-only, argmax",
    "jev-pointer-flyte": "decision, Flyte-only", "jev-pointer-flyte-argmax": "decision, Flyte-only, argmax",
    "rl-pointer": "RL", "rl-pointer-argmax": "RL, argmax",
    "pool-jev-pointer": "pool $\\times$ decision", "pool-jev-pointer-argmax": "pool $\\times$ decision, argmax",
    "pool-jev-pointer-k8s": "pool $\\times$ decision, Job-only",
    "pool-jev-pointer-flyte": "pool $\\times$ decision, Flyte-only",
    "pool-rl-pointer": "pool $\\times$ RL", "pool-rl-pointer-argmax": "pool $\\times$ RL, argmax",
    "jev-pointer-flyte-adapt": "decision, Flyte-only, adapted",
    "pool-jev-pointer-flyte-adapt": "pool $\\times$ decision, Flyte-only, adapted",
    "gate-rl-pointer": "gate (kNN, $q$ 0.95)", "gate-knn-q95": "gate (kNN, $q$ 0.95)",
    "gate-knn-raw-q95": "gate (k8s-1 kNN scale)", "gate-knn-numeric-q95": "gate (numeric kNN)",
    "gate-js-q50": "gate (disagreement)",
}
NAME["gate-rl-pointer"] = NAME["gate-knn-q95"] = "gate"
SEL = A1.get("selected", {})


def sel(v):
    return SEL.get(v, v)


#: the leaderboard: one read-out per approach, chosen on validation
LEAD = ["static", "rules", "rules-sized", "zeroshot-27b", "zeroshot-27b-json", "gbt-dist", sel("jev-pointer"),
        sel("rl-pointer"), sel("pool-jev-pointer"), sel("pool-rl-pointer"), "gate-rl-pointer"]
#: every read-out, for the tables that compare them
LEAD_ALL = ["static", "rules", "rules-sized", "zeroshot-27b", "zeroshot-27b-json", "gbt-dist", "jev-pointer",
            "jev-pointer-argmax", "rl-pointer", "rl-pointer-argmax", "pool-jev-pointer", "pool-jev-pointer-argmax",
            "pool-rl-pointer", "pool-rl-pointer-argmax", "gate-rl-pointer"]


def pct(x, d=1):
    if isinstance(x, (list, tuple)):
        x = x[0]
    return "---" if x is None else f"{100 * x:.{d}f}"


def cell(t, key, metric="regret", ci=True):
    x = (t or {}).get(key, {}).get(metric)
    if not x:
        return "---"
    return f"{pct(x[0])}\\,{{\\scriptsize[{pct(x[1])}, {pct(x[2])}]}}" if ci else pct(x[0])


def seeds(v, key):
    """mean (min--max) over the seeds that have v."""
    xs = [a["table"][v][key]["regret"][0] for a in A.values() if v in a["table"] and key in a["table"][v]]
    if len(xs) < 2:
        return "---"
    if max(xs) - min(xs) < 1e-9:
        return f"{pct(xs[0])}"
    return f"{pct(np.mean(xs))}\\,{{\\scriptsize({pct(min(xs))}--{pct(max(xs))})}}"


def best(rows, col):
    """Bold the smallest value in a column (rows: list of lists of strings with leading numbers)."""
    def num(s):
        try:
            return float(s.split("\\")[0])
        except ValueError:
            return float("inf")
    vals = [float("inf") if r[0].startswith("zero-shot") else num(r[col]) for r in rows]
    m = min(vals)
    for r, v in zip(rows, vals):
        if v == m:
            r[col] = "\\textbf{" + r[col].split("\\,")[0] + "}" + ("\\," + r[col].split("\\,", 1)[1] if "\\," in r[col] else "")
    return rows


def bold_inseparable(rows, variants, col, key):
    """Bold every row whose paired difference from the column's best includes zero (task-clustered)."""
    vb = A1.get("vs_best", {}).get(key)
    if not vb:
        return best(rows, col)
    for r, v in zip(rows, variants):
        d = vb.get(v)
        if v == vb["best"] or (d and d["diff"][1] <= 0 <= d["diff"][2]):
            head, _, tail = r[col].partition("\\,")
            r[col] = "\\textbf{" + head + "}" + ("\\," + tail if tail else "")
    return rows


def write(name, s):
    with open(os.path.join(OUT, name), "w") as f:
        f.write(s)
    print("wrote", name)


def tabular(cols, head, rows, groups=None):
    out = [f"\\begin{{tabular}}{{@{{}}{cols}@{{}}}}", "\\toprule"]
    if groups:
        out.append(groups)
    out += [" & ".join(head) + " \\\\", "\\midrule"]
    for r in rows:
        out.append("\\midrule" if r == "MID" else " & ".join(r) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(out)


# ── leaderboard: Job pods first, one read-out per approach ──────────────
def fam_ci(t, key):
    x = (t or {}).get(key, {}).get("regret_fam")
    return "---" if not x else f"{{\\scriptsize[{pct(x[1])}, {pct(x[2])}]}}"


rows, vs = [], []
for v in LEAD:
    if v not in A1["table"]:
        continue
    t = A1["table"][v]
    rows.append([NAME[v] + ("$^\\dagger$" if v.startswith("zeroshot") else ""), cell(t, "test/k8s"), cell(t, "ood/k8s"),
                 fam_ci(t, "ood/k8s"), cell(t, "test/flyte", ci=False), cell(t, "ood/flyte", ci=False),
                 cell(t, "test/k8s", "dollar_regret", ci=False), cell(t, "ood/k8s", "dollar_regret", ci=False),
                 seeds(v, "test/k8s"), seeds(v, "ood/k8s")])
    vs.append(v)
for c, key in ((1, "test/k8s"), (2, "ood/k8s"), (4, "test/flyte"), (5, "ood/flyte")):
    bold_inseparable(rows, vs, c, key)
write("tab_leaderboard.tex", tabular(
    "lrrrrrrrrr",
    ["approach", "test", "unseen", "by family", "test", "unseen", "test", "unseen", "test", "unseen"],
    rows,
    "& \\multicolumn{3}{c}{Job pods: regret (\\%)} & \\multicolumn{2}{c}{Flyte pods} & "
    "\\multicolumn{2}{c}{Job \\$ regret} & \\multicolumn{2}{c}{Job, 3 seeds: mean (range)} \\\\\n"
    "\\cmidrule(lr){2-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\\cmidrule(l){9-10}"))

# ── every read-out, and which one validation picked ─────────────────────
rows = []
for base in ("jev-pointer", "jev-pointer-k8s", "rl-pointer", "pool-jev-pointer", "pool-rl-pointer"):
    for v in (base, f"{base}-argmax"):
        t = A1["table"].get(v)
        if not t:
            continue
        mark = "\\checkmark" if SEL.get(base) == v else ""
        rows.append([NAME[v], cell(t, "val/k8s", ci=False), cell(t, "test/k8s", ci=False), cell(t, "ood/k8s", ci=False),
                     cell(t, "test/flyte", ci=False), cell(t, "ood/flyte", ci=False), mark])
    rows.append("MID")
write("tab_readouts.tex", tabular("lrrrrrc", ["approach", "val", "test", "unseen", "test", "unseen", "chosen"], rows[:-1],
                                  "& \\multicolumn{3}{c}{Job pods} & \\multicolumn{2}{c}{Flyte pods} & \\\\\n"
                                  "\\cmidrule(lr){2-4}\\cmidrule(lr){5-6}"))

# ── familiar families: test tasks whose code is in training, and the rest ─
if A1.get("code", {}).get("_n"):
    rows = []
    for v in LEAD:
        c = A1["code"].get(v)
        if not c or v.startswith("zeroshot"):
            continue
        rows.append([NAME[v], cell(c, "test/seen", "x", ci=False) if False else pct(c.get("test/seen")),
                     pct(c.get("test/novel")), pct(A1["table"][v]["ood/k8s"]["regret"])])
    n = A1["code"]["_n"]
    write("tab_code.tex", tabular("lrrr", ["approach", f"seen ({n['seen']:,})", f"new ({n['novel']:,})", "unseen families"],
                                  rows, "& \\multicolumn{2}{c}{test: code in training?} & \\\\\n\\cmidrule(lr){2-3}"))

# ── by corpus (Job pods) ────────────────────────────────────────────────
rows = []
for v in LEAD:
    c = A1["corpus"].get(v)
    if not c or v.startswith("zeroshot"):
        continue
    rows.append([NAME[v]] + [pct(c.get(f"{sp}/{src}")) for sp in ("test", "ood") for src in ("mem", "cpu", "gpu")])
write("tab_corpus.tex", tabular("lrrrrrr", ["approach", "Mem", "CPU", "GPU", "Mem", "CPU", "GPU"], rows,
                                "& \\multicolumn{3}{c}{test} & \\multicolumn{3}{c}{unseen} \\\\\n"
                                "\\cmidrule(lr){2-4}\\cmidrule(l){5-7}"))

# ── every approach on the zero-shot model's sample ──────────────────────
rows = []
for v in LEAD:
    z = A1["zs_sample"].get(v)
    if not z or not z.get("test"):
        continue
    rows.append([NAME[v], pct(z["test"]), pct(z["ood"])])
write("tab_zs.tex", tabular("lrr", ["approach", "test", "unseen"], rows))


# ── training data: both pod types, Jobs only, Flyte only ────────────────
FAM = [("trees", "gbt-dist", "gbt-dist-k8s", "gbt-dist-flyte"),
       ("decision", "jev-pointer", "jev-pointer-k8s", "jev-pointer-flyte"),
       ("decision, argmax", "jev-pointer-argmax", "jev-pointer-k8s-argmax", "jev-pointer-flyte-argmax"),
       ("pool $\\times$ decision", "pool-jev-pointer", "pool-jev-pointer-k8s", "pool-jev-pointer-flyte")]
rows = []
for label, *vs in FAM:
    for tr, v in zip(("both", "Jobs only", "Flyte only"), vs):
        if v not in A1["table"]:
            continue
        t = A1["table"][v]
        rows.append([label if tr == "both" else "", tr, cell(t, "test/k8s"), cell(t, "ood/k8s", ci=False),
                     cell(t, "test/flyte"), cell(t, "ood/flyte", ci=False),
                     pct(A1["table"][v].get("test/k8s-cpu", {}).get("first_ok"))])
    rows.append("MID")
rows = rows[:-1]
write("tab_training.tex", tabular(
    "llrrrrr", ["approach", "trained on", "test", "unseen", "test", "unseen", "first OK"], rows,
    "&& \\multicolumn{2}{c}{Job pods: regret (\\%)} & \\multicolumn{2}{c}{Flyte pods: regret (\\%)} & Job \\\\\n"
    "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}"))

# ── platform gap: Job minus Flyte, paired ───────────────────────────────
rows = []
for v in LEAD_ALL + ["jev-pointer-k8s", "jev-pointer-flyte"]:
    g = A1["gap"].get(v) or {}
    if not g.get("test"):
        continue
    f = lambda x: "---" if not x else f"{100*x['gap'][0]:+.1f}\\,{{\\scriptsize[{100*x['gap'][1]:+.1f}, {100*x['gap'][2]:+.1f}]}}"  # noqa: E731
    rows.append([NAME[v], f(g.get("test")), f(g.get("ood"))])
write("tab_gap.tex", tabular("lrr", ["approach", "test", "unseen"], rows,
                             "& \\multicolumn{2}{c}{Job $-$ Flyte regret, paired (pp)} \\\\\n\\cmidrule(l){2-3}"))

# ── real run/Job pairs ──────────────────────────────────────────────────
PAIRV = ["rules", "rules-sized", "zeroshot-27b", "gbt-dist", "gbt-dist-k8s", "gbt-dist-flyte", "jev-pointer",
         "jev-pointer-k8s", "jev-pointer-flyte", "rl-pointer", "pool-jev-pointer", "pool-rl-pointer-argmax",
         "gate-rl-pointer"]
rows = []
for v in PAIRV:
    p = (A1["pairs"].get(v) or {}).get("test")
    if not p:
        continue
    r = lambda k: pct(p[k]["rate"][0]) if k in p else "---"  # noqa: E731
    rows.append([NAME[v], r("moved"), r("moved_up"), r("moved_down"), r("unmoved"), r("all")])
s0 = A1["pairs"]["static"]["test"]
write("tab_pairs.tex", tabular("lrrrrr", ["approach", "moved", "up", "down", "unmoved, kept", "all"], rows,
                               "& \\multicolumn{3}{c}{label moved: direction right (\\%)} && \\\\\n\\cmidrule(lr){2-4}"))
write("num_pairs.tex", "\n".join([
    f"\\newcommand{{\\pairsN}}{{{s0['all']['n']:,}}}",
    f"\\newcommand{{\\pairsMoved}}{{{pct(s0['is_moved']['rate'][0])}}}",
    f"\\newcommand{{\\pairsUp}}{{{s0['moved_up']['n']:,}}}",
    f"\\newcommand{{\\pairsDown}}{{{s0['moved_down']['n']:,}}}", ""]))

# ── OOM-penalty sensitivity (Job pods) ──────────────────────────────────
rows = []
for v in LEAD:
    s = A1["sens"].get(v)
    if not s:
        continue
    rows.append([NAME[v]] + [pct((s[k] or {}).get(sp, [None])[0]) if s.get(k) else "---"
                             for k in ("oom0", "base", "oom5", "oom20") for sp in ("test", "ood")])
for c in range(1, 9):
    best(rows, c)
write("tab_sens.tex", tabular(
    "lrrrrrrrr", ["approach"] + ["test", "unseen"] * 4, rows,
    "& \\multicolumn{2}{c}{0} & \\multicolumn{2}{c}{1 (default)} & \\multicolumn{2}{c}{5} & \\multicolumn{2}{c}{20} \\\\\n"
    "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(l){8-9}"))

# ── cold vs warm (Job pods) ─────────────────────────────────────────────
rows = []
for v in LEAD:
    c = A1["cold"].get(v)
    if not c:
        continue
    rows.append([NAME[v], pct(c.get("test/k8s/cold")), pct(c.get("test/k8s/warm")),
                 pct(c.get("ood/k8s/cold")), pct(c.get("ood/k8s/warm"))])
for c in range(1, 5):
    best(rows, c)
write("tab_cold.tex", tabular("lrrrr", ["approach", "first run", "with history", "first run", "with history"], rows,
                              "& \\multicolumn{2}{c}{test} & \\multicolumn{2}{c}{unseen} \\\\\n\\cmidrule(lr){2-3}\\cmidrule(l){4-5}"))

# ── staging (Job CPU-node test pods) ────────────────────────────────────
rows = []
for v in LEAD:
    s = A1["staging"].get(v)
    if not s:
        continue
    rows.append([NAME[v]] + [pct((s.get(st) or {}).get("regret", [None])[0]) for st in
                             ("init-disk", "init-tmpfs", "boto3-disk", "boto3-tmpfs")])
for c in range(1, 5):
    best(rows, c)
write("tab_staging.tex", tabular("lrrrr", ["approach", "init-disk", "init-tmpfs", "boto3-disk", "boto3-tmpfs"], rows))

# ── seeds ───────────────────────────────────────────────────────────────
rows = []
for v in LEAD_ALL + ["jev-pointer-k8s", "jev-pointer-flyte", "gbt-dist-k8s", "gbt-dist-flyte"]:
    xs = {s: a["table"].get(v) for s, a in A.items()}
    if sum(1 for x in xs.values() if x) < 2:
        continue
    rows.append([NAME[v]] + [pct((xs.get(s) or {}).get(k, {}).get("regret", [None])[0]) if xs.get(s) else "---"
                             for k in ("test/k8s", "ood/k8s") for s in SEEDS])
write("tab_seeds.tex", tabular(
    "lrrrrrr", ["approach", "1", "2", "3", "1", "2", "3"], rows,
    "& \\multicolumn{3}{c}{Job test, by seed} & \\multicolumn{3}{c}{Job unseen, by seed} \\\\\n"
    "\\cmidrule(lr){2-4}\\cmidrule(l){5-7}"))

# ── counterfactual and names ────────────────────────────────────────────
cfs = {}
for run in ("rsk-k8s-2", "rsk-k8s-2-s2", "rsk-k8s-2-s3"):
    p = f"{CF}/{run}-counterfactual.json.gz"
    if os.path.exists(p):
        cfs[run] = json.load(gzip.open(p))
if cfs:
    cf = cfs["rsk-k8s-2"]
    CFV = ["static", "rules", "gbt-dist", "jev-pointer", "jev-pointer-k8s", "jev-pointer-flyte", "rl-pointer",
           "pool-jev-pointer", "pool-rl-pointer-argmax"]
    rows = []
    for v in CFV:
        x = cf["report"].get(v)
        if not x:
            continue
        r = []
        for sp in ("test", "ood"):
            m = x["split"][sp]["medium"]
            fe = x["split"][sp]["fetcher"]
            r += [pct(m["moved_direction"]), pct(m["unmoved_kept"]), pct(fe["moved_direction"])]
        rows.append([NAME[v]] + r)
    m0 = cf["report"]["static"]["split"]
    write("tab_cf.tex", tabular(
        "lrrrrrr", ["approach", "moved, right", "unmoved, kept", "fetcher moved", "moved, right", "unmoved, kept",
                    "fetcher moved"], rows,
        "& \\multicolumn{3}{c}{test} & \\multicolumn{3}{c}{unseen} \\\\\n\\cmidrule(lr){2-4}\\cmidrule(l){5-7}"))
    write("num_cf.tex", "\n".join([
        f"\\newcommand{{\\cfRuns}}{{{cf['n_runs']:,}}}",
        f"\\newcommand{{\\cfRend}}{{{cf['n_renditions']:,}}}",
        f"\\newcommand{{\\cfMediumMovedTest}}{{{pct(m0['test']['medium']['label_moved'])}}}",
        f"\\newcommand{{\\cfMediumMovedOod}}{{{pct(m0['ood']['medium']['label_moved'])}}}",
        f"\\newcommand{{\\cfFetchMovedTest}}{{{pct(m0['test']['fetcher']['label_moved'])}}}",
        f"\\newcommand{{\\cfFetchMovedOod}}{{{pct(m0['ood']['fetcher']['label_moved'])}}}", ""]))
    rows = []
    for v in CFV + ["zeroshot-27b"]:
        x = cf["names"].get(v)
        if not x or "test" not in x:
            continue
        rows.append([NAME[v]] + [f"{pct(x[sp]['named'])}" if sp in x else "---" for sp in ("test",)]
                    + [pct(x["test"]["neutral"]), f"{100*x['test']['diff']:+.1f}"]
                    + ([pct(x["ood"]["named"]), pct(x["ood"]["neutral"]), f"{100*x['ood']['diff']:+.1f}"] if "ood" in x
                       else ["---"] * 3))
    write("tab_names.tex", tabular(
        "lrrrrrr", ["approach", "named", "neutral", "$\\Delta$", "named", "neutral", "$\\Delta$"], rows,
        "& \\multicolumn{3}{c}{test regret (\\%)} & \\multicolumn{3}{c}{unseen regret (\\%)} \\\\\n"
        "\\cmidrule(lr){2-4}\\cmidrule(l){5-7}"))

# ── gates: signals, routing and regret ──────────────────────────────────
GATES = [("kNN, all features", "knn", "gate-knn-q95"), ("kNN, k8s-1 scaling", "knn-raw", "gate-knn-raw-q95"),
         ("kNN, numeric only", "knn-numeric", "gate-knn-numeric-q95"), ("disagreement (JS), $q$ 0.5", "js", "gate-js-q50")]
meta = next(x for v, x in json.load(gzip.open(f"{D}/rsk-k8s-2-picks.json.gz"))["metas"].items() if v.startswith("gate"))
rows = []
for label, sig, v in GATES:
    k = meta["signals"][sig]
    q = k["q"].get(v, {}).get("gated_out", {})
    t = A1["table"].get(v, {})
    rows.append([label, f"{k['auroc_val_vs_ood']:.3f}", pct(q.get("test")), pct(q.get("ood")),
                 cell(t, "test/k8s"), cell(t, "ood/k8s")])
write("tab_gate.tex", tabular("lrrrrr", ["signal", "AUROC", "test", "unseen", "test", "unseen"], rows,
                              "&& \\multicolumn{2}{c}{sent to RL (\\%)} & \\multicolumn{2}{c}{Job regret (\\%)} \\\\\n"
                              "\\cmidrule(lr){3-4}\\cmidrule(l){5-6}"))

# ── the numbers the prose quotes (round 24) ─────────────────────────────
KEY = {"static": "Static", "rules": "Rules", "rules-sized": "Sized", "gbt-dist": "Trees", "gbt-dist-k8s": "TreesJob",
       "gbt-dist-flyte": "TreesFlyte", "zeroshot-27b": "ZS", "zeroshot-27b-json": "ZSJ", "jev-pointer": "Dec",
       "jev-pointer-argmax": "DecArg", "jev-pointer-k8s": "DecJob", "jev-pointer-flyte": "DecFlyte",
       "jev-pointer-flyte-adapt": "DecAdapt", "rl-pointer": "RL", "rl-pointer-argmax": "RLArg",
       "pool-jev-pointer": "PoolDec", "pool-jev-pointer-argmax": "PoolDecArg", "pool-jev-pointer-k8s": "PoolDecJob",
       "pool-jev-pointer-flyte": "PoolDecFlyte", "pool-rl-pointer": "PoolRL", "pool-rl-pointer-argmax": "PoolRLArg",
       "gate-rl-pointer": "Gate"}
SPLITKEY = {"test/k8s": "Test", "ood/k8s": "Unseen", "test/flyte": "FTest", "ood/flyte": "FUnseen", "val/k8s": "Val"}
lines = []


_defined = set()


def mac(name, val):
    if name in _defined:  # e.g. usdCheap*, the same for every approach
        return
    _defined.add(name)
    lines.append(f"\\newcommand{{\\{name}}}{{{val}}}")


for v, k in KEY.items():
    t = A1["table"].get(v)
    if not t:
        continue
    for sk, sn in SPLITKEY.items():
        if sk in t:
            x = t[sk]["regret"]
            mac(f"r{sn}{k}", pct(x))
            mac(f"ci{sn}{k}", f"[{pct(x[1])}, {pct(x[2])}]")
    if "ood/k8s" in t:
        x = t["ood/k8s"]["regret_fam"]
        mac(f"ciFamUnseen{k}", f"[{pct(x[1])}, {pct(x[2])}]")
    for sk, sn in (("test/k8s", "Test"), ("ood/k8s", "Unseen")):
        xs = [a["table"][v][sk]["regret"][0] for a in A.values() if v in a["table"] and sk in a["table"][v]]
        if len(xs) == 3:
            mac(f"m{sn}{k}", pct(np.mean(xs)))
            mac(f"lo{sn}{k}", pct(min(xs)))
            mac(f"hi{sn}{k}", pct(max(xs)))
        if sk in t:
            mac(f"d{sn}{k}", pct(t[sk]["dollar_regret"]))
            mac(f"usd{sn}{k}", f"{t[sk]['cost_per_1k']:,.0f}")
            mac(f"usdCheap{sn}", f"{t[sk]['oracle_per_1k']:,.0f}")
    for sp, sn in (("test", "Test"), ("ood", "Unseen")):
        for s_, pn in (("oom0", "Zero"), ("oom5", "Five"), ("oom20", "Twenty")):
            x = (A1["sens"].get(v) or {}).get(s_, {}).get(sp)
            if x:
                mac(f"p{pn}{sn}{k}", pct(x))
    z = A1.get("zs_sample", {}).get(v)
    if z and z.get("test"):
        mac(f"zsTest{k}", pct(z["test"])), mac(f"zsUnseen{k}", pct(z["ood"]))
    c = A1.get("code", {}).get(v)
    if c and c.get("test/novel"):
        mac(f"seen{k}", pct(c["test/seen"])), mac(f"novel{k}", pct(c["test/novel"]))
    c = A1.get("corpus", {}).get(v)
    for sp, sn in (("test", "Test"), ("ood", "Unseen")):
        for src, cn in (("mem", "Mem"), ("cpu", "Cpu"), ("gpu", "Gpu")):
            if c and c.get(f"{sp}/{src}"):
                mac(f"c{sn}{cn}{k}", pct(c[f"{sp}/{src}"]))


def dmac(name, d, fam=False):
    x = d["diff_fam" if fam else "diff"]
    mac(name, f"{100*x[0]:+.1f}"), mac(name + "CI", f"[{100*x[1]:+.1f}, {100*x[2]:+.1f}]")


for k, d in A1["diff"].items():
    a, b = (KEY.get(z) for z in k.split(" - "))
    if not a or not b:
        continue
    for sk, sn in (("test/k8s", "Test"), ("ood/k8s", "Unseen")):
        if d.get(sk):
            dmac(f"diff{sn}{a}{b}", d[sk])
            if sn == "Unseen" and "diff_fam" in d[sk]:
                dmac(f"diffFam{sn}{a}{b}", d[sk], fam=True)
for k, d in A1.get("sens_diff", {}).items():
    a, b = (KEY.get(z) for z in k.split(" - "))
    for s_, pn in (("oom5", "Five"), ("oom20", "Twenty")):
        for sp, sn in (("test", "Test"), ("ood", "Unseen")):
            x = d.get(f"{s_}/{sp}")
            if x and a and b:
                dmac(f"sd{pn}{sn}{a}{b}", x), dmac(f"sdFam{pn}{sn}{a}{b}", x, fam=True)
for key, sn in (("test/k8s", "Test"), ("ood/k8s", "Unseen")):
    vb = A1.get("vs_best", {}).get(key, {})
    for v, d in vb.items():
        if v != "best" and d and KEY.get(v):
            dmac(f"vb{sn}{KEY[v]}", d), dmac(f"vbFam{sn}{KEY[v]}", d, fam=True)
if A1.get("code", {}).get("_n"):
    mac("codeSeenShare", pct(A1["code"]["_share_seen"], 0))
    mac("codeSeenN", f"{A1['code']['_n']['seen']:,}"), mac("codeNovelN", f"{A1['code']['_n']['novel']:,}")
write("num_main.tex", "\n".join(lines) + "\n")
