"""Write the paper's generated tables and figure data from the analysis files.

    python scripts/tables.py <analysis dir> <counterfactual dir>  ->  generated/*.tex, generated/*.dat

Inputs: ``k2s{1,2,3}-analysis.json`` (``analyze.py``) and
``rsk-k8s-2{,-s2,-s3}-counterfactual.json.gz`` (``paper_file`` on the
counterfactual station's output). Seed 1 gives the point estimates and
intervals; seeds 2 and 3 give the range across trainings.
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
LEAD = ["static", "rules", "rules-sized", "zeroshot-27b", "zeroshot-27b-json", "gbt-dist", "jev-pointer",
        "jev-pointer-argmax", "rl-pointer", "rl-pointer-argmax", "pool-jev-pointer", "pool-rl-pointer",
        "pool-rl-pointer-argmax", "gate-rl-pointer"]


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


# ── leaderboard: Job pods first ─────────────────────────────────────────
rows = []
for v in LEAD:
    if v not in A1["table"]:
        continue
    t = A1["table"][v]
    rows.append([NAME[v], cell(t, "test/k8s"), cell(t, "ood/k8s"), cell(t, "test/flyte", ci=False),
                 cell(t, "ood/flyte", ci=False), cell(t, "test/k8s", "dollar_regret", ci=False),
                 cell(t, "ood/k8s", "dollar_regret", ci=False), seeds(v, "test/k8s"), seeds(v, "ood/k8s")])
for c in (1, 2, 3, 4, 5, 6):
    best(rows, c)
write("tab_leaderboard.tex", tabular(
    "lrrrrrrrr",
    ["approach", "test", "unseen", "test", "unseen", "test", "unseen", "test", "unseen"],
    rows,
    "& \\multicolumn{2}{c}{Job pods: regret (\\%)} & \\multicolumn{2}{c}{Flyte pods} & "
    "\\multicolumn{2}{c}{Job \\$ regret} & \\multicolumn{2}{c}{Job, 3 seeds: mean (range)} \\\\\n"
    "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(l){8-9}"))

# figure data: Job-pod regret with intervals, test and unseen
with open(os.path.join(OUT, "fig_leaderboard.dat"), "w") as f:
    f.write("i name test tlo thi ood olo ohi\n")
    i = 0
    for v in LEAD:
        if v not in A1["table"]:
            continue
        t, o = A1["table"][v]["test/k8s"]["regret"], A1["table"][v]["ood/k8s"]["regret"]
        f.write(f"{i} {{{NAME[v]}}} {100*t[0]:.2f} {100*t[1]:.2f} {100*t[2]:.2f} {100*o[0]:.2f} {100*o[1]:.2f} {100*o[2]:.2f}\n")
        i += 1
write("fig_leaderboard_ticks.tex", ",".join("{" + NAME[v] + "}" for v in LEAD if v in A1["table"]))

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
for v in LEAD + ["jev-pointer-k8s", "jev-pointer-flyte"]:
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
for v in LEAD + ["jev-pointer-k8s", "jev-pointer-flyte", "gbt-dist-k8s", "gbt-dist-flyte"]:
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
