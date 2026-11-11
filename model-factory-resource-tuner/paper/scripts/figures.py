"""Draw the paper's charts from the analysis files (round 24).

    uv run --with matplotlib python scripts/figures.py <analysis dir>
        -> figures/fig_*.pdf (the paper) and figures/fig_*.svg (the HTML report)

The same drawing makes both. In the SVG, every color is one of the palette's
hex values below, which ``to_html.py`` swaps for the report's CSS variables,
so the charts follow its light and dark themes. Fonts: STIX in the PDF (it
matches the paper's Times); in the SVG, text stays text and takes the page's font.

Colors (validated with the dataviz skill's validator, light and dark, all pairs):
approach classes are trees (slot 1, blue), language models (slot 2, orange)
and combinations (slot 3, aqua); baselines are gray. Each class also has its own
marker, and every mark is labelled, so color is never the only cue.
"""
import gzip
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

D = sys.argv[1]
OUT = os.path.join(os.path.dirname(__file__), "..", "figures")
SEEDS = ("k2s1", "k2s2", "k2s3")
A = {s: json.load(open(f"{D}/{s}-analysis.json")) for s in SEEDS}
A1 = A["k2s1"]

INK, MUTED, GRID, PANEL = "#1F2328", "#6A737D", "#DCDFE8", "#FFFFFF"
S1, S2, S3, BASE = "#2A78D6", "#EB6834", "#1BAF7A", "#8A8FA0"
#: approach -> (label, class color, marker)
STYLE = {
    "static": ("static", BASE, "s"), "rules": ("rules", BASE, "s"), "rules-sized": ("rules-sized", BASE, "s"),
    "zeroshot-27b": ("zero-shot 27B", BASE, "D"),
    "gbt-dist": ("trees", S1, "o"),
    "jev-pointer": ("decision", S2, "^"), "rl-pointer": ("RL", S2, "^"),
    "pool-jev-pointer": ("pool × decision", S3, "D"), "pool-rl-pointer": ("pool × RL", S3, "D"),
    "gate-rl-pointer": ("gate", S3, "P"),
}


def sel(v):
    """The read-out validation chose for v (round 24); v itself if it has one read-out."""
    return A1.get("selected", {}).get(v, v)


def label(v):
    base = v.removesuffix("-argmax")
    name = STYLE.get(base, (base,))[0]
    return name + (", argmax" if v.endswith("-argmax") else "")


plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8.5,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none",
    "text.color": INK, "legend.frameon": False, "figure.facecolor": PANEL, "axes.facecolor": PANEL, "savefig.facecolor": PANEL,
})


def save(fig, name):
    for ext in ("pdf", "svg"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", name)


def grid(ax, axis="both"):
    ax.grid(True, axis=axis, color=GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)


# ── 1. familiar vs unseen code: the main result ─────────────────────────
#: label placement (points, points, alignment), chosen by eye against the seed-1 positions
OFFSETS = {"gate-rl-pointer": (-7, -5, "right"), "rl-pointer": (-7, 3, "right"), "pool-jev-pointer": (-7, 0, "right"),
           "pool-rl-pointer": (-7, 4, "right"), "jev-pointer": (7, -4, "left"), "gbt-dist": (7, 4, "left"),
           "rules-sized": (-7, -6, "right"), "rules": (6, 6, "left"), "zeroshot-27b": (7, -5, "left"),
           "static": (7, 0, "left")}
def fig_tradeoff():
    vs = ["static", "rules", "rules-sized", "zeroshot-27b", "gbt-dist", "jev-pointer", "rl-pointer",
          "pool-jev-pointer", "pool-rl-pointer", "gate-rl-pointer"]
    fig, ax = plt.subplots(figsize=(6.3, 3.4))
    for v in vs:
        s = sel(v)
        if s not in A1["table"]:
            continue
        t, o = A1["table"][s]["test/k8s"], A1["table"][s]["ood/k8s"]
        x, y = 100 * np.array(t["regret"]), 100 * np.array(o["regret"])
        yf = 100 * np.array(o.get("regret_fam", o["regret"]))
        _, col, mk = STYLE[v]
        hollow = v == "zeroshot-27b"
        ax.plot([yf[0]] * 0, [])
        ax.plot([x[0], x[0]], [yf[1], yf[2]], color=col, alpha=0.25, linewidth=4, solid_capstyle="butt", zorder=2)
        ax.errorbar(x[0], y[0], xerr=[[x[0] - x[1]], [x[2] - x[0]]], yerr=[[y[0] - y[1]], [y[2] - y[0]]],
                    fmt=mk, ms=6, color=col, mfc=PANEL if hollow else col, mec=col, mew=1.2, elinewidth=0.9,
                    capsize=0, zorder=3)
        dx, dy, ha = OFFSETS.get(v, (6, 3, "left"))
        ax.annotate(label(s), (x[0], y[0]), xytext=(dx, dy), textcoords="offset points", fontsize=8, color=INK,
                    ha=ha, va="center", zorder=4)
    ax.set_xscale("log"), ax.set_yscale("log")
    ticks = [10, 20, 50, 100]
    for setter in (ax.set_xticks, ax.set_yticks):
        setter(ticks)
    ax.set_xticklabels([str(t) for t in ticks]), ax.set_yticklabels([str(t) for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel("regret on familiar families (test, Job pods, %)")
    ax.set_ylabel("regret on held-out families (unseen, %)")
    grid(ax)
    ax.plot([1, 1000], [1, 1000], color=GRID, linewidth=0.8, linestyle=(0, (3, 3)), zorder=1)
    ax.text(150, 150, "same regret on both", fontsize=7, color=MUTED, rotation=33, ha="right", va="bottom")
    ax.set_xlim(8, 170), ax.set_ylim(15, 175)
    ax.text(0.99, 0.02, "thin bars: 95% interval, clustered by task\nshaded bar: unseen interval, clustered by family (7 families)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color=MUTED)
    save(fig, "fig_tradeoff")


# ── 2. the price of a retry ─────────────────────────────────────────────
def fig_penalty():
    vs = ["rules-sized", "gbt-dist", "rl-pointer", "pool-jev-pointer", "gate-rl-pointer"]
    dash = {"rules-sized": "-", "gbt-dist": "-", "rl-pointer": "-", "pool-jev-pointer": "-", "gate-rl-pointer": (0, (3, 2))}
    pen = [("oom0", 0), ("base", 1), ("oom5", 5), ("oom20", 20)]
    xs = np.arange(len(pen))
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.5), sharey=True)
    for ax, split, title in zip(axes, ("test", "ood"), ("familiar families (test)", "held-out families (unseen)")):
        for v in vs:
            s = sel(v)
            sens = A1["sens"].get(s)
            if not sens:
                continue
            ys = [100 * sens[k][split][0] for k, _ in pen]
            _, col, mk = STYLE[v]
            ax.plot(xs, ys, color=col, linestyle=dash[v], linewidth=1.6, marker=mk, ms=4.5, mfc=col, zorder=3,
                    label=label(s) if split == "test" else None)
        ax.set_yscale("log")
        ax.set_xticks(xs, [str(p) for _, p in pen])
        ax.set_xlim(-0.2, len(pen) - 0.8)
        ax.set_xlabel("cost of an OOM retry (in cheapest-request costs)")
        ax.set_title(title, fontsize=8.5, color=INK, loc="left")
        grid(ax, "y")
    axes[0].set_ylabel("regret on Job pods (%)")
    fig.legend(loc="upper center", ncol=5, fontsize=7.5, bbox_to_anchor=(0.5, 1.08), handlelength=2.2)
    for ax in axes:
        ax.set_yticks([5, 10, 20, 50, 100, 200, 500])
        ax.set_yticklabels(["5", "10", "20", "50", "100", "200", "500"])
        ax.minorticks_off()
    save(fig, "fig_penalty")


# ── 3. does the request follow the manifest? ────────────────────────────
def fig_counterfactual():
    cf = json.load(gzip.open(f"{D}/rsk-k8s-2-counterfactual.json.gz"))["report"]
    vs = [("static", "static", BASE, "s"), ("rules", "rules", BASE, "s"), ("gbt-dist", "trees", S1, "o"),
          ("jev-pointer", "decision", S2, "^"), ("jev-pointer-k8s", "decision, Job-only", S2, "^"),
          ("jev-pointer-flyte", "decision, Flyte-only", S2, "v"), ("rl-pointer", "RL", S2, "^"),
          ("pool-jev-pointer", "pool × decision", S3, "D")]
    vs = [v for v in vs if v[0] in cf]
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.6), sharey=True)
    ys = np.arange(len(vs))[::-1]
    for ax, split, title in zip(axes, ("test", "ood"), ("familiar families (test)", "held-out families (unseen)")):
        for y, (v, name, col, mk) in zip(ys, vs):
            m = cf[v]["split"][split]["medium"]
            right, kept = 100 * m["moved_direction"], 100 * m["unmoved_kept"]
            ax.plot([right, kept], [y, y], color=GRID, linewidth=1.2, zorder=1)
            ax.plot(right, y, marker=mk, ms=6, color=col, mfc=col, linestyle="none", zorder=3)
            ax.plot(kept, y, marker=mk, ms=6, color=col, mfc=PANEL, mew=1.2, linestyle="none", zorder=3)
        ax.set_xlim(-3, 103)
        ax.set_title(title, fontsize=8.5, color=INK, loc="left")
        ax.set_xlabel("share of first-run pairs (%)")
        grid(ax, "x")
    axes[0].set_yticks(ys, [v[1] for v in vs])
    axes[0].tick_params(axis="y", length=0)
    axes[1].plot([], [], marker="o", color=INK, linestyle="none", ms=5, label="label moved: request moved the same way")
    axes[1].plot([], [], marker="o", color=INK, mfc=PANEL, linestyle="none", ms=5, label="label stayed: request stayed")
    fig.legend(loc="upper center", ncol=2, fontsize=7.5, bbox_to_anchor=(0.55, 1.08))
    save(fig, "fig_counterfactual")


# ── 4. what the benchmark covers ────────────────────────────────────────
def fig_coverage():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from resource_tuner.bench_all import grid as g
    from resource_tuner.bench_gpu.grid import DEVICES

    rows = json.load(gzip.open(f"{D}/rsk-k8s-2-rows.json.gz"))
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.2), gridspec_kw={"width_ratios": [17, 16, 9]})
    corp = (("mem", "RightSizeMem", S1), ("cpu", "RightSizeCPU", S2))
    for ax, key, n, opts, title in ((axes[0], "label_mem", g.N_MEM, g.MEM_OPTIONS, "memory label (Job pods)"),
                                    (axes[1], "label_cpu", g.N_CPU, g.CPU_OPTIONS, "CPU label")):
        bottom = np.zeros(n)
        for src, name, col in corp:
            c = np.bincount([int(r[key]) for r in rows if r["source"] == src and r["platform"] == "k8s"], minlength=n)[:n]
            ax.bar(np.arange(n), c, bottom=bottom, color=col, width=0.8, edgecolor=PANEL, linewidth=0.6, label=name)
            bottom += c
        step = 4 if n > 16 else 3
        ax.set_xticks(np.arange(0, n, step), [opts[i] for i in range(0, n, step)], fontsize=7)
        ax.set_title(title, fontsize=8.5, color=INK, loc="left")
        grid(ax, "y")
    # GPU: the machine the label lands on
    names = [d.name for d in DEVICES]
    c = np.zeros(len(names))
    for r in rows:
        if r["source"] == "gpu" and r["platform"] == "k8s":
            c[g.device_of(int(r["label_gpu"]))] += 1
    axes[2].bar(np.arange(len(names)), c, color=S3, width=0.8, edgecolor=PANEL, linewidth=0.6, label="RightSizeGPU")
    axes[2].set_xticks(np.arange(len(names)), names, rotation=50, ha="right", fontsize=6.8)
    axes[2].set_title("cheapest GPU machine", fontsize=8.5, color=INK, loc="left")
    grid(axes[2], "y")
    axes[0].set_ylabel("evaluation pods")
    h0, l0 = axes[0].get_legend_handles_labels()
    h2, l2 = axes[2].get_legend_handles_labels()
    fig.legend(h0 + h2, l0 + l2, loc="upper center", ncol=3, fontsize=7.5, bbox_to_anchor=(0.5, 1.1))
    save(fig, "fig_coverage")


for f in (fig_tradeoff, fig_penalty, fig_counterfactual, fig_coverage):
    f()
