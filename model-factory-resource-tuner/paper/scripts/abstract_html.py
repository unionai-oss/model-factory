"""Write the one-page abstract (benchmarks/rightsize-bench-k8s-abstract.html).

    python scripts/abstract_html.py <analysis dir> <style source html> <out.html>

The prose is written here; the chart and table come from the seed analyses.
"""
import html
import json
import sys

import numpy as np

D, STYLE_SRC, OUT = sys.argv[1:4]
A = {s: json.load(open(f"{D}/{s}-analysis.json")) for s in ("k2s1", "k2s2", "k2s3")}
A1 = A["k2s1"]
ROWS = [("static", "static request", "--rule-c"), ("rules", "rules", "--rule-c"),
        ("rules-sized", "rules, sized by input and manifest", "--rule-c"),
        ("zeroshot-27b", "zero-shot 27B", "--warm"), ("gbt-dist", "gradient-boosted trees", "--tree"),
        ("jev-pointer", "decision model (1.7B)", "--accent"), ("rl-pointer", "decision model + RL", "--rl"),
        ("pool-jev-pointer", "trees × decision model", "--pool"),
        ("pool-rl-pointer-argmax", "trees × RL, argmax", "--pool"), ("gate-rl-pointer", "novelty gate", "--gate")]


def r(v, k, m="regret"):
    return A1["table"][v][k][m]


def seeds(v, k):
    xs = [a["table"][v][k]["regret"][0] for a in A.values() if v in a["table"]]
    return np.mean(xs), min(xs), max(xs), len(xs)


# chart: Job regret, test (filled) and unseen (open), with intervals
W, L, R_, T, RH = 760, 230, 20, 10, 30
H = T + RH * len(ROWS) + 44
x = lambda v: L + (W - L - R_) * min(v, 125) / 125  # noqa: E731
svg = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Regret on Job pods by approach">']
for g in range(0, 121, 20):
    svg.append(f'<line x1="{x(g):.1f}" x2="{x(g):.1f}" y1="{T}" y2="{T + RH * len(ROWS)}" stroke="var(--rule)"/>'
               f'<text x="{x(g):.1f}" y="{T + RH * len(ROWS) + 16}" text-anchor="middle" font-size="11" fill="var(--muted)">{g}</text>')
svg.append(f'<text x="{(L + W - R_) / 2:.0f}" y="{H - 6}" text-anchor="middle" font-size="12" fill="var(--muted)">regret on Job pods (%)</text>')
for i, (v, name, col) in enumerate(ROWS):
    y = T + RH * i + RH / 2
    svg.append(f'<text x="{L - 10}" y="{y + 4:.1f}" text-anchor="end" font-size="13" fill="var(--ink)">{html.escape(name)}</text>')
    for k, dy, fill in (("test/k8s", -5, f"var({col})"), ("ood/k8s", 5, "var(--panel)")):
        m, lo, hi = (100 * z for z in r(v, k))
        svg.append(f'<line x1="{x(lo):.1f}" x2="{x(hi):.1f}" y1="{y + dy:.1f}" y2="{y + dy:.1f}" stroke="var({col})" stroke-width="1.5"/>'
                   f'<circle cx="{x(m):.1f}" cy="{y + dy:.1f}" r="4.5" fill="{fill}" stroke="var({col})" stroke-width="2">'
                   f'<title>{html.escape(name)}, {"test" if k.startswith("test") else "unseen"}: {m:.1f}% [{lo:.1f}, {hi:.1f}]</title></circle>')
svg.append("</svg>")


def f(z):
    return f"{100 * z:.1f}"


trs = []
for v, name, col in ROWS:
    ms = seeds(v, "test/k8s"), seeds(v, "ood/k8s")
    sd = lambda s: f"{100*s[1]:.1f}–{100*s[2]:.1f}" if s[3] > 1 and s[2] - s[1] > 1e-9 else ("one training" if s[3] == 1 else "fixed")  # noqa: E731
    trs.append(f'<tr><td><span class="kind" style="background:var({col})"></span>{html.escape(name)}</td>'
               f'<td>{f(r(v, "test/k8s")[0])}</td><td>{f(r(v, "ood/k8s")[0])}</td>'
               f'<td>{f(r(v, "test/flyte")[0])}</td><td>{f(r(v, "ood/flyte")[0])}</td>'
               f'<td>{f(r(v, "test/k8s", "dollar_regret")[0])}</td>'
               f'<td class="muted">{sd(ms[0])}</td><td class="muted">{sd(ms[1])}</td></tr>')

src = open(STYLE_SRC).read()
head = src[:src.index("</style>")] + """
.fig-legend { display: flex; flex-wrap: wrap; gap: 6px 18px; font-size: 13px; color: var(--muted); margin-bottom: 8px; }
.dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; vertical-align: -1px; }
</style>"""
head = head.replace("<title>RightSize Bench K8s Abstract</title>", "<title>RightSize Bench K8s Abstract</title>")
body = f"""
<div class="wrap">
  <header>
    <div class="eyebrow">Union model factory · round 23 · November 2026</div>
    <h1>RightSize Bench: learning Kubernetes resource requests from code</h1>
    <span class="status">k8s-2 · 32,082 measured runs, each as a raw Job and as a Flyte task · 3 trainings</span>
  </header>

  <div class="abstract">
    <div class="label">Abstract</div>
    <p>Before a batch pod runs on Kubernetes, someone has to decide how much memory and CPU it gets, and whether
    it gets a GPU. Too little makes it fail or run slowly, and too much wastes the cluster. RightSize Bench scores
    approaches to this decision by what their requests would cost to get each run to finish, compared with the
    cheapest request that would have worked. It contains 32,082 measured runs of 33 families of generated Python
    tasks that use real libraries, from pandas and DuckDB to PyTorch training and sharded language models. Every run
    appears twice: once as a raw Kubernetes Job, with its inputs staged in one of four ways, and once as a task
    submitted through Flyte, an AI-native interface to Kubernetes that adds typed I/O, caching, retries and durable
    execution for about 70&nbsp;MiB of runtime per pod. We checked the labels in about 480 real pods, which found and
    fixed two errors in our first release.</p>
    <p>On new tasks from familiar families, pooling gradient-boosted trees with a fine-tuned 1.7B decision model
    gives <b>{f(r("pool-jev-pointer","test/k8s")[0])}% regret on Job pods</b>
    [{f(r("pool-jev-pointer","test/k8s")[1])}, {f(r("pool-jev-pointer","test/k8s")[2])}].
    On task families held out of training, every approach does much worse, and the leaders change: the decision model
    trained further with RL and a hand-written rule that scales by input size and reads the manifest tie at
    <b>39%</b>. When an out-of-memory failure is as expensive as resubmitting a Job by hand, that rule is the best
    approach on new code. Models trained on Jobs do read the manifest, and adding Flyte data does not make them
    better on Jobs.</p>
  </div>

  <section>
    <div class="label">Results</div>
    <h2>Regret on Job pods, by approach</h2>
    <div class="figure">
      <div class="fig-legend"><span><span class="dot" style="background:var(--ink)"></span>test: new tasks from familiar families</span>
      <span><span class="dot" style="border:2px solid var(--ink)"></span>unseen families</span></div>
      {"".join(svg)}
      <p class="caption">Regret is a run's cost under the approach's request divided by its cost under the oracle's,
      minus one, averaged over runs. Bars are 95% intervals that resample whole tasks. Seed 1; 5,095 test and 6,416
      unseen Job pods. Zero-shot is scored on a sample of about 300 runs per split.</p>
    </div>
    <div class="figure table-wrap"><table class="num">
      <thead><tr><th>approach</th><th>Job test</th><th>Job unseen</th><th>Flyte test</th><th>Flyte unseen</th><th>Job $ regret</th><th>3 seeds, test</th><th>3 seeds, unseen</th></tr></thead>
      <tbody>{"".join(trs)}</tbody></table>
      <p class="caption">Regret in percent, seed 1. Dollar regret weighs runs by what they cost. The last two columns
      give the range over three trainings; rules and trees are deterministic.</p></div>
  </section>

  <section>
    <h2>What we found</h2>
    <ol class="findings">
      <li><span class="n">70 MiB</span><span><b>Real pods changed the labels.</b> The Flyte runtime we had assumed
      was 180&nbsp;MiB measures about 70, and two-thirds of Flyte labels were a cell too high. Init containers in our
      first release were too small to hold what they staged, so 7% of Job rows described pods that could not start.
      Both are fixed in this release.</span></li>
      <li><span class="n">3 in 4</span><span><b>Models trained on Jobs read the manifest.</b> When we change only the
      staging volume from disk to <span class="mono">medium: Memory</span> on a first run, the trained models move their
      request the right way about three times in four. A model trained only on Flyte tasks does so about one time in
      five, and rules never do.</span></li>
      <li><span class="n">+0.6 pt</span><span><b>Flyte data does not help with Jobs.</b> A decision model trained only on
      Jobs is within a point of one trained on both kinds of pod, while one trained only on Flyte tasks is three times
      worse on Jobs. An adapter that applies the staging accounting recovers a third to a half of that gap.</span></li>
      <li><span class="n">39%</span><span><b>New code is the open problem.</b> On held-out families the best trained model
      and a careful rule tie at 39%, and the rule wins when failures are expensive. A novelty gate fixed in advance
      gets 8.1% on familiar code and 39.3% on new code, but its detector has an easy job here, since new families
      bring new libraries.</span></li>
    </ol>
  </section>

  <section>
    <div class="label">Setup</div>
    <h2>How the benchmark works</h2>
    <dl class="defs">
      <dt>Runs</dt><dd>32,082 usable runs from three corpora (RightSizeMem, RightSizeCPU and RightSizeGPU), each measured
      in Kubernetes pods for peak memory, CPU over the busiest 15 seconds, and peak GPU memory. Tasks are split 70/10/20
      into train, validation and test, and seven whole families are held out as the unseen split.</dd>
      <dt>Two pods per run</dt><dd>The same task body as a <span class="mono">job.py</span> with a
      <span class="mono">batch/v1</span> manifest, or as a Flyte task. A Job stages its inputs with an init container or
      boto3, onto node disk or a memory-backed volume. Its label is the workload's peak plus what the staging adds;
      a Flyte task's is the peak plus the 70&nbsp;MiB runtime.</dd>
      <dt>Approaches</dt><dd>A static request, two kinds of rules, gradient-boosted quantile trees, a zero-shot 27B model,
      a Qwen3-1.7B decision model with pointer heads (also trained on Jobs only and on Flyte tasks only), that model
      trained further with GRPO against the cost, pools of the trees with each model, and a novelty gate.</dd>
      <dt>Cost</dt><dd>Out-of-memory kills retry at double the memory, CPU throttling is replayed from the measured
      series, a too-small GPU retries one machine up, and retries and lateness add penalties. Prices are on-demand
      list prices.</dd>
      <dt>Oracle</dt><dd>The request the measurements say is right on each axis: the smallest memory cell that holds the
      need, the CPU cell that covers the busiest 15 seconds, and the GPU that lands on the cheapest machine the run fits.</dd>
    </dl>
  </section>

  <footer>
    <div>Full report: <a href="rightsize-bench-k8s.html">RightSize Bench K8s</a>. The paper (PDF and arXiv source) is in
    <span class="mono">model-factory-resource-tuner/paper/</span>.</div>
    <div>Research log <span class="mono">research_log/2026-11-08-round-23-review-real-pods.md</span> · runs
    <a href="https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rsk-k8s-2">rsk-k8s-2</a>,
    <a href="https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rsk-k8s-2-s2">-s2</a>,
    <a href="https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rsk-k8s-2-s3">-s3</a>
    and the real-pod check
    <a href="https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rt-k8spod-1">rt-k8spod-1</a>
    on demo.hosted.unionai.cloud.</div>
  </footer>
</div>
"""
with open(OUT, "w") as fh:
    fh.write(head + body)
print("wrote", OUT)
