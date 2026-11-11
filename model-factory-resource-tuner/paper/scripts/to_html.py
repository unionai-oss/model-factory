"""Render the paper as the HTML report, from the same LaTeX sources.

    make main.pdf && uv run --with pypandoc_binary python scripts/to_html.py <style source html> <out.html>

The PDF build must run first: section, table and figure numbers come from
main.aux, and citation numbers from main.bbl, so the report numbers everything
the way the PDF does. Pandoc converts the prose and tables; the pgfplots
figure is redrawn as inline SVG from the same data file.
"""
import html
import os
import re
import sys

import pypandoc

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
STYLE_SRC, OUT = sys.argv[1], sys.argv[2]


def read(p):
    with open(os.path.join(HERE, p)) as f:
        return f.read()


# ── numbers from the LaTeX build ────────────────────────────────────────
aux = read("main.aux")
LABELS = {k: (t, n) for k, t, n in re.findall(r"\\newlabel\{([^}@]+)@cref\}\{\{\[(\w+)\]\[[^]]*\]\[[^]]*\]([0-9A-Z.]+)\}", aux)}
CITE = {k: i + 1 for i, k in enumerate(re.findall(r"\\bibitem\[[^\]]*\]\{([^}]+)\}", read("main.bbl"), re.S))}
KIND = {"section": "Section", "subsection": "Section", "subsubsection": "Section", "appendix": "Appendix",
        "table": "Table", "figure": "Figure", "equation": "Equation", "lstlisting": "Figure"}


def cref(m):
    keys = [k.strip() for k in m.group(2).split(",")]
    parts = []
    for k in keys:
        t, n = LABELS.get(k, ("section", "?"))
        if t == "section" and n[:1].isalpha():
            t = "appendix"
        parts.append((KIND.get(t, "Section"), n, k))
    kinds = {p[0] for p in parts}
    links = [f'\\href{{#{k}}}{{{n}}}' for _, n, k in parts]
    if len(kinds) == 1:
        kind = parts[0][0] + ("s" if len(parts) > 1 else "")
        return f"{kind}~" + (" and ".join(links) if len(links) <= 2 else ", ".join(links[:-1]) + " and " + links[-1])
    return " and ".join(f"{p[0]}~{l}" for p, l in zip(parts, links))


def citep(m):
    return "[" + ", ".join(f"\\href{{#ref-{k.strip()}}}{{{CITE.get(k.strip(), '?')}}}" for k in m.group(1).split(",")) + "]"


# ── the LaTeX body, flattened and simplified for pandoc ─────────────────
def expand(text):
    def inp(m):
        p = m.group(1)
        p = p if p.endswith((".tex", ".txt", ".py", ".yaml")) else p + ".tex"
        return expand(read(p))
    return re.sub(r"\\input\{([^}]+)\}", inp, text)


def verbatim(m):
    return "\\begin{verbatim}\n" + read(m.group(1)).rstrip() + "\n\\end{verbatim}"


main = read("main.tex")
body = main[main.index("\\maketitle") + len("\\maketitle"):main.index("\\bibliographystyle")]
appendix = main[main.index("\\appendix") + len("\\appendix"):main.index("\\end{document}")]
abstract = expand(read("sections/abstract.tex"))
body = body.replace("\\input{sections/abstract}", "")
def number(t, letters=False):
    n = [0, 0]
    def sec(m):
        lvl = 0 if m.group(1) == "section" else 1
        n[lvl] += 1
        if lvl == 0:
            n[1] = 0
        top = chr(64 + n[0]) if letters else str(n[0])
        lab = top if lvl == 0 else f"{top}.{n[1]}"
        return f"\\{m.group(1)}{{{lab}~~{m.group(2)}}}"
    return re.sub(r"\\(section|subsection)\{([^}]*)\}", sec, t)


tex = number(expand(body)) + "\n" + number(expand(appendix), letters=True)
abstract_tex = abstract[abstract.index("\\begin{abstract}") + 16:abstract.index("\\end{abstract}")]

nums = "".join(read(f"generated/{f}") for f in ("num_pairs.tex", "num_cf.tex"))


def unwrap(t, opener):
    """Drop a one-argument wrapper like \\resizebox{..}{!}{X}, keeping X (brace-matched)."""
    while (i := t.find(opener)) >= 0:
        j, depth = i + len(opener), 1
        while depth:
            depth += {"{": 1, "}": -1}.get(t[j], 0)
            j += 1
        t = t[:i] + t[i + len(opener):j - 1].lstrip("%") + t[j:]
    return t


def number_captions(t):
    """Prefix each \\caption{..}\\label{key} with its number, as the PDF shows it."""
    out, i = [], 0
    while (j := t.find("\\caption{", i)) >= 0:
        k, depth = j + 9, 1
        while depth:
            depth += {"{": 1, "}": -1}.get(t[k], 0)
            k += 1
        m = re.match(r"\s*\\label\{([^}]+)\}", t[k:])
        pre = ""
        if m and m.group(1) in LABELS:
            typ, n = LABELS[m.group(1)]
            pre = f"\\textbf{{{KIND.get(typ, 'Table')} {n}.}} "
        out.append(t[i:j + 9] + pre)
        i = j + 9
    return "".join(out) + t[i:]


def simplify(t):
    t = number_captions(t)
    t = re.sub(r"\\cmidrule(\([^)]*\))?\{[^}]*\}", "", t)
    t = re.sub(r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}", "FIGURELEADERBOARD", t, flags=re.S)
    t = re.sub(r"\\lstinputlisting(?:\[[^\]]*\])?\{([^}]+)\}", verbatim, t)
    t = t.replace("\\begin{lstlisting}", "\\begin{verbatim}").replace("\\end{lstlisting}", "\\end{verbatim}")
    t = unwrap(t, "\\resizebox{\\linewidth}{!}{")
    t = re.sub(r"\\(C|c)ref\{([^}]+)\}", cref, t)
    t = re.sub(r"\\citep\{([^}]+)\}", citep, t)
    t = re.sub(r"\\paragraph\{([^}]+)\}", r"\\textbf{\1} ", t)
    t = t.replace("\\scriptsize", "").replace("\\small", "").replace("\\centering", "")
    t = re.sub(r"\{\\ttfamily ([^}]*)\}", r"\\texttt{\1}", t)
    t = re.sub(r"\\begin\{minipage\}(\[[^\]]*\])?\{[^}]*\}", "", t).replace("\\end{minipage}", "").replace("\\hfill", "")
    return t


pre = "\\newcommand{\\code}[1]{\\texttt{#1}}\n" + nums
to_html = lambda t: pypandoc.convert_text(pre + simplify(t), "html5", format="latex",  # noqa: E731
                                          extra_args=["--mathml", "--section-divs", "--wrap=none"])
body_html = to_html(tex)
abstract_html = to_html(abstract_tex)

# ── the figure, as SVG ──────────────────────────────────────────────────
rows = []
for ln in read("generated/fig_leaderboard.dat").splitlines()[1:]:
    m = re.match(r"(\d+) \{(.*)\} (.*)", ln)
    rows.append((m.group(2).replace("$\\times$", "×").replace("$q$", "q"), *map(float, m.group(3).split())))
W, L, R, T, RH = 760, 190, 20, 14, 26
H = T + RH * len(rows) + 46
x = lambda v: L + (W - L - R) * min(v, 125) / 125  # noqa: E731
svg = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Regret on Job pods by approach, test and unseen">']
for g in range(0, 121, 20):
    svg.append(f'<line x1="{x(g):.1f}" x2="{x(g):.1f}" y1="{T}" y2="{T + RH * len(rows)}" stroke="var(--rule)"/>')
    svg.append(f'<text x="{x(g):.1f}" y="{T + RH * len(rows) + 16}" text-anchor="middle" font-size="11" fill="var(--muted)">{g}</text>')
svg.append(f'<text x="{(L + W - R) / 2:.0f}" y="{H - 8}" text-anchor="middle" font-size="12" fill="var(--muted)">regret on Job pods (%)</text>')
for i, (name, t, tl, th, o, ol, oh) in enumerate(rows):
    y = T + RH * i + RH / 2
    svg.append(f'<text x="{L - 10}" y="{y + 4:.1f}" text-anchor="end" font-size="12.5" fill="var(--ink)">{html.escape(name)}</text>')
    for v, lo, hi, dy, col, fill in ((t, tl, th, -4, "var(--accent)", "var(--accent)"), (o, ol, oh, 4, "var(--rl)", "var(--panel)")):
        svg.append(f'<line x1="{x(lo):.1f}" x2="{x(hi):.1f}" y1="{y + dy:.1f}" y2="{y + dy:.1f}" stroke="{col}" stroke-width="1.5"/>')
        svg.append(f'<circle cx="{x(v):.1f}" cy="{y + dy:.1f}" r="4" fill="{fill}" stroke="{col}" stroke-width="1.8">'
                   f'<title>{html.escape(name)}: {v:.1f}% [{lo:.1f}, {hi:.1f}]</title></circle>')
svg.append("</svg>")
legend = ('<div class="legend"><span><span class="kind" style="background:var(--accent)"></span>test: new tasks, familiar families</span>'
          '<span><span class="kind" style="border:2px solid var(--rl);background:var(--panel)"></span>unseen families</span></div>')
body_html = re.sub(r"<p>FIGURELEADERBOARD</p>", '<div class="figure">' + legend + "\n".join(svg) + "</div>", body_html)

# ── house styling ───────────────────────────────────────────────────────
body_html = body_html.replace("<table>", '<div class="table-wrap"><table class="num">').replace("</table>", "</table></div>")
body_html = body_html.replace("<pre>", '<pre class="block">')
body_html = re.sub(r'<section id="([^"]+)" class="level1">', r'<section id="\1" class="level1">', body_html)
refs = re.findall(r"\\bibitem\[[^\]]*\]\{([^}]+)\}(.*?)(?=\\bibitem|\\end\{thebibliography\})", read("main.bbl"), re.S)
ref_items = []
for k, txt in refs:
    t = re.sub(r"\\newblock\s*", " ", txt)
    t = re.sub(r"\\(emph|textit)\{([^}]*)\}", r"<i>\2</i>", t)
    t = re.sub(r"\\url\{([^}]*)\}", r'<a href="\1">\1</a>', t)
    t = re.sub(r"\\doi\{([^}]*)\}", r'doi: <a href="https://doi.org/\1">\1</a>', t)
    t = re.sub(r"\\penalty0|\\natexlab\{[^}]*\}|\{\\v ?\{(\w)\}\}", "", t)
    t = t.replace("\\&", "&amp;").replace("{", "").replace("}", "").replace("~", " ").replace("\\", "")
    ref_items.append(f'<li id="ref-{k}">{" ".join(t.split())}</li>')
refs_html = '<section id="references"><h2>References</h2><ol class="refs">' + "".join(ref_items) + "</ol></section>"
body_html = body_html.replace('<section id="appendices" class="level1">', refs_html + '<section id="appendices" class="level1">', 1)

toc = "".join(f'<a href="#{k}">{html.escape(t)}</a>' for k, t in re.findall(r'<section id="([^"]+)" class="level1">\s*<h1>(?:<span[^>]*>[^<]*</span>)?\s*([^<]+)</h1>', body_html))
body_html = body_html.replace("<h2>", "<h3>").replace("</h2>", "</h3>")
body_html = body_html.replace("<h1>", "<h2>").replace("</h1>", "</h2>")

src = open(STYLE_SRC).read()
style = src[src.index("<style>"):src.index("</style>") + 8]
style = style.replace("</style>", """
.abstractbox { background: var(--panel); border: 1px solid var(--rule); border-radius: 6px; padding: 20px 22px; display: grid; gap: 12px; }
.abstractbox p { max-width: none; }
section.level1 { padding-block: 32px 4px; display: grid; gap: 14px; }
section.level2 { display: grid; gap: 12px; padding-top: 12px; }
h2 { margin-top: 6px; }
figure { margin: 0; display: grid; gap: 8px; background: var(--panel); border: 1px solid var(--rule); border-radius: 6px; padding: 14px 16px; min-width: 0; }
figcaption { font-size: 13px; color: var(--muted); }
figure pre.block { font-size: 11.5px; }
ol.refs { font-size: 14px; max-width: none; }
math { font-size: 1.05em; }
.displaymath, math[display="block"] { overflow-x: auto; max-width: 100%; }
p { max-width: 72ch; }
</style>""")
head = src[:src.index("<style>")].replace("<title>RightSize Bench K8s</title>", "<title>RightSize Bench K8s</title>")
page = f"""{head}{style}

<div class="wrap">
  <header>
    <div class="eyebrow">Union model factory · resource right-sizing · round 23 · November 2026</div>
    <h1>RightSize Bench: learning Kubernetes resource requests from code</h1>
    <p class="lede">How much memory, CPU and GPU should a batch pod ask for before it runs? We measured 32,082 runs of
    33 families of Python tasks, presented each one both as a raw Kubernetes Job and as a Flyte task, checked the labels in
    real pods, and scored eleven kinds of approach by what their requests would cost.</p>
    <span class="status">k8s-2 · 64,164 pods (32,082 runs × 2 ways to submit them) · 3 trainings</span>
    <div class="facts">
      <span><b>6.6%</b> best regret on familiar code (Job pods)</span>
      <span><b>39%</b> best regret on unseen families</span>
      <span><b>~70 MiB</b> measured Flyte runtime per pod</span>
      <span><b>~480</b> real pods checked</span>
    </div>
    <nav class="toc" aria-label="Sections">{toc}</nav>
  </header>
  <section id="abstract"><h2>Abstract</h2><div class="abstractbox">{abstract_html}</div></section>
{body_html}
  <footer>
    <span>PDF and LaTeX source: <span class="mono">model-factory-resource-tuner/paper/</span> (<span class="mono">make</span>, <span class="mono">make arxiv</span>). Research log: <span class="mono">research_log/2026-11-08-round-23-review-real-pods.md</span>.</span>
    <span>Reproduce: <span class="mono">flyte factory materialize rightsize-k8s-bench rt-k8sbench-leaderboard --partition release=k8s-2 --partition arm=static,rules,rules-sized,gbt-dist,gbt-dist-flyte,gbt-dist-k8s,zeroshot-27b,zeroshot-27b-json,jev-pointer,jev-pointer-flyte,jev-pointer-k8s</span></span>
  </footer>
</div>
"""
with open(OUT, "w") as f:
    f.write(page)
print("wrote", OUT, len(page))
