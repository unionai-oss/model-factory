"""Render the paper as the HTML report, from the same LaTeX sources.

    make main.pdf && uv run --with pypandoc_binary python scripts/to_html.py <style source html> <out.html>

The PDF build must run first: section, table and figure numbers come from
main.aux, and citation numbers from main.bbl, so the report numbers everything
the way the PDF does. Pandoc converts the prose and tables. Figures are
inlined as SVG: the charts from figures/fig_*.svg (scripts/figures.py), the
TikZ diagrams converted from their PDFs with pdftocairo. Their colors are
mapped to the report's CSS variables, so they follow its light and dark themes.
"""
import html
import os
import re
import subprocess
import sys
import tempfile

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

nums = "".join(read(f"generated/{f}") for f in ("num_main.tex", "num_pairs.tex", "num_cf.tex"))


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

# ── figures, as inline SVG ──────────────────────────────────────────────
#: the figures' palette (figures.py, figures/diagrams/common.tex) -> the report's CSS variables
TOKENS = {"1f2328": "--ink", "000000": "--ink", "6a737d": "--muted", "dcdfe8": "--rule", "d0d4dc": "--rule",
          "f4f5f8": "--tint", "ffffff": "--panel", "2a78d6": "--s1", "eb6834": "--s2", "1baf7a": "--s3",
          "8a8fa0": "--rule-c", "6250d6": "--s4"}


def _nearest(rgb):
    best = min(TOKENS, key=lambda h: sum((int(h[i:i + 2], 16) - c) ** 2 for i, c in zip((0, 2, 4), rgb)))
    d = sum((int(best[i:i + 2], 16) - c) ** 2 for i, c in zip((0, 2, 4), rgb))
    return f"var({TOKENS[best]})" if d < 400 else None


def tokenize(svg):
    def hexsub(m):
        h = m.group(1).lower()
        return _nearest([int(h[i:i + 2], 16) for i in (0, 2, 4)]) or m.group(0)

    def rgbsub(m):
        return _nearest([round(float(x) * 2.55) for x in m.groups()]) or m.group(0)
    svg = re.sub(r"#([0-9a-fA-F]{6})\b", hexsub, svg)
    return re.sub(r"rgb\(([\d.]+)%,\s*([\d.]+)%,\s*([\d.]+)%\)", rgbsub, svg)


def scope_ids(svg, prefix):
    """Prefix every id so several inline SVGs can share one page."""
    ids = set(re.findall(r'id="([^"]+)"', svg))
    for i in sorted(ids, key=len, reverse=True):
        svg = svg.replace(f'id="{i}"', f'id="{prefix}{i}"').replace(f"#{i})", f"#{prefix}{i})").replace(f'"#{i}"', f'"#{prefix}{i}"')
    return svg


def svg_for(path, n):
    """path: figures/fig_x.pdf (a chart: its .svg) or figures/diagrams/x.pdf (converted)."""
    svg_path = os.path.join(HERE, path[:-4] + ".svg")
    if "/diagrams/" in path or not os.path.exists(svg_path):
        out = os.path.join(tempfile.mkdtemp(), "d.svg")
        subprocess.run(["pdftocairo", "-svg", os.path.join(HERE, path), out], check=True)
        svg_path = out
    svg = open(svg_path).read()
    svg = svg[svg.index("<svg"):]
    svg = re.sub(r'<svg([^>]*?)\swidth="[^"]*"', r"<svg\1", svg, count=1)
    svg = re.sub(r'<svg([^>]*?)\sheight="[^"]*"', r"<svg\1", svg, count=1)
    svg = svg.replace("<svg", '<svg class="fig" role="img"', 1)
    return scope_ids(tokenize(svg), f"f{n}-")


_n = [0]


def inline(m):
    _n[0] += 1
    return svg_for(m.group(1), _n[0])


body_html = re.sub(r'<img src="(figures/[^"]+\.pdf)"[^>]*/?>', inline, body_html)
body_html = re.sub(r'<embed src="(figures/[^"]+\.pdf)"[^>]*/?>', inline, body_html)

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
svg.fig { width: 100%; height: auto; display: block; }
:root { --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --s4: #6250d6; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #9085e9; } }
:root[data-theme="dark"] { --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #9085e9; }
svg.fig text { font-family: var(--font-body); fill: var(--ink); }
ol.refs { font-size: 14px; max-width: none; }
math { font-size: 1.05em; }
.displaymath, math[display="block"] { overflow-x: auto; max-width: 100%; }
p { max-width: 72ch; }
</style>""")
head = src[:src.index("<style>")].replace("<title>RightSize Bench K8s</title>", "<title>RightSize Bench K8s</title>")
_m = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", read("generated/num_main.tex")))
FACT_TEST, FACT_UNSEEN = _m["rTestPoolDec"], _m["rUnseenSized"]
page = f"""{head}{style}

<div class="wrap">
  <header>
    <div class="eyebrow">Union model factory · resource right-sizing · round 24 · November 2026</div>
    <h1>RightSize Bench: learning Kubernetes resource requests from code</h1>
    <p class="lede">How much memory, CPU and GPU should a batch pod ask for before it runs? We measured 32,082 runs of
    33 families of Python tasks, presented each one as a raw Kubernetes Job and as a Flyte task, checked the labels in
    real pods, and scored each approach by what its requests cost against the cheapest request that would have run.</p>
    <span class="status">k8s-2 · 64,164 pods (32,082 runs × 2 ways to submit them) · 3 trainings</span>
    <div class="facts">
      <span><b>{FACT_TEST}%</b> best regret on familiar code (Job pods)</span>
      <span><b>{FACT_UNSEEN}%</b> best regret on held-out families</span>
      <span><b>~70 MiB</b> measured Flyte runtime per pod</span>
      <span><b>~480</b> real pods checked</span>
    </div>
    <nav class="toc" aria-label="Sections">{toc}</nav>
  </header>
  <section id="abstract"><h2>Abstract</h2><div class="abstractbox">{abstract_html}</div></section>
{body_html}
  <footer>
    <span>PDF and LaTeX source: <span class="mono">model-factory-resource-tuner/paper/</span> (<span class="mono">make</span>, <span class="mono">make arxiv</span>). Data and models: <a href="https://huggingface.co/datasets/unionai/rightsize-bench">huggingface.co/datasets/unionai/rightsize-bench</a>. Research log: <span class="mono">research_log/2026-11-10-round-24-review-release.md</span>.</span>
    <span>Reproduce: <span class="mono">flyte factory materialize rightsize-k8s-bench rt-k8sbench-leaderboard --partition release=k8s-2 --partition arm=static,rules,rules-sized,gbt-dist,gbt-dist-flyte,gbt-dist-k8s,zeroshot-27b,zeroshot-27b-json,jev-pointer,jev-pointer-flyte,jev-pointer-k8s</span></span>
  </footer>
</div>
"""
with open(OUT, "w") as f:
    f.write(page)
print("wrote", OUT, len(page))
