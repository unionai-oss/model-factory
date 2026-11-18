# Build the paper, and an arXiv source bundle (arXiv runs pdflatex + bibtex itself;
# the .bbl ships too so the bibliography never depends on its BibTeX version).
#   make            -> main.pdf
#   make tables A=<analysis dir>   -> regenerate generated/ from the analysis files
#   make arxiv      -> arxiv.tar.gz
# A: a directory holding k2s{1,2,3}-analysis.json (scripts/analyze.py) and the counterfactual exports.
A ?= $(error set A=<analysis dir>)

DIAGRAMS := $(patsubst %.tex,%.pdf,$(filter-out figures/diagrams/common.tex,$(wildcard figures/diagrams/*.tex)))

main.pdf: main.tex refs.bib sections/*.tex generated/*.tex figures/* $(DIAGRAMS)
	latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

# the TikZ diagrams, standalone (to_html.py turns each PDF into an inline SVG)
figures/diagrams/%.pdf: figures/diagrams/%.tex figures/diagrams/common.tex
	cd figures/diagrams && pdflatex -interaction=nonstopmode -halt-on-error $*.tex >/dev/null

# the charts: uv run --with matplotlib python scripts/figures.py <analysis dir>
figures:
	@test -n "$(A)"
	uv run --project .. --with matplotlib python scripts/figures.py $(A)

tables:
	@test -n "$(A)"
	uv run --project .. python scripts/tables.py $(A) $(A)

arxiv: main.pdf
	rm -rf arxiv && mkdir -p arxiv
	cp -r main.tex main.bbl refs.bib sections generated arxiv/
	mkdir -p arxiv/figures/diagrams
	cp figures/*.pdf figures/*.py figures/*.yaml figures/*.txt arxiv/figures/
	cp figures/diagrams/*.pdf arxiv/figures/diagrams/
	tar -czf arxiv.tar.gz -C arxiv .

clean:
	latexmk -C
	rm -rf arxiv arxiv.tar.gz

.PHONY: tables figures arxiv clean
