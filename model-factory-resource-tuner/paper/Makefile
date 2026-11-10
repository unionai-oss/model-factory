# Build the paper, and an arXiv source bundle (arXiv runs pdflatex + bibtex itself;
# the .bbl ships too so the bibliography never depends on its BibTeX version).
#   make            -> main.pdf
#   make tables A=<analysis dir>   -> regenerate generated/ from the analysis files
#   make arxiv      -> arxiv.tar.gz
# A: a directory holding k2s{1,2,3}-analysis.json (scripts/analyze.py) and the counterfactual exports.
A ?= $(error set A=<analysis dir>)

main.pdf: main.tex refs.bib sections/*.tex generated/*.tex figures/*
	latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

tables:
	@test -n "$(A)"
	uv run --project .. python scripts/tables.py $(A) $(A)

arxiv: main.pdf
	rm -rf arxiv && mkdir -p arxiv
	cp -r main.tex main.bbl refs.bib sections generated figures arxiv/
	tar -czf arxiv.tar.gz -C arxiv .

clean:
	latexmk -C
	rm -rf arxiv arxiv.tar.gz

.PHONY: tables arxiv clean
