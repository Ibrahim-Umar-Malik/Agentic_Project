# MicroRCA-Agent conference paper (IEEEtran)

Files
- `main.tex`: the paper (IEEEtran conference, two columns). Figures are drawn in TikZ/pgfplots, so there are no image files.
- `references.bib`: 33 references, each with a DOI link or an official URL.
- `MicroRCA-Agent-paper.pdf`: the compiled paper (9 pages including references).
- `analysis/`: scripts that produced the ablation, failure-category, and significance numbers. Run them from the repository root after `pip install -e .`.

Overleaf
1. New Project > Upload Project, and upload `main.tex` and `references.bib` (or a zip of them).
2. The compiler is pdfLaTeX, and the main document is `main.tex`. Overleaf runs BibTeX automatically.

Placeholders to fill (shown in red in the PDF)
- Project code: in `\title{...}`.
- Four email addresses: in the author blocks.
Search for `\placeholder` in `main.tex`. Delete the `\placeholder` macro once everything is filled in.
