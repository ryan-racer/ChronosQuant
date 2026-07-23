# Building the paper

`workshop.tex` is the workshop paper, typeset with the ICML 2025 style files
(`icml2025.sty`) **for formatting only** — the `[accepted]` option is intentionally
omitted, so it renders as an **anonymous submission** (no author info, review line
numbers, neutral "under review" notice; no venue branding).

```bash
cd paper
pdflatex workshop        # 1st pass
bibtex   workshop        # resolve refs.bib via icml2025.bst
pdflatex workshop        # 2nd pass
pdflatex workshop        # 3rd pass (finalize cross-refs)
```

The figure is regenerated from results by `uv run python scripts/make_figures.py`.
`workshop.bbl` is committed so a single `pdflatex workshop` also works (no bibtex needed).
Style files (`icml2025.sty`, `icml2025.bst`, `fancyhdr.sty`, `algorithm*.sty`) are the
official ICML 2025 bundle, vendored for reproducibility.

`workshop.md` is a plain-Markdown mirror of the content; `draft.md` is the longer working
draft with appendices.
