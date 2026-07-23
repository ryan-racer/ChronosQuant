"""Render the Markdown paper draft to a styled PDF (via headless Microsoft Edge).

Converts paper/draft.md -> HTML (academic styling) -> PDF, and injects the live
campaign results table (results/tables/campaign_master.csv) where the draft references
Table 2, so the PDF carries the actual numbers rather than a placeholder.

    uv run python scripts/build_paper_pdf.py

No network or extra system installs required (Edge ships with Windows 11).
"""

import shutil
import subprocess
import sys
from pathlib import Path

import markdown
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DRAFT = REPO_ROOT / "paper" / "draft.md"
MASTER_CSV = REPO_ROOT / "results" / "tables" / "campaign_master.csv"
HTML_OUT = REPO_ROOT / "paper" / "chronosquant_draft.html"
PDF_OUT = REPO_ROOT / "paper" / "chronosquant_draft.pdf"

EDGE_CANDIDATES = [
    Path(r"C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"),
    Path(r"C:/Program Files/Microsoft/Edge/Application/msedge.exe"),
]

CSS = """
@page { size: A4; margin: 18mm 16mm; }
* { box-sizing: border-box; }
body {
  font-family: "Georgia", "Times New Roman", serif;
  font-size: 10.2pt; line-height: 1.45; color: #111; max-width: 100%;
}
h1 { font-size: 19pt; line-height: 1.2; margin: 0 0 2pt; }
h2 { font-size: 13pt; border-bottom: 1px solid #bbb; padding-bottom: 3px;
     margin: 18px 0 8px; }
h3 { font-size: 11pt; margin: 14px 0 5px; }
h4 { font-size: 10.2pt; margin: 11px 0 4px; font-variant: small-caps;
     letter-spacing: .3px; }
p, li { text-align: justify; }
code, pre { font-family: "Consolas", "Courier New", monospace; font-size: 8.6pt; }
pre { background: #f5f5f5; padding: 8px 10px; border-radius: 4px; overflow-x: auto;
      border: 1px solid #e2e2e2; white-space: pre-wrap; }
code { background: #f0f0f0; padding: 1px 3px; border-radius: 3px; }
pre code { background: none; padding: 0; }
blockquote { border-left: 3px solid #4a6fa5; background: #f4f7fb; margin: 10px 0;
             padding: 6px 12px; }
blockquote p { margin: 5px 0; }
table { border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 8.4pt;
        font-family: "Helvetica Neue", Arial, sans-serif; }
th, td { border: 1px solid #ccc; padding: 3px 6px; text-align: left; }
th { background: #eef1f5; font-weight: 600; }
tr:nth-child(even) td { background: #fafbfc; }
a { color: #2a4d80; text-decoration: none; }
.results-table td:first-child { font-family: "Consolas", monospace; }
.pdf-meta { color: #666; font-size: 8.5pt; font-family: Arial, sans-serif;
            margin-bottom: 14px; }
hr { border: none; border-top: 1px solid #ddd; margin: 16px 0; }
"""


def results_table_html() -> str:
    if not MASTER_CSV.exists():
        return "<p><em>(campaign_master.csv not found — run the sweeps first.)</em></p>"
    df = pd.read_csv(MASTER_CSV)
    show = [
        "variant", "method", "weight_bits", "eff_bpw", "size_MB",
        "WQL_ret", "WQL_lo", "WQL_hi", "MASE_ret", "worst_task_WQL",
        "QCR", "cov80", "parity_with_fp32",
    ]  # fmt: skip
    show = [c for c in show if c in df.columns]
    html = df[show].to_html(index=False, border=0, classes="results-table",
                            float_format=lambda v: f"{v:.4f}")
    return (
        "<p><strong>Table 2 — Master accuracy/calibration matrix "
        "(Benchmark II, 27 tasks; retention vs fp32, bootstrap CI).</strong></p>"
        + html
    )


def find_edge() -> Path:
    for candidate in EDGE_CANDIDATES:
        if candidate.exists():
            return candidate
    sys.exit("Microsoft Edge not found; cannot render PDF.")


def main() -> None:
    text = DRAFT.read_text(encoding="utf-8")

    # Inject the live results table right after the Table 2 heading paragraph.
    marker = "**Table 2 — Main accuracy matrix"
    if marker in text:
        before, _, after = text.partition(marker)
        # drop the rest of that paragraph (up to the next blank line)
        _, _, rest = after.partition("\n\n")
        text = before + "@@RESULTS_TABLE@@\n\n" + rest

    body = markdown.markdown(
        text, extensions=["tables", "fenced_code", "toc", "sane_lists", "attr_list"]
    )
    body = body.replace("<p>@@RESULTS_TABLE@@</p>", results_table_html())

    git_sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
        capture_output=True, text=True,
    ).stdout.strip()
    meta = (
        f'<div class="pdf-meta">Rendered from paper/draft.md · git {git_sha} · '
        "results/tables/campaign_master.csv</div>"
    )

    html = (
        f"<!doctype html><html><head><meta charset='utf-8'>"
        f"<style>{CSS}</style></head><body>{meta}{body}</body></html>"
    )
    HTML_OUT.write_text(html, encoding="utf-8")

    edge = find_edge()
    if PDF_OUT.exists():
        PDF_OUT.unlink()
    subprocess.run(
        [
            str(edge), "--headless", "--disable-gpu", "--no-first-run",
            "--no-pdf-header-footer",
            f"--print-to-pdf={PDF_OUT}",
            HTML_OUT.as_uri(),
        ],
        check=True,
        timeout=120,
        capture_output=True,
    )
    if not PDF_OUT.exists():
        sys.exit("Edge ran but no PDF was produced.")
    size_kb = PDF_OUT.stat().st_size / 1024
    print(f"PDF written: {PDF_OUT}  ({size_kb:.0f} KB)")
    if shutil.which("wc"):  # rough page hint via pdfinfo-free heuristic
        pass


if __name__ == "__main__":
    main()
