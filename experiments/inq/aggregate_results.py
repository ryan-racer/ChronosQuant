"""Aggregate the KV-quantization dev runs into a retention table vs chronos2-fp32.

Run: uv run python experiments/inq/aggregate_results.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from chronosquant.analysis.aggregate import load_summaries, retention_table
from chronosquant.utils import REPO_ROOT

HERE = Path(__file__).parent
RAW = REPO_ROOT / "results" / "raw"

RUNS = ["chronos2_fp32_dev"] + [
    f"chronos2_kv{bits}_{method}_dev" for method in ("rtn_channel", "inq") for bits in (4, 3, 2)
]


def main():
    paths = [RAW / r / "summaries.csv" for r in RUNS if (RAW / r / "summaries.csv").exists()]
    print(f"aggregating {len(paths)} runs")
    summaries = load_summaries(paths)
    table = retention_table(summaries, reference_model="chronos2-fp32")
    table.to_csv(HERE / "retention.csv")
    print(table.round(4).to_string())

    # per-task WQL ratios for the interesting arms (which datasets break first?)
    piv = summaries.pivot_table(index="dataset_config", columns="model_name", values="WQL")
    rel = piv.div(piv["chronos2-fp32"], axis=0).drop(columns=["chronos2-fp32"]).round(3)
    rel.to_csv(HERE / "per_task_wql_ratio.csv")
    print("\nper-task WQL ratio vs fp32:")
    print(rel.to_string())


if __name__ == "__main__":
    main()
