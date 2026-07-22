"""Aggregate evaluation summaries into leaderboards / retention tables.

Usage:
    # Leaderboard (fev-bench methodology: skill score + win rate + bootstrap CIs)
    uv run python scripts/aggregate.py leaderboard results/raw/run_a results/raw/run_b \\
        --metric WQL --baseline seasonal_naive

    # Accuracy retention of quantized variants vs the fp32 parent
    uv run python scripts/aggregate.py retention results/raw/* --reference chronos2-fp32

    # Validate our results against published reference per-task numbers
    uv run python scripts/aggregate.py validate results/raw/seasonal_naive_dev \\
        --reference-csv results/reference/chronos_zeroshot/seasonal_naive.csv
"""

import argparse

import pandas as pd

from chronosquant.analysis import (
    leaderboard,
    load_summaries,
    retention_table,
    validate_against_reference,
)

pd.set_option("display.width", 200)
pd.set_option("display.float_format", lambda v: f"{v:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_lb = sub.add_parser("leaderboard", help="Skill score / win rate leaderboard")
    p_lb.add_argument("summaries", nargs="+", help="Summary CSVs or run directories")
    p_lb.add_argument("--metric", default="WQL", help="Metric column (WQL, MASE, test_error)")
    p_lb.add_argument("--baseline", default="seasonal_naive")
    p_lb.add_argument("--n-resamples", type=int, default=1000)
    p_lb.add_argument(
        "--missing-strategy",
        default="error",
        choices=["error", "drop", "impute"],
        help="'drop' restricts to tasks all models completed (e.g. vs 27-task reference CSVs)",
    )

    p_rt = sub.add_parser("retention", help="Accuracy retention vs a reference model")
    p_rt.add_argument("summaries", nargs="+")
    p_rt.add_argument("--reference", required=True, help="Reference model name (fp32 parent)")
    p_rt.add_argument("--metrics", nargs="+", default=["WQL", "MASE"])

    p_val = sub.add_parser("validate", help="Compare per-task results against a reference CSV")
    p_val.add_argument("summaries", nargs="+")
    p_val.add_argument("--reference-csv", required=True)
    p_val.add_argument("--metrics", nargs="+", default=["MASE", "WQL"])

    args = parser.parse_args()
    summaries = load_summaries(args.summaries)

    if args.command == "leaderboard":
        table = leaderboard(
            summaries,
            metric_column=args.metric,
            baseline_model=args.baseline,
            n_resamples=args.n_resamples,
            missing_strategy=args.missing_strategy,
        )
        print(table.to_string())
    elif args.command == "retention":
        table = retention_table(summaries, reference_model=args.reference, metrics=args.metrics)
        print(table.to_string())
    elif args.command == "validate":
        reference = pd.read_csv(args.reference_csv)
        table = validate_against_reference(summaries, reference, metrics=args.metrics)
        print(table.to_string())
        max_rel = table[[c for c in table.columns if c.endswith("_rel_diff")]].abs().max().max()
        print(f"\nMax |relative difference|: {max_rel:.2%}")


if __name__ == "__main__":
    main()
