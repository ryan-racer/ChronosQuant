"""CLI: post-hoc repair analysis (quantile rearrangement + split-conformal recalibration).

Runs entirely offline over the raw prediction parquets persisted by the evaluation
runner — no model inference, CPU only. Processes one task at a time (m4/m5/dominick
are large) and one run's parquet at a time.

Usage (from repo root):

    uv run python scripts/repair_analysis.py                       # all results/raw/*_full
    uv run python scripts/repair_analysis.py results/raw/chronos2_w4_hqq_full ...
    uv run python scripts/repair_analysis.py --tasks m5 ETTh       # task subset (debugging)

Outputs (to --out-dir, default results/tables):

- ``repair_per_task.csv``   — one row per (model, task, stage in before/sorted/conformal)
  with WQL, pinball, MACE, coverage[*], QCR. Stage "conformal" = recalibration + sorting.
- ``repair_sorting.csv``    — one row per model: before vs after sorting (WQL retention
  vs unrepaired fp32 with bootstrap CI, MACE, cov80, QCR).
- ``repair_conformal.csv``  — one row per model: before / sorted / conformal+sorted.

Validation: the recomputed "before" metrics are cross-checked against each run's
persisted summaries.csv (WQL, MACE, coverage[0.8], QCR) and the maximum absolute
deviation is printed — this gates trust in the offline recomputation.

Protocol details: see the docstring of ``chronosquant.analysis.repair``.
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from chronosquant.analysis.repair import DEFAULT_SEED, aggregate_repair, analyze_task  # noqa: E402

REFERENCE = "chronos2-fp32"
EXCLUDE_MODELS = {"seasonal_naive"}


def discover_runs(run_dirs: list[Path]) -> dict[str, Path]:
    """Map model_name -> run dir (from each run's summaries.csv), skipping baselines."""
    runs: dict[str, Path] = {}
    for run_dir in run_dirs:
        summaries = run_dir / "summaries.csv"
        if not summaries.exists() or not (run_dir / "predictions").is_dir():
            print(f"  skipping {run_dir.name}: no summaries.csv or predictions/")
            continue
        model_name = pd.read_csv(summaries, usecols=["model_name"])["model_name"].iloc[0]
        if model_name in EXCLUDE_MODELS:
            continue
        runs[model_name] = run_dir
    return runs


def validate_before_metrics(per_task: pd.DataFrame, runs: dict[str, Path]) -> pd.DataFrame:
    """Compare recomputed 'before' metrics with each run's persisted summaries.csv."""
    checks = []
    for model_name, run_dir in runs.items():
        persisted = pd.read_csv(run_dir / "summaries.csv").set_index("task_name")
        ours = per_task.query("model_name == @model_name and stage == 'before'").set_index(
            "task_name"
        )
        common = persisted.index.intersection(ours.index)
        for metric in ("WQL", "MACE", "coverage[0.8]", "QCR"):
            diff = (ours.loc[common, metric] - persisted.loc[common, metric]).abs()
            checks.append(
                {"model_name": model_name, "metric": metric, "max_abs_diff": float(diff.max())}
            )
    return pd.DataFrame(checks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "run_dirs",
        nargs="*",
        type=Path,
        help="Run directories (default: all results/raw/*_full)",
    )
    parser.add_argument("--out-dir", type=Path, default=REPO / "results" / "tables")
    parser.add_argument("--reference", default=REFERENCE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--tasks", nargs="*", default=None, help="Task subset (debugging)")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute everything (default: resume, skipping tasks already complete "
        "in repair_per_task.csv)",
    )
    args = parser.parse_args()

    run_dirs = args.run_dirs or sorted((REPO / "results" / "raw").glob("*_full"))
    runs = discover_runs(run_dirs)
    if args.reference not in runs:
        raise SystemExit(f"Reference model {args.reference!r} not found among {sorted(runs)}")
    print(f"Analyzing {len(runs)} runs: {sorted(runs)}")

    task_names = sorted(
        p.stem for p in (runs[args.reference] / "predictions").glob("*.parquet")
    )
    if args.tasks:
        task_names = [t for t in task_names if t in set(args.tasks)]
    print(f"{len(task_names)} tasks")

    # Incremental persistence: rows are appended to repair_per_task.csv after every task,
    # and completed tasks (all runs x 3 stages present) are skipped on restart.
    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_task_path = args.out_dir / "repair_per_task.csv"
    done_frames: list[pd.DataFrame] = []
    if per_task_path.exists() and not args.overwrite:
        previous = pd.read_csv(per_task_path)
        counts = previous.groupby("task_name")["model_name"].nunique()
        complete = set(counts[counts >= len(runs)].index)
        done_frames.append(previous[previous["task_name"].isin(complete)])
        skipped = [t for t in task_names if t in complete]
        task_names = [t for t in task_names if t not in complete]
        if skipped:
            print(f"Resuming: {len(skipped)} tasks already complete, {len(task_names)} to go")
    elif args.overwrite:
        per_task_path.unlink(missing_ok=True)

    if done_frames:
        pd.concat(done_frames, ignore_index=True).to_csv(per_task_path, index=False)

    for task_name in task_names:  # tasks outer, runs inner: bounded memory
        t0 = time.time()
        rows = []
        for model_name, run_dir in runs.items():
            frame = pd.read_parquet(run_dir / "predictions" / f"{task_name}.parquet")
            stage_metrics = analyze_task(frame, seed=args.seed)
            del frame
            for stage, metrics in stage_metrics.items():
                rows.append(
                    {"model_name": model_name, "task_name": task_name, "stage": stage, **metrics}
                )
        task_frame = pd.DataFrame(rows)
        header = not per_task_path.exists()
        task_frame.to_csv(per_task_path, mode="a", header=header, index=False)
        done_frames.append(task_frame)
        print(f"  {task_name}: {len(runs)} runs in {time.time() - t0:.1f}s", flush=True)

    per_task = pd.concat(done_frames, ignore_index=True)

    # Validation gate: recomputed unrepaired metrics must match the persisted summaries.
    checks = validate_before_metrics(per_task, runs)
    worst = checks.groupby("metric")["max_abs_diff"].max()
    print("\nValidation vs persisted summaries.csv (max abs diff over all runs/tasks):")
    print(worst.to_string())
    # Per-metric tolerances: WQL is a smooth average (float-precision agreement);
    # coverage/MACE/QCR are indicator means where a single exact y==q tie can flip one
    # point in a ~100k-point task (~1e-5) without any logic difference.
    tolerances = {"WQL": 1e-8}
    if any(worst[m] > tolerances.get(m, 2e-5) for m in worst.index):
        print("WARNING: recomputed 'before' metrics deviate from persisted summaries!")

    table = aggregate_repair(per_task, reference_model=args.reference)

    sorting_cols = [
        c
        for c in table.columns
        if c.endswith(("_before", "_sorted"))
        or ("_before_" in c or "_sorted_" in c)
        or c == "n_tasks"
    ]
    sorting = table[sorting_cols]
    conformal = table  # full before / sorted / conformal comparison

    sorting_path = args.out_dir / "repair_sorting.csv"
    conformal_path = args.out_dir / "repair_conformal.csv"
    sorting.to_csv(sorting_path)
    conformal.to_csv(conformal_path)

    with pd.option_context("display.width", 250, "display.max_columns", 50):
        print("\n=== Sorting repair (before vs after rearrangement) ===")
        print(
            sorting[
                ["WQL_rel_before", "WQL_rel_sorted", "QCR_before", "QCR_sorted",
                 "cov80_before", "cov80_sorted", "MACE_before", "MACE_sorted"]
            ].round(4).to_string()
        )
        print("\n=== Conformal recalibration (+sorting) ===")
        print(
            conformal[
                ["WQL_rel_before", "WQL_rel_conformal", "cov80_before", "cov80_sorted",
                 "cov80_conformal", "MACE_before", "MACE_conformal", "QCR_conformal"]
            ].round(4).to_string()
        )
    print(f"\nWrote {per_task_path}\n      {sorting_path}\n      {conformal_path}")


if __name__ == "__main__":
    main()
