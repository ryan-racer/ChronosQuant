"""Run a quantization-method sweep and report retention vs the fp32 reference.

Usage:
    uv run python scripts/sweep.py configs/evaluation/sweeps/standard_ptq.yaml --tier dev
    uv run python scripts/sweep.py <sweep.yaml> --tier full
    uv run python scripts/sweep.py <sweep.yaml> --tier dev --only chronos2-w4-hqq --overwrite

Behaviour:
- Each variant becomes one run: results/raw/<variant_name_with_underscores>_<tier>/
- Completed runs are skipped (resume-friendly) unless --overwrite is given.
- A variant whose transform or evaluation fails is reported and skipped; the sweep
  continues (per-variant failure isolation, mirroring the runner's per-task isolation).
- At the end, prints and saves the retention table vs the reference model
  (requires the matching fp32 reference run, e.g. chronos2_fp32_<tier>).
"""

import argparse
import logging
import traceback
from pathlib import Path

# Eager import: fail fast on import problems instead of failing every variant.
import chronos  # noqa: F401
import pandas as pd
import yaml

from chronosquant.analysis import retention_table
from chronosquant.evaluation import RunConfig, run_evaluation
from chronosquant.utils import REPO_ROOT

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("chronosquant.sweep")


def run_name_for(variant_name: str, tier: str) -> str:
    return variant_name.replace("-", "_") + "_" + tier


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep_config", help="Path to a sweep YAML")
    parser.add_argument("--tier", choices=["dev", "full"], default="dev")
    parser.add_argument("--only", nargs="+", help="Run only these variant names")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-report", action="store_true", help="Skip the retention report")
    args = parser.parse_args()

    with open(args.sweep_config) as f:
        sweep = yaml.safe_load(f)

    variants = sweep["variants"]
    if args.only:
        known = {v["name"] for v in variants}
        unknown = set(args.only) - known
        if unknown:
            raise SystemExit(f"Unknown variants {sorted(unknown)}. Available: {sorted(known)}")
        variants = [v for v in variants if v["name"] in args.only]

    task_names = sweep["dev_task_names"] if args.tier == "dev" else None
    completed_dirs: list[Path] = []
    failed: list[str] = []

    for i, variant in enumerate(variants, start=1):
        name = variant["name"]
        run_name = run_name_for(name, args.tier)
        output_dir = REPO_ROOT / "results" / "raw" / run_name
        if (output_dir / "summaries.csv").exists() and not args.overwrite:
            logger.info("[%d/%d] %s already complete, skipping", i, len(variants), name)
            completed_dirs.append(output_dir)
            continue

        predictor_config = dict(sweep["base_predictor"])
        predictor_config["name"] = name
        predictor_config["quantization"] = variant["quantization"]
        config = RunConfig(
            name=run_name,
            benchmark=sweep["benchmark"],
            predictor=predictor_config,
            task_names=task_names,
            overwrite=args.overwrite,
            notes=f"sweep={sweep['name']} tier={args.tier}",
        )
        logger.info("[%d/%d] running %s (%s)", i, len(variants), name, variant["quantization"])
        try:
            completed_dirs.append(run_evaluation(config))
        except Exception as err:  # noqa: BLE001 - per-variant isolation
            failed.append(name)
            logger.error("[%d/%d] %s FAILED: %r\n%s", i, len(variants), name, err,
                         traceback.format_exc())

    if failed:
        logger.warning("Failed variants: %s", failed)

    if args.skip_report or not completed_dirs:
        return

    reference_run = REPO_ROOT / "results" / "raw" / f"chronos2_fp32_{args.tier}"
    if not (reference_run / "summaries.csv").exists():
        logger.warning("Reference run %s missing; skipping retention report", reference_run)
        return

    frames = [pd.read_csv(reference_run / "summaries.csv")]
    frames += [pd.read_csv(d / "summaries.csv") for d in completed_dirs]
    summaries = pd.concat(frames, ignore_index=True)
    table = retention_table(summaries, reference_model=sweep["reference_model"])

    pd.set_option("display.width", 200)
    print("\n" + table.round(4).to_string())

    tables_dir = REPO_ROOT / "results" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    out_path = tables_dir / f"sweep_{sweep['name']}_{args.tier}_retention.csv"
    table.to_csv(out_path)
    print(f"\nRetention table saved to {out_path}")


if __name__ == "__main__":
    main()
