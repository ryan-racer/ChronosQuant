"""Run a benchmark evaluation from a config file.

Usage:
    uv run python scripts/evaluate.py configs/evaluation/runs/seasonal_naive_dev.yaml
    uv run python scripts/evaluate.py <config.yaml> --overwrite
    uv run python scripts/evaluate.py <config.yaml> --list-tasks
"""

import argparse
import logging

# Eager import: transformers' lazy import machinery has failed under mid-run process
# contention when first triggered deep inside a run; importing up front fails fast
# instead of failing 27 tasks.
import chronos  # noqa: F401

from chronosquant.evaluation import RunConfig, load_benchmark_tasks, run_evaluation

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
# Quiet per-request HTTP logging (404s on optional HF files are the normal path)
for _noisy in ("httpx", "huggingface_hub", "urllib3", "filelock"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Path to a run config YAML")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing output dir")
    parser.add_argument(
        "--list-tasks", action="store_true", help="List the tasks this config would run and exit"
    )
    args = parser.parse_args()

    overrides = {"overwrite": True} if args.overwrite else {}
    config = RunConfig.from_yaml(args.config, **overrides)

    if args.list_tasks:
        tasks = load_benchmark_tasks(config.benchmark, config.task_names)
        for task in tasks:
            print(f"{task.task_name}  (horizon={task.horizon}, seasonality={task.seasonality})")
        print(f"\n{len(tasks)} tasks")
        return

    output_dir = run_evaluation(config)
    print(f"\nResults written to {output_dir}")


if __name__ == "__main__":
    main()
