"""Config-driven evaluation runner.

Runs a predictor over a benchmark (a YAML list of fev tasks), producing per-task
evaluation summaries in the standard fev format so results are directly comparable
with (and aggregatable alongside) published reference results.

Design principles (mirroring the evaluation practice of the Chronos-2 / fev-bench and
quantized-LLM literature):

- **Provenance**: every run persists its resolved config, git SHA, package versions and
  hardware; every summary row carries the run name and git SHA.
- **Failure isolation**: a failing task is logged (with traceback) and skipped; the
  aggregation layer handles missing results explicitly rather than silently.
- **Incremental persistence**: summaries are flushed to disk after every task, so long
  runs are inspectable and interruption-safe.
- **Identical eval path for all models**: baseline and quantized variants flow through
  the same code; only the `Predictor` differs.
"""

import dataclasses
import json
import logging
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from fev import Benchmark, Task

from chronosquant.evaluation.metrics import DEFAULT_DIAGNOSTICS
from chronosquant.evaluation.predictors import Predictor, build_predictor
from chronosquant.utils import REPO_ROOT, ensure_truststore, environment_info

logger = logging.getLogger("chronosquant")

SUMMARIES_FILENAME = "summaries.csv"
METADATA_FILENAME = "run_metadata.yaml"
FAILURES_FILENAME = "failures.jsonl"


@dataclasses.dataclass
class RunConfig:
    """Configuration for a single evaluation run (one model over one benchmark)."""

    name: str
    benchmark: str
    predictor: dict[str, Any]
    task_names: list[str] | None = None
    output_root: str = "results/raw"
    per_quantile_scores: bool = True
    quantile_diagnostics: bool = True
    trained_on_this_dataset: bool = False
    overwrite: bool = False
    notes: str = ""

    @classmethod
    def from_yaml(cls, path: str | Path, **overrides) -> "RunConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)
        raw.update(overrides)
        return cls(**raw)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def load_benchmark_tasks(benchmark_path: str | Path, task_names: list[str] | None) -> list[Task]:
    """Load tasks from a benchmark YAML, optionally filtering to a named subset.

    Unknown names in `task_names` raise (a silently-missing task would corrupt
    cross-model comparisons).
    """
    path = Path(benchmark_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    benchmark = Benchmark.from_yaml(path)
    if task_names is None:
        return benchmark.tasks
    tasks_by_name = {t.task_name: t for t in benchmark.tasks}
    unknown = [name for name in task_names if name not in tasks_by_name]
    if unknown:
        raise ValueError(f"Unknown task names {unknown}. Available: {sorted(tasks_by_name)}")
    return [tasks_by_name[name] for name in task_names]


def _compute_quantile_diagnostics(
    task: Task, predictions_per_window: list, per_quantile_scores: bool
) -> dict[str, float]:
    """Compute MACE/QCR/coverage diagnostics, averaged across evaluation windows."""
    if not task.quantile_levels:
        return {}
    scores_per_window: dict[str, list[float]] = {}
    for predictions, window in zip(predictions_per_window, task.iter_windows(), strict=True):
        cleaned = task.clean_and_validate_predictions(predictions)
        window_scores = window.compute_metrics(
            cleaned,
            metrics=DEFAULT_DIAGNOSTICS,
            seasonality=task.seasonality,
            quantile_levels=task.quantile_levels,
            per_quantile_scores=per_quantile_scores,
        )
        for key, value in window_scores.items():
            scores_per_window.setdefault(key, []).append(value)
    return {key: float(np.mean(values)) for key, values in scores_per_window.items()}


def run_evaluation(config: RunConfig, predictor: Predictor | None = None) -> Path:
    """Run one model over a benchmark and persist fev-format summaries.

    Parameters
    ----------
    config
        The run configuration.
    predictor
        Optional pre-built predictor. If None, built from ``config.predictor``.
        Passing a pre-built predictor is how quantized models (constructed in code
        via ``model_transform``) enter the evaluation path.

    Returns
    -------
    Path to the run's output directory containing ``summaries.csv``,
    ``run_metadata.yaml`` and (if any) ``failures.jsonl``.
    """
    ensure_truststore()
    tasks = load_benchmark_tasks(config.benchmark, config.task_names)
    if predictor is None:
        predictor = build_predictor(config.predictor)

    output_dir = Path(config.output_root)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir = output_dir / config.name
    if output_dir.exists() and any(output_dir.iterdir()) and not config.overwrite:
        raise FileExistsError(
            f"Output directory {output_dir} already exists. Set overwrite=true to replace it."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    env = environment_info()
    metadata = {
        "config": config.to_dict(),
        "environment": env,
        "num_tasks": len(tasks),
        "started_at": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    with open(output_dir / METADATA_FILENAME, "w") as f:
        yaml.safe_dump(metadata, f, sort_keys=False)

    summaries: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for i, task in enumerate(tasks, start=1):
        logger.info("[%d/%d] evaluating %s on %s", i, len(tasks), predictor.name, task.task_name)
        start = time.perf_counter()
        try:
            predictions_per_window, inference_time_s = predictor.predict_task(task)
            summary = task.evaluation_summary(
                predictions_per_window,
                model_name=predictor.name,
                inference_time_s=inference_time_s,
                trained_on_this_dataset=config.trained_on_this_dataset,
                per_quantile_scores=config.per_quantile_scores,
                extra_info={
                    "run_name": config.name,
                    "git_sha": env.get("git_sha"),
                    "gpu_name": env.get("gpu_name"),
                },
            )
            if config.quantile_diagnostics:
                summary.update(
                    _compute_quantile_diagnostics(
                        task, predictions_per_window, config.per_quantile_scores
                    )
                )
            summaries.append(summary)
            # Incremental flush: long runs stay inspectable and interruption-safe
            pd.DataFrame(summaries).to_csv(output_dir / SUMMARIES_FILENAME, index=False)
            logger.info(
                "[%d/%d] %s done in %.1fs (test_error=%.4f)",
                i,
                len(tasks),
                task.task_name,
                time.perf_counter() - start,
                summary["test_error"],
            )
        except Exception as err:  # noqa: BLE001 - failure isolation is intentional
            failure = {
                "task_name": str(task.task_name),
                "error": repr(err),
                "traceback": traceback.format_exc(),
            }
            failures.append(failure)
            with open(output_dir / FAILURES_FILENAME, "a") as f:
                f.write(json.dumps(failure) + "\n")
            logger.error("[%d/%d] %s FAILED: %r", i, len(tasks), task.task_name, err)

    metadata["finished_at"] = pd.Timestamp.now(tz="UTC").isoformat()
    metadata["num_completed"] = len(summaries)
    metadata["num_failed"] = len(failures)
    with open(output_dir / METADATA_FILENAME, "w") as f:
        yaml.safe_dump(metadata, f, sort_keys=False)

    if not summaries:
        raise RuntimeError(f"All {len(tasks)} tasks failed; see {output_dir / FAILURES_FILENAME}")
    logger.info(
        "Run '%s' complete: %d/%d tasks succeeded -> %s",
        config.name,
        len(summaries),
        len(tasks),
        output_dir,
    )
    return output_dir
