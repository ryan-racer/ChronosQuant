"""Config-driven evaluation runner.

Runs a predictor over a benchmark (a YAML list of fev tasks), producing per-task
evaluation summaries in the standard fev format so results are directly comparable
with (and aggregatable alongside) published reference results.

Design principles (mirroring the evaluation practice of the Chronos-2 / fev-bench and
quantized-LLM literature):

- **Provenance**: every run persists its resolved config, git SHA, package versions,
  hardware, benchmark-file hash and a model card (params, dtypes, bits/param); every
  summary row carries the run name, git SHA and model characteristics.
- **Track everything a paper could need**: besides each task's own eval metric, the
  runner computes the full fev metric suite (WQL, SQL, MQL, MASE, MAE, RMSE, RMSSE,
  WAPE, SMAPE, MAPE — with per-quantile breakdowns) plus calibration diagnostics
  (MACE, coverage curves, QCR), records per-task peak GPU memory and inference time,
  and persists raw predictions + ground truth to parquet so *any* per-series metric
  (ΔWQL distributions, flips, difficulty slices) can be computed later without
  re-running the model.
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
import shutil
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from fev import Benchmark, Task
from fev.metrics import get_metric

from chronosquant.evaluation.metrics import DEFAULT_DIAGNOSTICS
from chronosquant.evaluation.predictors import (
    DATASET_NUM_PROC,
    Predictor,
    WindowPredictions,
    build_predictor,
)
from chronosquant.utils import REPO_ROOT, ensure_truststore, environment_info, file_sha256

logger = logging.getLogger("chronosquant")

SUMMARIES_FILENAME = "summaries.csv"
METADATA_FILENAME = "run_metadata.yaml"
FAILURES_FILENAME = "failures.jsonl"
PREDICTIONS_DIRNAME = "predictions"

#: Supplementary accuracy metrics computed for every task on top of the task's own
#: eval_metric/extra_metrics (which stay untouched so task identity matches published
#: reference CSVs). This is the full fev metric suite minus RMSLE (log1p of negative
#: targets is undefined and only emits noise) and MSE (dominated by RMSE for reporting).
DEFAULT_EXTRA_METRICS = ["SQL", "MQL", "MAE", "RMSE", "RMSSE", "WAPE", "SMAPE", "MAPE"]


@dataclasses.dataclass
class RunConfig:
    """Configuration for a single evaluation run (one model over one benchmark)."""

    name: str
    benchmark: str
    predictor: dict[str, Any]
    task_names: list[str] | None = None
    output_root: str = "results/raw"
    extra_metrics: list[str] = dataclasses.field(
        default_factory=lambda: list(DEFAULT_EXTRA_METRICS)
    )
    per_quantile_scores: bool = True
    quantile_diagnostics: bool = True
    save_predictions: bool = True
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


def _resolve_benchmark_path(benchmark_path: str | Path) -> Path:
    path = Path(benchmark_path)
    return path if path.is_absolute() else REPO_ROOT / path


def _supplementary_metric_objects(task: Task, extra_metric_names: list[str]) -> list:
    """Diagnostics + requested fev metrics not already computed by the task itself."""
    already_computed = {get_metric(m).name for m in [task.eval_metric] + task.extra_metrics}
    metrics = list(DEFAULT_DIAGNOSTICS)
    for name in extra_metric_names:
        metric = get_metric(name)
        if metric.name in already_computed:
            continue
        if metric.needs_quantiles and not task.quantile_levels:
            continue
        metrics.append(metric)
    return metrics


def _compute_supplementary_metrics(
    task: Task,
    predictions_per_window: list[WindowPredictions],
    extra_metric_names: list[str],
    include_diagnostics: bool,
    per_quantile_scores: bool,
) -> dict[str, float]:
    """Compute diagnostics + extra accuracy metrics, averaged across evaluation windows.

    Computed outside `task.evaluation_summary` so the task definition (and therefore
    task identity in cross-model pivot tables) stays byte-identical to the published
    benchmark configs.
    """
    metrics = _supplementary_metric_objects(task, extra_metric_names)
    if not include_diagnostics:
        metrics = [m for m in metrics if m not in DEFAULT_DIAGNOSTICS]
    if not metrics:
        return {}
    scores_per_window: dict[str, list[float]] = {}
    for predictions, window in zip(
        predictions_per_window, task.iter_windows(num_proc=DATASET_NUM_PROC), strict=True
    ):
        cleaned = task.clean_and_validate_predictions(predictions)
        window_scores = window.compute_metrics(
            cleaned,
            metrics=metrics,
            seasonality=task.seasonality,
            quantile_levels=task.quantile_levels,
            per_quantile_scores=per_quantile_scores,
        )
        for key, value in window_scores.items():
            scores_per_window.setdefault(key, []).append(value)
    return {key: float(np.mean(values)) for key, values in scores_per_window.items()}


def _predictions_to_frame(
    task: Task, predictions_per_window: list[WindowPredictions]
) -> pd.DataFrame:
    """Assemble raw predictions + ground truth into a long-format frame.

    Columns: window_idx, item_id, timestamp, step, y_true, point forecast, and one
    column per quantile level. With this artifact persisted, any per-series metric can
    be recomputed offline without re-running the model.
    """
    frames = []
    for window_idx, (predictions, window) in enumerate(
        zip(predictions_per_window, task.iter_windows(num_proc=DATASET_NUM_PROC), strict=True)
    ):
        ground_truth = window.get_ground_truth()
        item_ids = [row[task.id_column] for row in ground_truth]
        target_column = task.target_columns[0]
        horizon = task.horizon
        n_items = len(item_ids)

        frame = pd.DataFrame(
            {
                "window_idx": window_idx,
                "item_id": np.repeat(np.asarray(item_ids, dtype=object), horizon),
                "timestamp": np.concatenate(
                    [np.asarray(row[task.timestamp_column]) for row in ground_truth]
                ),
                "step": np.tile(np.arange(horizon), n_items),
                "y_true": np.concatenate(
                    [np.asarray(row[target_column], dtype=np.float64) for row in ground_truth]
                ),
                "point": np.concatenate(
                    [np.asarray(p["predictions"], dtype=np.float64) for p in predictions]
                ),
            }
        )
        for q in task.quantile_levels:
            frame[str(q)] = np.concatenate(
                [np.asarray(p[str(q)], dtype=np.float64) for p in predictions]
            )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _reset_peak_gpu_memory() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _peak_gpu_memory_mb() -> float | None:
    import torch

    if torch.cuda.is_available():
        return round(torch.cuda.max_memory_allocated() / 1e6, 1)
    return None


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
    ``run_metadata.yaml``, ``predictions/*.parquet`` and (if any) ``failures.jsonl``.
    """
    ensure_truststore()
    tasks = load_benchmark_tasks(config.benchmark, config.task_names)
    if predictor is None:
        predictor = build_predictor(config.predictor)

    output_dir = Path(config.output_root)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir = output_dir / config.name
    if output_dir.exists() and any(output_dir.iterdir()):
        if not config.overwrite:
            raise FileExistsError(
                f"Output directory {output_dir} already exists. Set overwrite=true to replace it."
            )
        # Clean slate: stale artifacts (esp. append-mode failures.jsonl) must not
        # leak into the new run's record
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if config.save_predictions:
        (output_dir / PREDICTIONS_DIRNAME).mkdir(exist_ok=True)

    env = environment_info()
    metadata = {
        "config": config.to_dict(),
        "environment": env,
        "benchmark_sha256": file_sha256(_resolve_benchmark_path(config.benchmark)),
        "num_tasks": len(tasks),
        "task_names": [t.task_name for t in tasks],
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
            _reset_peak_gpu_memory()
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
                    "peak_gpu_memory_mb": _peak_gpu_memory_mb(),
                    "task_wall_time_s": None,  # filled below
                },
            )
            summary.update(
                _compute_supplementary_metrics(
                    task,
                    predictions_per_window,
                    extra_metric_names=config.extra_metrics,
                    include_diagnostics=config.quantile_diagnostics,
                    per_quantile_scores=config.per_quantile_scores,
                )
            )
            if config.save_predictions:
                predictions_frame = _predictions_to_frame(task, predictions_per_window)
                predictions_frame.to_parquet(
                    output_dir / PREDICTIONS_DIRNAME / f"{task.task_name}.parquet", index=False
                )
            summary["task_wall_time_s"] = round(time.perf_counter() - start, 3)
            summaries.append(summary)
            # Incremental flush: long runs stay inspectable and interruption-safe
            pd.DataFrame(summaries).to_csv(output_dir / SUMMARIES_FILENAME, index=False)
            logger.info(
                "[%d/%d] %s done in %.1fs (test_error=%.4f)",
                i,
                len(tasks),
                task.task_name,
                summary["task_wall_time_s"],
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

    model_card = predictor.describe()
    metadata["model_card"] = model_card
    metadata["finished_at"] = pd.Timestamp.now(tz="UTC").isoformat()
    metadata["num_completed"] = len(summaries)
    metadata["num_failed"] = len(failures)
    with open(output_dir / METADATA_FILENAME, "w") as f:
        yaml.safe_dump(metadata, f, sort_keys=False)

    if not summaries:
        raise RuntimeError(f"All {len(tasks)} tasks failed; see {output_dir / FAILURES_FILENAME}")

    # Final write: replicate scalar model-card facts into summary columns so joins
    # (retention vs size, Pareto plots) need only the CSV
    summaries_df = pd.DataFrame(summaries)
    for key, value in model_card.items():
        if key == "name":  # already present as fev's model_name column
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            column = key if key.startswith("model") else f"model_{key}"
            summaries_df[column] = value
    summaries_df.to_csv(output_dir / SUMMARIES_FILENAME, index=False)

    logger.info(
        "Run '%s' complete: %d/%d tasks succeeded -> %s",
        config.name,
        len(summaries),
        len(tasks),
        output_dir,
    )
    return output_dir
