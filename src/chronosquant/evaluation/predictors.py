"""Predictor adapters: turn a model into fev-format window predictions.

Prediction format (per evaluation window, univariate task): a list with one dict per
series, each dict following `task.predictions_schema`:
``{"predictions": [H floats], "0.1": [H floats], ..., "0.9": [H floats]}``.

Conventions (following the Chronos papers):
- The point forecast (``"predictions"``) is the median (0.5-quantile) when available,
  since MASE is optimized by the median. Falls back to the model's mean otherwise.
- Predictions must be finite; predictors raise rather than silently emit NaN/Inf
  (fev would reject them downstream anyway).
"""

import time
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from fev.task import EvaluationWindow, Task

WindowPredictions = list[dict[str, Any]]

#: Dataset preprocessing parallelism. Kept at 1: `datasets` multiprocess map is flaky on
#: Windows once CUDA is initialized in the parent process (subprocess aborts), and
#: single-process preprocessing is deterministic. Results are cached by HF datasets, so
#: the cost is one-time per dataset.
DATASET_NUM_PROC = 1


class Predictor(ABC):
    """Base class for model adapters evaluated by the runner."""

    #: Row label used in results tables (e.g. "seasonal_naive", "chronos2-fp32").
    name: str

    @abstractmethod
    def predict_window(self, window: EvaluationWindow, task: Task) -> WindowPredictions:
        """Produce predictions for a single evaluation window."""

    def predict_task(self, task: Task) -> tuple[list[WindowPredictions], float]:
        """Produce predictions for all windows in the task.

        Returns
        -------
        predictions_per_window
            One entry per evaluation window, in `task.iter_windows()` order.
        inference_time_s
            Total wall-clock time spent in `predict_window` calls (excludes data loading,
            which is triggered eagerly before timing starts).
        """
        # Eager load: excludes dataset download/preprocessing from inference timing
        task.load_full_dataset(num_proc=DATASET_NUM_PROC)
        predictions_per_window: list[WindowPredictions] = []
        inference_time = 0.0
        for window in task.iter_windows(num_proc=DATASET_NUM_PROC):
            start = time.perf_counter()
            predictions_per_window.append(self.predict_window(window, task))
            inference_time += time.perf_counter() - start
        return predictions_per_window, inference_time

    @staticmethod
    def _extract_contexts(
        window: EvaluationWindow, task: Task, dtype: type = np.float32
    ) -> list[np.ndarray]:
        """Extract per-series historical target arrays from a window."""
        if task.is_multivariate:
            raise NotImplementedError("Multivariate tasks are not supported yet")
        target_column = task.target_columns[0]
        past_data, _ = window.get_input_data()
        return [np.asarray(row[target_column], dtype=dtype) for row in past_data]


def _validate_finite(array: np.ndarray, context: str) -> None:
    if not np.isfinite(array).all():
        raise ValueError(f"Non-finite forecast values produced for {context}")


class SeasonalNaivePredictor(Predictor):
    """Deterministic seasonal naive baseline.

    Forecast for step t is the observation one seasonal period back, with the last
    observed season tiled cyclically when the horizon exceeds the seasonality (the
    GluonTS ``SeasonalNaivePredictor`` convention used by the Chronos papers). All
    quantile levels equal the point forecast (deterministic forecaster).

    This baseline is the normalizer for all relative scores; its correctness is
    validated against the published per-task results from the fev repository
    (see tests and `results/reference/`).
    """

    def __init__(self, name: str = "seasonal_naive"):
        self.name = name

    def predict_window(self, window: EvaluationWindow, task: Task) -> WindowPredictions:
        # float64: this baseline normalizes all relative scores, so it must match the
        # published reference values to float precision (validated in the parity test)
        contexts = self._extract_contexts(window, task, dtype=np.float64)
        season_length = task.seasonality
        horizon = task.horizon

        predictions: WindowPredictions = []
        for i, y in enumerate(contexts):
            if len(y) >= season_length:
                template = y[-season_length:]
            else:
                template = y  # shorter than one season: tile the full history
            # Fill non-finite history values with the last preceding finite value
            template = np.asarray(template, dtype=np.float64)
            if not np.isfinite(template).all():
                template = self._ffill(template, fallback=self._last_finite(y))
            forecast = np.resize(template, horizon)  # cyclic tiling
            _validate_finite(forecast, f"series {i} of task {task.task_name}")
            item = {"predictions": forecast.tolist()}
            for q in task.quantile_levels:
                item[str(q)] = forecast.tolist()
            predictions.append(item)
        return predictions

    @staticmethod
    def _last_finite(y: np.ndarray) -> float:
        finite = y[np.isfinite(y)]
        return float(finite[-1]) if len(finite) else 0.0

    @staticmethod
    def _ffill(arr: np.ndarray, fallback: float) -> np.ndarray:
        out = arr.copy()
        last = fallback
        for j in range(len(out)):
            if np.isfinite(out[j]):
                last = out[j]
            else:
                out[j] = last
        return out


class Chronos2Predictor(Predictor):
    """Adapter for `Chronos2Pipeline` (fp32 reference and, later, quantized variants).

    Parameters
    ----------
    model_id
        HuggingFace model ID or local path (default ``amazon/chronos-2``).
    name
        Row label in results tables. Encode the precision/method here,
        e.g. ``chronos2-fp32``, ``chronos2-w8a16-rtn``.
    device_map
        Passed to ``from_pretrained`` (``"cuda"`` or ``"cpu"``).
    batch_size
        Forwarded to ``predict_quantiles``.
    context_length
        If set, overrides the pipeline's default context length.
    model_transform
        Optional callable applied to the loaded pipeline before first use. This is the
        hook where quantization methods plug in: it receives the pipeline and returns a
        (possibly modified) pipeline. Keeping quantization as a transform means the
        evaluation path is byte-identical for baseline and quantized runs.
    """

    def __init__(
        self,
        model_id: str = "amazon/chronos-2",
        name: str = "chronos2-fp32",
        device_map: str = "cuda",
        batch_size: int = 256,
        context_length: int | None = None,
        model_transform=None,
    ):
        self.model_id = model_id
        self.name = name
        self.device_map = device_map
        self.batch_size = batch_size
        self.context_length = context_length
        self.model_transform = model_transform
        self._pipeline = None

    @property
    def pipeline(self):
        if self._pipeline is None:
            from chronos import Chronos2Pipeline

            from chronosquant.utils import ensure_truststore

            ensure_truststore()
            pipeline = Chronos2Pipeline.from_pretrained(self.model_id, device_map=self.device_map)
            if self.model_transform is not None:
                pipeline = self.model_transform(pipeline)
            self._pipeline = pipeline
        return self._pipeline

    def predict_window(self, window: EvaluationWindow, task: Task) -> WindowPredictions:
        import torch

        if task.known_dynamic_columns or task.past_dynamic_columns or task.static_columns:
            raise NotImplementedError("Covariate tasks are not supported yet")

        contexts = [torch.as_tensor(c) for c in self._extract_contexts(window, task)]
        quantile_levels = list(task.quantile_levels)

        predict_kwargs: dict[str, Any] = {"batch_size": self.batch_size}
        if self.context_length is not None:
            predict_kwargs["context_length"] = self.context_length

        quantiles, mean = self.pipeline.predict_quantiles(
            contexts,
            prediction_length=task.horizon,
            quantile_levels=quantile_levels,
            **predict_kwargs,
        )

        # Point forecast: median if available (Chronos convention for MASE), else mean.
        median_idx = next(
            (j for j, q in enumerate(quantile_levels) if np.isclose(q, 0.5)), None
        )

        predictions: WindowPredictions = []
        for i in range(len(contexts)):
            q_i = quantiles[i].to(torch.float64).cpu().numpy()  # [n_variates=1, H, Q]
            if q_i.ndim != 3 or q_i.shape[0] != 1:
                raise ValueError(f"Unexpected quantile forecast shape {q_i.shape}")
            q_i = q_i[0]  # [H, Q]
            _validate_finite(q_i, f"series {i} of task {task.task_name}")
            if median_idx is not None:
                point = q_i[:, median_idx]
            else:
                point = mean[i].to(torch.float64).cpu().numpy().reshape(-1)
                _validate_finite(point, f"series {i} of task {task.task_name}")
            item = {"predictions": point.tolist()}
            for j, q in enumerate(quantile_levels):
                item[str(q)] = q_i[:, j].tolist()
            predictions.append(item)
        return predictions


PREDICTOR_TYPES: dict[str, type[Predictor]] = {
    "seasonal_naive": SeasonalNaivePredictor,
    "chronos2": Chronos2Predictor,
}


def build_predictor(config: dict[str, Any]) -> Predictor:
    """Build a predictor from a config dict with a ``type`` key.

    Example: ``{"type": "chronos2", "model_id": "amazon/chronos-2", "name": "chronos2-fp32"}``
    """
    config = dict(config)
    predictor_type = config.pop("type", None)
    if predictor_type not in PREDICTOR_TYPES:
        raise ValueError(
            f"Unknown predictor type {predictor_type!r}. Available: {sorted(PREDICTOR_TYPES)}"
        )
    return PREDICTOR_TYPES[predictor_type](**config)
