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

    def describe(self) -> dict[str, Any]:
        """Model card for run provenance: everything a paper table needs about this
        model variant. Called by the runner *after* evaluation (so lazily-loaded
        models are materialized and can be inspected).
        """
        return {"predictor_type": type(self).__name__, "name": self.name}

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
    quantization
        Optional quantization config dict, e.g. ``{"method": "rtn", "bits": 8}``
        (see `chronosquant.quantization.transforms.QUANTIZATION_METHODS`). Applied to
        ``pipeline.model`` after loading; resulting metadata is exposed in `describe()`.
        Keeping quantization as a declarative transform means the evaluation path is
        byte-identical for baseline and quantized runs.
    model_transform
        Optional callable applied to the loaded pipeline after quantization (escape
        hatch for transforms not expressible as configs). Receives and returns the
        pipeline.
    """

    def __init__(
        self,
        model_id: str = "amazon/chronos-2",
        name: str = "chronos2-fp32",
        device_map: str = "cuda",
        batch_size: int = 256,
        context_length: int | None = None,
        quantization: dict[str, Any] | None = None,
        model_transform=None,
    ):
        self.model_id = model_id
        self.name = name
        self.device_map = device_map
        self.batch_size = batch_size
        self.context_length = context_length
        self.quantization = quantization
        self.model_transform = model_transform
        self._pipeline = None
        self._load_time_s: float | None = None
        self._quant_info: dict[str, Any] | None = None

    def _load_pipeline(self):
        """Load the raw pipeline object (overridden by sibling model families)."""
        from chronos import Chronos2Pipeline

        return Chronos2Pipeline.from_pretrained(self.model_id, device_map=self.device_map)

    def _quantization_config(self) -> dict[str, Any] | None:
        """Quantization config as passed to `apply_quantization` (hook for families
        that inject default ``skip_modules``)."""
        return self.quantization

    @property
    def pipeline(self):
        if self._pipeline is None:
            from chronosquant.utils import ensure_truststore

            ensure_truststore()
            start = time.perf_counter()
            pipeline = self._load_pipeline()
            if self.quantization is not None:
                from chronosquant.quantization.transforms import apply_quantization

                _, self._quant_info = apply_quantization(
                    pipeline.model, self._quantization_config(), pipeline=pipeline
                )
            if self.model_transform is not None:
                pipeline = self.model_transform(pipeline)
            self._load_time_s = round(time.perf_counter() - start, 3)
            self._pipeline = pipeline
        return self._pipeline

    def describe(self) -> dict[str, Any]:
        info = super().describe()
        info.update(
            {
                "model_id": self.model_id,
                "device_map": self.device_map,
                "batch_size": self.batch_size,
                "context_length": self.context_length,
                "has_model_transform": self.model_transform is not None,
                "model_load_time_s": self._load_time_s,
            }
        )
        if self._pipeline is not None:
            from chronosquant.models.inspect import model_summary

            info.update(model_summary(self._pipeline.model))
        if self._quant_info is not None:
            info.update({f"quant_{k}": v for k, v in self._quant_info.items()})
        return info

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
            q_i = quantiles[i].to(torch.float64).cpu().numpy()
            q_i = self._series_quantiles(q_i, f"series {i} of task {task.task_name}")  # [H, Q]
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

    def _series_quantiles(self, q_i: np.ndarray, context: str) -> np.ndarray:
        """Normalize one series' quantile forecast to ``[H, Q]``.

        Chronos-2's ``predict_quantiles`` yields one ``[n_variates, H, Q]`` tensor per
        series (``n_variates == 1`` for the univariate tasks supported here).
        """
        if q_i.ndim != 3 or q_i.shape[0] != 1:
            raise ValueError(f"Unexpected quantile forecast shape {q_i.shape} for {context}")
        return q_i[0]


class _BoltPipelineAdapter:
    """Duck-typed wrapper giving `ChronosBoltPipeline` the Chronos-2 calling convention.

    The rest of the codebase (this module's predictors, GPTQ/GPTAQ calibration
    forwards and the GPTAQ execution-order probe) calls
    ``pipeline.predict_quantiles(series, prediction_length=..., quantile_levels=...,
    batch_size=..., [context_length=...])`` and reads ``pipeline.model``. Chronos-Bolt's
    own ``predict_quantiles`` accepts neither ``batch_size`` nor ``context_length`` —
    it forwards extra kwargs to ``predict``, which runs the whole input list as ONE
    padded batch. This adapter chunks the inputs into batches (bounding memory exactly
    like ``BaseChronosPipeline.predict_df`` does), optionally truncates contexts, and
    concatenates the stacked ``[batch, H, Q]`` / ``[batch, H]`` outputs.

    Deep-copyable (GPTAQ deep-copies the pipeline for its frozen fp32 stream); the
    wrapped pipeline and ``model`` stay consistent through ``copy.deepcopy`` memoization.
    """

    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.model = pipeline.model

    def __getattr__(self, name: str):
        # Only fires when normal lookup fails: delegate public attributes to the
        # wrapped pipeline. Underscored names raise so copy/pickle protocol probes
        # (__deepcopy__, __getstate__, ...) fall back to default behaviour.
        if name.startswith("_") or name == "pipeline":
            raise AttributeError(name)
        return getattr(self.pipeline, name)

    def predict_quantiles(
        self,
        inputs,
        prediction_length: int | None = None,
        quantile_levels=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        batch_size: int = 256,
        context_length: int | None = None,
        **kwargs,
    ):
        import torch

        series = [torch.as_tensor(s) for s in inputs]
        if context_length is not None:
            series = [s[..., -context_length:] for s in series]
        quantile_chunks, mean_chunks = [], []
        for start in range(0, len(series), batch_size):
            q, m = self.pipeline.predict_quantiles(
                series[start : start + batch_size],
                prediction_length=prediction_length,
                quantile_levels=list(quantile_levels),
                **kwargs,
            )
            quantile_chunks.append(q)
            mean_chunks.append(m)
        return torch.cat(quantile_chunks, dim=0), torch.cat(mean_chunks, dim=0)


class ChronosBoltPredictor(Chronos2Predictor):
    """Adapter for `ChronosBoltPipeline` (Phase C within-family size sweep).

    Same constructor, lazy loading, quantization hook, ``model_transform`` escape
    hatch and ``describe()`` as `Chronos2Predictor`; the family-specific differences:

    - **Output shape**: Bolt's ``predict_quantiles`` returns stacked tensors —
      quantiles ``[batch, H, Q]`` with no per-series ``n_variates`` dim (Chronos-2
      yields per-series ``[1, H, Q]``); see `_series_quantiles`. The point forecast is
      the 0.5-quantile channel (Bolt's native levels are exactly the nine deciles the
      benchmark scores, so levels are indexed directly with no interpolation, and the
      native prediction length of 64 covers the benchmark's max horizon of 56 in a
      single forward pass).
    - **Calling convention**: the raw pipeline is wrapped in `_BoltPipelineAdapter`
      so ``batch_size`` / ``context_length`` work identically to Chronos-2 for both
      evaluation and GPTQ/GPTAQ calibration forwards.
    - **Default quantization skips**: unless the quantization config specifies
      ``skip_modules``, the input/output patch-embedding residual blocks (the latter
      is the quantile head) stay in fp32 (`default_quant_skip_modules`), mirroring the
      protected-module treatment from the Chronos-2 sensitivity study. Pass
      ``skip_modules`` explicitly (possibly ``[]``) inside ``quantization`` to
      override. Norms are not ``nn.Linear`` and are never quantized by any method.
    """

    #: injected into the quantization config when it does not set ``skip_modules``
    default_quant_skip_modules: tuple[str, ...] = (
        "input_patch_embedding.*",
        "output_patch_embedding.*",
    )

    def __init__(
        self,
        model_id: str = "amazon/chronos-bolt-small",
        name: str = "chronos-bolt-fp32",
        **kwargs,
    ):
        super().__init__(model_id=model_id, name=name, **kwargs)

    def _load_pipeline(self):
        from chronos import BaseChronosPipeline, ChronosBoltPipeline

        pipeline = BaseChronosPipeline.from_pretrained(
            self.model_id, device_map=self.device_map
        )
        if not isinstance(pipeline, ChronosBoltPipeline):
            raise TypeError(
                f"{self.model_id!r} loaded as {type(pipeline).__name__}, expected a "
                "ChronosBoltPipeline (use predictor type 'chronos2' for Chronos-2 models)"
            )
        return _BoltPipelineAdapter(pipeline)

    def _quantization_config(self) -> dict[str, Any] | None:
        if self.quantization is None:
            return None
        config = dict(self.quantization)
        config.setdefault("skip_modules", list(self.default_quant_skip_modules))
        return config

    def _series_quantiles(self, q_i: np.ndarray, context: str) -> np.ndarray:
        """Bolt emits ``[H, Q]`` per series (stacked batch, no n_variates dim)."""
        if q_i.ndim != 2:
            raise ValueError(f"Unexpected quantile forecast shape {q_i.shape} for {context}")
        return q_i


class _TimesFMPipelineAdapter:
    """Duck-typed wrapper giving `TimesFM_2p5_200M_torch` the Chronos-2 calling
    convention used across the codebase (evaluation, GPTQ/GPTAQ calibration forwards,
    the GPTAQ execution-order probe): ``pipeline.predict_quantiles(series,
    prediction_length=..., quantile_levels=..., batch_size=..., [context_length=...])``
    returning ``(quantiles [B, H, Q], mean [B, H])`` torch tensors, plus ``.model``
    (the inner ``nn.Module``, quantization target).

    TimesFM specifics handled here:

    - **Quantile channels** (verified against ``timesfm_2p5_torch._compiled_decode``
      and ``TimesFM_2p5_200M_Definition``): ``forecast`` returns quantiles
      ``[B, H, 10]`` where channel 0 is the native *point/mean* head channel and
      channels 1..9 are the deciles 0.1..0.9 (``decode_index = 5``: TimesFM's own
      point output is literally ``full_forecast[..., 5]``, the median channel).
      Requested levels are mapped onto channels 1..9; the median comes from channel
      5, never channel 0. Non-decile levels raise.
    - **Compilation**: ``tfm.forecast`` requires a prior ``tfm.compile(ForecastConfig)``
      that fixes ``per_core_batch_size`` and ``max_horizon`` (rounded up to the
      128-step output patch internally). Compiling is a cheap closure rebuild (no
      ``torch.compile``), so the adapter lazily (re)compiles whenever the requested
      batch size or horizon exceeds the compiled state.
    - **``fix_quantile_crossing`` is pinned False in every arm**: it is a baked-in
      min/max sorting repair (the Chernozhukov-style rearrangement we study as an
      explicit *repair arm*) and would erase the QCR phenomenon under measurement.
    - **Deep copy**: GPTAQ deep-copies the pipeline for its frozen fp32 stream, but
      ``tfm.compiled_decode`` is a closure over the wrapper object — a naive deepcopy
      would leave the clone's decode bound to the *original* (progressively
      quantized) model. ``__deepcopy__`` copies the inner module and drops the
      compiled state so the clone recompiles against itself.
    - ``forecast`` pads the *caller's* input list in place to a batch multiple; the
      adapter always passes a private copy.
    """

    #: channel layout of the 10-channel forecast output (index 1..9 = decile/10)
    MEAN_CHANNEL = 0

    def __init__(self, tfm, max_context: int = 2048, max_horizon: int = 64):
        self.tfm = tfm
        self.model = tfm.model
        self.max_context = max_context
        self.base_max_horizon = max_horizon
        self._compiled_key: tuple[int, int] | None = None

    def __getattr__(self, name: str):
        # Delegate public attributes; underscored names raise so copy/pickle protocol
        # probes fall back to default behaviour (see _BoltPipelineAdapter).
        if name.startswith("_") or name == "tfm":
            raise AttributeError(name)
        return getattr(self.tfm, name)

    def __deepcopy__(self, memo):
        import copy

        clone = object.__new__(type(self))
        memo[id(self)] = clone
        tfm_clone = object.__new__(type(self.tfm))
        # Copy wrapper state except the module (copied explicitly) and the compiled
        # closure/config (a deepcopied function object would still close over *this*
        # adapter's wrapper and model — the GPTAQ-corrupting trap described above).
        for key, value in self.tfm.__dict__.items():
            if key in ("model", "compiled_decode", "forecast_config"):
                continue
            setattr(tfm_clone, key, copy.deepcopy(value, memo))
        tfm_clone.model = copy.deepcopy(self.tfm.model, memo)
        tfm_clone.compiled_decode = None
        tfm_clone.forecast_config = None
        clone.tfm = tfm_clone
        clone.model = tfm_clone.model
        clone.max_context = self.max_context
        clone.base_max_horizon = self.base_max_horizon
        clone._compiled_key = None  # force a fresh compile bound to the clone
        return clone

    def _ensure_compiled(self, batch_size: int, horizon: int) -> None:
        import timesfm

        max_horizon = max(self.base_max_horizon, horizon)
        key = (batch_size, max_horizon)
        if self._compiled_key == key:
            return
        self.tfm.compile(
            timesfm.ForecastConfig(
                max_context=self.max_context,
                max_horizon=max_horizon,  # rounded up to a 128 multiple internally
                normalize_inputs=True,
                per_core_batch_size=batch_size,
                use_continuous_quantile_head=True,
                force_flip_invariance=True,
                infer_is_positive=True,
                fix_quantile_crossing=False,  # NEVER enable (see class docstring)
                return_backcast=False,
            )
        )
        self._compiled_key = key

    @staticmethod
    def _quantile_channels(quantile_levels) -> list[int]:
        """Map requested levels onto the decile channels 1..9 (never the mean, 0)."""
        channels = []
        for q in quantile_levels:
            ch = int(round(float(q) * 10))
            if not (1 <= ch <= 9) or not np.isclose(ch / 10.0, q):
                raise ValueError(
                    f"TimesFM-2.5 emits the decile grid 0.1..0.9; cannot serve "
                    f"quantile level {q}"
                )
            channels.append(ch)
        return channels

    def predict_quantiles(
        self,
        inputs,
        prediction_length: int,
        quantile_levels=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        batch_size: int = 64,
        context_length: int | None = None,
    ):
        import torch

        series = []
        for s in inputs:
            arr = np.asarray(s.cpu() if isinstance(s, torch.Tensor) else s, dtype=np.float32)
            if arr.ndim != 1:
                raise ValueError(f"Expected 1-D context series, got shape {arr.shape}")
            if context_length is not None:
                arr = arr[-context_length:]
            series.append(arr)

        channels = self._quantile_channels(quantile_levels)
        self._ensure_compiled(batch_size, prediction_length)
        # private list: tfm.forecast pads it in place to a batch-size multiple
        _, quantiles = self.tfm.forecast(horizon=prediction_length, inputs=list(series))
        quantiles = torch.from_numpy(np.ascontiguousarray(quantiles))  # [B, H, 10]
        return quantiles[..., channels], quantiles[..., self.MEAN_CHANNEL]


class TimesFMPredictor(Chronos2Predictor):
    """Adapter for Google TimesFM-2.5 (200M, decoder-only AR; Phase C model family).

    Same constructor surface, lazy loading, quantization hook, ``model_transform``
    escape hatch and ``describe()`` as `Chronos2Predictor`; family specifics:

    - **Loading**: ``timesfm.TimesFM_2p5_200M_torch.from_pretrained(...,
      torch_compile=False)`` (eager mode keeps forward hooks and module swapping
      valid for quantization), moved to ``device_map`` explicitly (TimesFM otherwise
      auto-selects CUDA), then wrapped in `_TimesFMPipelineAdapter` for the shared
      calling convention.
    - **Output shape**: stacked ``[batch, H, Q]`` quantiles (no per-series
      n_variates dim), point forecast = the 0.5-quantile channel. The benchmark's
      nine levels are exactly TimesFM's native decile grid; the benchmark's max
      horizon (56) fits one 128-step output patch, so evaluation never enters the
      autoregressive decode loop (the adapter still recompiles for larger horizons).
    - **Default quantization skips**: the tokenizer (input embedding) and BOTH
      output heads (point + quantile projections) stay fp32 unless the config sets
      ``skip_modules`` explicitly — mirroring the protected-module treatment from
      the Chronos-2/Bolt studies (the quantile head was the calibration-damage
      mechanism there). This leaves the 80 linears of the 20 transformer blocks
      (qkv_proj/out/ff0/ff1, d=1280) as the quantization surface. Norms are not
      ``nn.Linear`` and are never quantized by any method.
    """

    #: injected into the quantization config when it does not set ``skip_modules``
    default_quant_skip_modules: tuple[str, ...] = (
        "tokenizer.*",
        "output_projection_point.*",
        "output_projection_quantiles.*",
    )

    def __init__(
        self,
        model_id: str = "google/timesfm-2.5-200m-pytorch",
        name: str = "timesfm25-fp32",
        batch_size: int = 64,
        max_context: int = 2048,
        **kwargs,
    ):
        super().__init__(model_id=model_id, name=name, batch_size=batch_size, **kwargs)
        self.max_context = max_context

    def _load_pipeline(self):
        import timesfm
        import torch

        tfm = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            self.model_id, torch_compile=False
        )
        device = torch.device(self.device_map)
        tfm.model.device = device  # decode reads this attribute for placement
        tfm.model.to(device)
        tfm.model.eval()
        return _TimesFMPipelineAdapter(tfm, max_context=self.max_context)

    def _quantization_config(self) -> dict[str, Any] | None:
        if self.quantization is None:
            return None
        config = dict(self.quantization)
        config.setdefault("skip_modules", list(self.default_quant_skip_modules))
        return config

    def _series_quantiles(self, q_i: np.ndarray, context: str) -> np.ndarray:
        """TimesFM emits ``[H, Q]`` per series (stacked batch, no n_variates dim)."""
        if q_i.ndim != 2:
            raise ValueError(f"Unexpected quantile forecast shape {q_i.shape} for {context}")
        return q_i

    def describe(self) -> dict[str, Any]:
        info = super().describe()
        info["max_context"] = self.max_context
        if self._pipeline is not None:
            fc = getattr(self._pipeline.tfm, "forecast_config", None)
            if fc is not None:
                # provenance for the calibration/QCR analysis: crossing repair OFF
                info["fix_quantile_crossing"] = fc.fix_quantile_crossing
                info["use_continuous_quantile_head"] = fc.use_continuous_quantile_head
                info["force_flip_invariance"] = fc.force_flip_invariance
        return info


PREDICTOR_TYPES: dict[str, type[Predictor]] = {
    "seasonal_naive": SeasonalNaivePredictor,
    "chronos2": Chronos2Predictor,
    "chronos_bolt": ChronosBoltPredictor,
    "timesfm": TimesFMPredictor,
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
