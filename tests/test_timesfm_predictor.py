"""Tests for the TimesFM-2.5 predictor adapter (Phase C model-family extension).

Unit tests run against a fake that reproduces the *exact* interface of
`timesfm.TimesFM_2p5_200M_torch` (no model download): ``forecast(horizon, inputs)``
requires a prior ``compile(ForecastConfig)`` fixing ``per_core_batch_size`` and
``max_horizon``, returns stacked ``(point [B, H], quantiles [B, H, 10])`` numpy
arrays whose channel 0 is the native point/mean head and channels 1..9 the deciles
0.1..0.9, and pads the caller's input list in place to a batch-size multiple.

The critical golden-value test: the median must come from decile channel 5, never
channel 0 (the mean channel) — the fake makes channel 0 a poison value so any
off-by-one channel mapping fails loudly.

The network-marked smoke test downloads google/timesfm-2.5-200m-pytorch (~1 GB)
and verifies the adapter end-to-end, including that ``fix_quantile_crossing``
(TimesFM's baked-in sorting repair, which would erase the QCR phenomenon under
study) is OFF on the compiled model.
"""

import copy

import numpy as np
import pytest
import torch
from torch import nn

from chronosquant.evaluation.predictors import (
    Chronos2Predictor,
    TimesFMPredictor,
    _TimesFMPipelineAdapter,
    build_predictor,
)
from chronosquant.quantization.gptq import (
    _stage_key,
    gptaq_quantize_model,
    gptq_quantize_model,
)
from tests.conftest import HORIZON, NUM_SERIES, QUANTILE_LEVELS

#: channel c of the fake's 10-channel output encodes: 0 -> poison mean, 1..9 -> decile c/10
MEAN_POISON = 55_555.0


class FakeTimesFM:
    """Analytic stand-in mimicking `TimesFM_2p5_200M_torch` faithfully.

    - ``forecast`` refuses to run uncompiled and enforces ``horizon <= max_horizon``
      (the real compiled_decode raises the same way).
    - Returns ``(point [B, H], quantiles [B, H, 10])`` numpy arrays. Channel 0 is a
      poison value (the real channel 0 is the mean head — mapping any quantile level
      onto it is the bug class under test); channel c in 1..9 encodes the decile
      c/10. Point output is channel 5 (the real ``full_forecast[..., 5]``).
    - Pads the input list in place to a ``global_batch_size`` multiple, exactly like
      ``TimesFM_2p5.forecast`` (the mutation the adapter must shield callers from).

    Forecast value for series ``s``, step ``h``, channel ``c >= 1``:
    ``s[-1] + 10 * (c / 10) + h``.
    """

    def __init__(self, model: nn.Module | None = None):
        self.model = model if model is not None else nn.Identity()
        self.forecast_config = None
        self.compiled_decode = None
        self.global_batch_size = 0
        self.compile_calls: list = []
        self.forecast_calls: list[list[int]] = []  # context lengths per call

    def compile(self, forecast_config):
        self.forecast_config = forecast_config
        self.global_batch_size = forecast_config.per_core_batch_size
        self.compiled_decode = lambda *a: None  # truthy sentinel
        self.compile_calls.append(forecast_config)

    def forecast(self, horizon, inputs):
        if self.compiled_decode is None:
            raise RuntimeError("Model is not compiled. Please call compile() first.")
        if horizon > self.forecast_config.max_horizon:
            raise ValueError("Horizon must be less than the max horizon.")
        num_inputs = len(inputs)
        if (w := num_inputs % self.global_batch_size) != 0:
            inputs += [np.array([0.0] * 3)] * (self.global_batch_size - w)  # in-place!
        self.forecast_calls.append([len(s) for s in inputs])
        quantiles = np.empty((len(inputs), horizon, 10), dtype=np.float32)
        for b, s in enumerate(inputs):
            base = float(np.asarray(s)[-1]) + np.arange(horizon, dtype=np.float32)
            quantiles[b, :, 0] = MEAN_POISON
            for c in range(1, 10):
                quantiles[b, :, c] = base + 10 * (c / 10)
        return quantiles[..., 5].copy(), quantiles


def make_adapter(fake: FakeTimesFM | None = None, **kwargs) -> _TimesFMPipelineAdapter:
    return _TimesFMPipelineAdapter(fake or FakeTimesFM(), **kwargs)


class TestRegistryDispatch:
    def test_builds_timesfm_lazily(self):
        predictor = build_predictor(
            {
                "type": "timesfm",
                "model_id": "google/timesfm-2.5-200m-pytorch",
                "name": "timesfm25-fp32",
            }
        )
        assert isinstance(predictor, TimesFMPredictor)
        assert isinstance(predictor, Chronos2Predictor)  # inherits the adapter stack
        assert predictor.name == "timesfm25-fp32"
        assert predictor._pipeline is None  # no download at construction time

    def test_defaults(self):
        predictor = TimesFMPredictor()
        assert predictor.model_id == "google/timesfm-2.5-200m-pytorch"
        assert predictor.batch_size == 64
        assert predictor.max_context == 2048
        assert predictor.quantization is None


class TestQuantileChannelSemantics:
    """The load-bearing mapping: level q -> channel round(10q) in 1..9; channel 0
    (the mean head) must never serve a quantile level."""

    def test_median_is_decile_channel_5_not_mean_channel(self):
        adapter = make_adapter()
        quantiles, mean = adapter.predict_quantiles(
            [torch.zeros(20)], prediction_length=4, quantile_levels=[0.5], batch_size=1
        )
        # median == fake channel 5 (value 10*0.5 + h), NOT the poison mean channel
        torch.testing.assert_close(quantiles[0, :, 0], 5.0 + torch.arange(4.0))
        assert not torch.any(quantiles == MEAN_POISON)
        # the "mean" return IS the mean channel (Chronos convention second output)
        assert torch.all(mean == MEAN_POISON)

    def test_all_nine_levels_map_to_their_decile_channels(self):
        adapter = make_adapter()
        levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        quantiles, _ = adapter.predict_quantiles(
            [torch.full((8,), 7.0)], prediction_length=3, quantile_levels=levels, batch_size=1
        )
        assert quantiles.shape == (1, 3, 9)
        for j, q in enumerate(levels):
            torch.testing.assert_close(
                quantiles[0, :, j], 7.0 + 10 * q + torch.arange(3.0)
            )

    def test_synthetic_channel_mapping_against_fake_tensor(self):
        """Golden mapping on a hand-built [B, H, 10] tensor: index j of the output
        equals input channel round(10 * level_j)."""
        channels = _TimesFMPipelineAdapter._quantile_channels(QUANTILE_LEVELS)
        assert channels == [1, 2, 3, 4, 5, 6, 7, 8, 9]
        assert _TimesFMPipelineAdapter._quantile_channels([0.5]) == [5]

    def test_non_decile_levels_raise(self):
        for bad in (0.05, 0.25, 0.95, 0.0, 1.0):
            with pytest.raises(ValueError, match="decile grid"):
                _TimesFMPipelineAdapter._quantile_channels([bad])


class TestAdapterCompileManagement:
    def test_forecast_requires_compile_premise(self):
        """Guards the premise: raw TimesFM refuses to forecast uncompiled, so the
        adapter's lazy compile is load-bearing."""
        fake = FakeTimesFM()
        with pytest.raises(RuntimeError, match="not compiled"):
            fake.forecast(4, [np.zeros(8)])

    def test_compiles_once_and_reuses(self):
        fake = FakeTimesFM()
        adapter = make_adapter(fake)
        for _ in range(3):
            adapter.predict_quantiles(
                [torch.zeros(16)], prediction_length=8, quantile_levels=[0.5], batch_size=4
            )
        assert len(fake.compile_calls) == 1
        fc = fake.compile_calls[0]
        assert fc.per_core_batch_size == 4
        assert fc.max_horizon == 64  # base default covers the benchmark max of 56
        assert fc.fix_quantile_crossing is False
        assert fc.use_continuous_quantile_head is True
        assert fc.normalize_inputs is True

    def test_recompiles_on_larger_horizon(self):
        fake = FakeTimesFM()
        adapter = make_adapter(fake)
        adapter.predict_quantiles(
            [torch.zeros(16)], prediction_length=8, quantile_levels=[0.5], batch_size=2
        )
        adapter.predict_quantiles(
            [torch.zeros(16)], prediction_length=200, quantile_levels=[0.5], batch_size=2
        )
        assert [fc.max_horizon for fc in fake.compile_calls] == [64, 200]

    def test_recompiles_on_batch_size_change(self):
        fake = FakeTimesFM()
        adapter = make_adapter(fake)
        adapter.predict_quantiles(
            [torch.zeros(16)], prediction_length=8, quantile_levels=[0.5], batch_size=2
        )
        adapter.predict_quantiles(
            [torch.zeros(16)], prediction_length=8, quantile_levels=[0.5], batch_size=32
        )
        assert [fc.per_core_batch_size for fc in fake.compile_calls] == [2, 32]


class TestAdapterCallingConvention:
    def test_caller_input_list_not_mutated(self):
        """tfm.forecast pads its input list in place; the adapter must shield the
        caller (the harness reuses context lists across quantile calls)."""
        adapter = make_adapter()
        series = [torch.zeros(10), torch.ones(10), torch.full((10,), 2.0)]
        adapter.predict_quantiles(
            series, prediction_length=4, quantile_levels=[0.5], batch_size=2
        )
        assert len(series) == 3  # would be 4 without the adapter's private copy

    def test_context_length_truncation(self):
        fake = FakeTimesFM()
        adapter = make_adapter(fake)
        adapter.predict_quantiles(
            [torch.arange(20.0), torch.arange(3.0)],
            prediction_length=4,
            quantile_levels=[0.5],
            batch_size=2,
            context_length=8,
        )
        assert fake.forecast_calls == [[8, 3]]  # truncated to last 8; short intact

    def test_rejects_non_1d_series(self):
        adapter = make_adapter()
        with pytest.raises(ValueError, match="1-D context series"):
            adapter.predict_quantiles(
                [torch.zeros(4, 2)], prediction_length=4, quantile_levels=[0.5], batch_size=1
            )

    def test_delegates_public_attrs(self):
        fake = FakeTimesFM()
        adapter = make_adapter(fake)
        assert adapter.global_batch_size == 0  # delegated to the wrapped tfm
        assert adapter.model is fake.model

    def test_deepcopy_detaches_model_and_compiled_state(self):
        """GPTAQ deep-copies the pipeline for its frozen fp32 stream; the clone must
        own an independent module and must NOT inherit the compiled closure (which
        closes over the original wrapper/model)."""
        model = nn.Linear(4, 4)
        fake = FakeTimesFM(model=model)
        adapter = make_adapter(fake)
        adapter.predict_quantiles(  # compile the original
            [torch.zeros(8)], prediction_length=4, quantile_levels=[0.5], batch_size=1
        )
        clone = copy.deepcopy(adapter)
        assert clone.model is not adapter.model
        assert clone.model is clone.tfm.model  # consistency GPTAQ relies on
        assert clone.tfm.compiled_decode is None  # stale closure dropped
        assert clone._compiled_key is None
        # clone still works: recompiles against itself
        q, _ = clone.predict_quantiles(
            [torch.zeros(8)], prediction_length=4, quantile_levels=[0.5], batch_size=1
        )
        assert q.shape == (1, 4, 1)
        assert len(clone.tfm.compile_calls) == 2  # shared history list is copied...
        assert len(adapter.tfm.compile_calls) == 1  # ...not aliased


class TestPredictWindow:
    def _predictor(self, fake: FakeTimesFM, batch_size: int = 2) -> TimesFMPredictor:
        predictor = TimesFMPredictor(name="timesfm-test", batch_size=batch_size)
        predictor._pipeline = _TimesFMPipelineAdapter(fake)
        return predictor

    def test_median_point_forecast_and_level_values(self, trend_task):
        fake = FakeTimesFM()
        predictor = self._predictor(fake)
        window = trend_task.get_window(0)
        predictions = predictor.predict_window(window, trend_task)

        assert len(predictions) == NUM_SERIES
        for i, item in enumerate(predictions):
            last = 83.0 + 1000 * i  # last context value of the trend fixture
            for q in QUANTILE_LEVELS:
                np.testing.assert_allclose(
                    item[str(q)], last + 10 * q + np.arange(HORIZON)
                )
            # point forecast = the 0.5-quantile channel (never the mean channel)
            np.testing.assert_allclose(item["predictions"], item["0.5"])
            assert not np.any(np.asarray(item["predictions"]) == MEAN_POISON)

    def test_format_accepted_by_fev(self, trend_task):
        predictor = self._predictor(FakeTimesFM())
        window = trend_task.get_window(0)
        predictions = predictor.predict_window(window, trend_task)
        cleaned = trend_task.clean_and_validate_predictions(predictions)
        assert set(cleaned.keys()) == {"target"}


class TestSeriesQuantileNormalization:
    def test_timesfm_accepts_2d(self):
        q = np.arange(12.0).reshape(4, 3)
        out = TimesFMPredictor()._series_quantiles(q, "test")
        np.testing.assert_array_equal(out, q)

    def test_timesfm_rejects_3d(self):
        with pytest.raises(ValueError, match="Unexpected quantile forecast shape"):
            TimesFMPredictor()._series_quantiles(np.zeros((1, 4, 3)), "test")


class TestSkipModuleDefaults:
    def test_defaults_injected_when_absent(self):
        predictor = TimesFMPredictor(quantization={"method": "rtn", "bits": 8})
        config = predictor._quantization_config()
        assert config["skip_modules"] == [
            "tokenizer.*",
            "output_projection_point.*",
            "output_projection_quantiles.*",
        ]
        # the user-supplied config object stays untouched
        assert "skip_modules" not in predictor.quantization

    def test_explicit_skip_modules_respected(self):
        predictor = TimesFMPredictor(
            quantization={"method": "rtn", "bits": 8, "skip_modules": []}
        )
        assert predictor._quantization_config()["skip_modules"] == []

    def test_patterns_match_timesfm_module_names(self):
        """Verified against the real model's named_modules (89 linears): tokenizer
        + both output heads are skipped; the 20x4 transformer linears are not."""
        from chronosquant.quantization.transforms import _matches_any

        skip = list(TimesFMPredictor.default_quant_skip_modules)
        for name in (
            "tokenizer.hidden_layer",
            "tokenizer.residual_layer",
            "output_projection_point.output_layer",
            "output_projection_quantiles.output_layer",  # the quantile head
            "output_projection_quantiles.residual_layer",
        ):
            assert _matches_any(name, skip), name
        for name in (
            "stacked_xf.0.attn.qkv_proj",
            "stacked_xf.7.attn.out",
            "stacked_xf.19.ff0",
            "stacked_xf.19.ff1",
        ):
            assert not _matches_any(name, skip), name


# ---------------------------------------------------------------------------
# GPTQ / GPTAQ compatibility (no network: tiny TimesFM-shaped fake)
# ---------------------------------------------------------------------------


class _TinyResidualBlock(nn.Module):
    """Same child names as timesfm.torch.dense.ResidualBlock."""

    def __init__(self, d_in: int, d_h: int, d_out: int):
        super().__init__()
        self.hidden_layer = nn.Linear(d_in, d_h, bias=False)
        self.output_layer = nn.Linear(d_h, d_out, bias=False)
        self.residual_layer = nn.Linear(d_in, d_out, bias=False)

    def forward(self, x):
        return self.output_layer(torch.relu(self.hidden_layer(x))) + self.residual_layer(x)


class _TinyXfBlock(nn.Module):
    """Same child names as timesfm.torch.transformer.Transformer (attn.qkv_proj,
    attn.out, ff0, ff1)."""

    def __init__(self, d: int):
        super().__init__()
        self.attn = nn.Module()
        self.attn.qkv_proj = nn.Linear(d, 3 * d, bias=False)
        self.attn.out = nn.Linear(d, d, bias=False)
        self.ff0 = nn.Linear(d, d, bias=False)
        self.ff1 = nn.Linear(d, d, bias=False)

    def forward(self, x):
        qkv = self.attn.qkv_proj(x)
        x = x + self.attn.out(qkv[..., : x.shape[-1]])
        return x + self.ff1(torch.relu(self.ff0(x)))


class TinyTimesFMModule(nn.Module):
    """TimesFM-2.5 module naming in miniature: tokenizer / stacked_xf.N / both
    output projections."""

    def __init__(self, d_in: int = 16, d: int = 16, h: int = 4, n_q: int = 10, n_blocks: int = 3):
        super().__init__()
        self.h, self.n_q = h, n_q
        self.tokenizer = _TinyResidualBlock(d_in, d, d)
        self.stacked_xf = nn.ModuleList([_TinyXfBlock(d) for _ in range(n_blocks)])
        self.output_projection_point = _TinyResidualBlock(d, d, h * n_q)
        self.output_projection_quantiles = _TinyResidualBlock(d, d, h * n_q)

    def forward(self, rows):  # [N, d_in] -> [N, h, n_q]
        x = self.tokenizer(rows)
        for block in self.stacked_xf:
            x = block(x)
        out = self.output_projection_point(x) + self.output_projection_quantiles(x)
        return out.view(-1, self.h, self.n_q)


class TinyTimesFM:
    """TimesFM-signature wrapper around TinyTimesFMModule: compile gate, stacked
    numpy outputs, in-place batch padding — the surface the adapter targets."""

    def __init__(self, model: TinyTimesFMModule | None = None, patch: int = 16):
        self.model = model if model is not None else TinyTimesFMModule()
        self.patch = patch
        self.forecast_config = None
        self.compiled_decode = None
        self.global_batch_size = 0

    def compile(self, forecast_config):
        self.forecast_config = forecast_config
        self.global_batch_size = forecast_config.per_core_batch_size
        self.compiled_decode = lambda *a: None

    @torch.no_grad()
    def forecast(self, horizon, inputs):
        if self.compiled_decode is None:
            raise RuntimeError("Model is not compiled. Please call compile() first.")
        num_inputs = len(inputs)
        if (w := num_inputs % self.global_batch_size) != 0:
            inputs += [np.array([0.0] * 3)] * (self.global_batch_size - w)
        rows = []
        for s in inputs:
            arr = torch.as_tensor(np.asarray(s, dtype=np.float32))
            if len(arr) < self.patch:
                arr = torch.nn.functional.pad(arr, (self.patch - len(arr), 0))
            rows.append(arr[-self.patch :])
        out = self.model(torch.asinh(torch.stack(rows)))  # [N, h, n_q]
        reps = -(-horizon // out.shape[1])
        out = out.repeat(1, reps, 1)[:, :horizon, :]
        return out[..., 5].numpy(), out.numpy()


CALIB = dict(
    n_calibration_series=8,
    calibration_context_length=64,
    calibration_seed=7,
)


class TestGptqCompat:
    def test_stacked_xf_stage_keys_are_per_block(self):
        """TimesFM's stacked_xf.N names group into one stage per transformer block
        (both modes: no `.layer.M` level exists), disjoint from each other; the
        tokenizer and output heads fall back to their own stages."""
        assert _stage_key("stacked_xf.3.attn.qkv_proj", "sublayer") == "stacked_xf.3"
        assert _stage_key("stacked_xf.3.ff1", "block") == "stacked_xf.3"
        assert _stage_key("stacked_xf.19.ff0", "sublayer") == "stacked_xf.19"
        assert _stage_key("stacked_xf.3.ff1", "sublayer") != _stage_key(
            "stacked_xf.4.ff1", "sublayer"
        )
        assert _stage_key("tokenizer.hidden_layer", "sublayer") == "tokenizer"
        assert _stage_key("output_projection_quantiles.output_layer", "sublayer") == (
            "output_projection_quantiles"
        )
        # regression: Chronos-2/Bolt T5-style names keep their existing grouping
        assert _stage_key("encoder.block.2.layer.1.SelfAttention.q", "sublayer") == (
            "encoder.block.2.layer.1"
        )

    def test_gptq_one_shot_through_adapter(self):
        torch.manual_seed(0)
        adapter = _TimesFMPipelineAdapter(TinyTimesFM())
        _, info = gptq_quantize_model(
            adapter.model, pipeline=adapter, bits=8, group_size=16, **CALIB
        )
        assert info["method"] == "gptq"
        # 3 residual blocks x 3 linears + 3 xf blocks x 4 linears = 21
        assert info["n_modules_quantized"] == 21
        for p in adapter.model.parameters():
            assert torch.isfinite(p).all()

    def test_gptaq_two_stream_through_adapter(self):
        """GPTAQ's deterministic paired forwards (probe + fp/quantized streams)
        through the TimesFM adapter, including the deepcopy of the compiled
        pipeline and per-block stacked_xf stages."""
        torch.manual_seed(1)
        model = TinyTimesFMModule()
        reference = {n: p.detach().clone() for n, p in model.named_parameters()}
        adapter = _TimesFMPipelineAdapter(TinyTimesFM(model))
        _, info = gptaq_quantize_model(
            adapter.model,
            pipeline=adapter,
            bits=4,
            group_size=16,
            calibration_batch_size=4,  # forces padding + multiple decode calls
            calibration_horizon=8,
            **CALIB,
        )
        assert info["method"] == "gptaq"
        # stages: tokenizer + 3 stacked_xf blocks + 2 output projections
        assert info["n_stages"] == 6
        assert info["n_modules_quantized"] == 21
        for name, p in model.named_parameters():
            assert torch.isfinite(p).all()
            assert not torch.equal(p, reference[name]), f"{name} was not quantized"

    def test_gptaq_skip_default_heads(self):
        """With the predictor's default skips applied, only the stacked_xf body is
        quantized -> 3 stages, 12 linears."""
        torch.manual_seed(2)
        adapter = _TimesFMPipelineAdapter(TinyTimesFM())
        _, info = gptaq_quantize_model(
            adapter.model,
            pipeline=adapter,
            bits=4,
            group_size=16,
            skip_modules=list(TimesFMPredictor.default_quant_skip_modules),
            calibration_batch_size=4,
            calibration_horizon=8,
            **CALIB,
        )
        assert info["n_stages"] == 3
        assert info["n_modules_quantized"] == 12
        assert info["n_modules_skipped"] == 9


# ---------------------------------------------------------------------------
# Network smoke test: real google/timesfm-2.5-200m-pytorch on CPU
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def timesfm_predictor():
    predictor = build_predictor(
        {
            "type": "timesfm",
            "name": "timesfm25-fp32",
            "device_map": "cpu",
            "batch_size": 2,  # exercises internal batching + padding on 3 series
        }
    )
    # Materialize the download BEFORE any `datasets` interaction (see
    # test_bolt_predictor.py: cold-cache downloads break after task fixtures).
    _ = predictor.pipeline
    return predictor


@pytest.mark.network
class TestTimesFMSmokeCPU:
    """End-to-end adapter check on the real checkpoint (~1 GB download), CPU only.

    Run with: uv run pytest tests/test_timesfm_predictor.py -q -m network
    """

    def test_short_cpu_forecast(self, timesfm_predictor, noisy_task):
        window = noisy_task.get_window(0)
        predictions = timesfm_predictor.predict_window(window, noisy_task)

        assert len(predictions) == NUM_SERIES
        for item in predictions:
            point = np.asarray(item["predictions"])
            assert point.shape == (HORIZON,)
            np.testing.assert_allclose(point, item["0.5"])  # median point forecast
            levels = np.stack([np.asarray(item[str(q)]) for q in QUANTILE_LEVELS])
            assert np.isfinite(levels).all()
            # the noisy fixture oscillates around 50 +- 10; forecasts must be sane
            assert 20.0 < point.mean() < 80.0
            # interval sanity: lower decile below upper decile on average
            assert levels[0].mean() < levels[-1].mean()

        cleaned = noisy_task.clean_and_validate_predictions(predictions)
        assert set(cleaned.keys()) == {"target"}

    def test_crossing_repair_is_off(self, timesfm_predictor):
        """`fix_quantile_crossing` must be False on the compiled model in every arm:
        it is TimesFM's baked-in sorting repair and would erase the QCR phenomenon.
        (fp32 outputs are typically well-ordered, so the flag state — not observed
        crossings — is the reliable check.)"""
        fc = timesfm_predictor.pipeline.tfm.forecast_config
        assert fc is not None, "model must be compiled after a forecast"
        assert fc.fix_quantile_crossing is False
        assert fc.use_continuous_quantile_head is True

        info = timesfm_predictor.describe()
        assert info["fix_quantile_crossing"] is False

    def test_describe_reports_model_card(self, timesfm_predictor):
        info = timesfm_predictor.describe()
        assert info["model_id"] == "google/timesfm-2.5-200m-pytorch"
        assert 200e6 < info["num_params"] < 260e6  # 2.5-200m checkpoint is ~231M
        assert info["param_dtypes"] == {"float32": info["num_params"]}
