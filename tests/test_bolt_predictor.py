"""Tests for the Chronos-Bolt predictor adapter (Phase C within-family size sweep).

Unit tests run against fakes that reproduce the *exact* interface quirks of
`ChronosBoltPipeline` (no model download): ``predict_quantiles`` forwards extra
kwargs to ``predict``, which accepts neither ``batch_size`` nor ``context_length``
and returns stacked ``[batch, Q, H]`` tensors with no per-series n_variates dim.
The network-marked smoke test downloads amazon/chronos-bolt-small (~190 MB) and
verifies the adapter end-to-end on CPU.
"""

import numpy as np
import pytest
import torch
from torch import nn

from chronosquant.evaluation.predictors import (
    Chronos2Predictor,
    ChronosBoltPredictor,
    _BoltPipelineAdapter,
    build_predictor,
)
from chronosquant.quantization.gptq import (
    _stage_key,
    gptaq_quantize_model,
    gptq_quantize_model,
)
from tests.conftest import HORIZON, NUM_SERIES, QUANTILE_LEVELS

NATIVE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


class FakeBoltPipeline:
    """Analytic stand-in mimicking `ChronosBoltPipeline`'s interface faithfully.

    - ``predict`` rejects ``batch_size``/``context_length`` (the incompatibility
      `_BoltPipelineAdapter` exists to fix) and runs all series as one batch.
    - ``predict_quantiles`` returns stacked ``[batch, H, Q]`` quantiles and a
      ``[batch, H]`` mean (the 0.5-quantile channel), with native levels indexed
      directly — no per-series n_variates dim, unlike Chronos-2.

    Forecast value for series ``s`` at step ``h``, level ``q``:
    ``s[-1] + 10 * q + h`` (deterministic, checkable per level).
    """

    quantiles = list(NATIVE_LEVELS)

    def __init__(self, model: nn.Module | None = None):
        self.model = model if model is not None else nn.Identity()
        self.batch_calls: list[list[int]] = []  # per-call list of context lengths

    def predict(self, inputs, prediction_length=None, limit_prediction_length=False):
        series = [torch.as_tensor(s, dtype=torch.float32) for s in inputs]
        self.batch_calls.append([len(s) for s in series])
        h = prediction_length if prediction_length is not None else 8
        out = torch.empty(len(series), len(self.quantiles), h)
        for b, s in enumerate(series):
            for j, q in enumerate(self.quantiles):
                out[b, j, :] = s[-1] + 10 * q + torch.arange(h)
        return out  # [batch, Q, H], as the real predict

    def predict_quantiles(
        self, inputs, prediction_length=None, quantile_levels=NATIVE_LEVELS, **predict_kwargs
    ):
        predictions = self.predict(
            inputs, prediction_length=prediction_length, **predict_kwargs
        ).swapaxes(1, 2)  # [batch, H, Q_native]
        idx = [self.quantiles.index(q) for q in quantile_levels]
        return predictions[..., idx], predictions[..., self.quantiles.index(0.5)]


class TestRegistryDispatch:
    def test_builds_chronos_bolt_lazily(self):
        predictor = build_predictor(
            {
                "type": "chronos_bolt",
                "model_id": "amazon/chronos-bolt-small",
                "name": "bolt-small-fp32",
            }
        )
        assert isinstance(predictor, ChronosBoltPredictor)
        assert isinstance(predictor, Chronos2Predictor)  # inherits the full adapter stack
        assert predictor.name == "bolt-small-fp32"
        assert predictor._pipeline is None  # no download at construction time

    def test_defaults(self):
        predictor = ChronosBoltPredictor()
        assert predictor.model_id == "amazon/chronos-bolt-small"
        assert predictor.quantization is None


class TestSeriesQuantileNormalization:
    """The families differ exactly here: Chronos-2 emits per-series [1, H, Q],
    Bolt emits [H, Q] rows of a stacked batch."""

    def test_bolt_accepts_2d(self):
        q = np.arange(12.0).reshape(4, 3)
        out = ChronosBoltPredictor()._series_quantiles(q, "test")
        np.testing.assert_array_equal(out, q)

    def test_bolt_rejects_3d(self):
        with pytest.raises(ValueError, match="Unexpected quantile forecast shape"):
            ChronosBoltPredictor()._series_quantiles(np.zeros((1, 4, 3)), "test")

    def test_chronos2_accepts_3d_singleton(self):
        q = np.arange(12.0).reshape(1, 4, 3)
        out = Chronos2Predictor()._series_quantiles(q, "test")
        np.testing.assert_array_equal(out, q[0])

    def test_chronos2_rejects_2d(self):
        with pytest.raises(ValueError, match="Unexpected quantile forecast shape"):
            Chronos2Predictor()._series_quantiles(np.zeros((4, 3)), "test")


class TestBoltPipelineAdapter:
    def test_real_signature_rejects_batch_size(self):
        """Guards the premise: the raw bolt interface chokes on batch_size, so the
        adapter is load-bearing for evaluation AND GPTQ/GPTAQ calibration forwards."""
        fake = FakeBoltPipeline()
        with pytest.raises(TypeError):
            fake.predict_quantiles([torch.zeros(10)], prediction_length=4, batch_size=2)

    def test_chunks_batches_and_concatenates(self):
        fake = FakeBoltPipeline()
        adapter = _BoltPipelineAdapter(fake)
        series = [torch.full((20,), float(i)) for i in range(5)]
        quantiles, mean = adapter.predict_quantiles(
            series, prediction_length=6, quantile_levels=NATIVE_LEVELS, batch_size=2
        )
        assert [len(c) for c in fake.batch_calls] == [2, 2, 1]
        assert quantiles.shape == (5, 6, 9)
        assert mean.shape == (5, 6)
        # values survive concatenation in order: series i, level q, step h
        for i in range(5):
            for j, q in enumerate(NATIVE_LEVELS):
                torch.testing.assert_close(
                    quantiles[i, :, j], float(i) + 10 * q + torch.arange(6.0)
                )
        torch.testing.assert_close(mean, quantiles[..., 4])

    def test_context_length_truncation(self):
        fake = FakeBoltPipeline()
        adapter = _BoltPipelineAdapter(fake)
        series = [torch.arange(20.0), torch.arange(3.0)]
        adapter.predict_quantiles(series, prediction_length=4, context_length=8, batch_size=4)
        assert fake.batch_calls == [[8, 3]]  # truncated to last 8; short series intact

    def test_delegates_public_attrs_and_deepcopies(self):
        import copy

        fake = FakeBoltPipeline()
        adapter = _BoltPipelineAdapter(fake)
        assert adapter.quantiles == NATIVE_LEVELS  # delegated to the wrapped pipeline
        clone = copy.deepcopy(adapter)
        assert clone.model is clone.pipeline.model  # consistency GPTAQ relies on
        assert clone.pipeline is not fake


class TestPredictWindow:
    def _predictor(self, fake: FakeBoltPipeline, batch_size: int = 2) -> ChronosBoltPredictor:
        predictor = ChronosBoltPredictor(name="bolt-test", batch_size=batch_size)
        predictor._pipeline = _BoltPipelineAdapter(fake)
        return predictor

    def test_median_is_half_quantile_channel(self, trend_task):
        fake = FakeBoltPipeline()
        predictor = self._predictor(fake)
        window = trend_task.get_window(0)
        predictions = predictor.predict_window(window, trend_task)

        assert len(predictions) == NUM_SERIES
        assert [len(c) for c in fake.batch_calls] == [2, 1]  # batch_size=2 chunking
        for i, item in enumerate(predictions):
            last = 83.0 + 1000 * i  # last context value of the trend fixture
            for q in QUANTILE_LEVELS:
                np.testing.assert_allclose(
                    item[str(q)], last + 10 * q + np.arange(HORIZON)
                )
            # point forecast = the 0.5-quantile channel (Chronos convention for MASE)
            np.testing.assert_allclose(item["predictions"], item["0.5"])

    def test_format_accepted_by_fev(self, trend_task):
        predictor = self._predictor(FakeBoltPipeline())
        window = trend_task.get_window(0)
        predictions = predictor.predict_window(window, trend_task)
        cleaned = trend_task.clean_and_validate_predictions(predictions)
        assert set(cleaned.keys()) == {"target"}


def _residual_block(in_dim: int, h_dim: int, out_dim: int) -> nn.Module:
    """Same child names as chronos_bolt.ResidualBlock (hidden/output/residual)."""
    block = nn.Module()
    block.hidden_layer = nn.Linear(in_dim, h_dim, bias=False)
    block.output_layer = nn.Linear(h_dim, out_dim, bias=False)
    block.residual_layer = nn.Linear(in_dim, out_dim, bias=False)
    block.forward = lambda x: block.output_layer(torch.relu(block.hidden_layer(x))) + (
        block.residual_layer(x)
    )
    return block


class TestSkipModuleDefaults:
    def test_defaults_injected_when_absent(self):
        predictor = ChronosBoltPredictor(quantization={"method": "rtn", "bits": 8})
        config = predictor._quantization_config()
        assert config["skip_modules"] == [
            "input_patch_embedding.*",
            "output_patch_embedding.*",
        ]
        # the user-supplied config object stays untouched
        assert "skip_modules" not in predictor.quantization

    def test_explicit_skip_modules_respected(self):
        predictor = ChronosBoltPredictor(
            quantization={"method": "rtn", "bits": 8, "skip_modules": []}
        )
        assert predictor._quantization_config()["skip_modules"] == []

    def test_chronos2_config_passes_through_unchanged(self):
        quantization = {"method": "rtn", "bits": 8}
        predictor = Chronos2Predictor(quantization=quantization)
        assert predictor._quantization_config() == quantization  # no injection

    def test_patterns_match_bolt_module_names(self):
        from chronosquant.quantization.transforms import _matches_any

        skip = list(ChronosBoltPredictor.default_quant_skip_modules)
        for name in (
            "input_patch_embedding.hidden_layer",
            "input_patch_embedding.residual_layer",
            "output_patch_embedding.output_layer",  # the quantile head
        ):
            assert _matches_any(name, skip), name
        for name in (
            "encoder.block.0.layer.0.SelfAttention.q",
            "decoder.block.1.layer.1.EncDecAttention.o",
            "decoder.block.0.layer.2.DenseReluDense.wi",
        ):
            assert not _matches_any(name, skip), name

    def test_quantization_hook_skips_patch_embeddings(self, trend_task):
        """End-to-end through the lazy `pipeline` property: RTN quantizes the body,
        skips the patch-embedding residual blocks, and forecasts stay finite."""
        model = nn.Module()
        model.input_patch_embedding = _residual_block(4, 6, 4)
        encoder = nn.Module()
        encoder.wi = nn.Linear(4, 8, bias=False)
        encoder.wo = nn.Linear(8, 4, bias=False)
        model.encoder = encoder
        model.output_patch_embedding = _residual_block(4, 6, 4)

        fake = FakeBoltPipeline(model=model)
        predictor = ChronosBoltPredictor(
            name="bolt-rtn-test", quantization={"method": "rtn", "bits": 8}
        )
        predictor._load_pipeline = lambda: _BoltPipelineAdapter(fake)

        _ = predictor.pipeline  # triggers load + quantization
        info = predictor._quant_info
        assert info["n_modules_skipped"] == 6  # 2 residual blocks x 3 linears
        assert info["n_modules_quantized"] == 2  # encoder.wi, encoder.wo
        # skipped linears untouched, body swapped for RTN modules
        assert isinstance(model.input_patch_embedding.hidden_layer, nn.Linear)
        assert not isinstance(model.encoder.wi, nn.Linear)

        window = trend_task.get_window(0)
        predictions = predictor.predict_window(window, trend_task)
        assert np.isfinite(np.asarray(predictions[0]["predictions"])).all()

        described = predictor.describe()
        assert described["quant_n_modules_skipped"] == 6


# ---------------------------------------------------------------------------
# GPTQ / GPTAQ encoder-decoder compatibility (no network: tiny enc-dec fake)
# ---------------------------------------------------------------------------


class _Branch(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.wi = nn.Linear(d, 2 * d, bias=False)
        self.wo = nn.Linear(2 * d, d, bias=False)

    def forward(self, x):
        return self.wo(torch.relu(self.wi(x)))


class _Block(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.layer = nn.ModuleList([_Branch(d), _Branch(d)])

    def forward(self, x):
        for sub in self.layer:
            x = x + sub(x)
        return x


class _Stack(nn.Module):
    def __init__(self, d: int, n_blocks: int):
        super().__init__()
        self.block = nn.ModuleList([_Block(d) for _ in range(n_blocks)])

    def forward(self, x):
        for blk in self.block:
            x = blk(x)
        return x


class TinyEncDecModel(nn.Module):
    """T5-shaped naming (encoder.block.N.layer.M / decoder.block.N.layer.M) around a
    single-step decoder — the Chronos-Bolt execution pattern in miniature."""

    def __init__(self, d_in: int = 16, d: int = 16, h: int = 8, n_q: int = 3):
        super().__init__()
        self.h, self.n_q = h, n_q
        self.input_patch_embedding = nn.Linear(d_in, d, bias=False)
        self.encoder = _Stack(d, 2)
        self.decoder = _Stack(d, 2)
        self.output_patch_embedding = nn.Linear(d, h * n_q, bias=False)

    def forward(self, rows):  # [N, d_in] -> [N, n_q, h]
        hidden = self.encoder(self.input_patch_embedding(rows))
        decoded = self.decoder(hidden)  # decoder runs once (single step)
        return self.output_patch_embedding(decoded).view(-1, self.n_q, self.h)


class TinyEncDecPipeline:
    """Bolt-signature pipeline around TinyEncDecModel: windows each series into
    fixed-size rows, forwards once, and (like the real bolt predict) refuses
    batch_size/context_length kwargs."""

    quantiles = [0.1, 0.5, 0.9]

    def __init__(self, model: TinyEncDecModel, in_features: int = 16):
        self.model = model
        self.in_features = in_features

    def _rows(self, series) -> torch.Tensor:
        rows = []
        for s in series:
            s = torch.as_tensor(s, dtype=torch.float32)
            n = (len(s) // self.in_features) * self.in_features
            rows.append(s[:n].reshape(-1, self.in_features))
        return torch.asinh(torch.cat(rows))

    @torch.no_grad()
    def predict(self, inputs, prediction_length=None, limit_prediction_length=False):
        out = self.model(self._rows(inputs))  # [N, n_q, h_native]
        h = prediction_length if prediction_length is not None else out.shape[-1]
        reps = -(-h // out.shape[-1])
        return out.repeat(1, 1, reps)[..., :h]

    def predict_quantiles(
        self, inputs, prediction_length=None, quantile_levels=(0.1, 0.5, 0.9), **predict_kwargs
    ):
        predictions = self.predict(
            inputs, prediction_length=prediction_length, **predict_kwargs
        ).swapaxes(1, 2)
        idx = [self.quantiles.index(q) for q in quantile_levels]
        return predictions[..., idx], predictions[..., self.quantiles.index(0.5)]


CALIB = dict(
    n_calibration_series=8,
    calibration_context_length=64,
    calibration_seed=7,
)


class TestGptqEncoderDecoderCompat:
    def test_decoder_stage_keys_are_distinct_stages(self):
        """The stage patterns already generalize: decoder.block.N maps to its own
        stages, disjoint from encoder.block.N."""
        name = "decoder.block.2.layer.1.EncDecAttention.q"
        assert _stage_key(name, "sublayer") == "decoder.block.2.layer.1"
        assert _stage_key(name, "block") == "decoder.block.2"
        assert _stage_key("decoder.block.0.layer.2.DenseReluDense.wo", "sublayer") == (
            "decoder.block.0.layer.2"
        )
        assert _stage_key("encoder.block.2.layer.1.SelfAttention.q", "sublayer") != (
            _stage_key(name, "sublayer")
        )
        # bolt patch-embedding residual blocks become their own stages
        assert _stage_key("output_patch_embedding.residual_layer", "sublayer") == (
            "output_patch_embedding"
        )

    def test_gptq_one_shot_through_adapter(self):
        """One-shot GPTQ calibration (which passes batch_size to predict_quantiles)
        works against the bolt calling convention via the adapter."""
        torch.manual_seed(0)
        adapter = _BoltPipelineAdapter(TinyEncDecPipeline(TinyEncDecModel()))
        _, info = gptq_quantize_model(
            adapter.model, pipeline=adapter, bits=8, group_size=16, **CALIB
        )
        assert info["method"] == "gptq"
        assert info["n_modules_quantized"] == 18  # 2 emb + 2x(2 blocks x 2 branches x 2)
        for p in adapter.model.parameters():
            assert torch.isfinite(p).all()

    def test_gptaq_two_stream_through_adapter(self):
        """GPTAQ's sequential two-stream capture (probe forward + paired fp/quantized
        passes, both needing batch_size) on an encoder-decoder model where the
        decoder runs once per forward."""
        torch.manual_seed(1)
        model = TinyEncDecModel()
        reference = {n: p.detach().clone() for n, p in model.named_parameters()}
        adapter = _BoltPipelineAdapter(TinyEncDecPipeline(model))
        _, info = gptaq_quantize_model(
            adapter.model,
            pipeline=adapter,
            bits=4,
            group_size=16,
            calibration_batch_size=4,  # forces the adapter's chunked batching
            calibration_horizon=8,
            **CALIB,
        )
        assert info["method"] == "gptaq"
        # stages: 2 patch embeddings + (encoder + decoder) x 2 blocks x 2 sublayers
        assert info["n_stages"] == 10
        assert info["n_modules_quantized"] == 18
        for name, p in model.named_parameters():
            assert torch.isfinite(p).all()
            assert not torch.equal(p, reference[name]), f"{name} was not quantized"


# ---------------------------------------------------------------------------
# Network smoke test: real amazon/chronos-bolt-small on CPU
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def bolt_small_predictor():
    predictor = build_predictor(
        {
            "type": "chronos_bolt",
            "model_id": "amazon/chronos-bolt-small",
            "name": "bolt-small-fp32",
            "device_map": "cpu",
            "batch_size": 2,  # exercises the adapter's chunking on 3 series
        }
    )
    # Materialize the download BEFORE any `datasets` interaction: in this env the
    # datasets-backed task fixtures can leave huggingface_hub's shared httpx client
    # closed, which breaks a cold-cache model download attempted afterwards.
    _ = predictor.pipeline
    return predictor


@pytest.mark.network
class TestBoltSmallSmokeCPU:
    """End-to-end adapter check on the real checkpoint (~190 MB download), CPU only.

    Run with: uv run pytest tests/test_bolt_predictor.py -q -m network
    """

    def test_short_cpu_forecast(self, bolt_small_predictor, noisy_task):
        window = noisy_task.get_window(0)
        predictions = bolt_small_predictor.predict_window(window, noisy_task)

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

    def test_describe_reports_model_card(self, bolt_small_predictor):
        info = bolt_small_predictor.describe()
        assert info["model_id"] == "amazon/chronos-bolt-small"
        assert 40e6 < info["num_params"] < 60e6  # bolt-small ~48M
        assert info["param_dtypes"] == {"float32": info["num_params"]}
