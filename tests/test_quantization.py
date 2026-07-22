"""Tests for quantization transforms: exact error bounds, accounting, determinism."""

import numpy as np
import pytest
import torch
from torch import nn

from chronosquant.models.inspect import model_summary
from chronosquant.quantization.transforms import (
    RTNQuantizedLinear,
    apply_quantization,
    rtn_quantize_model,
)


def small_model(seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 8),
    )


class TestRTNQuantizedLinear:
    def test_weight_reconstruction_error_bounded_by_half_scale(self):
        """RTN's defining property: |w - dequant(quant(w))| <= scale/2 elementwise."""
        torch.manual_seed(1)
        linear = nn.Linear(64, 32)
        qlinear = RTNQuantizedLinear(linear, granularity="per_channel")
        w_deq = qlinear.weight_q.to(torch.float32) * qlinear.scale
        err = (linear.weight.detach() - w_deq).abs()
        assert (err <= qlinear.scale / 2 + 1e-8).all()

    def test_int8_storage_and_scale_shapes(self):
        linear = nn.Linear(64, 32)
        qlinear = RTNQuantizedLinear(linear, granularity="per_channel")
        assert qlinear.weight_q.dtype == torch.int8
        assert qlinear.scale.shape == (32, 1)
        per_tensor = RTNQuantizedLinear(linear, granularity="per_tensor")
        assert per_tensor.scale.shape == (1, 1)

    def test_forward_close_to_original(self):
        torch.manual_seed(2)
        linear = nn.Linear(64, 32)
        qlinear = RTNQuantizedLinear(linear)
        x = torch.randn(8, 64)
        y_ref, y_q = linear(x), qlinear(x)
        assert y_q.shape == y_ref.shape
        # int8 per-channel on Gaussian weights: small relative error
        rel_err = (y_q - y_ref).norm() / y_ref.norm()
        assert rel_err < 0.02

    def test_bias_preserved(self):
        linear = nn.Linear(8, 4)
        qlinear = RTNQuantizedLinear(linear)
        torch.testing.assert_close(qlinear.bias, linear.bias)
        no_bias = RTNQuantizedLinear(nn.Linear(8, 4, bias=False))
        assert no_bias.bias is None


class TestRTNQuantizeModel:
    def test_replaces_all_linears_and_reports_metadata(self):
        model = small_model()
        model, info = rtn_quantize_model(model)
        assert isinstance(model[0], RTNQuantizedLinear)
        assert isinstance(model[2], RTNQuantizedLinear)
        assert info["method"] == "rtn"
        assert info["weight_bits"] == 8
        assert info["simulated"] is False
        assert info["n_modules_quantized"] == 2
        assert info["quantized_weight_numel"] == 32 * 16 + 16 * 8

    def test_effective_bits_per_weight_accounting(self):
        """BPW = (int8 bytes + fp32 scale bytes) * 8 / weight count, exactly."""
        model = small_model()
        _, info = rtn_quantize_model(model)
        n_weights = 32 * 16 + 16 * 8  # 640
        packed = n_weights  # int8: 1 byte each
        scales = (16 + 8) * 4  # per-channel fp32
        expected = (packed + scales) * 8 / n_weights
        assert info["effective_bits_per_weight"] == pytest.approx(expected, abs=1e-4)
        # storage actually shrinks: model_summary sees int8 buffers
        summary = model_summary(model)
        assert summary["bits_per_element"] < 10

    def test_skip_modules_pattern(self):
        model = small_model()
        model, info = rtn_quantize_model(model, skip_modules=["2"])
        assert isinstance(model[0], RTNQuantizedLinear)
        assert isinstance(model[2], nn.Linear)  # skipped
        assert info["n_modules_quantized"] == 1
        assert info["n_modules_skipped"] == 1

    def test_simulate_mode_fake_quantizes_in_place(self):
        model = small_model()
        w_before = model[0].weight.detach().clone()
        model, info = rtn_quantize_model(model, bits=4, simulate=True)
        assert isinstance(model[0], nn.Linear)  # no wrapper
        assert not torch.equal(model[0].weight, w_before)  # weights changed
        assert info["simulated"] is True
        assert info["effective_bits_per_weight"] is None  # no storage claim

    def test_low_bits_without_simulate_raises(self):
        with pytest.raises(ValueError, match="simulate"):
            rtn_quantize_model(small_model(), bits=4)

    def test_deterministic(self):
        m1, _ = rtn_quantize_model(small_model(seed=3))
        m2, _ = rtn_quantize_model(small_model(seed=3))
        x = torch.randn(4, 32)
        torch.testing.assert_close(m1(x), m2(x))

    def test_all_skipped_raises(self):
        with pytest.raises(ValueError, match="No Linear modules"):
            rtn_quantize_model(small_model(), skip_modules=["*"])


class TestApplyQuantization:
    def test_dispatches_by_method(self):
        model, info = apply_quantization(small_model(), {"method": "rtn", "bits": 8})
        assert info["method"] == "rtn"

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown quantization method"):
            apply_quantization(small_model(), {"method": "nope"})


class TestQuantizedChronos2SmokeViaSyntheticTask:
    """End-to-end: quantized predictor config flows through build_predictor.

    (Real-model quantized inference is exercised by the dev evaluation run;
    here we only verify the wiring is constructible offline.)
    """

    def test_build_predictor_accepts_quantization_config(self):
        from chronosquant.evaluation.predictors import build_predictor

        predictor = build_predictor(
            {
                "type": "chronos2",
                "name": "chronos2-w8a16-rtn",
                "quantization": {"method": "rtn", "bits": 8, "granularity": "per_channel"},
            }
        )
        assert predictor.quantization == {
            "method": "rtn",
            "bits": 8,
            "granularity": "per_channel",
        }
        assert predictor._pipeline is None  # still lazy

    def test_quant_metadata_lands_in_describe_after_manual_injection(self):
        """describe() exposes quant_* keys once _quant_info is populated."""
        from chronosquant.evaluation.predictors import Chronos2Predictor

        predictor = Chronos2Predictor(name="q")
        predictor._quant_info = {"method": "rtn", "weight_bits": 8}
        info = predictor.describe()
        assert info["quant_method"] == "rtn"
        assert info["quant_weight_bits"] == 8


def test_rtn_quantized_model_output_error_scales_with_bits():
    """Sanity on the simulate path: fewer bits -> strictly larger weight error."""
    errors = []
    for bits in [8, 4, 2]:
        model = small_model(seed=7)
        w_ref = model[0].weight.detach().clone()
        rtn_quantize_model(model, bits=bits, simulate=True)
        errors.append((model[0].weight.detach() - w_ref).norm().item())
    assert errors[0] < errors[1] < errors[2]
    assert np.isfinite(errors).all()
