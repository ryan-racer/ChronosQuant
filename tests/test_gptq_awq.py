"""Tests for the calibrated methods: GPTQ math invariants and AWQ fold exactness."""

import numpy as np
import pytest
import torch
from torch import nn

from chronosquant.quantization.awq import _rtn_per_channel_qdq, _search_unit_scales
from chronosquant.quantization.calibration import (
    LinearInputStats,
    synthetic_calibration_series,
)
from chronosquant.quantization.gptq import gptq_quantize_weight


def make_problem(out_f=16, in_f=64, n=256, seed=0):
    torch.manual_seed(seed)
    weight = torch.randn(out_f, in_f)
    x = torch.randn(n, in_f) * torch.linspace(0.5, 3.0, in_f)  # anisotropic inputs
    hessian = x.T @ x
    return weight, x, hessian


class TestGPTQWeight:
    def test_output_error_beats_rtn(self):
        """GPTQ's defining property: lower output MSE than RTN on the calibration dist."""
        weight, x, hessian = make_problem()
        w_gptq = gptq_quantize_weight(weight, hessian, bits=4, group_size=32)
        q_max = 7
        scale = (weight.abs().amax(dim=1, keepdim=True) / q_max).clamp(min=1e-10)
        w_rtn = torch.round(weight / scale).clamp(-q_max, q_max) * scale

        err_gptq = (x @ w_gptq.T - x @ weight.T).pow(2).mean()
        err_rtn = (x @ w_rtn.T - x @ weight.T).pow(2).mean()
        assert err_gptq < err_rtn

    def test_8bit_near_lossless(self):
        weight, x, hessian = make_problem(seed=1)
        w_q = gptq_quantize_weight(weight, hessian, bits=8, group_size=64)
        rel = (w_q - weight).norm() / weight.norm()
        assert rel < 0.01

    def test_deterministic(self):
        weight, _, hessian = make_problem(seed=2)
        a = gptq_quantize_weight(weight, hessian, bits=4)
        b = gptq_quantize_weight(weight, hessian, bits=4)
        torch.testing.assert_close(a, b)

    def test_act_order_toggle_runs(self):
        weight, _, hessian = make_problem(seed=3)
        for act_order in (True, False):
            w_q = gptq_quantize_weight(weight, hessian, bits=4, act_order=act_order)
            assert torch.isfinite(w_q).all()

    def test_dead_columns_zeroed(self):
        weight, x, _ = make_problem(seed=4)
        x[:, 0] = 0.0  # dead input feature
        hessian = x.T @ x
        w_q = gptq_quantize_weight(weight, hessian, bits=4)
        assert (w_q[:, 0] == 0).all()


class TestAWQSearch:
    def test_scale_search_beats_alpha_zero(self):
        """On outlier-channel inputs, some alpha>0 must beat no scaling (alpha=0)."""
        torch.manual_seed(5)
        weight = torch.randn(16, 64)
        x = torch.randn(512, 64)
        x[:, :4] *= 50.0  # strong activation outlier channels
        abs_mean = x.abs().mean(0)
        s, alpha = _search_unit_scales(
            [weight], x, abs_mean, bits=4, alpha_grid=(0.0, 0.25, 0.5, 0.75, 1.0)
        )
        assert alpha > 0.0
        # error with chosen s must be <= error at alpha=0 by construction of argmin
        ref = x @ weight.T
        err_s = ((x / s) @ _rtn_per_channel_qdq(weight * s, 4).T - ref).pow(2).mean()
        err_0 = (x @ _rtn_per_channel_qdq(weight, 4).T - ref).pow(2).mean()
        assert err_s <= err_0 + 1e-8


class TestAWQFoldExactness:
    """The fold must be output-invariant BEFORE quantization (exactness property)."""

    def test_norm_fold_invariance(self):
        """RMSNorm-weight fold: norm.weight/s then W*s leaves outputs unchanged."""
        torch.manual_seed(6)
        d = 32
        norm_weight = torch.rand(d) + 0.5
        linear = nn.Linear(d, 16, bias=False)
        x = torch.randn(64, d)

        def forward(nw, w):
            normed = x / x.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt() * nw
            return normed @ w.T

        y_ref = forward(norm_weight, linear.weight)
        s = torch.rand(d) + 0.5
        y_folded = forward(norm_weight / s, linear.weight * s)
        torch.testing.assert_close(y_ref, y_folded, rtol=1e-5, atol=1e-5)

    def test_relu_fold_invariance(self):
        """wi-rows fold through ReLU: exact for s > 0 (positive homogeneity)."""
        torch.manual_seed(7)
        wi = nn.Linear(32, 64, bias=False)
        wo = nn.Linear(64, 32, bias=False)
        x = torch.randn(16, 32)
        y_ref = torch.relu(x @ wi.weight.T) @ wo.weight.T

        s = torch.rand(64) + 0.5
        wi_folded = wi.weight / s.unsqueeze(1)
        wo_folded = wo.weight * s
        y_folded = torch.relu(x @ wi_folded.T) @ wo_folded.T
        torch.testing.assert_close(y_ref, y_folded, rtol=1e-5, atol=1e-5)


class TestCalibration:
    def test_synthetic_series_deterministic_and_diverse(self):
        a = synthetic_calibration_series(n_series=8, context_length=128, seed=9)
        b = synthetic_calibration_series(n_series=8, context_length=128, seed=9)
        assert all((x == y).all() for x, y in zip(a, b))
        scales = np.array([float(x.abs().mean()) for x in a])
        assert scales.max() / max(scales.min(), 1e-9) > 10  # scale diversity

    def test_input_stats_accumulate_exactly(self):
        stats = LinearInputStats(in_features=8, device=torch.device("cpu"))
        x1, x2 = torch.randn(10, 8), torch.randn(6, 8)
        stats.update(x1)
        stats.update(x2)
        x = torch.cat([x1, x2])
        torch.testing.assert_close(stats.hessian, x.T @ x)
        torch.testing.assert_close(stats.abs_mean, x.abs().mean(0))
        assert stats.n_rows == 16
        assert stats.sample_matrix.shape == (16, 8)


@pytest.mark.network
class TestCalibratedMethodsOnRealModel:
    """End-to-end on the real pipeline (network + GPU): the wiring must produce a
    working quantized model with finite forecasts."""

    @pytest.fixture(scope="class")
    def pipeline(self):
        from chronosquant.utils import ensure_truststore

        ensure_truststore()
        from chronos import Chronos2Pipeline

        # CPU: keeps this test independent of concurrent GPU evaluation runs
        return Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cpu")

    @pytest.mark.parametrize("method", ["gptq", "awq_rtn"])
    def test_quantize_and_forecast(self, pipeline, method):
        import copy

        from chronosquant.quantization.transforms import apply_quantization

        pipe = copy.deepcopy(pipeline)
        _, info = apply_quantization(
            pipe.model,
            {
                "method": method,
                "bits": 4,
                "n_calibration_series": 16,
                "calibration_context_length": 256,
            },
            pipeline=pipe,
        )
        assert info["n_modules_quantized"] == 126
        context = [torch.linspace(0, 10, 512) + torch.randn(512) * 0.1]
        quantiles, _ = pipe.predict_quantiles(
            context, prediction_length=24, quantile_levels=[0.1, 0.5, 0.9]
        )
        assert torch.isfinite(quantiles[0]).all()
