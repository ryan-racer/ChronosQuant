"""Tests for GPTAQ (GPTQv2, arXiv:2504.02692): asymmetric-calibration GPTQ.

Weight level: with coinciding streams (X_fp == X_q) the asymmetric path reduces
exactly to GPTQ; with diverging streams it lowers the asymmetric objective
||W X_fp - What X_q||. Model level: on a 2-layer toy model driven through a
duck-typed pipeline, the sequential two-stream machinery achieves lower
end-to-end output error than (sequential and one-shot) GPTQ — the paper's core
claim in miniature. Also covers the previously unexercised plain-GPTQ W2 path.
"""

import copy

import torch
from torch import nn

from chronosquant.quantization.gptq import (
    _stage_key,
    gptaq_quantize_model,
    gptq_quantize_model,
    gptq_quantize_weight,
)
from chronosquant.quantization.transforms import (
    CALIBRATED_METHODS,
    QUANTIZATION_METHODS,
)


def make_problem(out_f=16, in_f=64, n=256, seed=0, act_noise=0.0):
    """Weight + paired activation streams; X_q deviates from X_fp by act_noise."""
    torch.manual_seed(seed)
    weight = torch.randn(out_f, in_f)
    x_fp = torch.randn(n, in_f) * torch.linspace(0.5, 3.0, in_f)  # anisotropic
    if act_noise:
        x_q = x_fp + act_noise * torch.randn(n, in_f)
    else:
        x_q = x_fp.clone()
    hessian = x_q.T @ x_q
    dxxt = (x_fp - x_q).T @ x_q
    return weight, x_fp, x_q, hessian, dxxt


def asym_output_error(weight, w_q, x_fp, x_q):
    """The GPTAQ objective: quantized layer on X_q vs fp layer on X_fp."""
    return float((x_q @ w_q.T - x_fp @ weight.T).pow(2).mean())


class ToyPipeline:
    """Duck-typed pipeline: windows each series into rows, arcsinh, forwards.

    Deterministic and shape-stable across models with identical architecture —
    exactly the contract gptaq's paired two-stream capture requires.
    """

    def __init__(self, model: nn.Module, in_features: int):
        self.model = model
        self.in_features = in_features

    def _rows(self, series) -> torch.Tensor:
        rows = []
        for s in series:
            s = torch.as_tensor(s, dtype=torch.float32)
            n = (len(s) // self.in_features) * self.in_features
            rows.append(s[:n].reshape(-1, self.in_features))
        return torch.asinh(torch.cat(rows))  # tame the synthetic scale diversity

    @torch.no_grad()
    def predict_quantiles(self, series, prediction_length=8, quantile_levels=(0.5,), batch_size=32):
        out = self.model(self._rows(series))
        return out, out


def make_toy_model(d=32, hidden=48, out=16, seed=0) -> nn.Sequential:
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(d, hidden, bias=False),
        nn.ReLU(),
        nn.Linear(hidden, out, bias=False),
    )


class TestGPTAQWeight:
    def test_zero_stream_deviation_reduces_to_gptq(self):
        """X_fp == X_q => dxxt == 0 => byte-identical to the plain GPTQ path."""
        weight, _, _, hessian, dxxt = make_problem(seed=0, act_noise=0.0)
        assert dxxt.abs().max() == 0
        w_gptq = gptq_quantize_weight(weight, hessian, bits=4, group_size=32)
        w_gptaq = gptq_quantize_weight(weight, hessian, bits=4, group_size=32, dxxt=dxxt)
        torch.testing.assert_close(w_gptaq, w_gptq, rtol=0.0, atol=0.0)

    def test_alpha_zero_reduces_to_gptq(self):
        """alpha=0 disables the correction even for nonzero stream deviation."""
        weight, _, _, hessian, dxxt = make_problem(seed=1, act_noise=0.5)
        w_gptq = gptq_quantize_weight(weight, hessian, bits=4, group_size=32)
        w_a0 = gptq_quantize_weight(
            weight, hessian, bits=4, group_size=32, dxxt=dxxt, alpha=0.0
        )
        torch.testing.assert_close(w_a0, w_gptq, rtol=0.0, atol=0.0)

    def test_asymmetric_objective_improves_on_gptq(self):
        """Core claim, single layer: lower ||W X_fp - What X_q|| than GPTQ."""
        for bits, seed in [(4, 2), (3, 3), (2, 4)]:
            weight, x_fp, x_q, hessian, dxxt = make_problem(seed=seed, act_noise=0.3)
            w_gptq = gptq_quantize_weight(weight, hessian, bits=bits, group_size=32)
            w_gptaq = gptq_quantize_weight(
                weight, hessian, bits=bits, group_size=32, dxxt=dxxt
            )
            err_gptq = asym_output_error(weight, w_gptq, x_fp, x_q)
            err_gptaq = asym_output_error(weight, w_gptaq, x_fp, x_q)
            assert err_gptaq < err_gptq, f"bits={bits}: {err_gptaq} !< {err_gptq}"

    def test_lazy_cross_block_update(self):
        """Multi-block path (block_size < in_features), which production always hits.

        The lazy cross-block P update must (a) still beat GPTQ on the asymmetric
        objective and (b) stay close to the single-block result — block splitting is
        a second-order (quant-error x stream-deviation) approximation, not a rewrite.
        """
        weight, x_fp, x_q, hessian, dxxt = make_problem(in_f=64, seed=11, act_noise=0.3)
        w_gptq = gptq_quantize_weight(weight, hessian, bits=3, group_size=16, block_size=16)
        w_multi = gptq_quantize_weight(
            weight, hessian, bits=3, group_size=16, block_size=16, dxxt=dxxt
        )
        w_single = gptq_quantize_weight(
            weight, hessian, bits=3, group_size=16, block_size=64, dxxt=dxxt
        )
        err_gptq = asym_output_error(weight, w_gptq, x_fp, x_q)
        err_multi = asym_output_error(weight, w_multi, x_fp, x_q)
        err_single = asym_output_error(weight, w_single, x_fp, x_q)
        assert err_multi < err_gptq
        assert err_multi < err_single * 1.25  # same quality class as single-block

    def test_deterministic(self):
        weight, _, _, hessian, dxxt = make_problem(seed=5, act_noise=0.3)
        a = gptq_quantize_weight(weight, hessian, bits=4, dxxt=dxxt)
        b = gptq_quantize_weight(weight, hessian, bits=4, dxxt=dxxt)
        torch.testing.assert_close(a, b)

    def test_finite_all_bit_widths_and_act_order(self):
        weight, _, _, hessian, dxxt = make_problem(seed=6, act_noise=0.3)
        for bits in (8, 4, 3, 2):
            for act_order in (True, False):
                w_q = gptq_quantize_weight(
                    weight, hessian, bits=bits, act_order=act_order, dxxt=dxxt
                )
                assert torch.isfinite(w_q).all()

    def test_8bit_near_lossless(self):
        weight, _, _, hessian, dxxt = make_problem(seed=7, act_noise=0.05)
        w_q = gptq_quantize_weight(weight, hessian, bits=8, group_size=64, dxxt=dxxt)
        rel = (w_q - weight).norm() / weight.norm()
        assert rel < 0.02

    def test_dead_columns_zeroed(self):
        weight, x_fp, x_q, _, _ = make_problem(seed=8, act_noise=0.3)
        x_q[:, 0] = 0.0  # dead input feature in the quantized stream
        hessian = x_q.T @ x_q
        dxxt = (x_fp - x_q).T @ x_q
        w_q = gptq_quantize_weight(weight, hessian, bits=4, dxxt=dxxt)
        assert (w_q[:, 0] == 0).all()


class TestGPTQW2:
    """Plain GPTQ at 2 bits (the never-run cliff-curve point): nothing blocks it."""

    def test_w2_runs_finite_on_grid(self):
        weight, _, _, hessian, _ = make_problem(seed=9)
        w_q = gptq_quantize_weight(weight, hessian, bits=2, group_size=32)
        assert torch.isfinite(w_q).all()
        # symmetric 2-bit grid: at most 3 levels ({-s, 0, +s}) per group row.
        # act_order=False so groups are contiguous input columns.
        w_q_no_perm = gptq_quantize_weight(
            weight, hessian, bits=2, group_size=32, act_order=False
        )
        group = w_q_no_perm[:, :32]
        for row in group[:4]:
            assert len(torch.unique(row.abs())) <= 2  # {0, s} in magnitude

    def test_w2_output_error_beats_rtn(self):
        weight, _, x_q, hessian, _ = make_problem(seed=10)
        w_gptq = gptq_quantize_weight(weight, hessian, bits=2, group_size=32)
        q_max = 1
        scale = (weight.abs().amax(dim=1, keepdim=True) / q_max).clamp(min=1e-10)
        w_rtn = torch.round(weight / scale).clamp(-q_max, q_max) * scale
        err_gptq = (x_q @ w_gptq.T - x_q @ weight.T).pow(2).mean()
        err_rtn = (x_q @ w_rtn.T - x_q @ weight.T).pow(2).mean()
        assert err_gptq < err_rtn

    def test_w2_deterministic(self):
        weight, _, _, hessian, _ = make_problem(seed=11)
        a = gptq_quantize_weight(weight, hessian, bits=2)
        b = gptq_quantize_weight(weight, hessian, bits=2)
        torch.testing.assert_close(a, b)


class TestStageKey:
    def test_chronos_names(self):
        name = "encoder.block.3.layer.0.self_attention.q"
        assert _stage_key(name, "sublayer") == "encoder.block.3.layer.0"
        assert _stage_key(name, "block") == "encoder.block.3"
        assert _stage_key(name, "linear") == name
        assert _stage_key("input_patch_embedding.hidden_layer", "sublayer") == (
            "input_patch_embedding"
        )

    def test_toy_names_fall_back_per_linear(self):
        assert _stage_key("0", "sublayer") == "0"
        assert _stage_key("2", "sublayer") == "2"


class TestGPTAQModel:
    CALIB = dict(
        n_calibration_series=32,
        calibration_context_length=256,
        calibration_seed=7,
    )

    def test_registry(self):
        assert "gptaq" in QUANTIZATION_METHODS
        assert "gptaq" in CALIBRATED_METHODS

    def test_first_layer_reduces_to_gptq(self):
        """Single-linear model: the quantized stream equals the fp stream (nothing
        upstream is quantized), so GPTAQ must reproduce GPTQ exactly."""
        torch.manual_seed(12)
        base = nn.Sequential(nn.Linear(32, 16, bias=False))

        pipe_gptq = ToyPipeline(copy.deepcopy(base), in_features=32)
        _, info_gptq = gptq_quantize_model(
            pipe_gptq.model, pipeline=pipe_gptq, bits=3, group_size=16, **self.CALIB
        )
        pipe_gptaq = ToyPipeline(copy.deepcopy(base), in_features=32)
        _, info_gptaq = gptaq_quantize_model(
            pipe_gptaq.model, pipeline=pipe_gptaq, bits=3, group_size=16, **self.CALIB
        )
        torch.testing.assert_close(
            pipe_gptaq.model[0].weight, pipe_gptq.model[0].weight, rtol=0.0, atol=0.0
        )
        assert info_gptaq["method"] == "gptaq"
        assert info_gptq["method"] == "gptq"

    def test_two_layer_end_to_end_beats_gptq(self):
        """Paper's core claim in miniature: layer-1 quantization perturbs layer-2
        inputs; asymmetric calibration of layer 2 recovers end-to-end accuracy that
        both sequential GPTQ (alpha=0, same two-stream machinery) and the repo's
        one-shot GPTQ cannot."""
        from chronosquant.quantization.calibration import synthetic_calibration_series

        base = make_toy_model(seed=13)
        fp_pipe = ToyPipeline(copy.deepcopy(base), in_features=32)
        bits, group_size = 2, 16

        def quantize(method, **kwargs):
            pipe = ToyPipeline(copy.deepcopy(base), in_features=32)
            method(pipe.model, pipeline=pipe, bits=bits, group_size=group_size,
                   **self.CALIB, **kwargs)
            return pipe

        pipe_oneshot = quantize(gptq_quantize_model)
        pipe_seq = quantize(gptaq_quantize_model, alpha=0.0)  # sequential GPTQ
        pipe_gptaq = quantize(gptaq_quantize_model, alpha=1.0)

        series = synthetic_calibration_series(
            self.CALIB["n_calibration_series"],
            self.CALIB["calibration_context_length"],
            self.CALIB["calibration_seed"],
        )
        x = fp_pipe._rows(series)
        with torch.no_grad():
            y_fp = fp_pipe.model(x)
            errs = {
                "gptq_oneshot": float((pipe_oneshot.model(x) - y_fp).pow(2).mean()),
                "gptq_sequential": float((pipe_seq.model(x) - y_fp).pow(2).mean()),
                "gptaq": float((pipe_gptaq.model(x) - y_fp).pow(2).mean()),
            }
        print(f"\nend-to-end MSE vs fp (W{bits} g{group_size}): {errs}")
        assert errs["gptaq"] < errs["gptq_sequential"]
        assert errs["gptaq"] < errs["gptq_oneshot"]

    def test_metadata_and_determinism(self):
        base = make_toy_model(seed=14)
        results = []
        for _ in range(2):
            pipe = ToyPipeline(copy.deepcopy(base), in_features=32)
            _, info = gptaq_quantize_model(
                pipe.model, pipeline=pipe, bits=4, group_size=16, **self.CALIB
            )
            results.append((pipe.model[0].weight.clone(), pipe.model[2].weight.clone()))
            assert info["hessian_mode"] == "sequential_two_stream_asymmetric"
            assert info["asym_alpha"] == 1.0
            assert info["n_stages"] == 2  # toy names sequence per-linear
            assert info["n_modules_quantized"] == 2
            assert info["achievable_bits_per_weight"] == 5.0  # 4 + 16/16
            assert info["weight_rmse"] > 0
        torch.testing.assert_close(results[0][0], results[1][0])
        torch.testing.assert_close(results[0][1], results[1][1])

    def test_sublayer_stage_grouping_on_t5_style_names(self):
        """Multi-linear stages (Chronos-2's block.N.layer.M naming): the paired
        two-stream capture must handle several linears per stage and blockwise
        sequential propagation across stages."""

        class Branch(nn.Module):
            def __init__(self, d):
                super().__init__()
                self.wi = nn.Linear(d, 2 * d, bias=False)
                self.wo = nn.Linear(2 * d, d, bias=False)

            def forward(self, x):
                return self.wo(torch.relu(self.wi(x)))

        class Block(nn.Module):
            def __init__(self, d):
                super().__init__()
                self.layer = nn.ModuleList([Branch(d), Branch(d)])

            def forward(self, x):
                for sub in self.layer:
                    x = x + sub(x)
                return x

        class Net(nn.Module):
            def __init__(self, d, n_blocks):
                super().__init__()
                self.block = nn.ModuleList([Block(d) for _ in range(n_blocks)])
                self.head = nn.Linear(d, 8, bias=False)

            def forward(self, x):
                for blk in self.block:
                    x = blk(x)
                return self.head(x)

        torch.manual_seed(16)
        pipe = ToyPipeline(Net(32, 2), in_features=32)
        _, info = gptaq_quantize_model(
            pipe.model, pipeline=pipe, bits=4, group_size=16, **self.CALIB
        )
        # stages: block.{0,1}.layer.{0,1} (wi+wo each) + head = 5
        assert info["n_stages"] == 5
        assert info["n_modules_quantized"] == 9
        for p in pipe.model.parameters():
            assert torch.isfinite(p).all()

    def test_apply_quantization_dispatch(self):
        from chronosquant.quantization.transforms import apply_quantization

        base = make_toy_model(seed=15)
        pipe = ToyPipeline(copy.deepcopy(base), in_features=32)
        _, info = apply_quantization(
            pipe.model,
            {"method": "gptaq", "bits": 4, "group_size": 16, **self.CALIB},
            pipeline=pipe,
        )
        assert info["method"] == "gptaq"
        # weights actually changed (quantized)
        assert not torch.equal(pipe.model[0].weight, base[0].weight)
