"""AWQ-style activation-aware scale folding for Chronos-2 (+ RTN quantizer).

AWQ (Lin et al., arXiv:2306.00978) protects salient weights by scaling weight input
channels up (and the upstream producer down) before quantization, chosen per unit by a
grid search over s = E[|x|]^alpha. Chronos-2's architecture makes every fold **exact**:

- pre-norm `Chronos2LayerNorm` (RMSNorm with free per-channel weight) feeds q/k/v and
  wi -> fold s into the norm weight;
- `v -> o`: o's input channels are v's output channels (RoPE touches q/k only) -> fold
  into v's rows;
- `wi -> ReLU -> wo`: ReLU is positively homogeneous -> fold into wi's rows exactly.

The quantizer applied after folding is the same per-channel symmetric RTN used by the
plain `rtn` arm, so the AWQ-vs-RTN comparison isolates the effect of scale folding.
Head ResidualBlocks (biased, no clean upstream) are quantized without folding.
"""

import logging
from typing import Any

import torch
from torch import nn

from chronosquant.quantization.calibration import collect_linear_input_stats

logger = logging.getLogger("chronosquant")

DEFAULT_ALPHA_GRID = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


def _rtn_per_channel_qdq(weight: torch.Tensor, bits: int) -> torch.Tensor:
    """Symmetric per-channel quantize-dequantize (same grid as the `rtn` arm)."""
    q_max = 2 ** (bits - 1) - 1
    scale = (weight.abs().amax(dim=1, keepdim=True) / q_max).clamp(min=1e-10)
    return torch.round(weight / scale).clamp(-q_max, q_max) * scale


def _search_unit_scales(
    weights: list[torch.Tensor],
    x_sample: torch.Tensor,
    abs_mean: torch.Tensor,
    bits: int,
    alpha_grid: tuple[float, ...],
) -> tuple[torch.Tensor, float]:
    """Grid-search alpha; return (best_s, best_alpha) minimizing joint output MSE."""
    best_err, best_s, best_alpha = float("inf"), None, 0.0
    x = x_sample.to(torch.float32)
    references = [x @ w.T for w in weights]
    for alpha in alpha_grid:
        s = abs_mean.clamp(min=1e-8).pow(alpha)
        s = (s / (s.max() * s.min()).sqrt().clamp(min=1e-8)).clamp(min=1e-4)
        x_scaled = x / s
        err = 0.0
        for w, ref in zip(weights, references):
            w_q = _rtn_per_channel_qdq(w * s, bits)
            err += float((x_scaled @ w_q.T - ref).pow(2).mean())
        if err < best_err:
            best_err, best_s, best_alpha = err, s, alpha
    assert best_s is not None
    return best_s, best_alpha


def _discover_fold_units(model: nn.Module) -> list[dict]:
    """Find (upstream, [targets]) fold units in the Chronos-2 encoder.

    Returns dicts: {"kind": norm|rows, "upstream": module, "targets": [(name, module)]}.
    Runtime-verified: raises if the expected structure is absent.
    """
    modules = dict(model.named_modules())
    units = []
    for name, module in modules.items():
        if name.endswith("self_attention"):
            prefix = name.rsplit(".", 1)[0]
            norm = modules[f"{prefix}.layer_norm"]
            q, k, v, o = (modules[f"{name}.{x}"] for x in ("q", "k", "v", "o"))
            assert q.bias is None and v.bias is None, "expected bias-free attention linears"
            units.append(
                {
                    "kind": "norm",
                    "upstream": norm,
                    "targets": [(f"{name}.q", q), (f"{name}.k", k), (f"{name}.v", v)],
                }
            )
            units.append({"kind": "rows", "upstream": v, "targets": [(f"{name}.o", o)]})
        elif name.endswith(".mlp"):
            prefix = name.rsplit(".", 1)[0]
            norm = modules[f"{prefix}.layer_norm"]
            wi, wo = modules[f"{name}.wi"], modules[f"{name}.wo"]
            assert wi.bias is None, "expected bias-free MLP"
            units.append({"kind": "norm", "upstream": norm, "targets": [(f"{name}.wi", wi)]})
            units.append({"kind": "rows", "upstream": wi, "targets": [(f"{name}.wo", wo)]})
    if not units:
        raise ValueError("No foldable units found; unexpected model structure")
    return units


@torch.no_grad()
def awq_rtn_quantize_model(
    model: nn.Module,
    *,
    pipeline=None,
    bits: int = 4,
    alpha_grid: tuple[float, ...] = DEFAULT_ALPHA_GRID,
    simulate: bool = True,
    skip_modules: list[str] | None = None,
    n_calibration_series: int = 128,
    calibration_context_length: int = 1024,
    calibration_seed: int = 123,
) -> tuple[nn.Module, dict[str, Any]]:
    """AWQ scale folding + per-channel RTN over all Linears.

    `simulate=True` stores dequantized grid values (accuracy arm; matches `rtn`
    simulate semantics). Requires the pipeline for calibration forwards.
    """
    from chronosquant.quantization.transforms import (
        rtn_quantize_model,
    )

    if pipeline is None:
        raise ValueError("awq_rtn requires the pipeline for calibration forwards")

    skip_modules = skip_modules or []
    units = _discover_fold_units(model)
    fold_target_names = [n for unit in units for n, _ in unit["targets"]]
    stats = collect_linear_input_stats(
        pipeline,
        fold_target_names,
        n_series=n_calibration_series,
        context_length=calibration_context_length,
        seed=calibration_seed,
    )

    alphas = []
    for unit in units:
        first_target_name = unit["targets"][0][0]
        stat = stats[first_target_name]
        s, alpha = _search_unit_scales(
            [m.weight.data for _, m in unit["targets"]],
            stat.sample_matrix,
            stat.abs_mean,
            bits=bits,
            alpha_grid=alpha_grid,
        )
        alphas.append(alpha)
        for _, target in unit["targets"]:
            target.weight.data.mul_(s)  # scale input channels up
        upstream = unit["upstream"]
        if unit["kind"] == "norm":
            upstream.weight.data.div_(s)
        else:  # rows: divide upstream output channels
            upstream.weight.data.div_(s.unsqueeze(1))

    # Quantize everything (folded encoder + unfolded heads) with the shared RTN grid
    _, rtn_info = rtn_quantize_model(
        model,
        bits=bits,
        granularity="per_channel",
        skip_modules=skip_modules,
        simulate=simulate or bits < 8,
    )

    info: dict[str, Any] = {
        **rtn_info,
        "method": "awq_rtn",
        "n_fold_units": len(units),
        "alpha_mean": round(sum(alphas) / len(alphas), 4),
        "alpha_min": min(alphas),
        "alpha_max": max(alphas),
        "calib_n_series": n_calibration_series,
        "calib_context_length": calibration_context_length,
        "calib_seed": calibration_seed,
        "calib_source": "synthetic",
    }
    return model, info
