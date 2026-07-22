"""GPTQ for Chronos-2: Hessian-aware error-compensating weight quantization.

Implementation of the GPTQ algorithm (Frantar et al., arXiv:2210.17323) operating
directly on named Linear modules — no HF-LLM harness, no tracing framework. Per-linear
input Hessians are captured in one pass of the fp32 pipeline over synthetic calibration
series ("one-shot" / parallel GPTQ: all layers use fp-model activations rather than the
original's sequential block re-forwarding; documented in the metadata and cheap to
tighten later).

Quantized weights are written back as their dequantized grid values (accuracy-exact
simulation of the quantized model; storage containers are a separate concern — the
achievable BPW incl. group scales is recorded in the metadata).
"""

import logging
from typing import Any

import torch
from torch import nn

from chronosquant.quantization.calibration import (
    collect_linear_input_stats,
    synthetic_calibration_series,  # noqa: F401  (re-export for ablations)
)

logger = logging.getLogger("chronosquant")


@torch.no_grad()
def gptq_quantize_weight(
    weight: torch.Tensor,
    hessian: torch.Tensor,
    bits: int,
    group_size: int = 128,
    percdamp: float = 0.01,
    act_order: bool = True,
    block_size: int = 128,
) -> torch.Tensor:
    """Quantize one weight matrix [out, in] with GPTQ; returns dequantized weights.

    Symmetric per-group grids (group_size input columns share a scale, computed from the
    max-abs of the group), error feedback via the Cholesky inverse of the (damped)
    Hessian, optional activation-order permutation (descending diag(H)).
    """
    out_features, in_features = weight.shape
    w = weight.detach().to(torch.float32).clone()
    h = hessian.detach().to(torch.float32).clone()

    # Dead input columns: no signal in H -> weight value is irrelevant; zero them
    dead = torch.diag(h) == 0
    h[dead, dead] = 1.0
    w[:, dead] = 0.0

    if act_order:
        perm = torch.argsort(torch.diag(h), descending=True)
        w = w[:, perm]
        h = h[perm][:, perm]
        inv_perm = torch.argsort(perm)

    damp = percdamp * torch.mean(torch.diag(h))
    h += torch.eye(in_features, device=h.device) * damp

    # Inverse via Cholesky; upper-triangular factor of H^-1 (standard GPTQ trick)
    h_inv = torch.cholesky_inverse(torch.linalg.cholesky(h))
    h_inv = torch.linalg.cholesky(h_inv, upper=True)

    q_max = 2 ** (bits - 1) - 1
    scales = torch.zeros(out_features, 1, device=w.device)

    for block_start in range(0, in_features, block_size):
        block_end = min(block_start + block_size, in_features)
        w_block = w[:, block_start:block_end].clone()
        err_block = torch.zeros_like(w_block)
        h_inv_block = h_inv[block_start:block_end, block_start:block_end]

        for j in range(block_end - block_start):
            col = block_start + j
            if col % group_size == 0:
                group_end = min(col + group_size, in_features)
                scales = (
                    w[:, col:group_end].abs().amax(dim=1, keepdim=True) / q_max
                ).clamp(min=1e-10)
            w_col = w_block[:, j]
            q_col = torch.clamp(torch.round(w_col / scales.squeeze(1)), -q_max, q_max)
            dq_col = q_col * scales.squeeze(1)
            err = (w_col - dq_col) / h_inv_block[j, j]
            # error feedback into the remaining columns of this block
            w_block[:, j:] -= err.unsqueeze(1) * h_inv_block[j, j:].unsqueeze(0)
            w_block[:, j] = dq_col
            err_block[:, j] = err
        w[:, block_start:block_end] = w_block
        # propagate accumulated block error into all later columns
        w[:, block_end:] -= err_block @ h_inv[block_start:block_end, block_end:]

    if act_order:
        w = w[:, inv_perm]
    return w.to(weight.dtype)


def gptq_quantize_model(
    model: nn.Module,
    *,
    pipeline=None,
    bits: int = 4,
    group_size: int = 128,
    percdamp: float = 0.01,
    act_order: bool = True,
    skip_modules: list[str] | None = None,
    n_calibration_series: int = 128,
    calibration_context_length: int = 1024,
    calibration_seed: int = 123,
) -> tuple[nn.Module, dict[str, Any]]:
    """GPTQ over all Linear modules of the pipeline's model.

    Requires the `pipeline` (calibration forwards run through its real preprocessing).
    """
    from chronosquant.quantization.transforms import _iter_linear_parents, _matches_any

    if pipeline is None:
        raise ValueError("gptq requires the pipeline for calibration forwards")

    skip_modules = skip_modules or []
    targets = [
        (full_name, child)
        for _, _, full_name, child in _iter_linear_parents(model)
        if not _matches_any(full_name, skip_modules)
    ]
    skipped = [
        full_name
        for _, _, full_name, _ in _iter_linear_parents(model)
        if _matches_any(full_name, skip_modules)
    ]
    if not targets:
        raise ValueError("No Linear modules to quantize (check skip_modules patterns)")

    logger.info(
        "GPTQ calibration: %d series x ctx %d over %d linears",
        n_calibration_series,
        calibration_context_length,
        len(targets),
    )
    stats = collect_linear_input_stats(
        pipeline,
        [name for name, _ in targets],
        n_series=n_calibration_series,
        context_length=calibration_context_length,
        seed=calibration_seed,
    )

    total_sq_err = 0.0
    total_weight_numel = 0
    for name, module in targets:
        w_q = gptq_quantize_weight(
            module.weight.data,
            stats[name].hessian,
            bits=bits,
            group_size=group_size,
            percdamp=percdamp,
            act_order=act_order,
        )
        total_sq_err += float((w_q - module.weight.data).pow(2).sum())
        total_weight_numel += module.weight.numel()
        module.weight.data.copy_(w_q)
        del stats[name]  # free the Hessian as we go

    info: dict[str, Any] = {
        "method": "gptq",
        "weight_bits": bits,
        "activation_bits": None,
        "granularity": f"group_{group_size}",
        "simulated": True,  # dequantized grid values stored in fp32 containers
        "act_order": act_order,
        "percdamp": percdamp,
        "hessian_mode": "one_shot_fp_activations",
        "n_modules_quantized": len(targets),
        "n_modules_skipped": len(skipped),
        "quantized_weight_numel": total_weight_numel,
        # achievable storage: b-bit codes + one fp16 scale per group
        "effective_bits_per_weight": None,
        "achievable_bits_per_weight": round(bits + 16 / group_size, 4),
        "weight_rmse": round((total_sq_err / total_weight_numel) ** 0.5, 8),
        "calib_n_series": n_calibration_series,
        "calib_context_length": calibration_context_length,
        "calib_seed": calibration_seed,
        "calib_source": "synthetic",
    }
    return model, info
