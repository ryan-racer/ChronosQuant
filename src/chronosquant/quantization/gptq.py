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

GPTAQ / GPTQv2 (Li et al., arXiv:2504.02692, ICML 2025) is implemented as an extension
of the same column loop: *asymmetric calibration* minimizes ``||W x_fp - What x_q||``
(quantized layer on quantized-stream activations vs the full-precision layer on clean
activations) instead of GPTQ's symmetric ``||W x - What x||``. The closed-form
correction enters as one extra rank-structured term per column (Algorithm 1 of the
paper); calibration propagates TWO activation streams stage-by-stage in execution
order. ``gptaq_quantize_model`` orchestrates; ``gptq_quantize_weight(dxxt=...)`` is
the per-matrix solver shared by both methods.
"""

import copy
import logging
import re
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
    dxxt: torch.Tensor | None = None,
    alpha: float = 1.0,
) -> torch.Tensor:
    """Quantize one weight matrix [out, in] with GPTQ; returns dequantized weights.

    Symmetric per-group grids (group_size input columns share a scale, computed from the
    max-abs of the group), error feedback via the Cholesky inverse of the (damped)
    Hessian, optional activation-order permutation (descending diag(H)).

    GPTAQ asymmetric mode (arXiv:2504.02692): pass ``dxxt`` = sum_t (x_fp - x_q) x_q^T,
    the cross-correlation between the fp-stream activation deviation DX = X_fp - X_q and
    the quantized-stream activations X_q (the ``hessian`` must then be X_q X_q^T). The
    residual-correction matrix

        P = ((DX X_q^T @ L) * M_U) @ L^T          (paper Theorem 4.2)

    with L the lower Cholesky factor of the damped H^-1 and M_U the strictly-upper mask,
    is added to GPTQ's error feedback column by column:

        W[:, j:] += W[:, j] (outer) P[j, j:]      (paper Algorithm 1)

    which greedily minimizes ``||W X_fp - What X_q||^2`` instead of the symmetric
    objective. ``alpha`` scales the correction (0 = plain GPTQ on the X_q Hessian);
    ``dxxt=None`` keeps the exact original GPTQ code path.
    """
    out_features, in_features = weight.shape
    w = weight.detach().to(torch.float32).clone()
    h = hessian.detach().to(torch.float32).clone()
    dx = None
    if dxxt is not None:
        dx = dxxt.detach().to(torch.float32).clone()
        if alpha != 1.0:
            dx = dx * alpha

    # Dead input columns: no signal in H -> weight value is irrelevant; zero them
    dead = torch.diag(h) == 0
    h[dead, dead] = 1.0
    w[:, dead] = 0.0

    if act_order:
        perm = torch.argsort(torch.diag(h), descending=True)
        w = w[:, perm]
        h = h[perm][:, perm]
        if dx is not None:
            dx = dx[perm][:, perm]
        inv_perm = torch.argsort(perm)

    damp = percdamp * torch.mean(torch.diag(h))
    h += torch.eye(in_features, device=h.device) * damp

    # Inverse via Cholesky; upper-triangular factor of H^-1 (standard GPTQ trick)
    h_inv = torch.cholesky_inverse(torch.linalg.cholesky(h))
    h_inv = torch.linalg.cholesky(h_inv, upper=True)

    p = None
    if dx is not None:
        # P = ((DX X_q^T L) ⊙ M_U) L^T with H^-1 = L L^T, i.e. L = h_inv.T (lower).
        # Strictly-upper mask: corrections flow only into not-yet-quantized columns.
        p = torch.triu(dx @ h_inv.T, diagonal=1) @ h_inv

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
            if p is not None:
                # value at quantization time (already updated by previous feedback):
                # this is the W[:, q] of the residual decomposition R = sum_q W[:,q] DX[q,:]
                w_col_fp = w_col.clone()
            # error feedback into the remaining columns of this block
            w_block[:, j:] -= err.unsqueeze(1) * h_inv_block[j, j:].unsqueeze(0)
            if p is not None:
                # asymmetric residual correction within the block (P[col, col] == 0)
                w_block[:, j:] += w_col_fp.unsqueeze(1) * p[col, col:block_end].unsqueeze(0)
            w_block[:, j] = dq_col
            err_block[:, j] = err
        w[:, block_start:block_end] = w_block
        # propagate accumulated block error into all later columns
        w[:, block_end:] -= err_block @ h_inv[block_start:block_end, block_end:]
        if p is not None:
            # lazy-batched residual correction beyond the block (Algorithm 1, last line)
            w[:, block_end:] += w[:, block_start:block_end] @ p[block_start:block_end, block_end:]

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


#: stage grouping for GPTAQ's sequential two-stream calibration. Chronos-2 linears are
#: named T5-style (encoder.block.N.layer.M.{self_attention,mlp}.*); TimesFM-2.5 names
#: its transformer stack stacked_xf.N.* (20 blocks -> 20 stages); unmatched names
#: (input/output patch embeddings, tokenizer/output heads, toy models) fall back per
#: _stage_key.
_STAGE_PATTERNS = (
    re.compile(r"^(.*?(?:block|blocks|layers|h|stacked_xf)\.\d+\.layer\.\d+)"),
    re.compile(r"^(.*?(?:block|blocks|layers|h|stacked_xf)\.\d+)"),
)


def _stage_key(name: str, mode: str) -> str:
    """Which capture stage a linear belongs to.

    'linear' = exact per-linear sequencing (2 pipeline passes per linear);
    'sublayer' groups by encoder.block.N.layer.M (default; attention qkv+o or mlp
    share one capture); 'block' groups by encoder.block.N. Unmatched names use their
    first dotted component (patch embeddings become their own stages; bare toy-model
    names sequence per-linear).
    """
    if mode == "linear":
        return name
    if mode not in ("sublayer", "block"):
        raise ValueError(f"Unknown stage_mode {mode!r} (linear|sublayer|block)")
    patterns = _STAGE_PATTERNS if mode == "sublayer" else _STAGE_PATTERNS[1:]
    for pattern in patterns:
        m = pattern.match(name)
        if m:
            return m.group(1)
    return name.split(".")[0]


@torch.no_grad()
def _linear_execution_order(
    pipeline, target_names: list[str], context_length: int, seed: int
) -> list[str]:
    """Order in which the target linears first fire during one tiny probe forward."""
    modules = dict(pipeline.model.named_modules())
    order: list[str] = []
    seen: set[str] = set()
    hooks = []

    def make_hook(name: str):
        def hook(mod, args):
            if name not in seen:
                seen.add(name)
                order.append(name)

        return hook

    for name in target_names:
        hooks.append(modules[name].register_forward_pre_hook(make_hook(name)))
    try:
        series = synthetic_calibration_series(2, min(context_length, 256), seed)
        pipeline.predict_quantiles(
            series, prediction_length=8, quantile_levels=[0.5], batch_size=2
        )
    finally:
        for hook in hooks:
            hook.remove()
    missing = [n for n in target_names if n not in seen]
    if missing:
        logger.warning(
            "GPTAQ: %d target linears not exercised by the probe forward "
            "(appended last, will collect no activations): %s",
            len(missing),
            missing[:5],
        )
        order.extend(missing)
    return order


@torch.no_grad()
def _collect_two_stream_stats(
    fp_pipeline,
    q_pipeline,
    stage_names: list[str],
    series: list[torch.Tensor],
    horizon: int,
    batch_size: int,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Paired-activation statistics for one stage of GPTAQ calibration.

    Pass 1 runs the frozen fp32 pipeline and stores each pre-hook activation X_fp per
    call; pass 2 runs the partially quantized pipeline and pairs calls one-to-one
    (both passes are deterministic and identically batched), accumulating
    H = sum x_q x_q^T and DXX^T = sum (x_fp - x_q) x_q^T.
    """
    fp_modules = dict(fp_pipeline.model.named_modules())
    q_modules = dict(q_pipeline.model.named_modules())
    predict_kwargs = dict(
        prediction_length=horizon, quantile_levels=[0.1, 0.5, 0.9], batch_size=batch_size
    )

    fp_acts: dict[str, list[torch.Tensor]] = {n: [] for n in stage_names}
    hooks = []

    def make_fp_hook(name: str):
        def hook(mod, args):
            x = args[0].detach().reshape(-1, args[0].shape[-1]).to(torch.float32)
            fp_acts[name].append(x.cpu())

        return hook

    for name in stage_names:
        hooks.append(fp_modules[name].register_forward_pre_hook(make_fp_hook(name)))
    try:
        fp_pipeline.predict_quantiles(series, **predict_kwargs)
    finally:
        for hook in hooks:
            hook.remove()

    hessians: dict[str, torch.Tensor] = {}
    crosses: dict[str, torch.Tensor] = {}
    counters = dict.fromkeys(stage_names, 0)
    for name in stage_names:
        in_f = q_modules[name].in_features
        device = q_modules[name].weight.device
        hessians[name] = torch.zeros(in_f, in_f, dtype=torch.float32, device=device)
        crosses[name] = torch.zeros(in_f, in_f, dtype=torch.float32, device=device)

    hooks = []

    def make_q_hook(name: str):
        def hook(mod, args):
            x_q = args[0].detach().reshape(-1, args[0].shape[-1]).to(torch.float32)
            i = counters[name]
            calls = fp_acts[name]
            if i >= len(calls) or calls[i].shape != x_q.shape:
                raise RuntimeError(
                    f"GPTAQ two-stream mismatch on {name!r} (call {i}): the fp and "
                    "quantized calibration passes must be deterministic and "
                    "identically batched"
                )
            x_fp = calls[i].to(x_q.device)
            calls[i] = torch.empty(0)  # free as we go
            counters[name] = i + 1
            hessians[name] += x_q.T @ x_q
            crosses[name] += (x_fp - x_q).T @ x_q

        return hook

    for name in stage_names:
        hooks.append(q_modules[name].register_forward_pre_hook(make_q_hook(name)))
    try:
        q_pipeline.predict_quantiles(series, **predict_kwargs)
    finally:
        for hook in hooks:
            hook.remove()

    for name in stage_names:
        if counters[name] != len(fp_acts[name]):
            raise RuntimeError(
                f"GPTAQ: {name!r} fired {counters[name]} times in the quantized pass "
                f"vs {len(fp_acts[name])} in the fp pass"
            )
    return hessians, crosses


def gptaq_quantize_model(
    model: nn.Module,
    *,
    pipeline=None,
    bits: int = 4,
    group_size: int = 128,
    percdamp: float = 0.01,
    act_order: bool = True,
    alpha: float = 1.0,
    stage_mode: str = "sublayer",
    skip_modules: list[str] | None = None,
    n_calibration_series: int = 128,
    calibration_context_length: int = 1024,
    calibration_seed: int = 123,
    calibration_batch_size: int = 32,
    calibration_horizon: int = 64,
) -> tuple[nn.Module, dict[str, Any]]:
    """GPTAQ / GPTQv2 (arXiv:2504.02692) over all Linear modules of the pipeline's model.

    Sequential asymmetric calibration: a frozen fp32 deep copy of the pipeline provides
    the clean activation stream X_fp while the working pipeline — quantized stage by
    stage in execution order — provides X_q. Each linear is quantized to minimize
    ``||W X_fp - What X_q||^2`` (match the quantized layer's output on quantized-input
    activations to the full-precision model's output on clean activations), which
    corrects the inter-layer error that GPTQ's symmetric objective accumulates.

    ``alpha`` scales the asymmetric correction; ``alpha=0`` degrades gracefully to
    *sequential* GPTQ (X_q Hessians, no correction) — the natural ablation baseline.
    ``stage_mode`` controls capture granularity ('sublayer' default: 38 stages = 2x38
    pipeline passes on Chronos-2; 'linear' is exact per-linear at 2 passes per linear).
    Within a stage, linears are captured before any of them is quantized (the standard
    GPTQ block-parallel approximation).
    """
    from chronosquant.quantization.transforms import _iter_linear_parents, _matches_any

    if pipeline is None:
        raise ValueError("gptaq requires the pipeline for calibration forwards")

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

    # Frozen fp32 reference for the clean activation stream (copied before any edit).
    fp_pipeline = copy.deepcopy(pipeline)
    # Both streams must be deterministic: any train-mode stochasticity (dropout) would
    # corrupt the paired statistics while passing the shape/call-count guards.
    pipeline.model.eval()
    fp_pipeline.model.eval()
    series = synthetic_calibration_series(
        n_calibration_series, calibration_context_length, calibration_seed
    )
    order = _linear_execution_order(
        fp_pipeline, [name for name, _ in targets], calibration_context_length, calibration_seed
    )
    stages: dict[str, list[str]] = {}
    for name in order:
        stages.setdefault(_stage_key(name, stage_mode), []).append(name)

    logger.info(
        "GPTAQ calibration: %d series x ctx %d over %d linears in %d stages "
        "(2 passes per stage, alpha=%.3g)",
        n_calibration_series,
        calibration_context_length,
        len(targets),
        len(stages),
        alpha,
    )

    modules = dict(model.named_modules())
    total_sq_err = 0.0
    total_weight_numel = 0
    for stage_names in stages.values():
        hessians, crosses = _collect_two_stream_stats(
            fp_pipeline,
            pipeline,
            stage_names,
            series,
            horizon=calibration_horizon,
            batch_size=calibration_batch_size,
        )
        for name in stage_names:
            module = modules[name]
            w_q = gptq_quantize_weight(
                module.weight.data,
                hessians.pop(name),
                bits=bits,
                group_size=group_size,
                percdamp=percdamp,
                act_order=act_order,
                dxxt=crosses.pop(name),
                alpha=alpha,
            )
            total_sq_err += float((w_q - module.weight.data).pow(2).sum())
            total_weight_numel += module.weight.numel()
            module.weight.data.copy_(w_q)
    del fp_pipeline

    info: dict[str, Any] = {
        "method": "gptaq",
        "weight_bits": bits,
        "activation_bits": None,
        "granularity": f"group_{group_size}",
        "simulated": True,  # dequantized grid values stored in fp32 containers
        "act_order": act_order,
        "percdamp": percdamp,
        "hessian_mode": "sequential_two_stream_asymmetric",
        "asym_alpha": alpha,
        "stage_mode": stage_mode,
        "n_stages": len(stages),
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
