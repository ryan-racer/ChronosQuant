"""Quantization transforms applied to a Chronos2Pipeline via the predictor hook.

Two families:

- ``rtn`` — our reference round-to-nearest implementation (symmetric weight-only,
  per-channel or per-tensor). Dependency-free, identical behaviour on CPU/GPU, and
  exact storage accounting (int8 container + fp32 scales) → the transparent baseline
  method every quantization paper needs. ``bits=8`` stores real int8 weights;
  ``bits<8`` requires ``simulate=true`` (fake-quantize in fp32: accuracy studies only,
  no storage claim — proper sub-8-bit packing comes from library methods).
- ``torchao_*`` — torchao's optimized paths (int8 weight-only for now; int4/FP8 later).

Every transform returns quantization metadata (method, bits, modules quantized/skipped,
effective bits per weight incl. scale overhead) that the predictor exposes in its model
card, so paper tables get honest BPW numbers per variant.
"""

import fnmatch
from collections.abc import Callable
from typing import Any

import torch
from torch import nn

TransformResult = tuple[nn.Module, dict[str, Any]]


class RTNQuantizedLinear(nn.Module):
    """Linear layer with symmetric round-to-nearest int8 weights, dequantized at forward.

    W8A16/W8A32 semantics: weights stored int8 (+ scales), activations and compute stay
    in the input dtype. Memory shrinks ~4x; speed is NOT claimed (dequant + dense GEMM),
    which the efficiency protocol measures rather than assumes.
    """

    def __init__(self, linear: nn.Linear, granularity: str = "per_channel"):
        super().__init__()
        weight = linear.weight.detach()
        if granularity == "per_channel":
            max_abs = weight.abs().amax(dim=1, keepdim=True)  # [out, 1]
        elif granularity == "per_tensor":
            max_abs = weight.abs().amax().reshape(1, 1)
        else:
            raise ValueError(f"Unknown granularity {granularity!r}")
        scale = (max_abs / 127.0).clamp(min=torch.finfo(torch.float32).tiny)
        weight_q = torch.round(weight / scale).clamp(-127, 127).to(torch.int8)

        self.register_buffer("weight_q", weight_q)
        self.register_buffer("scale", scale.to(torch.float32))
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.detach().clone())
        else:
            self.bias = None
        self.in_features = linear.in_features
        self.out_features = linear.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = (self.weight_q.to(torch.float32) * self.scale).to(x.dtype)
        return nn.functional.linear(x, weight, self.bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bits=8"


def _fake_quantize_(linear: nn.Linear, bits: int, granularity: str) -> None:
    """In-place quantize-dequantize of weights (simulated low-bit; storage unchanged)."""
    weight = linear.weight.detach()
    q_max = 2 ** (bits - 1) - 1
    if granularity == "per_channel":
        max_abs = weight.abs().amax(dim=1, keepdim=True)
    elif granularity == "per_tensor":
        max_abs = weight.abs().amax().reshape(1, 1)
    else:
        raise ValueError(f"Unknown granularity {granularity!r}")
    scale = (max_abs / q_max).clamp(min=torch.finfo(torch.float32).tiny)
    weight_dq = torch.round(weight / scale).clamp(-q_max, q_max) * scale
    linear.weight.data.copy_(weight_dq)


def _iter_linear_parents(model: nn.Module):
    for parent_name, parent in model.named_modules():
        for child_name, child in parent.named_children():
            if isinstance(child, nn.Linear):
                full_name = f"{parent_name}.{child_name}" if parent_name else child_name
                yield parent, child_name, full_name, child


def _matches_any(name: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(name, p) or p in name for p in patterns)


def rtn_quantize_model(
    model: nn.Module,
    bits: int = 8,
    granularity: str = "per_channel",
    skip_modules: list[str] | None = None,
    simulate: bool = False,
) -> TransformResult:
    """Quantize all nn.Linear weights with symmetric RTN; returns (model, metadata).

    ``skip_modules``: list of glob patterns or substrings matched against the module's
    dotted path; matching Linears stay in full precision (used by sensitivity studies).
    """
    skip_modules = skip_modules or []
    if bits == 8 and not simulate:
        real = True
    elif 2 <= bits <= 8 and simulate:
        real = False
    else:
        raise ValueError(
            f"rtn supports bits=8 (real int8 storage) or bits in [2, 8] with simulate=true "
            f"(got bits={bits}, simulate={simulate})"
        )

    quantized, skipped = [], []
    orig_weight_numel = 0
    packed_bytes = 0
    scale_bytes = 0
    for parent, child_name, full_name, child in list(_iter_linear_parents(model)):
        if _matches_any(full_name, skip_modules):
            skipped.append(full_name)
            continue
        orig_weight_numel += child.weight.numel()
        if real:
            new_module = RTNQuantizedLinear(child, granularity=granularity)
            packed_bytes += new_module.weight_q.numel()  # int8 = 1 byte
            scale_bytes += new_module.scale.numel() * 4
            setattr(parent, child_name, new_module)
        else:
            _fake_quantize_(child, bits=bits, granularity=granularity)
        quantized.append(full_name)

    if not quantized:
        raise ValueError("No Linear modules were quantized (check skip_modules patterns)")

    info: dict[str, Any] = {
        "method": "rtn",
        "weight_bits": bits,
        "activation_bits": None,  # weight-only
        "granularity": granularity,
        "simulated": not real,
        "n_modules_quantized": len(quantized),
        "n_modules_skipped": len(skipped),
        "quantized_weight_numel": orig_weight_numel,
    }
    if real:
        info["effective_bits_per_weight"] = round(
            (packed_bytes + scale_bytes) * 8 / orig_weight_numel, 4
        )
    else:
        # storage is unchanged in simulate mode; record the *logical* bits only
        info["effective_bits_per_weight"] = None
    return model, info


def torchao_int8wo_quantize_model(
    model: nn.Module,
    skip_modules: list[str] | None = None,
) -> TransformResult:
    """torchao Int8WeightOnly (per-channel symmetric, optimized kernels)."""
    from torchao.quantization import Int8WeightOnlyConfig, quantize_

    skip_modules = skip_modules or []
    linear_names = [name for _, _, name, _ in _iter_linear_parents(model)]
    to_quantize = [n for n in linear_names if not _matches_any(n, skip_modules)]
    orig_weight_numel = sum(
        m.weight.numel() for _, _, n, m in _iter_linear_parents(model) if n in set(to_quantize)
    )

    def filter_fn(module: nn.Module, name: str) -> bool:
        return isinstance(module, nn.Linear) and not _matches_any(name, skip_modules)

    quantize_(model, Int8WeightOnlyConfig(), filter_fn=filter_fn)
    info = {
        "method": "torchao_int8wo",
        "weight_bits": 8,
        "activation_bits": None,
        "granularity": "per_channel",
        "simulated": False,
        "n_modules_quantized": len(to_quantize),
        "n_modules_skipped": len(linear_names) - len(to_quantize),
        "quantized_weight_numel": orig_weight_numel,
        # int8 data + per-channel fp scales, same accounting as rtn
        "effective_bits_per_weight": None,  # computed from serialized size if needed
    }
    return model, info


#: method name -> (callable(model, **params) -> (model, info))
QUANTIZATION_METHODS: dict[str, Callable[..., TransformResult]] = {
    "rtn": rtn_quantize_model,
    "torchao_int8wo": torchao_int8wo_quantize_model,
}


def apply_quantization(model: nn.Module, config: dict[str, Any]) -> TransformResult:
    """Apply the quantization described by a config dict: ``{"method": ..., **params}``."""
    config = dict(config)
    method = config.pop("method", None)
    if method not in QUANTIZATION_METHODS:
        raise ValueError(
            f"Unknown quantization method {method!r}. Available: {sorted(QUANTIZATION_METHODS)}"
        )
    return QUANTIZATION_METHODS[method](model, **config)
