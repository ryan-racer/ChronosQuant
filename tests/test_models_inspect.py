"""Tests for model characterization (exact hand-computed values)."""

import pytest
import torch

from chronosquant.models.inspect import model_summary


def test_linear_fp32_exact_accounting():
    module = torch.nn.Linear(4, 3)  # 4*3 weights + 3 bias = 15 params
    summary = model_summary(module)
    assert summary["num_params"] == 15
    assert summary["param_bytes"] == 15 * 4
    assert summary["num_buffers"] == 0
    assert summary["bits_per_param"] == 32.0
    assert summary["param_dtypes"] == {"float32": 15}
    assert summary["module_types"] == {"Linear": 1}


def test_buffers_count_toward_effective_bits():
    """Quantization scales/zero-points live in buffers and must count toward BPW."""

    class WithBuffer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 3)
            self.register_buffer("scales", torch.ones(15))

    summary = model_summary(WithBuffer())
    assert summary["num_params"] == 15
    assert summary["num_buffers"] == 15
    assert summary["total_bytes"] == 15 * 4 + 15 * 4
    assert summary["bits_per_param"] == 64.0  # params + equal-size buffers


def test_half_precision_bpw():
    module = torch.nn.Linear(4, 3).to(torch.bfloat16)
    summary = model_summary(module)
    assert summary["bits_per_param"] == 16.0
    assert summary["param_dtypes"] == {"bfloat16": 15}


def test_mixed_dtype_breakdown():
    class Mixed(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = torch.nn.Linear(4, 3)  # fp32, 15 params
            self.b = torch.nn.Linear(4, 3).to(torch.bfloat16)  # bf16, 15 params

    summary = model_summary(Mixed())
    assert summary["num_params"] == 30
    assert summary["param_dtypes"] == {"float32": 15, "bfloat16": 15}
    assert summary["bits_per_param"] == pytest.approx(24.0)  # (15*32 + 15*16) / 30
