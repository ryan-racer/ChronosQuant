"""Unit tests for the InQ KV codecs: the proposal's Prop. 4.1 (no drift),
Prop. 3.3 (RoPE conjugation identity), Thm. 4.3 (prediction gain), and
equivalence of the patched MHA forward at high precision."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))

from inq_kv import (  # noqa: E402
    _quantized_mha_forward,
    kv_inq,
    kv_open_delta,
    kv_rtn_channel,
    lag1_coefficient,
    make_codec,
)


def ar1(rho: float, shape: tuple[int, int, int, int], seed: int = 0) -> torch.Tensor:
    """AR(1) process along dim=2 with unit marginal variance."""
    g = torch.Generator().manual_seed(seed)
    b, h, s, d = shape
    noise = torch.randn(b, h, s, d, generator=g) * (1 - rho**2) ** 0.5
    x = torch.empty(shape)
    x[:, :, 0] = torch.randn(b, h, d, generator=g)
    for t in range(1, s):
        x[:, :, t] = rho * x[:, :, t - 1] + noise[:, :, t]
    return x


def test_closed_loop_error_is_flat_open_loop_drifts():
    """Prop. 4.1: closed-loop per-position error is uniform in t; open-loop
    cumulative decoding accumulates error within a block."""
    x = ar1(0.99, (2, 2, 512, 8), seed=1)
    err_inq = ((x - kv_inq(x, bits=3)) ** 2).mean(dim=(0, 1, 3))
    err_open = ((x - kv_open_delta(x, bits=3, anchor_every=None)) ** 2).mean(dim=(0, 1, 3))
    early = slice(1, 101)
    late = slice(-100, None)
    # closed loop: late error comparable to early error
    assert err_inq[late].mean() < 3 * err_inq[early].mean()
    # open loop: late error much larger than early error
    assert err_open[late].mean() > 5 * err_open[early].mean()
    # and closed loop beats open loop overall at equal bits
    assert err_inq.mean() < err_open.mean()


def test_prediction_gain_matches_theory():
    """Thm. 4.3: for AR(1) with rho=0.95, innovation coding at equal bits should
    shrink MSE by roughly (1 - rho^2) ~ 0.1 vs direct per-channel quantization."""
    x = ar1(0.95, (4, 2, 1024, 8), seed=2)
    mse_rtn = ((x - kv_rtn_channel(x, bits=4)) ** 2).mean()
    mse_inq = ((x - kv_inq(x, bits=4)) ** 2).mean()
    gain = (mse_rtn / mse_inq).item()
    assert gain > 3.0, f"expected large prediction gain, got {gain:.2f}"


def test_lag1_coefficient_recovers_rho():
    x = ar1(0.9, (2, 2, 2048, 4), seed=3)
    a = lag1_coefficient(x)
    assert (a - 0.9).abs().median() < 0.05


def test_rope_conjugation_identity():
    """Prop. 3.3 rests on R_t = R_i o R_{t-i}: rotating by position (t-i) then
    advancing by i steps equals rotating by position t."""
    from chronos.chronos2.config import Chronos2CoreConfig
    from chronos.chronos2.layers import Chronos2RotaryEmbedding

    config = Chronos2CoreConfig(d_model=32, d_kv=16, num_heads=2, attn_implementation="eager")
    rope = Chronos2RotaryEmbedding(config)
    x = torch.randn(1, 2, 1, 16)  # [b, h, seq=1, d]

    def rotate(v, pos):
        pos_ids = torch.tensor([[pos]])
        cos, sin = rope(v, pos_ids)
        _, out = Chronos2RotaryEmbedding.apply_rotary_pos_emb(v, v, cos, sin)
        return out

    t, i = 17, 5
    direct = rotate(x, t)
    composed = rotate(rotate(x, t - i), i)
    torch.testing.assert_close(direct, composed, atol=1e-5, rtol=1e-5)


def test_patched_mha_matches_original_at_high_bits():
    """The patched forward with an 8-bit codec should be numerically close to the
    original MHA forward; with method='none' it should match exactly."""
    from chronos.chronos2.config import Chronos2CoreConfig
    from chronos.chronos2.layers import MHA

    torch.manual_seed(0)
    config = Chronos2CoreConfig(
        d_model=32, d_kv=8, num_heads=4, dropout_rate=0.0, attn_implementation="eager"
    )
    mha = MHA(config, use_rope=True).eval()
    b, s = 2, 24
    hidden = torch.randn(b, s, 32)
    mask = torch.zeros(b, 1, s, s)
    pos = torch.arange(s).unsqueeze(0)

    with torch.no_grad():
        ref = mha(hidden, mask=mask, position_ids=pos).hidden_states
        fwd_none = _quantized_mha_forward(mha, make_codec("none", 0), make_codec("none", 0))
        out_none = fwd_none(hidden, mask=mask, position_ids=pos).hidden_states
        fwd_q8 = _quantized_mha_forward(
            mha, make_codec("rtn_channel", 8), make_codec("rtn_channel", 8)
        )
        out_q8 = fwd_q8(hidden, mask=mask, position_ids=pos).hidden_states

    torch.testing.assert_close(out_none, ref)
    assert (out_q8 - ref).abs().max() < 0.05


def test_inq_static_coefficients_accepted():
    x = ar1(0.9, (2, 3, 64, 4), seed=4)
    a_static = torch.full((3, 4), 0.9)
    out = kv_inq(x, bits=4, a=a_static)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_anchored_blocks():
    x = ar1(0.95, (1, 2, 130, 4), seed=5)  # not a multiple of the block size
    for fn in (lambda: kv_inq(x, bits=3, anchor_every=32),
               lambda: kv_open_delta(x, bits=3, anchor_every=32)):
        out = fn()
        assert out.shape == x.shape
        assert torch.isfinite(out).all()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
