"""Innovation Quantization (InQ) codecs for Chronos-2 time-attention K/V activations.

Empirical test of the "InQ" proposal (innovation-quantization-tsfm.md): predictive
(DPCM) coding of transformer K/V state along the patch-time axis, versus standard
KIVI/KVQuant-style uniform quantization.

Architecture note: Chronos-2 is encoder-only — it has NO KV cache at inference
(each forward re-encodes the full context). "KV quantization" here therefore means
simulated low-bit representation of the K/V tensors inside the encoder pass (the
proposal's §7.3 "encoder mode"), exactly analogous to how this repo simulates
sub-8-bit weight quantization. This tests the statistical theory end to end; it
does not by itself save memory in this architecture.

All codecs operate on attention states shaped [batch, n_heads, seq, d_head] and
return dequantized float tensors. Keys are quantized PRE-RoPE (the de-rotated
frame of the proposal's Prop. 3.3); re-rotation is an isometry so errors carry
through unchanged.

Codecs:
- rtn_channel : uniform min/max per (batch, head, channel) over the seq axis —
                the KIVI/KVQuant key-cache granularity.
- rtn_token   : uniform min/max per (batch, head, position) over channels —
                KIVI's value-cache granularity.
- open_delta  : anchor + open-loop first-difference coding (CacheGen-style);
                decoded by cumulative sum, so errors accumulate within a block.
- inq         : closed-loop DPCM on lag-1 innovations (the proposal's core).
                The predictor runs on reconstructed values, so the per-position
                error equals that position's scalar quantization error
                (proposal Prop. 4.1) — no drift.

The lag-1 coefficient for `inq` is either fit per (batch, head, channel) from
the tensor itself (dynamic; a legitimate two-pass encode in encoder mode) or a
static per-(module, head, channel) table calibrated offline.
"""

from __future__ import annotations

import torch
from einops import rearrange

from chronos.chronos2.layers import (
    AttentionOutput,
    Chronos2RotaryEmbedding,
    TimeSelfAttention,
)

EPS = 1e-12


def uniform_qdq(x: torch.Tensor, bits: int, dim: int) -> torch.Tensor:
    """Simulated asymmetric uniform quantization with min/max range along `dim`."""
    n_levels = 2**bits - 1
    lo = x.amin(dim=dim, keepdim=True)
    hi = x.amax(dim=dim, keepdim=True)
    scale = (hi - lo).clamp_min(EPS) / n_levels
    q = torch.round((x - lo) / scale).clamp_(0, n_levels)
    return lo + q * scale


def kv_rtn_channel(x: torch.Tensor, bits: int) -> torch.Tensor:
    """KIVI/KVQuant-style per-channel uniform quantization (range over seq axis)."""
    return uniform_qdq(x, bits, dim=2)


def kv_rtn_token(x: torch.Tensor, bits: int) -> torch.Tensor:
    """KIVI-style per-token uniform quantization (range over channel axis)."""
    return uniform_qdq(x, bits, dim=3)


def lag1_coefficient(x: torch.Tensor) -> torch.Tensor:
    """Least-squares lag-1 predictor coefficient per (batch, head, channel).

    Uncentered fit (the DPCM predictor has no intercept; a mean level is handled
    by a -> 1). Clamped to [-1, 1], the stability region of proposal Prop. 4.2.
    """
    x0, x1 = x[:, :, :-1], x[:, :, 1:]
    num = (x0 * x1).sum(dim=2)
    den = (x0 * x0).sum(dim=2).clamp_min(EPS)
    return (num / den).clamp_(-1.0, 1.0)  # [b, h, d]


def kv_open_delta(x: torch.Tensor, bits: int, anchor_every: int | None = 64) -> torch.Tensor:
    """CacheGen-style open-loop delta coding: 8-bit per-token anchors every
    `anchor_every` positions, first differences quantized per (b, h, channel)
    within each block, decoded by cumulative sum (errors accumulate in-block)."""
    b, h, s, d = x.shape
    step = anchor_every or s
    out = torch.empty_like(x)
    for start in range(0, s, step):
        blk = x[:, :, start : start + step]
        anchor = uniform_qdq(blk[:, :, :1], 8, dim=3)
        if blk.shape[2] == 1:
            out[:, :, start : start + 1] = anchor
            continue
        deltas = blk[:, :, 1:] - blk[:, :, :-1]
        dq = uniform_qdq(deltas, bits, dim=2)
        rec = torch.cat([anchor, anchor + dq.cumsum(dim=2)], dim=2)
        out[:, :, start : start + blk.shape[2]] = rec
    return out


def kv_inq(
    x: torch.Tensor,
    bits: int,
    a: torch.Tensor | None = None,
    anchor_every: int | None = None,
    loop_margin: float = 1.1,
) -> torch.Tensor:
    """Closed-loop DPCM innovation quantization along the seq (patch-time) axis.

    - anchor at block start: 8-bit per-token.
    - residual quantizer range per (b, h, channel): min/max of the OPEN-loop
      innovations, widened by `loop_margin` to absorb the closed-loop residual
      inflation of proposal Prop. 4.2 (~9% at these operating points).
    - `a`: predictor coefficients, [b, h, d] (dynamic) or [h, d] (static
      calibrated). None -> fit dynamically from `x` (two-pass encoder mode).
    """
    b, h, s, d = x.shape
    if a is None:
        a = lag1_coefficient(x)
    if a.dim() == 2:
        a = a.unsqueeze(0).expand(b, -1, -1)
    a = a.to(x.dtype)
    n_levels = 2**bits - 1
    step = anchor_every or s
    out = torch.empty_like(x)
    for start in range(0, s, step):
        blk = x[:, :, start : start + step]
        T = blk.shape[2]
        anchor = uniform_qdq(blk[:, :, :1], 8, dim=3).squeeze(2)  # [b,h,d]
        out[:, :, start] = anchor
        if T == 1:
            continue
        e = blk[:, :, 1:] - a.unsqueeze(2) * blk[:, :, :-1]  # open-loop innovations
        lo = e.amin(dim=2)
        hi = e.amax(dim=2)
        mid = (hi + lo) / 2
        half = (hi - lo).clamp_min(EPS) * (loop_margin / 2)
        lo = mid - half
        scale = (2 * half) / n_levels
        prev = anchor
        for t in range(1, T):
            pred = a * prev
            r = blk[:, :, t] - pred
            q = torch.round((r - lo) / scale).clamp_(0, n_levels)
            prev = pred + lo + q * scale
            out[:, :, start + t] = prev
    return out


def make_codec(method: str, bits: int, a: torch.Tensor | None = None, anchor_every: int | None = None):
    """Build a codec fn(x) -> x_hat for states [b, h, s, d]."""
    if method == "rtn_channel":
        return lambda x: kv_rtn_channel(x, bits)
    if method == "rtn_token":
        return lambda x: kv_rtn_token(x, bits)
    if method == "open_delta":
        return lambda x: kv_open_delta(x, bits, anchor_every=anchor_every or 64)
    if method == "inq":
        return lambda x: kv_inq(x, bits, a=a, anchor_every=anchor_every)
    if method == "none":
        return lambda x: x
    raise ValueError(f"Unknown KV codec method {method!r}")


def _quantized_mha_forward(mha, codec_k, codec_v):
    """Replacement for `MHA.forward` (self-attention path) with K/V quantization
    inserted after the k/v projections, BEFORE RoPE (de-rotated frame)."""

    def forward(
        hidden_states: torch.Tensor,
        mask: torch.Tensor,
        encoder_states: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        output_attentions: bool = False,
    ) -> AttentionOutput:
        assert encoder_states is None, "KV codec is only patched onto self-attention"

        def shape(states):
            return rearrange(states, "b s (h d) -> b h s d", h=mha.n_heads, d=mha.kv_proj_dim)

        def unshape(states):
            return rearrange(states, "b h s d -> b s (h d)")

        query_states = shape(mha.q(hidden_states))
        key_states = shape(mha.k(hidden_states))
        value_states = shape(mha.v(hidden_states))

        key_states = codec_k(key_states)
        value_states = codec_v(value_states)

        if mha.use_rope:
            assert position_ids is not None
            cos, sin = mha.rope_embed(value_states, position_ids)
            query_states, key_states = Chronos2RotaryEmbedding.apply_rotary_pos_emb(
                query_states, key_states, cos, sin
            )

        attn_implementation = mha.config._attn_implementation
        if output_attentions:
            attn_implementation = "eager"
        if attn_implementation == "sdpa":
            attn_output, attn_weights = mha._sdpa_attention(query_states, key_states, value_states, mask)
        else:
            attn_output, attn_weights = mha._eager_attention(query_states, key_states, value_states, mask)

        attn_output = mha.o(unshape(attn_output))
        return AttentionOutput(
            hidden_states=attn_output,
            attn_weights=attn_weights if output_attentions else None,
        )

    return forward


def patch_time_attention(
    model,
    method: str,
    bits: int,
    static_a: dict[str, dict[str, torch.Tensor]] | None = None,
    anchor_every: int | None = None,
) -> list[str]:
    """Patch every TimeSelfAttention MHA in `model` to quantize K/V with `method`.

    GroupSelfAttention (whose attention axis is the unordered series-in-batch
    axis) is left untouched in all arms so comparisons are apples-to-apples on
    the patch-time axis where the InQ theory applies.

    Returns the list of patched module names.
    """
    patched = []
    for name, module in model.named_modules():
        if isinstance(module, TimeSelfAttention):
            mha = module.self_attention
            a_k = a_v = None
            if static_a is not None:
                a_k = static_a[name]["k"]
                a_v = static_a[name]["v"]
            codec_k = make_codec(method, bits, a=a_k, anchor_every=anchor_every)
            codec_v = make_codec(method, bits, a=a_v, anchor_every=anchor_every)
            mha.forward = _quantized_mha_forward(mha, codec_k, codec_v)
            patched.append(name)
    if not patched:
        raise RuntimeError("No TimeSelfAttention modules found to patch")
    return patched


def make_pipeline_transform(method: str, bits: int, anchor_every: int | None = None):
    """`model_transform` hook for Chronos2Predictor: patches time-attention K/V."""

    def transform(pipeline):
        patched = patch_time_attention(pipeline.model, method, bits, anchor_every=anchor_every)
        print(f"[inq_kv] patched {len(patched)} TimeSelfAttention modules: method={method} bits={bits}")
        return pipeline

    return transform
