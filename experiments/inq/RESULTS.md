# InQ (Innovation Quantization) — empirical test on Chronos-2

Test of the proposal in *"Innovation Quantization (InQ): Process-Matched Predictive
Coding of Transformer State in Time Series Foundation Models"* (2026-07-30), executed
2026-07-30 on this repo's evaluation harness (Chronos Benchmark II, dev tier, 8 tasks,
`amazon/chronos-2` 120M, RTX 2000 Ada).

## What was tested

The proposal's core claim: TSFM K/V attention states inherit strong temporal
redundancy from the input process along the patch-time axis, so closed-loop DPCM
("innovation") coding should beat KIVI/KVQuant-style uniform quantization by
1.7–4.5 bits at equal distortion (Thm. 4.3, predicted from ρ ∈ [0.95, 0.995]).

Implementation (`inq_kv.py`, unit-tested in `test_inq_kv.py`, 7/7 pass):
- **inq**: closed-loop lag-1 DPCM per (series, head, channel) along patch-time,
  keys quantized pre-RoPE (Prop. 3.3 de-rotated frame), min/max residual range
  with Prop. 4.2 loop margin, best-case dynamic (two-pass) predictor fit.
- **rtn_channel / rtn_token**: KIVI/KVQuant-granularity uniform baselines.
- **open_delta**: CacheGen-style open-loop delta coding (anchors + cumsum decode).
- Applied to all 12 `TimeSelfAttention` modules (K pre-RoPE + V); group attention
  (unordered series axis) untouched in all arms.

Architecture caveat stated up front: **Chronos-2 is encoder-only and has no KV
cache** — each forward re-encodes the context. Quantization is simulated inside
the encoder pass (the proposal's own §7.3 "encoder mode"). The proposal's headline
deployment motivation (cache memory) does not exist in this model family at all.

## Results vs the proposal's falsifiable predictions

| Prediction | Verdict | Evidence |
|---|---|---|
| **P1** — median lag-1 autocorr of K/V channels > 0.8 on most datasets | **FALSIFIED** | Measured (context patches, per-(series,head,channel), centered): ercot K 0.42 / V 0.18; exchange_rate K 0.61 / V 0.46; covid K 0.46 / V 0.38; hospital ≈ −0.2; m1_quarterly ≈ −0.2. Fraction of channels with ρ > 0.8: 2–17%. (`stats.csv`) |
| **P2** — InQ at 2–3 bits ≈ baselines at 4 bits | **FALSIFIED (both halves)** | (a) InQ's equivalent-bits gain over per-channel RTN is **−0.35 to +0.2 bits** (median ≈ −0.08): at equal bits InQ is typically slightly *worse* (`rd_gains.csv`). (b) The premise is also vacuous downstream: plain 2-bit RTN already retains ~100% WQL (below). |
| **P4** — closed-loop error flat in t, open-loop drifts | **CONFIRMED** | On real K tensors: closed-loop per-position error ratio last/first = 1.0; open-loop = 26× (`rd_position.csv`). Matches Prop. 4.1 exactly. But open-loop delta is so bad (rel-MSE > 1 at 2–3 bits) that the whole delta direction loses to plain RTN. |
| **P5** — channel autocorr non-decreasing with depth | **MIXED** | ercot: rises 0.06 (layer 0) → ~0.45 (mid), falls at layer 11. exchange_rate: *falls* 0.92 → ~0.6 — depth whitens the one strongly-correlated case. |
| **Prop. 3.3** — RoPE conjugation | Identity confirmed (unit test); practical value **~0.1 bits** (InQ@3b rel-MSE 0.0143 pre-RoPE vs 0.0161 post-RoPE on exchange_rate; similar elsewhere). |
| **Cor. 4.4 law** — compressibility tracks input spectral shape | **Directionally right, quantitatively empty** | Dataset ordering of ρ/SFM matches intuition (near-unit-root exchange_rate most structured, layer-0 ρ = 0.92, SFM 0.09; short seasonal monthly/quarterly least). But post-depth magnitudes leave < 0.3 bits on the table everywhere. |
| Group-attention control | As theorized: ρ ≈ 0 along the unordered series axis. The time axis *is* the special axis — there's just little redundancy left in it. |

### Why P1 fails: the patch interface destroys the redundancy the theory needs

The proposal's own Prop. 3.2 (polyphase blocking), carried through honestly, predicts
this: Chronos-2 patches p = 16 steps per token. Blocking decimates the input spectrum
16-fold — an hourly series' strong lag-1-hour correlation becomes lag-16-hours at the
token level, daily seasonality lands at a non-integer patch lag (1.5), and instance
norm + the learned embedding (which mixes 16 values, time encodings, and a mask
channel) flatten what remains. The measured redundancy (ρ ≈ 0.4 ⇒ theoretical lag-1
ceiling of −½log₂(1−ρ²) ≈ 0.13 bits) is then fully consumed by the innovation's wider
min/max range and the closed-loop margin. The proposal computed its headline numbers
from ρ ∈ [0.95, 0.995] — a regime that exists only at layer 0 on near-random-walk
data (exchange_rate: ρ = 0.92, ~1.3 bits available) and nowhere after depth.

## Downstream accuracy (dev tier, 8 tasks, geometric-mean retention vs chronos2-fp32)

| arm | WQL_rel [95% CI] | MASE_rel [95% CI] |
|---|---|---|
| kv4 rtn_channel | 0.998 [0.988, 1.009] | 0.999 [0.995, 1.001] |
| kv4 inq | 1.002 [0.990, 1.014] | 1.004 [0.998, 1.012] |
| kv3 rtn_channel | 1.009 [0.992, 1.026] | 1.017 [0.998, 1.048] |
| kv3 inq | 1.000 [0.989, 1.009] | 1.008 [1.001, 1.016] |
| kv2 rtn_channel | 1.003 [0.965, 1.054] | 1.015 [0.998, 1.038] |
| kv2 inq | 0.995 [0.960, 1.033] | 1.038 [1.005, 1.084] |

Every arm is within noise of fp32 on WQL — **including 2-bit plain RTN**. All
InQ-vs-RTN differences are inside the bootstrap CIs. (For scale: the repo's *weight*
sweeps put W4 GPTQ at ~1.01 WQL_rel — K/V activations are simply not a sensitive
surface in this model.)

## Comparison to other techniques (this experiment + repo context)

- **vs KIVI/KVQuant-style uniform (rtn_channel)**: InQ loses or ties at equal bits
  (−0.35 to +0.2 equivalent bits offline; noise downstream). The baseline it must
  beat already solves the problem at 2 bits.
- **vs CacheGen-style open-loop delta**: InQ wins decisively (26× drift eliminated) —
  the closed-loop math is the most solid part of the proposal, but it is textbook
  DPCM (Jayant & Noll 1984), and both delta variants lose to plain RTN here.
- **vs per-token quant (rtn_token)**: per-channel beats per-token ~3× in rel-MSE on
  keys, consistent with the KVQuant literature; orthogonal to InQ's claims.
- **vs this repo's weight-PTQ track**: KV-activation quantization is far easier than
  weight quantization on Chronos-2 (W2 RTN weights badly degrade; KV2 is free).

## Verdict: novel? breakthrough?

**Novelty: partially yes. Breakthrough: no — the empirical thesis is falsified on
this model family.**

- The *combination* is genuinely unoccupied: no TSFM-specific cache/state
  compression method surfaced in a fresh literature check (AQUA-KV = ICML'25
  predict-then-quantize across *layers* for LLMs, arXiv:2501.19392; nothing on the
  time axis for TSFMs). The proposal's novelty audit is honest and accurate.
- The mathematical scaffolding is correct — every proposition tested (4.1, 3.3,
  4.2 stability, Thm. 4.3 on synthetic AR(1)) verifies numerically.
- But the load-bearing empirical premise (P1) is false for Chronos-2: patching,
  instance norm, and depth whiten the patch-time axis, leaving ≤ 0.3 bits of
  exploitable redundancy — and the proposal's own "near-white ⇒ InQ degrades to a
  KIVI-class quantizer" escape clause turns out to be the *typical* case, not the
  edge case. AQUA-KV's negative result on token-axis prediction in LLMs does **not**
  invert for this TSFM; it approximately holds here too.
- Independently fatal for the motivation: the tested model family has no KV cache,
  and even simulated K/V quantization is a non-problem (2-bit ≈ fp32), so there is
  no accuracy gap for the machinery to close.

**What would resurrect it** (untested here): TimesFM-2.5-class decoder-only models
(real KV cache, 16k contexts) — though TimesFM patches 32 steps per token, so the
same decimation argument applies; original Chronos-T5 (per-timestep value tokens, no
patching — the one family whose token axis retains raw temporal redundancy, but it's
deprecated); and the layer-0-only regime on near-unit-root data. The honest reduced
claim would be: *"closed-loop beats open-loop for any state-delta coding (confirmed),
and token-axis redundancy in patched TSFMs is far smaller than input-signal intuition
suggests (new negative result)."*

## Files

- `inq_kv.py` — codecs + MHA patch; `test_inq_kv.py` — 7 unit tests (all pass)
- `collect_stats.py` → `stats.csv`, `captures/*.pt` (P1/P5, 5 datasets)
- `rate_distortion.py` → `rd_table.csv`, `rd_gains.csv`, `rd_position.csv`, `rd_conjugation.csv`
- `run_eval.py` → `results/raw/chronos2_kv{2,3,4}_{rtn_channel,inq}_dev/`
- `aggregate_results.py` → `retention.csv`, `per_task_wql_ratio.csv`
