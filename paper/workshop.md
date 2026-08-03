# How Low Can Chronos-2 Go? Post-Training Quantization of a Time-Series Foundation Model

*Anonymous submission — under review.*

## Abstract

Time-series foundation models (TSFMs) such as Chronos-2 deliver strong zero-shot
probabilistic forecasts, but their deployment cost is rarely studied and, unlike large
language models, *nothing* is known about how they respond to post-training
quantization (PTQ). We present the first systematic PTQ study of TSFMs: 39
weight-quantization configurations spanning seven method families (round-to-nearest,
torchao, bitsandbytes INT8/NF4, HQQ, GPTQ, GPTAQ, and activation-aware scale folding)
and two-to-eight bits, evaluated zero-shot on the 27-dataset Chronos benchmark across
four models and three architectures (encoder-only Chronos-2, encoder–decoder
Chronos-Bolt, decoder-only TimesFM-2.5). Beyond point (MASE) and probabilistic (weighted
quantile loss, WQL) accuracy, we introduce two calibration-aware diagnostics — mean
absolute coverage error and a quantile-crossing rate — because a probabilistic
forecaster can degrade in ways aggregate accuracy cannot see. Five findings emerge.
(i) 8-bit quantization is free for every method. (ii) Hessian-aware GPTQ retains
full-precision accuracy down to **3 bits** (WQL retention 1.05), and
asymmetric-calibration GPTAQ extends the parity frontier (1.016); every PTQ method
collapses at 2 bits. (iii) The *method* matters far more than the *bit-width*: at 4
bits, retention spans 0.996–1.66. (iv) Interval calibration and quantile monotonicity
degrade *before* accuracy does — and we trace this damage *entirely* to the 3-linear
quantile head: protecting 3.2% of the weights eliminates it at every bit-width, whereas
accuracy damage is distributed across the encoder. (v) Two cheap repairs follow:
post-hoc quantile sorting (free, provably never worse) eliminates crossing, and
split-conformal recalibration restores nominal coverage for *every* method at *every*
bit-width — even fp32 improves. We release the framework, which reproduces published
per-task benchmark numbers to <10⁻⁶, and all per-series predictions.

## 1. Introduction

Time-series foundation models (TSFMs) — Chronos [Ansari et al., 2024], Chronos-2 [Ansari
et al., 2025], TimesFM, Moirai — forecast unseen series zero-shot and are increasingly
deployed on CPU-only servers, edge devices, and high-throughput batch pipelines where
full-precision inference is costly. Quantization is the standard efficiency lever for
large language models (LLMs), but its lessons do not obviously transfer to TSFMs, which
are small (Chronos-2 has 120M parameters, a regime where quantization bites harder
[Li et al., 2024]), encoder-only and non-autoregressive (no KV cache, so LLM
decode-centric methods do not apply), and — critically — emit *probabilistic* forecasts
whose calibration and quantile monotonicity matter, not just point accuracy.

Despite an extensive LLM-quantization literature, **no published work quantizes the
weights of any Chronos-family or comparable TSFM** (in this literature "quantization"
usually denotes Chronos-1's input value-*binning*, a distinct notion). We close that gap
with a controlled study across method families, bit-widths, and models, and argue that
evaluating a *quantized probabilistic* forecaster demands metrics beyond those on TSFM
leaderboards.

**Contributions.** (1) The first systematic PTQ study of TSFMs: 39 configurations × 7
method families × 5 bit-widths × 4 models spanning 3 architectures (Chronos-2,
Chronos-Bolt small/base, TimesFM-2.5), zero-shot on 27 datasets, with bootstrap
confidence intervals on every aggregate. (2) A
calibration-aware evaluation protocol for compressed probabilistic forecasters, adding
interval-coverage error and a quantile-crossing rate, which reveal failure modes
invisible to WQL/MASE. (3) A *mechanism*: module-family sensitivity and weight-space SNR
diagnostics showing calibration damage is gated exclusively by the 3-linear quantile
head, while accuracy damage is distributed. (4) Two mitigations with a deployment
recipe: an fp32 head (prevention, 3.2% of weights) and post-hoc sorting +
split-conformal recalibration (cure, at any bit-width). (5) An open, reproducible
framework that reproduces published benchmark numbers to floating-point precision and
releases all per-series predictions, so any further metric is computable without
re-running models.

## 2. Setup

**Models.** Chronos-2 [Ansari et al., 2025] (`amazon/chronos-2`) is a 119.5M-parameter,
T5-derived *encoder-only* transformer (12 layers, $d_\text{model}{=}768$, RoPE,
alternating "time" and "group" attention). It ingests patches of a per-instance-
normalized series through a residual embedding (no input tokenization) and emits, in one
non-autoregressive forward pass, direct multi-step forecasts at 21 quantile levels. It
ships in fp32 (478 MB). Quantization targets all 126 `nn.Linear` weights — including
the patch embeddings and the 3-linear quantile head (3.2% of weights), except in arms
that explicitly protect the head; LayerNorm stays in fp32, and bf16 (not fp16, which
overflows in T5-lineage models) is the deployment baseline. As cross-model arms we
also quantize Chronos-Bolt small/base [Ansari et al., 2024] (48M/205M),
*encoder–decoder* T5 forecasters, through the same harness (fp32 reproduces their
published per-task numbers to 0.34%/0.72%; their quantile head is protected by
default), and TimesFM-2.5 (`google/timesfm-2.5-200m-pytorch`, 200M), a *decoder-only*
autoregressive forecaster with a continuous quantile head, via a torch-only adapter
(deterministic single decode step at horizons ≤56; head and tokenizer protected as in
our "+head" arms). TimesFM-2.5 has no published per-task reference on this suite (we
sanity-anchor the adapter by its dev-subset skill exceeding Chronos-2's); its built-in
`fix_quantile_crossing` auto-sort is disabled in all arms so QCR is measured honestly.

**Methods.** We evaluate weight-only PTQ across seven families runnable on commodity
hardware: **RTN** (symmetric round-to-nearest, per-channel/per-tensor, plus a simulated
sub-8-bit sweep); **torchao** INT8 weight-only and INT8 dynamic-activation (W8A8);
**bitsandbytes** LLM.int8() [Dettmers et al., 2022] and NF4 [Dettmers et al., 2023];
**HQQ** [Badri & Shao, 2023], a calibration-free half-quadratic solver; **GPTQ**
[Frantar et al., 2023], a Hessian-aware error-compensating quantizer implemented
directly on the model's linear layers with one-shot calibration; **GPTAQ** [Li et al.,
2025], its asymmetric-calibration extension that corrects each layer toward the
*full-precision* network's activations; and **AWQ-style** activation-aware scale folding
[Lin et al., 2024]. Methods are exercised at W8, W4 (group sizes 64/128), W3, and W2 —
39 full-benchmark configurations in total.

**Protocol & metrics.** We evaluate zero-shot on the 27-dataset Chronos benchmark
[Ansari et al., 2024] (190,674 forecasts) using the `fev` library — the harness the
Chronos-2 authors use — so our numbers are directly comparable with the public
leaderboard. Point accuracy is MASE and probabilistic accuracy is weighted quantile loss
(WQL, 9 quantile levels); our full-precision Chronos-2 attains 0.425 WQL skill and an
88% win rate over seasonal-naive, and as a reproduction gate our seasonal-naive matches
the published per-task MASE/WQL to <10⁻⁶.

For quantized models the key quantity is **retention**: the geometric-mean ratio
$\text{WQL}(\text{quant})/\text{WQL}(\text{fp32})$ over the 27 tasks (1.00 = parity;
1.05 = 5% worse), with a 1000-resample bootstrap 95% CI. Because a probabilistic
forecaster can lose quality without moving WQL, we add two diagnostics. **MACE** (mean
absolute coverage error) averages $|\text{empirical}-\text{nominal}|$ over the central
80/60/40/20% prediction intervals. **QCR** (quantile-crossing rate) is the fraction of
forecast points whose predicted quantiles are non-monotone. Two further protocols: a
*module-family sensitivity* sweep (dev tier: 8-task subset) that partitions the 126
linears into six disjoint families, quantized or protected one family at a time; and an
offline *repair* arm applying monotone rearrangement (quantile sorting) [Chernozhukov et
al., 2010; Fakoor et al., 2023] and 2-fold series-split conformal recalibration [Romano
et al., 2019] to the persisted predictions of every full-tier arm.

## 3. Results

Table 1 summarizes representative configurations; Figure 1 plots the accuracy–bit
curve and calibration. The complete 39-variant matrix is released.

**Table 1.** Representative variants, full 27-task benchmark. Retention = geometric-mean
WQL(quant)/WQL(fp32), [95% bootstrap CI]. *cov80* = empirical coverage of the nominal-80%
interval (fp32: 0.750). *QCR* = quantile-crossing rate (fp32: 0.039). BPW = effective
bits/weight incl. scales; †accuracy-exact simulation (dequantized grid in fp32) — BPW is
the *achievable* footprint. Size is on-disk MB for storage-real methods. "+head" = all
linears quantized except the 3-linear quantile head (kept fp32).

| Variant | Bits | BPW | Size MB | WQL ret. [CI] | MASE ret. | cov80 | QCR |
|---|---|---|---|---|---|---|---|
| fp32 (reference) | 32 | 32.0 | 478 | 1.000 | 1.000 | 0.750 | 0.039 |
| RTN W8 | 8 | 8.03 | 120 | 0.999 [.997, 1.00] | 1.001 | 0.749 | 0.149 |
| GPTQ W8 | 8 | 8.13† | — | 1.000 [.990, 1.01] | 1.003 | 0.749 | 0.064 |
| HQQ W8 | 8 | 9.00 | 120 | 1.002 [1.00, 1.00] | 1.000 | 0.746 | 0.086 |
| **RTN W8+head** | 8 | 8.03 | 131 | **0.999 [.998, 1.00]** | 1.000 | 0.751 | 0.037 |
| **GPTQ W4** | 4 | 4.13† | — | **1.002 [.985, 1.02]** | 1.011 | 0.756 | 0.254 |
| GPTAQ W4 | 4 | 4.25† | — | 1.000 [.987, 1.01] | 1.008 | 0.745 | 0.226 |
| HQQ W4 | 4 | 5.00 | 60 | 1.060 [1.02, 1.12] | 1.027 | 0.630 | 0.420 |
| NF4 W4 | 4 | 4.13 | 60 | 1.065 [1.00, 1.15] | 1.026 | 0.686 | 0.396 |
| HQQ W4+head | 4 | 5.00 | 73 | 1.020 [.985, 1.06] | 1.016 | 0.751 | 0.021 |
| NF4 W4+head | 4 | 4.13 | 73 | 1.045 [.984, 1.13] | 1.025 | 0.761 | 0.022 |
| AWQ-fold W4 | 4 | 4.03† | — | 1.639 [1.38, 2.01] | 1.409 | 0.481 | 0.599 |
| GPTQ W3 | 3 | 3.25† | — | 1.047 [.988, 1.14] | 1.059 | 0.791 | 0.366 |
| **GPTAQ W3** | 3 | 3.25† | — | **1.016 [.988, 1.05]** | 1.034 | 0.770 | 0.379 |
| HQQ W3 | 3 | 4.20 | 48 | 1.229 [1.09, 1.48] | 1.143 | 0.666 | 0.589 |
| GPTAQ W2 | 2 | 2.25† | — | 1.937 [1.55, 2.48] | 1.816 | 0.642 | 0.833 |
| RTN W2 (sim) | 2 | 2.0† | — | 4.082 [3.08, 5.57] | 3.108 | 0.060 | 0.523 |

**F1 — 8-bit is free, for every method.** All six W8 configurations retain WQL within
[0.999, 1.013] and MASE within [1.000, 1.011] of fp32 (Table 1; Fig. 1a), at 4×
compression. Even W8A8 with dynamic-activation quantization costs only 1.3%. Eight-bit
quantization of Chronos-2 is, for practical purposes, lossless regardless of method.

**F2 — error-compensating PTQ reaches full precision at 3 bits; everything collapses at
2.** GPTQ retains parity at 4 bits (retention 1.00, both group sizes) and at **3 bits**
(1.047, CI [0.988, 1.137] — statistically indistinguishable from fp32), where HQQ loses
23% and naive RTN 70% (Fig. 1a). GPTAQ extends the frontier: W3 retention **1.016**
[0.988, 1.053] and W4 parity (1.000). Its asymmetry strength matters: the official
α=0.25 beats α=1.0 (1.016 vs 1.087 at W3). At 2 bits every PTQ method collapses —
GPTAQ 1.94, GPTQ 2.17, RTN-sim 4.08 — so sub-3-bit deployment needs quantization-aware
training (future work). (A dev-subset signal that GPTQ *beat* fp32 resolved to parity
at full scale; we report the confirmed result.)

**F3 — method dominates bit-width.** At a fixed 4 bits, WQL retention spans 0.996 (GPTQ)
to 1.66 (RTN) — a 66-point gap — whereas moving a *good* method from 8 to 4 bits costs
≤6% (Fig. 1a). For a TSFM, *how* you quantize matters far more than *how much*. This
inverts the intuition that bit-width is the primary knob.

**F4 — calibration degrades before accuracy, and the quantile head is the entire
mechanism.** At 8 bits, accuracy is perfectly retained yet the quantile-crossing rate
already quadruples (0.039→0.149 for RTN) — a structural defect WQL and MASE cannot see;
at 4 bits HQQ and NF4 sag to cov80 0.630/0.686 (a nominal "80%" band that holds 63%)
while GPTQ keeps 0.756 (Fig. 1b). Sensitivity experiments explain *why*. Partitioning
the 126 linears into disjoint families (dev tier, W4 RTN-sim): quantizing *only* the
3-linear quantile head — 3.2% of weights — reproduces the full collapse (WQL 2.115, vs
2.130 with all 126 quantized), while protecting only the head recovers to 1.143 with
*better-than-fp32* calibration (cov80 0.778, QCR 0.002). Accuracy damage is instead
distributed (only-group-attention 1.001, only-FFN 1.086), and
first/last-block-protection folklore does not hold here. Full-benchmark confirmation
(Table 1, "+head"): an fp32 head returns W8-RTN's QCR to 0.037 (fp32: 0.039) — the F4
quadrupling is *entirely* head-gated — and repairs W4 coverage: HQQ 0.630→0.751 (WQL
1.020), NF4 0.686→0.761 (1.045), at a cost of ≈11–13 MB. Accuracy-only evaluation
therefore overstates a compressed forecaster's quality [Dettmers et al., 2024] — but
prevention costs 3% of the weight budget.

**F5 — damage is tail-concentrated.** Mean retention hides worst cases: HQQ-W3 averages
1.23 but its worst task is 8.2× and 44% of tasks exceed 1.10; GPTQ-W4-g64 has a median
per-task ratio of 0.998 and only 4% of tasks above 1.10. Per-series distributions —
released for every run — are the honest picture.

**F6 — no LLM-style outliers: the head fails on weight-space SNR, explaining the AWQ
negative result.** For Chronos-2 the AWQ scale fold is *exact* (we verify the fold
numerically) — yet AWQ-fold + RTN at 4 bits (1.64) barely differs from
plain RTN (1.66) and is far behind GPTQ (Fig. 1a). Activation diagnostics show why:
Chronos-2 has *no* massive-magnitude activation outliers (global |x|max ≈ 11; the
fraction of channels exceeding |x|>6 is zero in most layers and at most 1.6%), only
ReLU-sparsity channel heterogeneity (per-channel kurtosis is modest; pooled kurtosis is
inflated by dead channels). The head's failure is instead a weight-space signal-to-noise
problem: at W4 its output layer's relative reconstruction error is 7.83 — quantization
noise ≈8× the output signal, over 100× any other linear (all <0.07) — because small
quantile differences ride on large weights with no downstream layer to compensate.
Activation *scaling* cannot fix a weight-space SNR deficit; Hessian-aware
*compensation* (GPTQ/GPTAQ) can.

**F7 — post-hoc repairs: sorting is free; conformal restores coverage everywhere.**
Sorting predicted quantiles (monotone rearrangement) provably never increases quantile
loss [Chernozhukov et al., 2010; Fakoor et al., 2023]; empirically it zeroes QCR for
*every* arm at never-worse WQL, and even recovers accuracy where crossing is severe
(HQQ-W3 1.229→1.170). Coverage is only partially restored (HQQ-W4 cov80 0.630→0.703):
crossing is cosmetic damage; the residual under-coverage is genuine distributional
corruption. Split-conformal recalibration (2-fold series split, sort-before-calibrate)
fixes that: cov80 returns to 0.78–0.80 for every method at every bit-width, from
RTN-W2-sim (0.060→0.782) to AWQ-W3 (0.480→0.795) — and raw fp32 itself under-covers
(0.750; MACE 0.059→0.017 conformal), so a quantized-and-recalibrated model is better
calibrated than the raw fp32 reference. Conformal retention is reported against the
conformalized fp32 (same stage), leaving accuracy essentially unchanged. TimesFM-2.5
supplies independent evidence that this repair is a general TSFM need rather than a
quantization artifact: with its auto-sort disabled, its *fp32* model already emits
crossed quantiles (QCR 0.109, cov80 0.709) — the vendor ships exactly this sorting
repair, on by default. Caveat: the conformal guarantee is series-level (residuals
pooled across horizons), not per-horizon.

**F8 — the findings transfer across architectures.** On encoder–decoder Chronos-Bolt
small (48M) and base (205M), with the quantile head protected by default, *every*
method is at parity: small W4-HQQ 0.994 and W3-GPTQ 0.986; base 0.9997–1.005 across all
arms; calibration untouched (base cov80 0.782 = fp32, QCR unchanged). On decoder-only
TimesFM-2.5, likewise head-protected, W8-RTN 0.998, W4-GPTQ 0.999 [0.950, 1.034], and
W4-HQQ 1.009 are at parity, while W3-GPTQ degrades (1.101 [1.054, 1.155]) — its
accuracy cliff sits one bit higher than Chronos-2's, a genuine cross-model difference.
Calibration is again untouched in every arm (cov80 0.686–0.715 vs fp32's 0.709; QCR
0.097–0.115 vs 0.109) — third-architecture confirmation of the head mechanism.

**Efficiency.** Storage-real 4-bit models occupy ≈60 MB (8× smaller) and cut peak
batch-1 GPU memory from 493 MB (fp32) to 87 MB for NF4 (5.7×). Throughput, however, is
flat (~120 series/s at batch 256, vs 110 for fp32): our reference dequantize-then-GEMM
paths buy footprint, not speed. Memory-bound and edge deployments benefit immediately;
compute-bound throughput does not.

![Figure 1](figures/fig1_accuracy_calibration.png)

**Figure 1.** *(a)* WQL retention vs weight bits by method (log scale; lower is better,
1.0 = fp32). GPTQ tracks fp32 to 3 bits; the method spread at 4 bits exceeds the 8→4-bit
gap of any single method; AWQ-fold sits on the RTN line. *(b)* Empirical vs nominal
interval coverage: GPTQ-W4 tracks fp32 while HQQ/NF4 systematically under-cover.

## 4. Discussion, Limitations, and Conclusion

**Recommendation.** (a) At 8 bits use any toolkit, but keep the 3-linear quantile head
in fp32 (retention 0.9995 with fp32-level calibration). (b) For 4–3 bits use GPTQ or
GPTAQ (GPTAQ-W3 1.016); below 3 bits PTQ collapses — use QAT. (c) Always sort predicted
quantiles post-hoc: it is free, provably never worse, and zeroes quantile crossing.
(d) When calibrated intervals matter, wrap the deployed model in split-conformal
recalibration: it restores nominal coverage at any bit-width and even improves fp32.
Report calibration and quantile-crossing alongside WQL whenever a probabilistic model is
compressed.

**Limitations.** We study univariate zero-shot tasks; covariate/multivariate tasks and
quantization-aware training (the indicated sub-3-bit path) are future work.
Module-family sensitivity numbers are dev-tier (8 tasks). GPTQ/GPTAQ/AWQ arms are
accuracy-exact simulations; realized speed needs packed kernels we do not implement
(achievable footprints reported). Latency is hardware-specific (one laptop GPU);
retention ratios are the transferable claim. Conformal guarantees are series-level, pooled across horizons. We
assess but exclude KV-cache/vector methods such as TurboQuant [Zandieh et al., 2025],
which do not apply to a cache-free encoder.

**Conclusion.** Across three architectures, TSFMs quantize remarkably well: 8-bit is
free and error-compensating PTQ holds full-precision accuracy to 3–4 bits. The failure mode that
does appear — calibration and monotonicity degrading before accuracy — is gated by 3.2%
of the weights (the quantile head) and is preventable at the source (fp32 head) or
curable post hoc (sorting + split-conformal) at any bit-width. Compressing a
probabilistic forecaster is not merely an accuracy-retention problem — but with the
head protected and quantiles recalibrated, it is close to a solved one down to 3 bits.

## Reproducibility

Code, configs, environment lockfile, vendored benchmark definitions, per-task summaries,
and all per-series predictions are released. The harness reproduces published per-task
seasonal-naive MASE/WQL to <10⁻⁶ (continuously tested) and Chronos-Bolt's published
per-task numbers to <1%. All results in this paper are regenerated by
`python -m chronosquant.analysis.campaign`.

## References

Ansari et al. Chronos: Learning the Language of Time Series. TMLR 2024. arXiv:2403.07815.
· Ansari et al. Chronos-2: From Univariate to Universal Forecasting. 2025.
arXiv:2510.15821. · Badri & Shao. Half-Quadratic Quantization (HQQ). 2023. ·
Chernozhukov et al. Quantile and Probability Curves Without Crossing. Econometrica 2010.
· Dettmers et al. LLM.int8(). NeurIPS 2022. arXiv:2208.07339. · Dettmers et al. QLoRA
(NF4). NeurIPS 2023. arXiv:2305.14314. · Dettmers et al. Accuracy is Not All You Need.
NeurIPS 2024. arXiv:2407.09141. · Fakoor et al. Flexible Model Aggregation for Quantile
Regression. JMLR 2023. · Frantar et al. GPTQ. ICLR 2023. arXiv:2210.17323. · Aksu et al.
GIFT-Eval. 2024. arXiv:2410.10393. · Li et al. Evaluating Quantized LLMs. 2024.
arXiv:2402.16775. · Li et al. GPTAQ: Efficient Finetuning-Free Quantization for
Asymmetric Calibration. 2025. arXiv:2504.02692. · Lin et al. AWQ. MLSys 2024.
arXiv:2306.00978. · Romano et al. Conformalized Quantile Regression. NeurIPS 2019.
arXiv:1905.03222. · Shchur et al. fev-bench. 2025. arXiv:2509.26468. · Xiao et al.
SmoothQuant. ICML 2023. arXiv:2211.10438. · Zandieh et al. TurboQuant. 2025.
arXiv:2504.19874.
