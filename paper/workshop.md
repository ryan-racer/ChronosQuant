# Accuracy Survives, Calibration Doesn't: Post-Training Quantization of Time-Series Foundation Models

*Anonymous submission — under review.*

## Abstract

Time-series foundation models (TSFMs) deliver strong zero-shot probabilistic forecasts,
but little is known about how they respond to post-training quantization (PTQ). We study
39 weight-quantization configurations spanning seven method families (RTN, torchao,
bitsandbytes INT8/NF4, HQQ, GPTQ, GPTAQ, AWQ-style scale folding) at two to eight bits,
zero-shot on the 27-dataset Chronos benchmark across four models and three architectures
(encoder-only Chronos-2, encoder–decoder Chronos-Bolt, decoder-only TimesFM-2.5), adding
two calibration-aware diagnostics because a probabilistic forecaster can degrade in ways
aggregate accuracy cannot see. Our central result is a *mechanism*: interval calibration
and quantile monotonicity degrade *before* accuracy does, and in Chronos-2 the 3-linear
quantile head — 3.06% of the quantized weights — accounts for essentially all of that
damage; an fp32 head restores fp32-level calibration at both bit-widths we test (W8, W4).
Around it: 8-bit is practically free (all eight W8 arms within 1.3% WQL), yet one
per-tensor 8-bit arm loses four points of coverage while looking lossless on WQL;
error-compensating PTQ stays near parity at 3 bits (GPTAQ 1.016 [0.988, 1.053]) and every
method collapses at 2; and method matters more than bit-width (0.996–1.66 at 4 bits).
Post-hoc sorting and approximate recalibration remove crossing and restore nominal
coverage — but recalibration repairs a model's *claims* about its uncertainty, not its
forecasts.

## 1. Introduction

Time-series foundation models (TSFMs) — Chronos [Ansari et al., 2024], Chronos-2 [Ansari
et al., 2025], TimesFM, Moirai — forecast unseen series zero-shot and are increasingly
deployed where full-precision inference is costly: CPU-only servers, edge devices,
high-throughput batch pipelines. Quantization is the standard efficiency lever for large
language models (LLMs), but its lessons do not obviously transfer to TSFMs, which are
small (Chronos-2 has 120M parameters, a regime where quantization bites harder [Li et
al., 2024]), non-autoregressive (no KV cache, so LLM decode-centric methods do not
apply), and — critically — emit *probabilistic* forecasts whose calibration and quantile
monotonicity matter, not just point accuracy.

**Relation to concurrent work.** Weight quantization of TSFMs has only just begun to be
studied, and its probabilistic side is untouched. Concurrent work (TQS-PTQ; [Pavlova et
al., 2026]) weight-quantizes TimesFM-2.5 — our third architecture — plus Aurora-small and
Pangu-Weather at W2–W4 against RTN/GPTQ/GPTAQ/QEP, on the ETT/Exchange/Weather suite
with point metrics only. Ours is, to our knowledge, the first PTQ
study of a TSFM under the standard *zero-shot* protocol, the first for the Chronos family
and encoder-only quantile-head architectures, and the first to evaluate — and repair — the
*probabilistic* output. The studies converge on where damage lives: TQS-PTQ ranks
TimesFM's tokenizer and point head most sensitive, independent evidence for the
I/O-boundary mechanism we isolate. The nearest calibration neighbour is a conformal
benchmark over quantized and sparse LLMs [Tong et al., 2026], classification-only and
diagnostic rather than corrective.

**Contributions.** (1) A systematic PTQ study of TSFMs under a zero-shot probabilistic
protocol: 39 configurations, 7 method families, 5 bit-widths, 4 models, 3 architectures,
27 datasets, with bootstrap confidence intervals on every aggregate. (2) A
calibration-aware evaluation protocol for compressed probabilistic forecasters —
interval-coverage error and a quantile-crossing rate — revealing failure modes invisible
to WQL/MASE. (3) A *mechanism*: module-family sensitivity and weight-space SNR
diagnostics identify the 3-linear quantile head (3.06% of quantized weights) as the gate
on calibration damage in Chronos-2. (4) Two mitigations with a deployment recipe — an fp32
head (prevention) and post-hoc sorting plus approximate recalibration (cure) — with an
explicit account of what recalibration does *not* fix.

## 2. Setup

**Models.** Chronos-2 [Ansari et al., 2025] (`amazon/chronos-2`) is a 119.5M-parameter,
T5-derived *encoder-only* transformer (12 layers, $d_\text{model}{=}768$, RoPE,
alternating "time" and "group" attention) emitting, in one non-autoregressive pass, direct
multi-step forecasts at 21 quantile levels; it ships in fp32 (478 MB). Quantization
targets all 126 `nn.Linear` weights — including the patch embeddings and the 3-linear
quantile head (3,649,536 of 119,439,360 linear parameters, 3.06%) — except in arms that
explicitly protect the head; LayerNorm stays in fp32 and bf16 (not fp16, which overflows
in T5-lineage models) is the deployment baseline. As cross-model arms we also quantize
Chronos-Bolt small/base [Ansari et al., 2024] (48M/205M), *encoder–decoder* T5 forecasters
(in fp32 our harness reproduces their published per-task WQL to 0.34%/0.72% and MASE to
0.61%/1.38%, small/base; head protected by default), and TimesFM-2.5
(`google/timesfm-2.5-200m-pytorch`; 231M parameters as loaded), a *decoder-only*
forecaster with a continuous quantile head, via a torch-only adapter (head and tokenizer
protected). Its built-in `fix_quantile_crossing` auto-sort is disabled in all arms so QCR
is measured honestly.

**Methods.** We evaluate weight-only PTQ across seven families runnable on commodity
hardware: **RTN** (symmetric round-to-nearest, per-channel/per-tensor, plus a simulated
sub-8-bit sweep); **torchao** INT8 weight-only and INT8 dynamic-activation (W8A8);
**bitsandbytes** LLM.int8() [Dettmers et al., 2022] and NF4 [Dettmers et al., 2023];
**HQQ** [Badri & Shao, 2023], a calibration-free half-quadratic solver; **GPTQ** [Frantar
et al., 2023], a Hessian-aware error-compensating quantizer with one-shot calibration;
**GPTAQ** [Li et al., 2025], its asymmetric-calibration extension correcting each layer
toward the *full-precision* network's activations; and **AWQ-style** activation-aware
scale folding [Lin et al., 2024]. Methods are exercised at W8, W4 (group sizes 64/128),
W3, and W2 — 39 full-benchmark configurations in total.

**Protocol & metrics.** We evaluate zero-shot on the 27-dataset Chronos benchmark [Ansari
et al., 2024] (190,674 forecasts) with the `fev` library — the harness the Chronos-2
authors use — so our numbers are comparable with the public leaderboard. Point accuracy is
MASE, probabilistic accuracy weighted quantile loss (WQL, 9 levels); our fp32 Chronos-2
attains 0.425 WQL skill over seasonal-naive, beating it on WQL on all 27 tasks and on
MASE on 88.9%, and our seasonal-naive matches the published per-task MASE/WQL to <10⁻⁶.
The key quantity is **retention**: the geometric-mean ratio
$\text{WQL}(\text{quant})/\text{WQL}(\text{fp32})$ over the 27 tasks (1.00 = parity), with
a 1000-resample bootstrap 95% CI. Because a probabilistic forecaster can lose quality
without moving WQL, we add **MACE**, the mean $|\text{empirical}-\text{nominal}|$ coverage
error over the central 80/60/40/20% intervals, and **QCR**, the fraction of forecast
points with non-monotone predicted quantiles. Two further protocols: a *module-family
sensitivity* sweep (dev tier: 8 tasks) partitioning the 126 linears into *five* disjoint
families — patch embed (3), time attn (48), group attn (48), FFN (24), head (3) —
quantized or protected one at a time, plus two positional controls (first/last block) that
deliberately cut *across* families; and an offline *repair* arm applying quantile sorting
and approximate 2-fold series-split recalibration to the persisted predictions of 22
full-tier Chronos-2 arms.

## 3. Results

Table 1 summarizes representative configurations; Figure 1 plots the accuracy–bit curve
and calibration. All numbers are sourced from one bootstrap over the full 27-task
benchmark; the complete 39-variant matrix is released.

**Table 1.** Representative variants, full 27-task benchmark. Retention = geometric-mean
WQL(quant)/WQL(fp32), [95% bootstrap CI]; *c80* = coverage of the nominal-80% interval
(fp32 0.750); *QCR* = quantile-crossing rate (fp32 0.039). BPW = effective bits/weight
incl. scales; †accuracy-exact simulation — BPW is the *achievable* footprint. "+head" =
all linears quantized except the 3-linear head; "pt" = per-tensor scales.

| Variant | BPW | WQL ret. [CI] | MASE ret. | c80 | QCR |
|---|---|---|---|---|---|
| fp32 (reference) | 32.0 | 1.000 | 1.000 | .750 | .039 |
| RTN W8 | 8.03 | 0.999 [.997, 1.001] | 1.001 | .749 | .149 |
| RTN W8 (pt) | 8.00 | 1.007 [1.000, 1.014] | 1.011 | .709 | .256 |
| HQQ W8 | 9.00 | 1.002 [1.0001, 1.004] | 1.000 | .746 | .086 |
| **RTN W8+head** | 8.03 | **0.999 [.998, 1.001]** | 1.000 | .751 | .037 |
| **GPTQ W4** | 4.13† | **1.002 [.985, 1.023]** | 1.011 | .756 | .254 |
| HQQ W4 | 5.00 | 1.060 [1.017, 1.124] | 1.027 | .630 | .420 |
| NF4 W4 | 4.13 | 1.065 [1.001, 1.149] | 1.026 | .686 | .396 |
| HQQ W4+head | 5.00 | 1.020 [.986, 1.067] | 1.016 | .751 | .021 |
| NF4 W4+head | 4.13 | 1.045 [.979, 1.122] | 1.025 | .761 | .022 |
| GPTQ W3 | 3.25† | 1.047 [.988, 1.137] | 1.059 | .791 | .366 |
| **GPTAQ W3** | 3.25† | **1.016 [.987, 1.050]** | 1.034 | .770 | .379 |
| HQQ W3 | 4.20 | 1.229 [1.091, 1.481] | 1.143 | .666 | .589 |
| RTN W2 (sim) | 2.0† | 4.082 [3.083, 5.566] | 3.108 | .060 | .523 |

**F1 — 8-bit is practically, not literally, free.** All eight W8 configurations land
within 1.3% of fp32 on WQL (0.999–1.013) and 1.1% on MASE, at 4× compression — practical
equivalence at a 1.3% margin, not statistical parity: five of the eight (HQQ, torchao
INT8-wo and W8A8, per-tensor RTN, bitsandbytes INT8) have bootstrap CIs whose lower bound
reaches 1.000. Accuracy also hides calibration: the per-tensor RTN arm is within 0.7% on
WQL yet drops c80 from 0.750 to 0.709 and lifts QCR to 0.256 (Table 1) — an 8-bit model
that looks lossless and is not.

**F2 — error-compensating PTQ stays near parity at 3 bits; everything collapses at 2.**
GPTQ is at parity at 4 bits (1.002, both group sizes). At 3 bits GPTAQ reaches **1.016**
[0.988, 1.053] — the tightest sub-4-bit interval we obtain — and GPTQ 1.047 [0.988,
1.137]: we cannot reject parity for either, but GPTQ's interval still admits ~14%
degradation, so GPTAQ *tightens* the W3 result rather than provably beating it (intervals
overlap heavily; no paired test was run). At 3 bits HQQ loses 23% and naive RTN 70%
(Fig. 1a), and GPTAQ's asymmetry strength matters (α=0.25: 1.016 vs α=1.0: 1.087). At 2
bits every method collapses — GPTAQ 1.94, GPTQ 2.17, RTN-sim 4.08 — so sub-3-bit
deployment needs QAT.

**F3 — method dominates bit-width.** At 4 bits, WQL retention spans 0.996 (GPTQ) to 1.66
(RTN), whereas moving a *good* method from 8 to 4 bits costs ≤6% (Fig. 1a): *how* you
quantize matters far more than *how much*, inverting the intuition that bit-width is the
primary knob.

**F4 — calibration degrades before accuracy, and the quantile head gates it.** At 8 bits
accuracy is retained yet the quantile-crossing rate already quadruples (0.039→0.149 for
RTN) — a defect WQL and MASE cannot see; at 4 bits HQQ and NF4 sag to c80 0.630/0.686 — a
nominal "80%" band holding 63% — while GPTQ keeps 0.756. Sensitivity experiments locate
the cause; *numbers marked (d) are dev-tier* (8 tasks, W4 RTN-sim). Quantizing *only* the
3-linear head — 3.06% of weights — costs WQL 2.115 [1.436, 3.116] (d), indistinguishable
from quantizing all 126 linears (2.130 [1.455, 3.239] (d)); protecting only the head
recovers to 1.143 [0.957, 1.420] (d) with *better-than-fp32* calibration (c80 0.778, QCR
0.002). The dev tier exaggerates absolute damage — all-quantized is 2.130 (d) but 1.656
full-tier for the same configuration — so that comparison holds within-tier only. Within
it the head accounts for 87% of the excess WQL and 89% of the excess MASE: at W4-RTN it
gates accuracy too. What distinguishes calibration is that head protection restores it
for *every* method on the full benchmark, whereas accuracy retention still varies by
method (Table 1, "+head"): an fp32 head returns W8-RTN's QCR to 0.037 (fp32: 0.039) and
repairs W4 coverage — HQQ 0.630→0.751 (WQL 1.020), NF4 0.686→0.761 (1.045) — for ≈11–13 MB.
Damage is not positional: only-group-attn 1.001 (d), only-FFN 1.086 (d), and the
first/last-block controls change nothing (2.098/2.103 (d)). Prevention is verified at W8
and W4; head-protected W3/W2 arms are not yet run. Accuracy-only evaluation thus
overstates compressed forecaster quality [Dettmers et al., 2024].

**F5 — damage is tail-concentrated.** HQQ-W3 averages 1.23 but its worst task is 8.2× and
44% of tasks exceed 1.10; GPTQ-W4's median per-task ratio is 0.998 with only 4% above
1.10. Report the distribution, not the mean.

**F6 — no LLM-style outliers: the head fails on weight-space SNR, explaining the AWQ
negative result.** The AWQ scale fold is *exact* for Chronos-2 (verified numerically),
yet AWQ-fold + RTN at 4 bits (1.64) barely differs from plain RTN (1.66). Chronos-2 has
*no* massive-magnitude activation outliers (global |x|max ≈ 11; at most 1.6% of channels
exceed |x|>6), only ReLU-sparsity heterogeneity. The head instead fails on weight-space
signal-to-noise: at W4 its output layer's relative reconstruction error is 7.83, over
100× any other linear (all <0.07), because small quantile differences ride on large
weights with no downstream layer to compensate. Activation *scaling* cannot fix a
weight-space SNR deficit; Hessian-aware *compensation* (GPTQ/GPTAQ) can.

**F7 — post-hoc repairs, and their limits.** Sorting predicted quantiles never increases
pinball loss;¹ empirically it zeroes QCR for *every* arm at never-worse WQL and recovers
accuracy where crossing is severe (HQQ-W3 1.229→1.170). It is already standard practice —
TimesFM ships an auto-sort on by default — so our contribution is the negative result that
the *standard* sort suffices; the need is also general rather than quantization-specific
(with that auto-sort disabled, TimesFM's *fp32* model already crosses, QCR 0.109).
Sorting only partly restores coverage (HQQ-W4 c80 0.630→0.703); approximate per-level
recalibration (2-fold series split, sort-before-calibrate) closes the rest, returning c80
to the 0.78–0.81 band for all 22 Chronos-2 arms, from RTN-W2-sim (0.060→0.782) to AWQ-W3
(0.480→0.795), with even raw fp32 improving (MACE 0.059→0.017). *But this repairs the
model's claims, not the model.* RTN-W2-sim ends nominally covered with WQL retention
still 3.41 (from 4.08) — a forecaster 3.4× worse than fp32 wearing honest intervals,
because they were widened; AWQ-W3 moves only 1.50→1.41. Coverage must be read subject to
sharpness [Gneiting et al., 2007], so we report WQL at every stage. The procedure is also
approximate: it pools residuals across horizon steps within a series and applies a
point-level ⌈(n+1)τ⌉ order-statistic correction (per-level one-sided residual quantiles,
not CQR), so the exchangeable unit is really the trajectory and no finite-sample
split-conformal guarantee holds under within-series dependence — 98 of 594 (model, task)
cells get *worse* coverage afterwards, including fp32 on
`monash_australian_electricity` (0.675→0.608). Horizon-stratified calibration
[Stankevičiūtė et al., 2021] is the principled fix, and is future work.

> ¹ For fixed *y* and levels t₁<⋯<t_K, ∂/∂q of the pinball loss ρ_t(y−q) is
> **1**{q>y} − t, nondecreasing in *q* and decreasing in *t*, making the assignment cost
> submodular in (q, t); the sorted arrangement therefore minimizes Σ_k ρ_{t_k}(y − q_(k))
> pointwise, hence in any aggregate. Chernozhukov et al. (2010) instead prove
> rearrangement weakly reduces L^p distance to the *true* quantile curve; Fakoor et al.
> (2023), improvement of the weighted interval score.

**F8 — head-protected quantization transfers across architectures.** On encoder–decoder
Chronos-Bolt small/base (48M/205M), whose quantile head our harness protects by default,
*every* method is at parity (small W4-HQQ 0.994, W3-GPTQ 0.986; base 0.9997–1.005). On
decoder-only TimesFM-2.5 (231M), likewise head- and tokenizer-protected, W8-RTN 0.998,
W4-GPTQ 0.999 [0.947, 1.030] and W4-HQQ 1.009 are at parity while W3-GPTQ degrades
(1.101 [1.058, 1.152]) — its cliff sits one bit higher than Chronos-2's, a genuine
cross-model difference. Calibration moves by method: RTN and HQQ leave Bolt coverage
untouched (base c80 0.782→0.782/0.783; small 0.773→0.773/0.774), but the Bolt GPTQ arms
shift it *up* 3–4 points into over-coverage (base 0.815/0.815 at W4/W3; small
0.811/0.817); TimesFM stays within 0.686–0.715 of its fp32 0.709. Since the head is
protected in every arm here, these results establish that *head-protected* quantization
retains accuracy and calibration across three architectures; they cannot test the
mechanism itself, which is demonstrated only on Chronos-2. A head-unprotected arm on
another family is the obvious next experiment.

**Efficiency.** Storage-real 4-bit models occupy ≈60 MB (8× smaller) and cut peak batch-1
GPU memory 493 to 87 MB, but throughput is flat (~120 vs 110 series/s at batch 256):
dequantize-then-GEMM buys footprint, not speed. (INT8-HQQ: 120 MB on disk, 176 MB peak;
batch-1 latency 30→45 ms.)

![Figure 1](figures/fig1_accuracy_calibration.png)

**Figure 1.** *(a)* WQL retention vs weight bits by method (log scale; lower is better,
1.0 = fp32). GPTQ tracks fp32 to 3 bits; the method spread at 4 bits exceeds the 8→4-bit
gap of any single method; AWQ-fold sits on the RTN line. *(b)* Empirical vs nominal
interval coverage: GPTQ-W4 tracks fp32 while HQQ/NF4 systematically under-cover.

## 4. Discussion, Limitations, and Conclusion

**Recommendation.** (a) At 8 bits use any toolkit, but keep the quantile head in fp32 and
prefer per-channel to per-tensor scales. (b) For 4–3 bits use GPTQ or GPTAQ; below 3 bits
PTQ collapses — use QAT. (c) Always sort predicted quantiles post-hoc: free, provably
never worse, zeroes crossing. (d) When calibrated intervals matter, recalibrate — but as
an *offline diagnostic*: as evaluated it draws residuals from other series' realized
futures, so it is not an online wrapper, and it restores coverage without restoring
forecast quality. Report calibration, sharpness, and crossing alongside WQL whenever a
probabilistic model is compressed.

**Limitations.** We study univariate zero-shot tasks; covariate/multivariate tasks and
QAT are future work. Module-family numbers are dev-tier (8 tasks), overstating absolute
damage. The head mechanism is established on Chronos-2 only; head-protected W3/W2 arms
and head-unprotected arms on Bolt/TimesFM are not yet run, and W2 is represented only by
RTN-sim. The repair arm covers 22 Chronos-2 arms — no Bolt or TimesFM — and its
recalibration is approximate: residuals are exchangeable at the trajectory level but
treated pointwise, so no finite-sample guarantee holds. GPTQ/GPTAQ/AWQ arms are
accuracy-exact simulations (realized speed needs packed kernels we do not implement), and
latency is hardware-specific — retention ratios are the transferable claim. We run no
head-to-head against TQS-PTQ [Pavlova et al., 2026], whose benchmark and metrics differ,
and exclude KV-cache/vector methods such as TurboQuant [Zandieh et al., 2025].

**Conclusion.** On accuracy, TSFMs quantize remarkably well across three architectures:
8-bit is practically free and error-compensating PTQ holds near-full-precision accuracy
at 3–4 bits. The failure mode that does appear — calibration and monotonicity degrading
before accuracy — is, in Chronos-2, gated by 3.06% of the weights, preventable at W8 and
W4 with an fp32 head and partly curable post hoc. But the cure is honest only about the
model's claims: it restores coverage, not quality.

## Reproducibility

Code, configs, lockfile, benchmark definitions, per-task summaries and run metadata are
released; the 1,692 per-series prediction parquets are available on request. Regenerate
tables and figures with `python -m chronosquant.analysis.campaign` (master table),
`python -m chronosquant.quantization.sensitivity analyze|report` (module sensitivity),
`scripts/repair_analysis.py` (repair tables), `scripts/aggregate.py`
(retention/leaderboard), and `scripts/make_figures.py` (figures).

## References

Ansari et al. Chronos: Learning the Language of Time Series. TMLR 2024. arXiv:2403.07815.
· Ansari et al. Chronos-2: From Univariate to Universal Forecasting. 2025.
arXiv:2510.15821. · Badri & Shao. Half-Quadratic Quantization (HQQ). 2023. ·
Chernozhukov et al. Quantile and Probability Curves Without Crossing. Econometrica 2010.
· Dettmers et al. LLM.int8(). NeurIPS 2022. arXiv:2208.07339. · Dettmers et al. QLoRA
(NF4). NeurIPS 2023. arXiv:2305.14314. · Dettmers et al. Accuracy is Not All You Need.
NeurIPS 2024. arXiv:2407.09141. · Fakoor et al. Flexible Model Aggregation for Quantile
Regression. JMLR 2023. · Frantar et al. GPTQ. ICLR 2023. arXiv:2210.17323. · Gneiting,
Balabdaoui & Raftery. Probabilistic Forecasts, Calibration and Sharpness. JRSS-B 2007. ·
Li et al. Evaluating Quantized LLMs. 2024. arXiv:2402.16775. · Li et al. GPTAQ: Efficient
Finetuning-Free Quantization for Asymmetric Calibration. 2025. arXiv:2504.02692. · Lin et
al. AWQ. MLSys 2024. arXiv:2306.00978. · Pavlova et al. Quantizing Time-Series Models As
Dynamical Systems: Trajectory-Based Quantization Sensitivity Score (TQS-PTQ). 2026.
arXiv:2606.13300. · Shchur et al. fev-bench. 2025. arXiv:2509.26468. · Stankevičiūtė,
Alaa & van der Schaar. Conformal Time-Series Forecasting. NeurIPS 2021. · Tong et al.
Does Compression Preserve Uncertainty? A Unified Benchmark for Quantized and Sparse LLMs
via Conformal Prediction. 2026. arXiv:2606.01850. · Xiao et al. SmoothQuant. ICML 2023.
arXiv:2211.10438. · Zandieh et al. TurboQuant. 2025. arXiv:2504.19874.
