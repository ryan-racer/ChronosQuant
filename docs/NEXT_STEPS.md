# Next Steps — synthesis of the 2026-07-30 deep-research sweep

Three parallel literature sweeps (SOTA PTQ methods · TSFM landscape/novelty · calibration-repair
& publication scope) as of 2026-07-30. This file is the decision record; details and full
citation lists below.

## Headline: the novelty clock is running

**TQS-PTQ** ([arXiv:2606.13300](https://arxiv.org/abs/2606.13300), Imperial College, June 2026,
ICML 2026 Workshop on Forecasting) weight-quantizes **TimesFM-2.5** (plus Aurora-small,
Pangu-Weather) with a trajectory-based sensitivity score. The "first-ever TSFM weight
quantization" claim is gone. Still open and defensible for us: **first systematic PTQ study of a
TSFM on standard zero-shot benchmarks; first for Chronos-family / any encoder-only quantile-head
model; first probabilistic (WQL/calibration/QCR) treatment** — TQS-PTQ evaluates only on the
ETT/Exchange/Weather long-horizon suite with no probabilistic metrics. Their paper is citable
evidence the topic is timely, and a head-to-head baseline for any TimesFM extension.
Also relevant: a July 2026 survey ([arXiv:2607.20002](https://arxiv.org/abs/2607.20002)) now
names "compression" a TSFM post-training category → **submit the workshop paper ASAP.**

## Verdict on the rotation family (SpinQuant/QuaRot & successors)

**Deprioritize for the current weight-only agenda.** The 2025-26 literature converged: rotations
(QuaRot [2404.00456], SpinQuant [2405.16406], FlatQuant [2410.09426], DartQuant [2511.04063])
are *activation-outlier* technology — they pay off at W4A4/W4A8/KV4, not weight-only scalar
quantization. This corroborates our AWQ-fold ≈ RTN negative result (cite OptRot
[arXiv:2512.24124] — the exception that proves the rule: rotations must target *weight* fourth
moments to help weight-only). Chronos-2 is architecturally the easy case for rotation fusion
(RMSNorm, bias-free, no KV cache), so a rotation arm is justified **only if** we open a W4A4
activation-quant track later; then the order is DartQuant (cheap calibration) + a
register-token experiment ([arXiv:2510.04547], PrefixQuant-style for encoders).

## Ranked plan

### Phase A — ship + cheap high-novelty additions (workshop → strong workshop/ICASSP)

1. **Submit the current workshop paper now.** The ICML 2026 Forecasting workshop already accepted
   a TS-quantization paper; the niche is filling.
2. **Repair-and-recalibrate arm** (highest leverage per line of code):
   - *Post-hoc quantile sorting* (Chernozhukov rearrangement, [arXiv:0704.3649]) — provably
     never worsens pinball/WQL (Fakoor et al., JMLR 2023) and zeroes QCR. Report
     QCR/WQL/coverage before+after sorting for every arm → separates cosmetic crossing from
     structural distribution damage.
   - *Split-conformal / CQR recalibration* on a small calibration window per quantized arm —
     can it restore HQQ-W4's 63%→80% coverage? Nobody has recalibrated a compressed
     forecaster (nearest neighbor: [arXiv:2606.01850], classification-only, measures but
     doesn't repair). Converts our diagnosis into a deployment recipe. Caveat: per-level
     recalibration itself induces crossing (MultiQT [arXiv:2512.23671]) — combine with sorting.
3. **RQ3 executed as a dual-metric sensitivity study**: module-family leave-one-out at W4/W3/W2
   (patch embed / time attn / group attn / FFN / quantile head) × HAWQ-V2 Hessian traces
   [arXiv:1911.03852] × per-module activation-outlier stats — scored separately on WQL *and*
   coverage/QCR. No published sensitivity study ranks modules by calibration damage; likely
   explains *why* GPTQ preserves calibration. Pre-empts TQS-PTQ's calibration-free score.
   At 12 layers, exact leave-one-out is affordable; feeds a mixed-precision Pareto arm
   ("W2 body + W4 first/last/head").

### Phase B — method upgrades (push the cliff below 3 bits)

4. **GPTAQ** ([arXiv:2504.02692], ICML 2025) — ~20-line asymmetric-calibration upgrade to our
   hand-rolled GPTQ; validated on encoder ViTs. Then **Qronos** ([arXiv:2505.11695], NeurIPS
   2025) — provably subsumes GPTQ+GPTAQ.
5. **AutoRound/SignRound(V2)** ([arXiv:2309.05516], [arXiv:2512.04746]) — consensus best cheap
   W3/W2 weight-only (10-20% absolute over GPTQ-class at INT2/3); V2's gradient-based per-layer
   bit allocation doubles as the automated mixed-precision comparison for item 3.
6. **Quantile-loss-guided quantization** (GuidedQuant [arXiv:2505.07004] adapted): weight the
   layerwise Hessian by gradients of the actual pinball/interval loss instead of activation
   second moments. Most novel methodological angle available — directly optimizes for the
   calibration-preservation property we showed proxies miss. Potential title-level contribution.
7. **W2 endpoint via short QAT** (ParetoQ [arXiv:2502.02631] recipe, torchao QAT flow) —
   validated at exactly ~125M scale, trivially affordable on the 8 GB Ada; positioned as "what
   PTQ cannot do, 10%-budget QAT can". Data need: a slice of the Chronos pretraining mixture,
   not 128 windows.

### Phase C — multi-model extension (the main-conference version)

Reviewer calibration (from ICML 2024 [2402.18158], NeurIPS 2024 [2407.09141], ACL Findings
[2402.16775]): main-conference studies need breadth (≥2-3 model families) or a new
metric/phenomenon + mechanism + mitigation. We have the metric (QCR/MACE), mechanism (RQ3), and
mitigation (repair arm); add families:

1. **TiRex/TiRex-2** (NX-AI, 35-44M, xLSTM, quantile head, Apache-2.0, tops fev-bench) — first
   quantization of a *recurrent* TSFM; error accumulation through recurrent state is open.
2. **TimesFM-2.5** (Google, 200M, decoder-only AR + KV cache) — head-to-head with TQS-PTQ under
   a proper zero-shot probabilistic protocol; adds the KV-cache axis Chronos-2 lacks.
3. **Toto-2.0** (Datadog, 4M-2.5B single-recipe family, Student-T mixture sampling head,
   Apache-2.0, #1 on GIFT-Eval) — bit-width-vs-scale scaling law + does quantization distort a
   *sampled* predictive distribution.
4. *(cheap)* **Chronos-Bolt family** (9M-205M) — within-family size sweep on existing pipeline.

This lineup covers encoder-only / recurrent / decoder-AR / MoE-optional and
quantile / sampling output heads — the 2026 TSFM design space.

### Benchmarks

Add **GIFT-Eval** and/or **fev-bench** (v2, June 2026; run by the Chronos team — natural home)
alongside Benchmark II for the full paper; Benchmark II has contamination criticism
(TSFMAudit [arXiv:2605.26161], TIME [arXiv:2602.12147]) — note explicitly that retention
*ratios* are contamination-robust (same data both arms). Quantized variants can be submitted to
the fev-bench/GIFT-Eval leaderboards as distinct entries — free visibility.

### Explicitly deprioritized

- SpinQuant/QuaRot-family rotations for weight-only (see verdict above).
- FP4/MXFP4/NVFP4 — no hardware path on cc 8.9 (Blackwell-only tensor cores); emulation only.
- VQ/trellis W2 (QTIP/AQLM/QuIP#) — LLM-scale machinery, poor amortization at d=768/478 MB;
  QTIP [2406.11235] citable as the PTQ rate-distortion frontier.
- TurboQuant — already excluded (no KV cache).

## Motivation ammunition (deployment demand)

AutoGluon 1.5 ships Chronos-2 presets + SageMaker deployment (June 2026); community INT8 ONNX
Chronos-2 exists with *zero* published accuracy characterization (kashif/chronos-2-onnx) — the
exact gap we fill; TimesFM productized in BigQuery ML/Vertex; Datadog Toto in production
observability; Reverso [2602.17634] and TSFM speculative decoding [2511.18191] both open with
inference-cost motivation.

## Positioning note

Frame the calibration finding as the forecasting-side entry in the "accuracy is not all you
need" lineage (flips [2407.09141] — whose "GPTQ was the only scheme without large flips"
directly parallels our GPTQ-preserves-calibration result; Hooker PIEs [1911.05248]; compressed
multilingual forgetting [2205.10828]; VLM ECE [2509.21173]; conformal compression benchmark
[2606.01850]). Quantile crossing is the qualitatively new failure mode: the output ceases to be
a valid distribution — no classification analogue.
