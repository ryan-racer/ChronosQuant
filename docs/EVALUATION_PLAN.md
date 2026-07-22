# Evaluation Framework Plan — Quantized Chronos-2

**Status:** v1 (2026-07-22) · synthesized from a literature sweep of TSFM benchmarks (2024–2026) and LLM/encoder quantization evaluation practice. Citations in [RELATED_WORK.md](RELATED_WORK.md).

## 1. Goal and positioning

Quantify how post-training quantization affects **`amazon/chronos-2`** (120M, fp32, encoder-only, direct 21-quantile head) along two axes — *forecast quality retention* and *inference efficiency* — with the rigor of the quantized-LLM evaluation literature applied to the standard TSFM benchmark stack. No peer-reviewed work quantizes a Chronos-family model as of 2026-07; the closest artifacts are a community ONNX-INT8 export and a non-peer-reviewed MLX blog showing **int8 ≈ free, int4 degrades quantile calibration** — a hypothesis this framework is designed to test properly.

Research questions RQ1–RQ4 are defined in the [README](../README.md).

## 2. Systems under evaluation

| Group | Variant | Purpose |
|---|---|---|
| Reference | `amazon/chronos-2` fp32 | ships in fp32 → the ground-truth reference |
| Baseline | bf16 (never fp16 — T5-lineage overflow risk) | the "lossless" deployment baseline all retention numbers cite |
| PTQ sweep | W8A16, W8A8-dynamic, FP8 (E4M3), W4A16 (g=32/64/128), W4A8, optional W3/W2 (cliff-finding) × methods: RTN (torchao), HQQ, bitsandbytes int8/NF4, Quanto, ONNX-RT dynamic INT8, NNCF static INT8 | the core matrix (RQ1, RQ2) |
| Adapted PTQ (stretch) | GPTQ / SmoothQuant via llm-compressor tracing; AWQ-style scaling | contribution-worthy ports (RQ2) |
| QAT (contingent) | torchao QAT (8da4w) | only if W4 PTQ misses the accuracy bar |
| Anchors | `autogluon/chronos-2-small` (28M fp32), `amazon/chronos-bolt-small/base` fp32, `kashif/chronos-2-onnx` INT8 | answers "quantized-large vs small-fp32" — the practically decisive comparison |

Ada GPU (cc 8.9) supports FP8 and INT8 natively; CPU runs cover the ONNX/OpenVINO deployment story.

## 3. Benchmarks & datasets (three tiers + dev subset)

| Tier | Suite | Composition | Metrics / aggregation | Why |
|---|---|---|---|---|
| **Primary** | **fev-bench** (`pip install fev`; arXiv:2509.26468) | 100 tasks / 96 datasets; 46 covariate, 35 multivariate; energy, retail, cloud-obs, econ, health, mobility, nature | **SQL** (9 quantiles 0.1–0.9) + **MASE**; **win rate** + **skill score** vs Seasonal Naive; 95% bootstrap CIs (1,000 resamples) | The Chronos-2 authors' own harness → our fp32 numbers are directly checkable against the published leaderboard (91.4% win rate / 47.3% skill); exercises covariates + multivariate paths (group attention) that other suites miss |
| **Secondary** | **GIFT-Eval** (arXiv:2410.10393; HF `Salesforce/GiftEval`) | 97 task configs; 7 domains, 10 frequencies, short/medium/long terms; rolling windows stride H, ≤20 windows | **MASE** + **CRPS** normalized by Seasonal Naive, geometric mean; mean rank | De-facto public leaderboard → external comparability; long-horizon terms stress the 1024-step direct decoding |
| **Tertiary** | **Chronos Benchmark II** (arXiv:2403.07815; HF `autogluon/chronos_datasets`, fev `chronos_zeroshot` configs) | 27 zero-shot datasets, single last window, short histories | **WQL** + **MASE**, relative to Seasonal Naive, geometric mean | Canonical, cheap (~fast full pass) → the workhorse for sweeps/ablations before promoting configs to Tier 1/2 |
| **Dev** | ~10-task subset of Benchmark II (fixed seed, fixed windows) | subset spanning frequencies/domains | WQL + MASE | Harness iteration only; never reported |

**Leakage stance:** we evaluate *relative* degradation (quantized vs same-weights fp32), so pretraining leakage largely cancels; still, we disclose the Chronos-2 training-corpus overlaps exactly as the Chronos-2 paper does, mark leaderboard flags accordingly, and (stretch) repeat headline tables with `autogluon/chronos-2-synth` for strictly-zero-shot framing. GIFT-Eval task-count ambiguity (97 vs 98 rows / 23 vs 55 datasets) is stated explicitly in our reporting.

## 4. Metric suite

### 4.1 Accuracy (per benchmark's native convention)
- **Probabilistic:** WQL (Chronos convention) / SQL (fev-bench) / CRPS (GIFT-Eval, GluonTS `mean_weighted_sum_quantile_loss`) — all on the 9-quantile grid {0.1,…,0.9} even though the model emits 21 levels.
- **Point:** MASE (seasonal-naive-scaled).
- **Aggregation:** per-task normalization by Seasonal Naive → geometric mean (clip [10⁻², 10²] per fev) → skill score; pairwise **win rates** with bootstrap CIs; mean rank for GIFT-Eval.

### 4.2 Quantization-specific (the paper's novel measurement surface)
1. **Accuracy retention:** metric(quant)/metric(fp32) per cell of the bit-width × method matrix.
2. **Quantile calibration:** empirical coverage of 80%/90% prediction intervals + weighted-interval/MSIS deltas per bit-width — *the* metric the MLX evidence says fails first at int4; a probabilistic-forecasting-specific contribution beyond LLM-quantization practice.
3. **Distributional fidelity / "flips"** (per *Accuracy is Not All You Need*, arXiv:2407.09141): per-series ΔWQL distribution (not just the mean), % of tasks where quantized flips from beating→losing to Seasonal Naive, and divergence between quantized and fp32 predictive quantiles.
4. **Win rate of quantized vs its own fp32 parent** (head-to-head, per task).
5. **Difficulty slicing** (Whisper precedent, arXiv:2511.08093): degradation stratified by series volatility/seasonality strength — expect damage concentrated on hard series.

### 4.3 Efficiency (RQ4) — on named hardware
- **Hardware:** NVIDIA RTX 2000 Ada Laptop (8 GB, cc 8.9) for GPU; the same machine's CPU for ONNX-RT/OpenVINO INT8. Backend/kernel named in every row (torchao + torch.compile, ORT EP, OpenVINO, bnb kernels).
- **Size:** checkpoint bytes + **effective bits/weight including scales/zero-points** (e.g., 4-bit g128 = 4.125 BPW).
- **Memory:** peak inference VRAM/RAM at fixed batch.
- **Latency:** batch-1 median + p90 per forecast (ctx 2048, h 64), warm.
- **Throughput:** series/s at large batch (mirroring the paper's A10G convention: batch 1024, ctx 2048, h 64 — scaled to fit 8 GB), GPU and CPU.
- Report "quantized-but-slower" cases honestly (documented bnb small-model slowdowns).

### 4.4 Statistical hygiene
Fixed seeds for window selection & calibration sampling; calibration reruns across ≥3 seeds with std; bootstrap CIs on all aggregate deltas; per-task results persisted as CSV (fev/GIFT-Eval submission formats) under `results/raw/` with config + git SHA.

## 5. Sensitivity & ablation protocols (RQ3)

1. **Per-module sensitivity:** torchao per-module-FQN configs; quantize one module class at a time / leave-one-out in high precision: {attention QKV/O} × {FFN wi} × {FFN wo (expected cliff — GLU outlier locus)} × {patch-embed residual net} × {quantile head} × {time-attention vs **group-attention** layers (no prior data — novel)} × encoder depth (early/mid/late).
2. **Always-high-precision set:** norms, input standardization/arcsinh path, residual adds, softmax (per PR #197 precedent); ablate to confirm necessity.
3. **Calibration ablations** (for calibrated methods): set size {32, 128, 512, 1024 series} × domain (in-domain / cross-domain / synthetic) × seed variance (arXiv:2311.09755 shows both matter).
4. **Granularity ablations:** per-tensor vs per-channel weights; static vs dynamic activations (Whisper evidence: dynamic ≫ static); group size {32, 64, 128}.

## 6. Deliverable tables & figures

| Artifact | Content |
|---|---|
| T1 | Main matrix: method × bit-width → SQL/WQL/MASE retention on all three tiers, with CIs |
| T2 | Efficiency: size/BPW, peak mem, latency, throughput (GPU + CPU) per variant |
| T3 | Calibration: 80/90% PI coverage per bit-width |
| T4 | Sensitivity: per-module ΔWQL heatmap (module × bit-width) |
| T5 | Anchors: Chronos-2-W4/W8 vs chronos-2-small/bolt fp32 |
| F1 | Pareto: accuracy retention vs model size and vs latency |
| F2 | Per-series ΔWQL distributions (violin) per bit-width |
| F3 | Quality-vs-bit-width curve with the degradation cliff |
| F4 | Calibration curves (nominal vs empirical coverage) |

## 7. Implementation phases

- **P0 — Harness bring-up (first):** integrate `fev` + Benchmark II configs; `src/chronosquant/evaluation/` runner: config → pipeline → per-task CSV → aggregation (skill/win-rate/bootstrap, mirroring fev-bench's definitions). **Gate:** fp32 Chronos-2 reproduces published Benchmark II ballpark.
  **✅ Done (2026-07-22).** Gate results: our Seasonal Naive reproduces the published per-task MASE/WQL **exactly** (< 1e-6 rel. diff on all 8 dev tasks, enforced continuously by `tests/test_reference_parity.py`); Chronos-2 fp32 ranks first vs published anchors on the dev subset (83% win rate / 0.383 WQL skill vs chronos_bolt_base 58%/0.329). Notable baseline measurement: fp32 Chronos-2 already exhibits nonzero quantile crossing on covid_deaths (QCR 0.178) — quantized QCR/coverage deltas must be reported relative to the fp32 baseline, which `retention_table` does by construction.
- **P1 — Efficiency profiler:** `scripts/profile_model.py` (latency/throughput/memory protocol above).
  **✅ Done (2026-07-22).** fp32 baseline on RTX 2000 Ada (ctx 2048, h 64): batch-1 median 30 ms (p90 51 ms), 112.6 series/s @ batch 32, peak 2.3 GB @ batch 256; CPU: batch-1 155 ms, 13.3 series/s @ batch 256. Full 27-task baselines complete: Seasonal Naive matches published per-task numbers exactly (27/27), Chronos-2 fp32 = **88.0% win rate / 0.425 WQL skill** (vs chronos_bolt_base 67.6%/0.376, non-overlapping win-rate CIs). Note: our WQL skill is ~4 pts below the paper's 79.8%-win-pool value (46.6%) — likely context-length protocol difference (we use the pipeline default); retention claims use our own fp32 as reference, so this does not gate quantization results.
- **P2 — Weight-only PTQ sweep:** torchao RTN {W8, W4 g32/64/128}, HQQ {8/4/3}, bnb {int8, NF4} → T1/T2 on Tier 3, promote survivors to Tiers 1–2.
- **P3 — Activation quant + CPU story:** W8A8-dynamic, FP8, ONNX-RT dynamic INT8, NNCF static INT8 (attention-node exclusions per known pitfalls).
- **P4 — Sensitivity + calibration ablations** (§5) → T3/T4.
- **P5 — Stretch:** llm-compressor GPTQ/SmoothQuant port with time-series calibration sets; QAT only if W4 fails; `chronos-2-synth` zero-shot repeat.
- **P6 — Analysis & writing:** `analysis/` produces all tables/figures from `results/raw/` deterministically.

## 8. Risks & mitigations

| Risk | Mitigation |
|---|---|
| 8 GB VRAM limits batch-1024 throughput protocol | scale batch to fit, report the batch used; absolute numbers are per-hardware anyway — retention ratios are the headline |
| GPTQ/AWQ tooling assumes decoder LLMs | treat as stretch (P5); core claims rest on architecture-agnostic toolkits |
| Full GIFT-Eval/fev-bench runtime on a laptop | sweep on Tier 3 (cheap), promote only surviving configs; medium/long GIFT-Eval terms batched overnight |
| "Quantization" terminology collision (Chronos-1 input binning) | disambiguate in abstract/intro; keyword hygiene in the paper |
| Windows quirks in quant kernels (triton/bnb) | ONNX-RT/OpenVINO paths are Windows-native; torchao CPU fallbacks; WSL2 as fallback |
