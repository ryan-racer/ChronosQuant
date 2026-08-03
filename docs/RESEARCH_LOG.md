# Research Log

Dated log of experiments, decisions, and findings. Newest entries first.

## 2026-07-31 — Full-tier confirmations (27 tasks, 0 failures across all sweeps)

- **GPTAQ**: W4-g64 0.9998 [0.987, 1.012] = exact parity; **W3 1.016 [0.988, 1.053] — beats
  GPTQ-W3 (1.047)**, extending the parity frontier; alpha=1.0 ablation 1.087 (official 0.25
  default confirmed at scale); W2-GPTAQ 1.937 vs W2-GPTQ 2.168 — 10% relative gain but both
  collapsed → sub-3-bit needs QAT.
- **fp32-head recipe**: w8a16-rtn+fp32head WQL 0.9995, **QCR 0.037 ≈ fp32 0.039** (F4's QCR
  quadrupling fully attributed to the head and eliminated); w4-hqq+fp32head 1.020 / cov80
  0.751 (from 0.630) / QCR 0.021; w4-nf4+fp32head 1.045 / 0.761 / 0.022.
- **Chronos-Bolt (48M + 205M, head protected by default)**: every method ≈ parity — small
  w4-hqq 0.994, w3-gptq 0.986 (CI incl. 1.0); base all within 0.9997–1.005; calibration
  untouched (base cov80 0.783 vs fp32 0.782). First cross-model evidence; Bolt quantizes
  even more gracefully than Chronos-2.
- Repair tables regenerated under the reviewed sort-before-calibrate protocol: conformal
  cov80 lands 0.78–0.80 (nominal) for every arm incl. w2-rtn-sim (0.062→0.782); guarantee
  now attaches to the evaluated object; `_samestage` retention column added (repair also
  improves fp32 by ~0.4%).
- Campaign master regenerated. Paper updated (8 findings, 4pp+refs, all numbers CSV-verified).
- **TimesFM-2.5 (200M decoder-only AR, 3rd architecture family) integrated and full-tier
  evaluated (27/27 × 5 arms, 0 failures)**: W8 0.998, **W4-GPTQ 0.999 = parity**, W4-HQQ
  1.009, W3-GPTQ 1.101 (cliff one bit higher than Chronos-2). Calibration untouched under
  quantization with head+tokenizer protected (cov80 0.686–0.715 vs fp32 0.709). **fp32
  TimesFM natively emits crossed quantiles (QCR 0.109) once its built-in
  fix_quantile_crossing auto-sort is disabled** — the vendor ships our sorting repair by
  default; crossing is a general TSFM phenomenon, not a quantization artifact. Adapter
  gotchas handled: forecast() mutates caller's input list; deepcopy of compiled model
  aliases the quantized stream (custom __deepcopy__); stacked_xf stage patterns added to
  gptq.py (20 stages × 4 linears). TiRex/TiRex-2 (recurrent xLSTM, 4th family) remains
  the top extension for the main-conference version (transformers<5 pin needs isolation).

## 2026-07-30 — Phase A–C campaign: repair arm, RQ3 answered, GPTAQ, Chronos-Bolt

Multi-agent campaign executing docs/NEXT_STEPS.md Phases A–C (parallel implementation
agents + adversarial code review per stage; full-tier promotions running as of this entry).

**RQ3 ANSWERED — the quantile head is the mechanism (dev tier, W4 RTN-sim, 13 arms).**
- Families partition all 126 linears (verified vs the real model + in metadata). Quantizing
  ONLY the 3-linear quantile head (3.2% of weights) reproduces the full collapse (WQL_rel
  2.115 vs all-quantized 2.130); PROTECTING only it recovers to 1.143 with calibration
  better than fp32 (cov80 0.778, QCR 0.0016). Calibration damage is *exclusively*
  head-gated; accuracy damage is distributed (only-ffn 1.086 > only-patch 1.097* > time
  1.018 > group 1.001; *wide CI). First/last-block protection folklore does NOT hold here.
- Head-interaction sweep (real methods + fp32 head, dev): HQQ-W4 QCR 0.419→0.018, cov80
  0.650→0.738; NF4 0.409→0.013/0.751; W8-RTN QCR 0.102→0.022 (= fp32). **Solves the F4
  mystery: the 8-bit QCR quadrupling was entirely head-quantization.** Table:
  results/tables/sweep_head_interaction_dev_retention.csv (+ full tier queued).
- Diagnostics (results/tables/sensitivity_{hessian,outliers}.csv): the head's
  output_layer has rel_recon_err 7.83 — quantization noise ≈ 8x its output signal, 3
  orders of magnitude above every other module (small quantile differences riding on
  large weights; no downstream layer to compensate). Family-level Hessian sums are
  convention-dependent and do NOT flag the head — only the per-module SNR view does
  (itself a finding: cheap local diagnostics under-flag the head). Chronos-2 has NO
  LLM-style massive-magnitude activation outliers (global |x|max 11.3; |x|>6 channels
  ~0-0.2%); what exists is ReLU-sparsity channel heterogeneity (pooled kurtosis up to
  ~99k driven by dead channels; per-channel kurtosis modest). Reconciles the AWQ≈RTN
  negative result: the damage is a weight-space SNR problem in the head, which
  activation-aware scaling cannot address.

**Repair-and-recalibrate arm (FULL tier, 27 tasks × 20 runs, offline on persisted
predictions; results/tables/repair_{sorting,conformal,per_task}.csv).**
- Chernozhukov sorting: QCR → exactly 0 for every arm at never-worse WQL (confirmed
  empirically arm-by-arm; even recovers accuracy — HQQ-W3 1.229→1.170). Coverage only
  partially restored (HQQ-W4 cov80 0.630→0.703): crossing is cosmetic damage; the
  remaining under-coverage is genuine distribution corruption.
- 2-fold series-split conformal (+sorting): restores cov80 to ~0.80 nominal for every
  method — and reveals raw fp32 itself under-covers (0.750, MACE 0.059→0.018 conformal).
  A quantized+recalibrated model is better calibrated than raw fp32. Deployment recipe:
  sort (free) + small conformal window; or prevent at source with the fp32 head.

**GPTAQ (arXiv:2504.02692) implemented** (two-stream asymmetric calibration in gptq.py;
math review vs the authors' reference code: sign-off; 19 tests). Dev: ≈GPTQ at W4/W3;
alpha ablation confirms the official 0.25 default (alpha=1.0 degrades W3 0.964→1.089);
W2 improves over GPTQ (2.51 vs 2.71) but both collapse → QAT is the sub-3-bit path
(ParetoQ-style, future work). W2-GPTQ cliff point added.

**Chronos-Bolt adapter** (predictors.py subclass + _BoltPipelineAdapter; 172 offline
tests): fp32 dev parity vs published per-task CSVs 0.34%/0.72% (small/base). Dev PTQ
(head protected by default, 6 modules skipped): HQQ-W4 at parity (1.000 base/0.989
small) vs Chronos-2's head-quantized 1.06-1.10 — cross-model confirmation of the head
mechanism. Two infra fixes: Bolt's predict_quantiles rejects batch_size (adapter chunks
manually); transformers T5 MLP reads `.weight` in an int8-aware dtype guard →
RTNQuantizedLinear now exposes int8 `weight` property (bnb convention; regression test).
sweep.py generalized: per-variant base-predictor overrides + per-reference retention
tables (existing sweeps byte-identical, reviewed).

**Phase C recon:** TiRex/TiRex-2 viable on Windows via pure-PyTorch sLSTM backends
(tirex-2 pins transformers<5 — isolate first); TimesFM-2.5 clean torch-only install,
but `fix_quantile_crossing` must be OFF (it is literally our sorting repair baked in —
and a free "vendor repair" comparison arm); its KV-cache axis needs horizon >128.

**Reviews:** GPTAQ (no blockers; 3 should-fixes applied), sensitivity+Bolt (no blockers;
all paper-bound numbers reproduced by hand; 3 diagnostic-table fixes applied: raw
ch_absmax_median instead of clamp-artifact ratios, per-channel vs pooled kurtosis,
dual-convention Hessian family sums). Repair module review pending.

**Also noted:** independent InQ experiment (experiments/inq/, not part of this campaign)
falsified DPCM K/V-state coding on Chronos-2 — K/V activations are insensitive (2-bit
KV ≈ fp32) and patching+depth whiten temporal redundancy. Complements our story: on
Chronos-2 the sensitive surface is weights (specifically the head), not states.

**Running:** full-tier chain (GPTAQ 5 variants → head_interaction 3 → Bolt fp32 ×2 →
Bolt PTQ 8) for paper-grade numbers.

## 2026-07-23 — ICML-workshop paper (LaTeX, 3 pp.), figures, efficiency table

- Rewrote the paper as a tight two-column ICML-workshop submission: `paper/workshop.tex` → `paper/workshop.pdf` (**3 pages, under the 4-page limit**), typeset with MiKTeX `pdflatex`. Markdown source retained at `paper/workshop.md`; long working draft at `paper/draft.md`.
- Every number pulled directly from result CSVs (no transcription): abstract, main results table (Table 1, 11 representative variants), efficiency table (Table 2), findings F1–F6.
- Figure: `scripts/make_figures.py` → `paper/figures/fig1_accuracy_calibration.png` — 2-panel (accuracy-vs-bits by method; calibration curves), Okabe-Ito CVD-safe palette + distinct markers/linestyles (print-safe), one axis per panel.
- Visual review loop: PyMuPDF rasterizes each page → I inspect layout (no overfull boxes, tables/figure placement, page count) → iterate. Added Table 2 (efficiency) to use page-3 whitespace and make the deployment claims concrete.
- Build: `pdflatex` at `%LOCALAPPDATA%/Programs/MiKTeX/...`; missing packages installed via `mpm` (network slow but works; on-the-fly installer hangs behind the proxy so pre-install or use `--disable-installer`).

## 2026-07-23 — Full quantization campaign complete (18 variants × 27 tasks)

Master table: `results/tables/campaign_master.csv` (via `python -m chronosquant.analysis.campaign`).

- **Both full sweeps finished, 0 failures.** Recovered twice: (1) bnb-int8 native process crash → per-variant process isolation + metadata-based completeness check; (2) user closed the tab mid-`w3-gptq` → resumed cleanly (completed variants skipped, partial reran).
- **Confirmed headline results (full-tier, bootstrap CIs):**
  1. **8-bit is free, method-agnostic** — WQL retention 0.999–1.013 across all 6 W8 methods.
  2. **GPTQ = fp32 parity to 3 bits.** W4-GPTQ 0.996/1.002 (CI incl. 1.0); **W3-GPTQ 1.047 [0.988–1.137] = still parity** (vs W3-HQQ 1.229, W3-RTN 1.695). Real 3-bit at no accuracy cost is the campaign's strongest result.
  3. **Dev "beats fp32" (0.96) did NOT replicate** at full scale — resolved to parity for both GPTQ W4 group sizes. Flagging-before-claiming paid off.
  4. **Method ≫ bit-width**: 66-pt WQL spread at 4 bits (GPTQ 0.996 → RTN 1.66); ≤6% cost 8→4 with a good method.
  5. **Only GPTQ preserves calibration at W4**: coverage[0.8] 0.756 vs fp32 0.750, while HQQ 0.630 / NF4 0.686. **QCR rises under every method even when accuracy is perfect** (the earliest, accuracy-invisible damage signal — our metric).
  6. **AWQ-fold ≈ RTN** (W4 1.64 vs 1.66) despite provably-exact folds — clean negative result: compensation works, scaling alone doesn't.
- **Efficiency (storage-real winners, GPU):** peak batch-1 memory fp32 493 → W8-hqq 176 → W4-hqq 117 → W4-nf4 87 MB (5.7×); throughput flat (~120 series/s) — dequant-at-forward buys footprint not speed. Profiles in `results/profiles/`.
- Storage caveat recorded: GPTQ/AWQ/sim variants store dequantized grids in fp32 containers (accuracy arms); achievable footprint = eff_bpw column. HQQ/NF4/bnb/RTN-real carry true reduced size.

## 2026-07-22 — Leading-methods dev sweep (GPTQ, AWQ-fold); TurboQuant excluded

- **TurboQuant (arXiv:2504.19874, ICLR 2026) assessed and excluded**: online *vector* quantization for KV caches/retrieval — Chronos-2 has no KV cache; as a weight method it reduces to rotation + scalar quant (subset of QuaRot-class). Cited in related work instead.
- Implemented hand-rolled **GPTQ** (one-shot fp-activation Hessians, act-order, group grids) and **AWQ-style exact scale folding** (+ shared RTN grid), both calibrated on 128 synthetic series (leakage-clean; source/size/seed are P4 ablation axes).
- **Dev results (8 tasks, CPU eval):**
  | variant | WQL_rel | MASE_rel |
  |---|---|---|
  | w4-gptq-g64 | **0.963** [0.94–0.98] | 0.994 |
  | w3-gptq | **0.965** [0.89–1.04] | 0.976 |
  | w4-gptq (g128) | 0.980 | 1.001 |
  | w8-gptq | 1.000 | 1.008 |
  | w4-awq-rtn | 2.065 | 1.458 |
  | w3-awq-rtn | 1.768 | 1.452 |
- **Findings (pending full-27 confirmation):**
  1. **GPTQ >> HQQ (1.10) >> NF4 (1.16) >> AWQ-fold ≈ plain RTN (~2.1) at W4** — Hessian error compensation is what preserves quality on this model; activation-aware scaling alone does almost nothing at per-channel granularity.
  2. **GPTQ-W4/W3 measure *better* than fp32 on the dev subset** (CI excludes 1.0 at W4-g64). Treat skeptically: 8 tasks, possible regularization artifact; full tier must confirm before any claim.
  3. GPTQ's dev ranking at W3 (0.965) vs HQQ-W3 (1.257): the accuracy cliff moves at least one bit lower with error compensation.
- Full-tier leading sweep queued behind the standard full sweep (GPU-bound; big datasets infeasible on CPU).

## 2026-07-22 — Standard PTQ sweep (12 variants), dev tier

- New backends verified on Windows CUDA: torchao int8dyn (W8A8), HQQ 2–8 bit, bnb LLM.int8() (fp16-cast wrapper), bnb NF4. torchao int4wo unusable on Windows (`mslk` kernel dep) → real W4 via HQQ/NF4.
- **Dev retention (WQL ratio vs fp32, 8 tasks):**
  | variant | WQL_rel | note |
  |---|---|---|
  | w8a16-rtn (per-channel) | 0.9994* | best (*full-tier number) |
  | w8-hqq (g64) | 1.005 | |
  | w8a16-torchao | 1.009 | |
  | w8a16-rtn-pt (per-tensor) | 1.013 | granularity matters |
  | w8-bnb-int8 | 1.016 | fp16 activations cost |
  | w8a8-torchao-dyn | 1.023 | activation quant ≈ 2% |
  | w4-hqq | 1.101 | best W4 |
  | w4-bnb-nf4 | 1.156 | see divergence below |
  | w3-hqq | 1.257 | |
  | w4-rtn-sim | 2.130 | plain RTN collapses at W4 |
  | w2-rtn-sim | 5.824 | full collapse |
- **Findings:**
  1. **W4 is method-dominated**: HQQ 1.10 vs plain per-channel RTN 2.13 — optimized rounding/grouping is worth ~2× WQL at 4 bits (LLM literature confirmed on a TSFM).
  2. **NF4 probabilistic/point divergence**: WQL +15.6% while MASE only +0.6% — quantile-head damage that point metrics completely miss. Strengthens the paper's "evaluate probabilistically" thesis.
  3. **Calibration cliff precedes accuracy cliff**: coverage[0.8] 0.744 (fp32) → 0.65 (w4-hqq, −9 pts) while WQL only +10%; QCR rises monotonically with fewer bits (0.02 → 0.61 at w3).
  4. HQQ's g64 fp32 scale+zero overhead is heavy at high bits: w8-hqq = **9.0 BPW** (8 + 2×32/64); w4-hqq = 5.0 BPW. Larger groups or quantized metadata would trim this.
  5. Storage-accounting caveat: torchao/bnb-int8 tensors are partially invisible to `model_summary` (tensor subclasses / plain attrs) — `quant_effective_bits_per_weight` from the transform is authoritative; torchao BPW ≈ RTN's 8.03.
- Full 27-task sweep launched (12 variants, ~3.5 h).

## 2026-07-22 — P2: W8A16 RTN full-benchmark results (first quantized Chronos-2 numbers)

- **Full Benchmark II (27/27 tasks): W8A16 RTN is accuracy-neutral.** WQL retention 0.9994 [CI 0.997–1.002], MASE 1.0008 [0.999–1.004]; WQL skill 0.4252 vs fp32's 0.4248; win rate vs fp32 ≈ coin toss (0.41/0.37).
- **QCR finding confirmed at scale: crossings quadruple (mean 0.039 → 0.149), increasing on 25/27 tasks** (max Δ +0.53 on car_parts) — while MACE (0.0594 → 0.0599) and coverage[0.8] (0.750 → 0.749) are unchanged. Refined claim: W8 preserves both accuracy and interval calibration but systematically breaks quantile monotonicity; crossings must be small in magnitude (else WQL/MACE would move). TODO: quantify crossing magnitude from persisted predictions; add post-hoc quantile-sorting repair arm.
- **Efficiency (reference dequant-at-forward implementation, honest numbers):** storage 478 → 120 MB (3.98×, 8.034 BPW); peak GPU memory (batch 1) 493 → 154 MB; but batch-1 GPU latency 30 → 77 ms (dequant overhead) and throughput ≈ parity (109.6 → 115.7 series/s @ batch 256). Weight-only RTN buys footprint, not speed — as the LLM literature predicts for unfused dequant paths; optimized-kernel methods (torchao int8wo, ONNX-RT INT8) are the speed comparison arms.

## 2026-07-22 — P2: first quantized variant (W8A16 RTN) — dev results

- Implemented `chronosquant.quantization`: reference **RTN** (symmetric weight-only int8, per-channel/per-tensor, glob-based module skipping, simulate mode for sub-8-bit accuracy studies) + **torchao int8wo** adapter (verified working on Windows CUDA, torchao 0.17). Quantization is declarative predictor config; metadata (bits, modules, effective BPW incl. scales) flows into the model card and summary columns.
- **Dev subset (8 tasks), chronos2-w8a16-rtn vs fp32:**
  - Storage: 478 MB → **120 MB** (3.98×), effective **8.034 bits/weight**.
  - Accuracy: **WQL ratio 1.0007** [CI 0.994–1.006], MASE 0.9982 → **accuracy-neutral**, consistent with the int8-is-free hypothesis.
  - **⚠ Finding: quantile crossing rate (QCR) increases on 6/8 tasks** despite neutral accuracy — covid_deaths 0.178 → **0.511**, m1_quarterly 0 → 0.088, ercot 0 → 0.078; MACE up slightly (covid 0.020 → 0.079). Weight quantization perturbs quantile *ordering* even when quantile *loss* is unchanged — invisible to WQL/MASE-only evaluations. Candidate paper contribution; suggests a with/without post-hoc quantile-sorting repair arm.
- 73 tests passing (20 new for quantization: reconstruction bounds ≤ scale/2, BPW accounting exact, determinism, skip patterns, simulate mode monotone error growth).

## 2026-07-22 — Full 27-task baselines locked; P1 profiler run; ready for P2

- **Seasonal Naive full run: 27/27 tasks, exact parity with published reference** (max rel. diff 0.00% on MASE and WQL for every task, incl. ETTh/ETTm and the large datasets m4/m5/dominick).
- **Chronos-2 fp32 full run: 27/27 tasks, 0 failures.** WQL: 88.0% win rate [CI 80.6–94.4], 0.425 skill [0.346–0.509]; MASE: 86.4% win, 0.241 skill — above chronos_bolt_base (67.6%/0.376 WQL) with non-overlapping win-rate CIs. Caveat logged: WQL skill ≈4 pts below the Chronos-2 paper's value; suspect context-length protocol difference (we use pipeline default; consider a context_length=8192 sensitivity check). Retention analyses use our fp32 as reference, so quantization results are unaffected.
- **P1 efficiency profile (fp32 reference rows of paper Table 4):** GPU (RTX 2000 Ada, ctx 2048/h 64): batch-1 30 ms median / 51 ms p90, 112.6 series/s @32, 109.6 @256, peak 493 MB/660 MB/2317 MB @ batch 1/32/256. CPU: batch-1 155 ms, 6.5→13.3 series/s @1→256. CPU INT8 is the highest-leverage efficiency target (community ONNX artifact claims 3–8×).
- Canonical run artifacts (summaries.csv + run_metadata.yaml) now tracked in git; predictions parquets remain untracked (large, regenerable).
- **Next: P2 — first quantized variant (torchao W8A16 RTN) through the `model_transform` hook.**

## 2026-07-22 — Track-everything harness, paper draft, P1 profiler; full-run infrastructure debugging

- **Harness now captures ~90 columns/task**: full fev metric suite (WQL/SQL/MQL/MASE/MAE/RMSE/RMSSE/WAPE/SMAPE/MAPE) with per-quantile breakdowns, calibration diagnostics, peak GPU memory, wall/inference/load times, and a full model card (params, bytes, effective bits/param, module inventory) replicated into summary columns. Raw per-series predictions + ground truth persisted per task as parquet → any future metric computable offline. Tracking contract: paper/draft.md Appendix D.
- **Paper draft v0.1** (paper/draft.md): structure follows multi-technique quantization studies (arXiv:2402.16775, arXiv:2511.08093, GPTQ/AWQ conventions); 7 tables + 5 figures fully specified with data provenance.
- **P1 profiler** (scripts/profile.py): ctx 2048 / h 64 protocol, batch {1, 32, 256}, warm latency percentiles, throughput, peak memory, deterministic seeded inputs.
- **Infrastructure failures found & fixed during full-run bring-up:**
  1. `datasets` 4.x removed script-dataset support → ETTh/ETTm (chronos_datasets_extra) unloadable. Pinned `datasets>=3.6,<4` + `HF_DATASETS_TRUST_REMOTE_CODE=1`.
  2. The prepared-dataset cache is **not compatible across datasets major versions** (4.x writes `List` feature types 3.x can't parse) → moved to a project-local cache (`data/hf_datasets_cache`, set in `ensure_truststore`) to prevent silent cross-version poisoning.
  3. `failures.jsonl` (append-mode) leaked across overwritten runs → `overwrite` now rmtree's the run dir.
  4. `transformers` lazy-import failure (`AutoModelForCausalLM`) that killed every Chronos-2 full-run attempt: **`scripts/profile.py` shadowed Python's stdlib `profile` module** (script dir heads `sys.path`), and the transformers import chain trips over it, surfacing as a masked lazy-import error. Renamed to `scripts/profile_model.py`; the eager `import chronos` in evaluate.py (added as fail-fast insurance) is what turned a mid-run 27-task failure into an instant, diagnosable crash. Rule: never name scripts after stdlib modules.
  5. `HF_DATASETS_CACHE`/trust env vars were originally set in `ensure_truststore()`, which runs *after* `import datasets` reads them → moved verification of env ordering into the entry-point import sequence (env vars are set before `datasets` import via `ensure_truststore` being called at script top / package import).

## 2026-07-22 — Evaluation framework built & validated (P0 complete)

- Built `src/chronosquant/evaluation/` on `fev` 0.9 (the harness the Chronos-2 authors use): config-driven runner with provenance (git SHA, env, hardware), incremental persistence, per-task failure isolation; predictors for Seasonal Naive and Chronos-2 (with a `model_transform` hook where quantization will plug in); `analysis/` aggregation reusing fev's exact skill-score/win-rate/bootstrap code, plus `retention_table` and `validate_against_reference`.
- Added quantization-specific diagnostics as fev-compatible metrics: **MACE** (interval-coverage calibration error) and **QCR** (quantile crossing rate).
- **Reproduction gate passed:** Seasonal Naive matches the published fev per-task MASE/WQL exactly (<1e-6 rel.) on the dev subset; enforced by a network-marked pytest (`pytest -m network`).
- Chronos-2 fp32 dev run: 83% win rate / 0.383 WQL skill, above chronos_bolt_base/small published anchors — ordering consistent with the Chronos-2 paper.
- **Findings:** (1) fev's `dataset_fingerprint` changed algorithm between 0.6 and 0.9 → cross-version fingerprint comparison is invalid; validation is version-aware. (2) `datasets` multiprocess map is flaky on Windows with CUDA initialized → dataset preprocessing pinned to `num_proc=1`. (3) fp32 Chronos-2 already emits crossed quantiles on covid_deaths (QCR 0.178) → quantized diagnostics must be baselined against fp32, not against zero.
- 41 tests total: analytic closed-form checks (SN MASE ≡ 1.0 on trend series), hand-computed leaderboard math, runner determinism/failure-isolation, reference parity.

## 2026-07-22 — Literature review & evaluation plan

- Deep research sweep (3 parallel agents): Chronos-2 internals, TSFM benchmark landscape, quantization methodology. Distilled into [RELATED_WORK.md](RELATED_WORK.md).
- Key findings: Chronos-2 is encoder-only (not enc-dec), fp32, no input tokenization; **no peer-reviewed quantization of any Chronos-family model exists** → novelty gap. Community artifacts: `kashif/chronos-2-onnx` (INT8, unevaluated), tsfm.ai MLX blog (int8 ≈ free; **int4 degrades PI coverage** — our central hypothesis to test).
- Evaluation framework drafted: [EVALUATION_PLAN.md](EVALUATION_PLAN.md) — fev-bench (primary), GIFT-Eval (secondary), Chronos Benchmark II (sweep workhorse); WQL/SQL/MASE retention + calibration + per-module sensitivity + efficiency Pareto.
- Added `fev` as the primary harness dependency.

## 2026-07-22 — Project initialized

- Created repository structure (src layout, configs, results, paper).
- Environment: Windows 11, Python 3.12 (uv-managed), NVIDIA RTX 2000 Ada (8 GB, driver 596.58), CUDA 12.8 torch wheels.
- Installed Chronos-2 (`chronos-forecasting`) and verified inference.
- Drafted evaluation framework plan from literature review (see [EVALUATION_PLAN.md](EVALUATION_PLAN.md)).
