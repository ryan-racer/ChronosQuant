# Research Log

Dated log of experiments, decisions, and findings. Newest entries first.

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
