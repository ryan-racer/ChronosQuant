# Research Log

Dated log of experiments, decisions, and findings. Newest entries first.

## 2026-07-22 — Track-everything harness, paper draft, P1 profiler; full-run infrastructure debugging

- **Harness now captures ~90 columns/task**: full fev metric suite (WQL/SQL/MQL/MASE/MAE/RMSE/RMSSE/WAPE/SMAPE/MAPE) with per-quantile breakdowns, calibration diagnostics, peak GPU memory, wall/inference/load times, and a full model card (params, bytes, effective bits/param, module inventory) replicated into summary columns. Raw per-series predictions + ground truth persisted per task as parquet → any future metric computable offline. Tracking contract: paper/draft.md Appendix D.
- **Paper draft v0.1** (paper/draft.md): structure follows multi-technique quantization studies (arXiv:2402.16775, arXiv:2511.08093, GPTQ/AWQ conventions); 7 tables + 5 figures fully specified with data provenance.
- **P1 profiler** (scripts/profile.py): ctx 2048 / h 64 protocol, batch {1, 32, 256}, warm latency percentiles, throughput, peak memory, deterministic seeded inputs.
- **Infrastructure failures found & fixed during full-run bring-up:**
  1. `datasets` 4.x removed script-dataset support → ETTh/ETTm (chronos_datasets_extra) unloadable. Pinned `datasets>=3.6,<4` + `HF_DATASETS_TRUST_REMOTE_CODE=1`.
  2. The prepared-dataset cache is **not compatible across datasets major versions** (4.x writes `List` feature types 3.x can't parse) → moved to a project-local cache (`data/hf_datasets_cache`, set in `ensure_truststore`) to prevent silent cross-version poisoning.
  3. `failures.jsonl` (append-mode) leaked across overwritten runs → `overwrite` now rmtree's the run dir.
  4. Transient `transformers` lazy-import failure (`AutoModelForCausalLM`) when heavy processes ran concurrently with an eval run → eager `import chronos` in scripts/evaluate.py (fail-fast) + operational rule: keep the machine quiet during timed runs.

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
