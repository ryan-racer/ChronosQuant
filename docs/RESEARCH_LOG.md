# Research Log

Dated log of experiments, decisions, and findings. Newest entries first.

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
