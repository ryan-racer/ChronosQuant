# Research Log

Dated log of experiments, decisions, and findings. Newest entries first.

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
