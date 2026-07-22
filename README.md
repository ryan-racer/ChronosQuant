# ChronosQuant

**Quantizing the Chronos-2 time series foundation model: methods, evaluation framework, and accuracy/efficiency benchmarks.**

Chronos-2 (Amazon) is a state-of-the-art zero-shot time series foundation model. This project studies how post-training quantization (PTQ) at various bit-widths affects its forecasting quality and inference efficiency, and provides a reproducible evaluation harness for compressed time series foundation models (TSFMs).

## Research questions

1. **RQ1 — Accuracy retention:** How much probabilistic (WQL/CRPS) and point (MASE) forecasting accuracy does Chronos-2 retain under W8A8, W8A16, W4A16, … quantization, evaluated zero-shot on standard benchmarks?
2. **RQ2 — Method comparison:** Which quantization families (weight-only round-to-nearest, activation-aware PTQ, FP8) work best for Chronos-2's architecture?
3. **RQ3 — Sensitivity:** Which components (input embedding/patching, attention, FFN, output/quantile head) are most sensitive to reduced precision?
4. **RQ4 — Efficiency:** What are the realized latency, throughput, and memory gains on GPU and CPU, and where is the accuracy/efficiency Pareto frontier?

## Repository layout

```
├── src/chronosquant/       # Library code
│   ├── models/             #   Model loading & quantized model wrappers
│   ├── quantization/       #   Quantization method implementations/adapters
│   ├── evaluation/         #   Benchmark harness, metrics, efficiency profiling
│   └── analysis/           #   Result aggregation, tables, plots
├── scripts/                # CLI entry points (quantize, evaluate, profile)
├── configs/                # YAML configs for quantization & evaluation runs
├── tests/                  # Unit tests
├── notebooks/              # Exploratory analysis
├── data/                   # Dataset cache (gitignored)
├── results/                # raw/ (gitignored) · tables/ · figures/
├── docs/                   # Research plan & evaluation protocol
└── paper/                  # Manuscript sources & camera-ready figures
```

## Setup

Requires Python ≥ 3.11 and [uv](https://docs.astral.sh/uv/). On Windows with an NVIDIA GPU, CUDA-enabled PyTorch is pulled automatically (see `[tool.uv.sources]` in `pyproject.toml`).

```bash
uv sync --all-extras
```

> **Corporate-proxy note:** this project enables `system-certs` for uv (see `pyproject.toml`) and calls `truststore.inject_into_ssl()` in scripts so that PyPI/HuggingFace downloads work behind TLS-intercepting proxies.

Smoke-test Chronos-2 inference:

```bash
uv run python scripts/smoke_test.py
```

## Running evaluations

Evaluations are config-driven and produce [fev](https://github.com/autogluon/fev)-format
per-task summaries (directly comparable with published reference results):

```bash
# Run a model over a benchmark (Chronos Benchmark II configs are vendored)
uv run python scripts/evaluate.py configs/evaluation/runs/chronos2_fp32_dev.yaml

# Leaderboard (fev-bench methodology: skill score + win rate + bootstrap CIs)
uv run python scripts/aggregate.py leaderboard results/raw/chronos2_fp32_dev \
    results/reference/chronos_zeroshot/seasonal_naive.csv --metric WQL --missing-strategy drop

# Accuracy retention of quantized variants vs the fp32 parent
uv run python scripts/aggregate.py retention results/raw/* --reference chronos2-fp32

# Reproduction gate: verify our harness matches published per-task numbers
uv run python scripts/aggregate.py validate results/raw/seasonal_naive_dev \
    --reference-csv results/reference/chronos_zeroshot/seasonal_naive.csv
```

Tests: `uv run pytest` (offline, synthetic data) and `uv run pytest -m network`
(reference-parity gate against published results; downloads small HF datasets).

## Documentation

- [docs/EVALUATION_PLAN.md](docs/EVALUATION_PLAN.md) — the evaluation framework: benchmarks, datasets, metrics, protocols, and efficiency measurement methodology, grounded in the recent TSFM and quantization literature.
- [docs/RESEARCH_LOG.md](docs/RESEARCH_LOG.md) — dated log of experiments and decisions.

## Reproducibility

- Environment is locked via `uv.lock`; all experiments are driven by YAML configs under `configs/` and log their config + git SHA into `results/raw/`.
- Camera-ready tables/figures under `results/tables/` and `results/figures/` are tracked in git.

## Citation

See [CITATION.cff](CITATION.cff). This project builds on [Chronos-2](https://github.com/amazon-science/chronos-forecasting) (Apache-2.0).

## License

Apache-2.0 — see [LICENSE](LICENSE).
