"""Smoke test: load Chronos-2 and produce a zero-shot probabilistic forecast.

Usage:
    uv run python scripts/smoke_test.py [--device cuda|cpu] [--model amazon/chronos-2]
"""

import argparse
import time

import numpy as np
import pandas as pd
import torch
import truststore

# Trust the OS cert store (needed behind corporate TLS interception); must run
# before any HuggingFace Hub client is created.
truststore.inject_into_ssl()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="amazon/chronos-2")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--prediction-length", type=int, default=24)
    args = parser.parse_args()

    print(f"torch {torch.__version__} | cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"device: {torch.cuda.get_device_name(0)}")

    from chronos import Chronos2Pipeline

    t0 = time.perf_counter()
    pipeline = Chronos2Pipeline.from_pretrained(args.model, device_map=args.device)
    print(f"loaded {args.model} ({type(pipeline).__name__}) in {time.perf_counter() - t0:.1f}s")

    n_params = sum(p.numel() for p in pipeline.model.parameters())
    dtypes = {str(p.dtype) for p in pipeline.model.parameters()}
    print(f"parameters: {n_params / 1e6:.1f}M | dtypes: {dtypes}")

    # Synthetic seasonal series: daily pattern + trend + noise
    rng = np.random.default_rng(0)
    t = np.arange(512)
    context = 10 + 0.01 * t + 2 * np.sin(2 * np.pi * t / 24) + 0.3 * rng.standard_normal(len(t))

    # Tensor API
    t0 = time.perf_counter()
    quantiles, mean = pipeline.predict_quantiles(
        [torch.tensor(context, dtype=torch.float32)],
        prediction_length=args.prediction_length,
        quantile_levels=[0.1, 0.5, 0.9],
    )
    q = quantiles[0]
    dt = time.perf_counter() - t0
    print(f"tensor API forecast in {dt:.2f}s | quantiles shape: {tuple(q.shape)}")
    assert torch.isfinite(q).all(), "non-finite forecast values"

    # DataFrame API (the interface the eval harness will use)
    df = pd.DataFrame(
        {
            "item_id": "series_0",
            "timestamp": pd.date_range("2025-01-01", periods=len(t), freq="h"),
            "target": context,
        }
    )
    t0 = time.perf_counter()
    pred_df = pipeline.predict_df(
        df,
        prediction_length=args.prediction_length,
        quantile_levels=[0.1, 0.5, 0.9],
    )
    print(f"predict_df forecast in {time.perf_counter() - t0:.2f}s")
    print(pred_df.head(3))
    print("\nSmoke test PASSED")


if __name__ == "__main__":
    main()
