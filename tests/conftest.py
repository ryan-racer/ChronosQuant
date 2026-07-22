"""Shared fixtures: offline synthetic datasets (no network, no model downloads)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fev import Task

QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
SEASONALITY = 24
HORIZON = 12
HISTORY_LENGTH = 96
NUM_SERIES = 3


def _write_long_parquet(path: Path, frames: list[pd.DataFrame]) -> Path:
    pd.concat(frames, ignore_index=True).to_parquet(path)
    return path


@pytest.fixture(scope="session")
def trend_dataset_path(tmp_path_factory) -> Path:
    """Deterministic linear-trend series: y_i[t] = t + 1000*i.

    Constructed so that metrics have closed-form values:
    - in-sample seasonal error   = SEASONALITY (all seasonal diffs equal the period)
    - seasonal-naive error       = SEASONALITY at every horizon step
    - hence seasonal-naive MASE  = 1.0 exactly.
    """
    frames = []
    for i in range(NUM_SERIES):
        t = np.arange(HISTORY_LENGTH, dtype=np.float64)
        frames.append(
            pd.DataFrame(
                {
                    "id": f"series_{i}",
                    "timestamp": pd.date_range("2024-01-01", periods=HISTORY_LENGTH, freq="h"),
                    "target": t + 1000.0 * i,
                }
            )
        )
    return _write_long_parquet(tmp_path_factory.mktemp("data") / "trend.parquet", frames)


@pytest.fixture(scope="session")
def noisy_dataset_path(tmp_path_factory) -> Path:
    """Seasonal + noise series for non-degenerate probabilistic behaviour."""
    rng = np.random.default_rng(42)
    frames = []
    for i in range(NUM_SERIES):
        t = np.arange(HISTORY_LENGTH, dtype=np.float64)
        values = (
            50.0
            + 10.0 * np.sin(2 * np.pi * t / SEASONALITY)
            + rng.normal(0, 1.0, HISTORY_LENGTH)
        )
        frames.append(
            pd.DataFrame(
                {
                    "id": f"series_{i}",
                    "timestamp": pd.date_range("2024-01-01", periods=HISTORY_LENGTH, freq="h"),
                    "target": values,
                }
            )
        )
    return _write_long_parquet(tmp_path_factory.mktemp("data") / "noisy.parquet", frames)


def make_task(dataset_path: Path, horizon: int = HORIZON, **kwargs) -> Task:
    defaults = dict(
        dataset_path=str(dataset_path),
        horizon=horizon,
        seasonality=SEASONALITY,
        eval_metric="MASE",
        extra_metrics=["WQL"],
        quantile_levels=QUANTILE_LEVELS,
    )
    defaults.update(kwargs)
    return Task(**defaults)


@pytest.fixture()
def trend_task(trend_dataset_path) -> Task:
    return make_task(trend_dataset_path)


@pytest.fixture()
def noisy_task(noisy_dataset_path) -> Task:
    return make_task(noisy_dataset_path)
