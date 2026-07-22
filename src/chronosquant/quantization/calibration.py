"""Calibration data generation and activation capture for calibrated PTQ methods.

Calibration series are **synthetic** by default (diverse trend/seasonal/AR/intermittent
mixtures, seeded): they exercise the model's input distribution without touching any
evaluation data (leakage hygiene), and calibration-source ablations are a planned P4
axis. Activation statistics are captured by forward hooks on named Linear modules while
the full pipeline (patching, normalization, arcsinh) runs its real preprocessing.
"""

from typing import Any

import numpy as np
import torch
from torch import nn


def synthetic_calibration_series(
    n_series: int = 128,
    context_length: int = 1024,
    seed: int = 123,
) -> list[torch.Tensor]:
    """Deterministic, diverse synthetic series for calibration.

    Mixes the canonical time-series regimes: trend + (multi-)seasonality + noise,
    random walks, AR(1), and intermittent count series, across widely varying scales
    (the pipeline's instance normalization sees scale diversity anyway, but arcsinh
    responds to it).
    """
    rng = np.random.default_rng(seed)
    t = np.arange(context_length, dtype=np.float64)
    series: list[torch.Tensor] = []
    for i in range(n_series):
        kind = i % 4
        scale = 10.0 ** rng.uniform(-1, 4)
        if kind == 0:  # trend + up to 2 seasonalities + noise
            period_1 = rng.choice([24, 12, 7, 52, 168])
            values = (
                rng.uniform(-0.05, 0.05) * t
                + rng.uniform(0.5, 3.0) * np.sin(2 * np.pi * t / period_1 + rng.uniform(0, 6))
                + rng.uniform(0, 1.5)
                * np.sin(2 * np.pi * t / (period_1 * rng.choice([2, 4])) + rng.uniform(0, 6))
                + rng.normal(0, rng.uniform(0.1, 1.0), context_length)
            )
        elif kind == 1:  # random walk with drift
            values = np.cumsum(rng.normal(rng.uniform(-0.1, 0.1), 1.0, context_length))
        elif kind == 2:  # AR(1)
            phi = rng.uniform(0.7, 0.99)
            noise = rng.normal(0, 1.0, context_length)
            values = np.zeros(context_length)
            for j in range(1, context_length):
                values[j] = phi * values[j - 1] + noise[j]
        else:  # intermittent counts (retail-like)
            values = rng.poisson(rng.uniform(0.2, 3.0), context_length).astype(np.float64)
        series.append(torch.as_tensor(values * scale, dtype=torch.float32))
    return series


class LinearInputStats:
    """Accumulates per-linear input statistics: Hessian H = sum(x x^T), abs-mean, samples."""

    def __init__(self, in_features: int, device: torch.device, max_samples: int = 512):
        self.hessian = torch.zeros(in_features, in_features, dtype=torch.float32, device=device)
        self.abs_sum = torch.zeros(in_features, dtype=torch.float32, device=device)
        self.n_rows = 0
        self.max_samples = max_samples
        self.samples: list[torch.Tensor] = []
        self._n_sampled = 0

    def update(self, x: torch.Tensor) -> None:
        x = x.detach().reshape(-1, x.shape[-1]).to(torch.float32)
        self.hessian += x.T @ x
        self.abs_sum += x.abs().sum(dim=0)
        self.n_rows += x.shape[0]
        if self._n_sampled < self.max_samples:
            take = min(self.max_samples - self._n_sampled, x.shape[0])
            self.samples.append(x[:take].clone())
            self._n_sampled += take

    @property
    def abs_mean(self) -> torch.Tensor:
        return self.abs_sum / max(self.n_rows, 1)

    @property
    def sample_matrix(self) -> torch.Tensor:
        return torch.cat(self.samples, dim=0) if self.samples else torch.empty(0)


@torch.no_grad()
def collect_linear_input_stats(
    pipeline,
    target_names: list[str],
    n_series: int = 128,
    context_length: int = 1024,
    horizon: int = 64,
    batch_size: int = 32,
    seed: int = 123,
) -> dict[str, LinearInputStats]:
    """Run calibration series through the pipeline, capturing input stats per Linear."""
    model = pipeline.model
    modules = dict(model.named_modules())
    missing = [n for n in target_names if n not in modules]
    if missing:
        raise ValueError(f"Target modules not found: {missing[:5]}")

    stats: dict[str, LinearInputStats] = {}
    hooks = []
    for name in target_names:
        module = modules[name]
        assert isinstance(module, nn.Linear), f"{name} is not nn.Linear"
        stats[name] = LinearInputStats(module.in_features, module.weight.device)

        def make_hook(stat: LinearInputStats):
            def hook(mod, args):
                stat.update(args[0])

            return hook

        hooks.append(module.register_forward_pre_hook(make_hook(stats[name])))

    try:
        series = synthetic_calibration_series(n_series, context_length, seed)
        pipeline.predict_quantiles(
            series,
            prediction_length=horizon,
            quantile_levels=[0.1, 0.5, 0.9],
            batch_size=batch_size,
        )
    finally:
        for hook in hooks:
            hook.remove()
    return stats


def calibration_config_summary(**kwargs: Any) -> dict[str, Any]:
    """Standardized record of calibration settings for run metadata."""
    return {f"calib_{k}": v for k, v in kwargs.items()}
