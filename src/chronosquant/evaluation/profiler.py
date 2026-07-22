"""Efficiency profiler (P1): the systems half of the paper's efficiency table.

Measures, for one model variant on one device:

- model card: params, bytes, effective bits/param, load time
- batch-1 latency (median / p90 / mean / std over warm repetitions)
- throughput (series/s) at configurable batch sizes
- peak GPU memory per batch size; process RSS delta (best-effort, psutil)

Protocol follows the Chronos-2 paper's convention (context 2048, horizon 64) and the
quantized-LLM reporting checklist (named hardware + backend, warm measurements, medians
with dispersion). Deterministic synthetic input (seeded) so all variants are profiled on
identical data. Absolute numbers are hardware-specific; ratios between variants profiled
on the same machine are the transferable claim.
"""

import dataclasses
import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np

from chronosquant.evaluation.predictors import Chronos2Predictor
from chronosquant.utils import REPO_ROOT, environment_info

QUANTILE_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


@dataclasses.dataclass
class ProfilerConfig:
    context_length: int = 2048  # Chronos-2 paper throughput convention
    horizon: int = 64
    batch_sizes: tuple[int, ...] = (1, 32, 256)
    n_warmup: int = 3
    n_reps_batch1: int = 20
    n_reps_batched: int = 5
    seed: int = 42

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def latency_stats(times_s: list[float]) -> dict[str, float]:
    """Summary statistics for a list of per-call wall-clock times (seconds)."""
    if not times_s:
        raise ValueError("times_s must be non-empty")
    times = sorted(times_s)
    return {
        "n_reps": len(times),
        "median_s": statistics.median(times),
        "p90_s": times[min(len(times) - 1, int(round(0.9 * (len(times) - 1))))],
        "mean_s": statistics.fmean(times),
        "std_s": statistics.stdev(times) if len(times) > 1 else 0.0,
        "min_s": times[0],
    }


def _synthetic_contexts(batch_size: int, context_length: int, seed: int) -> list:
    """Deterministic synthetic series (trend + daily seasonality + noise), float32.

    Same seed -> identical profiling inputs for every model variant.
    """
    import torch

    rng = np.random.default_rng(seed)
    t = np.arange(context_length, dtype=np.float32)
    contexts = []
    for i in range(batch_size):
        values = (
            50.0
            + 0.01 * t
            + 10.0 * np.sin(2 * np.pi * (t + 17 * i) / 24)
            + rng.normal(0, 1.0, context_length).astype(np.float32)
        )
        contexts.append(torch.as_tensor(values, dtype=torch.float32))
    return contexts


def profile_predictor(predictor: Chronos2Predictor, config: ProfilerConfig) -> dict[str, Any]:
    """Profile one model variant on the device it is configured for."""
    import torch

    device_is_cuda = str(predictor.device_map).startswith("cuda") and torch.cuda.is_available()

    try:
        import psutil

        process = psutil.Process()
        rss_before_load_mb = process.memory_info().rss / 1e6
    except ImportError:
        process = None
        rss_before_load_mb = None

    pipeline = predictor.pipeline  # triggers (timed) load

    def _sync():
        if device_is_cuda:
            torch.cuda.synchronize()

    def _run(contexts):
        pipeline.predict_quantiles(
            contexts,
            prediction_length=config.horizon,
            quantile_levels=QUANTILE_GRID,
            batch_size=max(config.batch_sizes),
        )

    results_per_batch: dict[str, Any] = {}
    for batch_size in config.batch_sizes:
        contexts = _synthetic_contexts(batch_size, config.context_length, config.seed)
        n_reps = config.n_reps_batch1 if batch_size == 1 else config.n_reps_batched

        for _ in range(config.n_warmup):
            _run(contexts)
        _sync()

        if device_is_cuda:
            torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(n_reps):
            start = time.perf_counter()
            _run(contexts)
            _sync()
            times.append(time.perf_counter() - start)

        stats = latency_stats(times)
        stats["throughput_series_per_s"] = batch_size / stats["median_s"]
        if device_is_cuda:
            stats["peak_gpu_memory_mb"] = round(torch.cuda.max_memory_allocated() / 1e6, 1)
        if process is not None:
            stats["process_rss_mb"] = round(process.memory_info().rss / 1e6, 1)
        results_per_batch[str(batch_size)] = stats

    from chronosquant.models.inspect import model_summary

    return {
        "name": predictor.name,
        "model_id": predictor.model_id,
        "device": predictor.device_map,
        "backend": "torch-" + ("cuda" if device_is_cuda else "cpu"),
        "torch_num_threads": torch.get_num_threads(),
        "config": config.to_dict(),
        "model": {
            **model_summary(pipeline.model),
            "load_time_s": predictor._load_time_s,
            "rss_before_load_mb": rss_before_load_mb,
        },
        "latency": results_per_batch,
        "environment": environment_info(),
    }


def save_profile(profile: dict[str, Any], output_dir: str | Path = "results/profiles") -> Path:
    output_dir = Path(output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device_slug = str(profile["device"]).replace(":", "")
    path = output_dir / f"{profile['name']}_{device_slug}.json"
    with open(path, "w") as f:
        json.dump(profile, f, indent=2)
    return path
