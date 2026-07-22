"""Quantization-specific forecast diagnostics, implemented as `fev`-compatible metrics.

Standard accuracy metrics (MASE, WQL, SQL) come from `fev.metrics`. This module adds
diagnostics that the quantized-LLM literature shows are necessary beyond aggregate
accuracy (`Accuracy is Not All You Need`, arXiv:2407.09141) and that probabilistic
forecasting requires specifically:

- ``MACE``  — mean absolute coverage error of central prediction intervals. Empirical
  evidence on quantized TSFMs (tsfm.ai MLX experiments) shows interval *calibration*
  degrades before point accuracy does, making this the early-warning metric for
  quantization damage.
- ``QCR``   — quantile crossing rate. A direct-quantile-head model (Chronos-2 emits
  monotone quantiles in fp32) can start emitting crossed quantiles under low-bit
  weights; this measures that structural failure.
"""

import numpy as np
from fev.metrics import Metric


def central_interval_pairs(quantile_levels: list[float]) -> list[tuple[float, float]]:
    """Return symmetric (lower, upper) quantile pairs forming central intervals.

    E.g. for the standard 9-quantile grid {0.1, ..., 0.9} this yields
    (0.1, 0.9), (0.2, 0.8), (0.3, 0.7), (0.4, 0.6) with nominal coverages
    0.8, 0.6, 0.4, 0.2.
    """
    levels = sorted(quantile_levels)
    pairs = []
    for lo in levels:
        hi = round(1.0 - lo, 10)
        if lo < 0.5 and any(np.isclose(hi, level) for level in levels):
            pairs.append((lo, hi))
    return pairs


class MACE(Metric):
    """Mean absolute coverage error over symmetric central prediction intervals.

    For each symmetric quantile pair (q, 1-q) present in `quantile_levels`, computes the
    empirical coverage ``P(q_pred_lo <= y <= q_pred_hi)`` over all finite ground-truth
    points and reports the mean absolute deviation from nominal coverage ``(1 - 2q)``.

    `compute_scores` additionally reports each interval's empirical coverage as
    ``coverage[<nominal>]`` (e.g. ``coverage[0.8]``), enabling calibration curves.
    """

    needs_quantiles: bool = True

    def _coverages(
        self,
        y_true: np.ndarray,
        q_pred: np.ndarray,
        quantile_levels: list[float],
    ) -> dict[float, float]:
        """Empirical coverage per nominal level. y_true [N,H,D], q_pred [N,H,D,Q]."""
        pairs = central_interval_pairs(quantile_levels)
        if not pairs:
            raise ValueError(
                f"MACE requires at least one symmetric quantile pair; got {quantile_levels}"
            )
        levels = sorted(quantile_levels)
        finite = np.isfinite(y_true)
        coverages: dict[float, float] = {}
        for lo, hi in pairs:
            lo_idx, hi_idx = levels.index(lo), levels.index(hi)
            inside = (q_pred[..., lo_idx] <= y_true) & (y_true <= q_pred[..., hi_idx])
            coverages[round(hi - lo, 10)] = float(inside[finite].mean())
        return coverages

    def compute(
        self,
        *,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_past: np.ndarray,
        y_past_lengths: np.ndarray,
        q_pred: np.ndarray,
        seasonality: int,
        quantile_levels: list[float],
    ) -> float:
        coverages = self._coverages(y_true, q_pred, quantile_levels)
        return float(np.mean([abs(cov - nominal) for nominal, cov in coverages.items()]))

    def compute_scores(
        self,
        *,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_past: np.ndarray,
        y_past_lengths: np.ndarray,
        q_pred: np.ndarray,
        seasonality: int,
        quantile_levels: list[float],
        per_quantile_scores: bool = False,
    ) -> dict[str, float]:
        coverages = self._coverages(y_true, q_pred, quantile_levels)
        scores = {
            self.name: float(np.mean([abs(cov - nominal) for nominal, cov in coverages.items()]))
        }
        for nominal in sorted(coverages, reverse=True):
            scores[f"coverage[{nominal}]"] = coverages[nominal]
        return scores


class QCR(Metric):
    """Quantile crossing rate.

    Fraction of forecast points (series x horizon step x target dim) where the predicted
    quantile vector is not monotone non-decreasing in the quantile level. A well-formed
    probabilistic forecast has QCR = 0.
    """

    needs_quantiles: bool = True

    def compute(
        self,
        *,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_past: np.ndarray,
        y_past_lengths: np.ndarray,
        q_pred: np.ndarray,
        seasonality: int,
        quantile_levels: list[float],
    ) -> float:
        if q_pred.shape[-1] < 2:
            raise ValueError("QCR requires at least two quantile levels")
        # quantile_levels are sorted by fev.Task, so q_pred's last axis is level-ordered
        crossed = (np.diff(q_pred, axis=-1) < 0).any(axis=-1)  # [N, H, D]
        return float(crossed.mean())


#: Diagnostics computed by the runner for every evaluated model (in addition to the
#: task's own eval_metric / extra_metrics).
DEFAULT_DIAGNOSTICS: list[Metric] = [MACE(), QCR()]
