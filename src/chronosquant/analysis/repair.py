"""Post-hoc repair of quantized-model forecast distributions (offline, CPU-only).

Operates on the raw per-series prediction parquets persisted by the evaluation runner
(``results/raw/<run>/predictions/<task>.parquet``: long format, one row per
(window_idx, item_id, step) with ``y_true`` and one column per quantile level), so no
model re-runs are needed. Two repairs are implemented:

1. **Quantile rearrangement** (Chernozhukov, Fernandez-Val & Galichon, arXiv:0704.3649):
   sort the predicted quantile vector at every (series, horizon) point. Sorting provably
   never increases pinball loss at any level pair aggregate (Fakoor et al., JMLR 2023)
   and makes the quantile crossing rate exactly zero. Comparing WQL/MACE/coverage before
   vs after separates *cosmetic* crossing (reordered but nearly-coincident quantiles)
   from *structural* distribution damage.

2. **Split-conformal recalibration** (CQR-style per-level additive offsets,
   Romano et al. 2019). Exact protocol, chosen to be leakage-clean:

   - Within each task, the set of **series** (unique ``item_id``) — not time steps — is
     randomly split into two halves A/B with a fixed seed (default ``20260730``),
     so calibration and evaluation never share a series.
   - On the calibration half, for each nominal quantile level ``tau``, residuals
     ``r = y_true - q_hat_tau`` are pooled across all series and all horizon steps
     (horizon-agnostic pooling: one offset per level per task; uniform protocol across
     tasks whose calibration halves range from ~7 to ~50k series). Non-finite residuals
     (missing ground truth) are dropped.
   - The additive offset is the split-conformal empirical quantile of the residuals:
     the ``k``-th order statistic with ``k = ceil((n+1) * tau)`` for ``tau >= 0.5`` and
     ``k = floor((n+1) * tau)`` for ``tau < 0.5`` (clipped to ``[1, n]``) — the
     one-sided finite-sample-valid choice on each tail, reducing to the plain empirical
     quantile as ``n`` grows.
   - Offsets fitted on A are applied to B and vice versa (2-fold swap), so **every
     series is evaluated exactly once**, always with offsets fitted on the other half.
     Metrics are computed on the pooled recalibrated predictions of all series (same
     point set as the unrepaired metrics, keeping numbers directly comparable).
   - Quantile rearrangement (repair 1) is applied **after** recalibration: per-level
     offsets can themselves induce crossing (MultiQT, arXiv:2512.23671).

Metric conventions are reused verbatim from the evaluation stack (``fev.metrics.WQL``,
``chronosquant.evaluation.metrics.MACE/QCR``); multi-window runs are handled like the
runner handles them (metrics per evaluation window, arithmetic mean across windows).
WQL retention aggregation follows the campaign convention: per-task ratio vs the fp32
reference (its *unrepaired* WQL, the campaign's fixed reference), clipped to
[1e-2, 1e2], geometric mean over tasks, percentile-bootstrap 95% CI. MACE / coverage /
QCR aggregate as plain arithmetic means over tasks.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd
import scipy.stats
from fev.metrics import WQL

from chronosquant.evaluation.metrics import MACE, QCR

__all__ = [
    "DEFAULT_SEED",
    "quantile_level_columns",
    "pinball_loss",
    "rearrange",
    "forecast_metrics",
    "split_series",
    "conformal_offsets",
    "conformal_repair",
    "analyze_task",
    "aggregate_repair",
]

#: Fixed seed for the series-level calibration/evaluation split (2-fold swap makes the
#: evaluated set independent of the seed; the seed only fixes which half calibrates which).
DEFAULT_SEED = 20260730

#: fev-bench clipping bounds for relative errors (campaign convention).
MIN_REL, MAX_REL = 1e-2, 1e2
N_RESAMPLES = 1000

#: Metric stages produced by `analyze_task`.
STAGES = ("before", "sorted", "conformal")


# ---------------------------------------------------------------------------
# Parquet schema helpers
# ---------------------------------------------------------------------------

def quantile_level_columns(frame: pd.DataFrame) -> list[str]:
    """Columns of `frame` that are quantile levels (float strings in (0, 1)), sorted."""
    cols = []
    for col in frame.columns:
        try:
            level = float(col)
        except (TypeError, ValueError):
            continue
        if 0.0 < level < 1.0:
            cols.append(col)
    if len(cols) < 2:
        raise ValueError(f"Expected >=2 quantile columns, found {cols}")
    return sorted(cols, key=float)


# ---------------------------------------------------------------------------
# Repair 1: quantile rearrangement
# ---------------------------------------------------------------------------

def rearrange(q_pred: np.ndarray) -> np.ndarray:
    """Chernozhukov rearrangement: sort the quantile vector along the last axis."""
    return np.sort(q_pred, axis=-1)


def pinball_loss(y_true: np.ndarray, q_pred: np.ndarray, levels: Sequence[float]) -> float:
    """Mean pinball (quantile) loss over all finite points and levels.

    Same elementwise formula as ``fev.metrics._quantile_loss`` (factor 2 included).
    y_true [P], q_pred [P, Q].
    """
    lev = np.asarray(levels, dtype=np.float64)
    y = y_true[:, None]
    ql = 2.0 * np.abs((y - q_pred) * ((y <= q_pred) - lev[None, :]))
    return float(np.nanmean(ql))


# ---------------------------------------------------------------------------
# Metrics (reusing the evaluation stack's implementations exactly)
# ---------------------------------------------------------------------------

def _window_metrics(
    y_true: np.ndarray, q_pred: np.ndarray, levels: list[float]
) -> dict[str, float]:
    """WQL / MACE / coverage[*] / QCR for one evaluation window. y [P], q [P, Q]."""
    kwargs = dict(
        y_true=y_true.reshape(-1, 1, 1),
        y_pred=None,
        y_past=None,
        y_past_lengths=None,
        q_pred=q_pred.reshape(-1, 1, 1, len(levels)),
        seasonality=1,
        quantile_levels=list(levels),
    )
    scores = MACE().compute_scores(**kwargs)
    scores["QCR"] = QCR().compute(**kwargs)
    scores["WQL"] = WQL().compute(**kwargs)
    scores["pinball"] = pinball_loss(y_true, q_pred, levels)
    return scores


def forecast_metrics(
    y_true: np.ndarray,
    q_pred: np.ndarray,
    levels: list[float],
    window_idx: np.ndarray | None = None,
) -> dict[str, float]:
    """Metrics matching the runner's convention: per evaluation window, then averaged.

    ``window_idx`` may be omitted for single-window data (all full-tier runs).
    """
    if window_idx is None or len(np.unique(window_idx)) == 1:
        return _window_metrics(y_true, q_pred, levels)
    per_window: dict[str, list[float]] = {}
    for w in np.unique(window_idx):
        mask = window_idx == w
        for key, value in _window_metrics(y_true[mask], q_pred[mask], levels).items():
            per_window.setdefault(key, []).append(value)
    return {key: float(np.mean(vals)) for key, vals in per_window.items()}


# ---------------------------------------------------------------------------
# Repair 2: split-conformal recalibration (protocol in the module docstring)
# ---------------------------------------------------------------------------

def split_series(item_ids: np.ndarray, seed: int = DEFAULT_SEED) -> tuple[np.ndarray, np.ndarray]:
    """Randomly split the unique series ids into two disjoint, exhaustive halves."""
    unique = np.unique(item_ids)
    if len(unique) < 2:
        raise ValueError("Conformal split requires at least 2 series")
    perm = np.random.default_rng(seed).permutation(len(unique))
    half = len(unique) // 2
    return unique[perm[:half]], unique[perm[half:]]


def conformal_offsets(
    y_cal: np.ndarray, q_cal: np.ndarray, levels: Sequence[float]
) -> np.ndarray:
    """Per-level additive offsets from calibration residuals (split-conformal quantile).

    For each level tau, the offset is the k-th order statistic of the residuals
    ``y - q_hat_tau`` with ``k = ceil((n+1)*tau)`` for tau >= 0.5 and
    ``k = floor((n+1)*tau)`` for tau < 0.5 (clipped to [1, n]): the one-sided
    finite-sample conformal choice on each tail. Non-finite residuals are dropped;
    a level with no finite residuals gets offset 0.
    """
    offsets = np.zeros(len(levels), dtype=np.float64)
    for j, tau in enumerate(levels):
        residuals = y_cal - q_cal[:, j]
        residuals = residuals[np.isfinite(residuals)]
        n = residuals.size
        if n == 0:
            continue
        if tau >= 0.5:
            k = min(n, math.ceil((n + 1) * tau))
        else:
            k = max(1, math.floor((n + 1) * tau))
        offsets[j] = np.sort(residuals)[k - 1]
    return offsets


def conformal_repair(
    frame: pd.DataFrame,
    levels_cols: Sequence[str] | None = None,
    seed: int = DEFAULT_SEED,
    sort_after: bool = True,
) -> np.ndarray:
    """2-fold split-conformal recalibration of a task's prediction frame.

    Returns the repaired quantile matrix [P, Q] aligned with ``frame``'s rows: every
    row is recalibrated with offsets fitted on the *other* series-half, then (by
    default) rearranged. See the module docstring for the full protocol.
    """
    if levels_cols is None:
        levels_cols = quantile_level_columns(frame)
    levels = [float(c) for c in levels_cols]
    y_true = frame["y_true"].to_numpy(dtype=np.float64)
    # Calibrate the SORTED object: fitting offsets on raw (possibly crossed) quantiles
    # and sorting afterwards would attach the conformal guarantee to a different object
    # than the one evaluated — the post-hoc sort widens the outer band and inflates
    # coverage beyond nominal on high-crossing arms. Sorting first makes the evaluated
    # object the calibrated one; the final sort only fixes rare offset-induced crossings.
    q_pred = rearrange(frame[list(levels_cols)].to_numpy(dtype=np.float64))

    # Integer-coded equivalent of `split_series` + `np.isin` (hash-based factorize with
    # sort=True reproduces np.unique's sorted order, so the split is identical but the
    # per-row membership test is O(P) integer indexing — matters on 800k-row tasks).
    codes, uniques = pd.factorize(frame["item_id"], sort=True)
    if (codes < 0).any():
        raise ValueError("item_id contains missing values; refusing a silent fold wrap")
    if len(uniques) < 2:
        raise ValueError("Conformal split requires at least 2 series")
    perm = np.random.default_rng(seed).permutation(len(uniques))
    in_a = np.zeros(len(uniques), dtype=bool)
    in_a[perm[: len(uniques) // 2]] = True
    mask_a = in_a[codes]

    repaired = np.full_like(q_pred, np.nan)
    written = np.zeros(len(q_pred), dtype=bool)
    for cal_mask, eval_mask in ((mask_a, ~mask_a), (~mask_a, mask_a)):
        offsets = conformal_offsets(y_true[cal_mask], q_pred[cal_mask], levels)
        assert not written[eval_mask].any(), "row repaired twice"
        repaired[eval_mask] = q_pred[eval_mask] + offsets[None, :]
        written[eval_mask] = True
    assert written.all(), "every row must be repaired exactly once"
    return rearrange(repaired) if sort_after else repaired


# ---------------------------------------------------------------------------
# Per-task analysis and cross-task aggregation
# ---------------------------------------------------------------------------

def analyze_task(frame: pd.DataFrame, seed: int = DEFAULT_SEED) -> dict[str, dict[str, float]]:
    """Compute metrics for one task at each repair stage.

    Returns ``{"before": {...}, "sorted": {...}, "conformal": {...}}`` where each inner
    dict has WQL, pinball, MACE, coverage[*], QCR. "conformal" = recalibration + sorting.
    """
    level_cols = quantile_level_columns(frame)
    levels = [float(c) for c in level_cols]
    y_true = frame["y_true"].to_numpy(dtype=np.float64)
    q_pred = frame[level_cols].to_numpy(dtype=np.float64)
    window_idx = frame["window_idx"].to_numpy() if "window_idx" in frame else None

    q_conformal = conformal_repair(frame, level_cols, seed=seed, sort_after=True)
    return {
        "before": forecast_metrics(y_true, q_pred, levels, window_idx),
        "sorted": forecast_metrics(y_true, rearrange(q_pred), levels, window_idx),
        "conformal": forecast_metrics(y_true, q_conformal, levels, window_idx),
    }


def _gmean_ci(ratios: np.ndarray, seed: int = 123) -> tuple[float, float, float]:
    """Geometric mean + 95% percentile-bootstrap CI (campaign convention)."""
    point = float(scipy.stats.gmean(ratios))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ratios), (N_RESAMPLES, len(ratios)))
    boot = scipy.stats.gmean(ratios[idx], axis=1)
    return point, float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def aggregate_repair(
    per_task: pd.DataFrame,
    reference_model: str = "chronos2-fp32",
    stages: Sequence[str] = STAGES,
) -> pd.DataFrame:
    """Aggregate per-task repair metrics into one row per model (retention-table style).

    ``per_task`` needs columns: model_name, task_name, stage, WQL, MACE, coverage[0.8],
    QCR. For each model and stage this reports ``WQL_rel_<stage>`` (+ 95% CI bounds):
    geometric mean over tasks of WQL / WQL(reference, stage="before") — the campaign's
    fixed fp32-unrepaired reference — plus arithmetic task-means of MACE, coverage[0.8]
    (as ``cov80``) and QCR.
    """
    ref = (
        per_task.query("model_name == @reference_model and stage == 'before'")
        .set_index("task_name")["WQL"]
    )
    if ref.empty:
        raise ValueError(f"Reference model {reference_model!r} (stage 'before') not found")
    # Same-stage reference for the repaired stages: repair also improves fp32 itself
    # (~0.4% WQL), so quantized-repaired vs fp32-UNrepaired flatters retention. The
    # `_samestage` columns divide by fp32 at the same repair stage.
    ref_by_stage = {
        stage: per_task.query("model_name == @reference_model and stage == @stage")
        .set_index("task_name")["WQL"]
        for stage in stages
    }

    rows = []
    for model_name, group in per_task.groupby("model_name", sort=False):
        row: dict[str, object] = {"model_name": model_name}
        for stage in stages:
            sub = group[group["stage"] == stage].set_index("task_name")
            if sub.empty:
                continue
            missing = ref.index.difference(sub.index)
            if len(missing) > 0:
                raise ValueError(f"{model_name}/{stage} missing tasks: {list(missing)}")
            ratios = (sub["WQL"] / ref).clip(MIN_REL, MAX_REL).to_numpy()
            point, lo, hi = _gmean_ci(ratios)
            row[f"WQL_rel_{stage}"] = point
            row[f"WQL_rel_{stage}_lower"] = lo
            row[f"WQL_rel_{stage}_upper"] = hi
            ref_same = ref_by_stage[stage]
            if stage != "before" and not ref_same.empty:
                same = (sub["WQL"] / ref_same).clip(MIN_REL, MAX_REL).to_numpy()
                row[f"WQL_rel_{stage}_samestage"] = _gmean_ci(same)[0]
            row[f"MACE_{stage}"] = float(sub["MACE"].mean())
            row[f"cov80_{stage}"] = float(sub["coverage[0.8]"].mean())
            row[f"QCR_{stage}"] = float(sub["QCR"].mean())
        row["n_tasks"] = int(group["task_name"].nunique())
        rows.append(row)
    table = pd.DataFrame(rows).set_index("model_name")
    return table.sort_values(f"WQL_rel_{stages[0]}")
