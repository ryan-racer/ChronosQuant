"""Synthetic-data tests for the post-hoc repair arm (sorting + split-conformal).

Proves the three properties the analysis relies on:

(a) quantile rearrangement never increases pinball loss and zeroes QCR;
(b) split-conformal per-level offsets restore coverage on a synthetically
    miscalibrated forecaster;
(c) the 2-fold swap covers every series exactly once in each direction and always
    applies offsets fitted on the *other* half.
"""

import numpy as np
import pandas as pd
import pytest
import scipy.stats

from chronosquant.analysis.repair import (
    DEFAULT_SEED,
    analyze_task,
    conformal_offsets,
    conformal_repair,
    forecast_metrics,
    pinball_loss,
    quantile_level_columns,
    rearrange,
    split_series,
)

LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
LEVEL_COLS = [str(q) for q in LEVELS]
Z = scipy.stats.norm.ppf(LEVELS)  # standard normal quantiles


def make_frame(y: np.ndarray, q: np.ndarray, item_ids: np.ndarray) -> pd.DataFrame:
    """Long-format frame matching the persisted parquet schema."""
    frame = pd.DataFrame({"window_idx": 0, "item_id": item_ids, "y_true": y})
    for j, col in enumerate(LEVEL_COLS):
        frame[col] = q[:, j]
    return frame


def synthetic_forecaster(
    n_series: int = 200, horizon: int = 10, scale: float = 1.0, bias: float = 0.0, seed: int = 0
) -> pd.DataFrame:
    """Gaussian data with quantile forecasts distorted by `scale` (shrink) and `bias`.

    True distribution per point: N(mu_i, 1) with a per-series mean mu_i. Forecast
    quantiles: mu_i + bias + scale * z_tau — perfectly calibrated iff scale=1, bias=0.
    """
    rng = np.random.default_rng(seed)
    mu = np.repeat(rng.normal(0, 5, n_series), horizon)
    y = mu + rng.standard_normal(n_series * horizon)
    q = mu[:, None] + bias + scale * Z[None, :]
    item_ids = np.repeat([f"T{i:04d}" for i in range(n_series)], horizon)
    return make_frame(y, q, item_ids)


class TestRearrangement:
    """(a) Sorting never increases pinball loss and zeroes QCR."""

    @pytest.mark.parametrize("seed", range(5))
    def test_pinball_never_increases_and_qcr_zero(self, seed):
        rng = np.random.default_rng(seed)
        y = rng.normal(0, 3, 500)
        q = rng.normal(0, 3, (500, len(LEVELS)))  # heavily crossed
        before = forecast_metrics(y, q, LEVELS)
        after = forecast_metrics(y, rearrange(q), LEVELS)
        assert before["QCR"] > 0.5  # the synthetic input really is crossed
        assert after["QCR"] == 0.0
        assert after["pinball"] <= before["pinball"] + 1e-12
        assert after["WQL"] <= before["WQL"] + 1e-12

    def test_pointwise_pinball_never_increases(self):
        """The rearrangement theorem holds at every single forecast point."""
        rng = np.random.default_rng(42)
        y = rng.normal(size=300)
        q = rng.normal(size=(300, len(LEVELS)))
        lev = np.asarray(LEVELS)

        def per_point(qm):
            return (2 * np.abs((y[:, None] - qm) * ((y[:, None] <= qm) - lev))).sum(axis=1)

        assert (per_point(rearrange(q)) <= per_point(q) + 1e-12).all()

    def test_monotone_input_is_fixed_point(self):
        q = np.cumsum(np.random.default_rng(1).uniform(0.1, 1, (50, len(LEVELS))), axis=1)
        np.testing.assert_array_equal(rearrange(q), q)
        assert forecast_metrics(np.zeros(50), q, LEVELS)["QCR"] == 0.0

    def test_pinball_matches_manual(self):
        y = np.array([1.0])
        q = np.array([[0.0] * 9])  # forecast 0 at all levels, y=1: loss = 2*tau*1
        assert pinball_loss(y, q, LEVELS) == pytest.approx(2 * np.mean(LEVELS))


class TestConformalOffsets:
    def test_recovers_known_bias(self):
        frame = synthetic_forecaster(n_series=2000, horizon=10, bias=5.0, seed=3)
        q = frame[LEVEL_COLS].to_numpy()
        offsets = conformal_offsets(frame["y_true"].to_numpy(), q, LEVELS)
        np.testing.assert_allclose(offsets, -5.0, atol=0.05)

    def test_recovers_shrunk_scale(self):
        frame = synthetic_forecaster(n_series=5000, horizon=10, scale=0.4, seed=4)
        q = frame[LEVEL_COLS].to_numpy()
        offsets = conformal_offsets(frame["y_true"].to_numpy(), q, LEVELS)
        np.testing.assert_allclose(offsets, (1 - 0.4) * Z, atol=0.05)

    def test_nan_residuals_dropped(self):
        y = np.array([np.nan, 0.0, 1.0, 2.0, np.nan])
        q = np.zeros((5, len(LEVELS)))
        offsets = conformal_offsets(y, q, LEVELS)
        assert np.isfinite(offsets).all()
        assert offsets[-1] == 2.0  # 0.9-quantile of {0,1,2}, conformal order statistic

    def test_all_nan_gives_zero_offsets(self):
        offsets = conformal_offsets(np.full(4, np.nan), np.zeros((4, 9)), LEVELS)
        np.testing.assert_array_equal(offsets, 0.0)


class TestConformalRepairRestoresCoverage:
    """(b) A synthetically miscalibrated forecaster is recalibrated to nominal coverage."""

    def test_overconfident_forecaster(self):
        frame = synthetic_forecaster(n_series=400, horizon=10, scale=0.4, seed=7)
        metrics = analyze_task(frame, seed=DEFAULT_SEED)
        # Shrunk intervals: nominal-80% covers P(|Z| <= 0.4*z_.9) ~ 0.39
        assert metrics["before"]["coverage[0.8]"] < 0.5
        assert metrics["conformal"]["coverage[0.8]"] == pytest.approx(0.8, abs=0.03)
        assert metrics["conformal"]["MACE"] < 0.03 < metrics["before"]["MACE"]
        assert metrics["conformal"]["QCR"] == 0.0

    def test_biased_forecaster(self):
        frame = synthetic_forecaster(n_series=400, horizon=10, bias=3.0, seed=8)
        metrics = analyze_task(frame)
        assert metrics["before"]["coverage[0.8]"] < 0.4
        assert metrics["conformal"]["coverage[0.8]"] == pytest.approx(0.8, abs=0.03)
        # Removing a 3-sigma bias must also slash WQL
        assert metrics["conformal"]["WQL"] < 0.5 * metrics["before"]["WQL"]

    def test_calibrated_forecaster_stays_calibrated(self):
        frame = synthetic_forecaster(n_series=400, horizon=10, seed=9)
        metrics = analyze_task(frame)
        assert metrics["conformal"]["coverage[0.8]"] == pytest.approx(
            metrics["before"]["coverage[0.8]"], abs=0.04
        )
        # Recalibrating an already-calibrated forecaster should not hurt WQL much
        assert metrics["conformal"]["WQL"] < 1.05 * metrics["before"]["WQL"]

    def test_sorting_applied_after_recalibration(self):
        frame = synthetic_forecaster(n_series=100, horizon=5, scale=0.4, seed=10)
        repaired = conformal_repair(frame, LEVEL_COLS, sort_after=True)
        assert (np.diff(repaired, axis=1) >= 0).all()


class TestFoldSwap:
    """(c) 2-fold swap: disjoint halves, every series evaluated exactly once, cross-fitted."""

    def test_split_is_disjoint_and_exhaustive(self):
        for n in (2, 3, 14, 101):
            ids = np.array([f"s{i}" for i in range(n)])
            half_a, half_b = split_series(ids, seed=DEFAULT_SEED)
            assert set(half_a) | set(half_b) == set(ids)
            assert set(half_a) & set(half_b) == set()
            assert abs(len(half_a) - len(half_b)) <= 1

    def test_split_deterministic_and_seed_sensitive(self):
        ids = np.array([f"s{i}" for i in range(20)])
        a1, _ = split_series(ids, seed=1)
        a2, _ = split_series(ids, seed=1)
        a3, _ = split_series(ids, seed=2)
        np.testing.assert_array_equal(a1, a2)
        assert set(a1) != set(a3)

    def test_single_series_raises(self):
        with pytest.raises(ValueError, match="at least 2 series"):
            split_series(np.array(["only"] * 10))

    def test_offsets_are_cross_fitted(self):
        """Each half must be repaired with offsets fitted on the *other* half."""
        n_series, horizon = 20, 4
        item_ids = np.repeat([f"s{i:02d}" for i in range(n_series)], horizon)
        half_a, half_b = split_series(item_ids, seed=DEFAULT_SEED)
        # y = +10 on half A, -10 on half B; all forecast quantiles are 0
        y = np.where(np.isin(item_ids, half_a), 10.0, -10.0)
        frame = make_frame(y, np.zeros((len(y), len(LEVELS))), item_ids)
        repaired = conformal_repair(frame, LEVEL_COLS, seed=DEFAULT_SEED, sort_after=False)
        # Offsets fitted on A are all +10 -> applied to B's rows (and vice versa)
        mask_a = np.isin(item_ids, half_a)
        np.testing.assert_allclose(repaired[~mask_a], 10.0)
        np.testing.assert_allclose(repaired[mask_a], -10.0)

    def test_every_row_repaired_exactly_once(self):
        frame = synthetic_forecaster(n_series=31, horizon=3, seed=11)  # odd series count
        repaired = conformal_repair(frame, LEVEL_COLS, sort_after=False)
        assert np.isfinite(repaired).all()
        assert repaired.shape == frame[LEVEL_COLS].to_numpy().shape


class TestSchemaHelpers:
    def test_quantile_level_columns(self):
        frame = synthetic_forecaster(n_series=5, horizon=2)
        assert quantile_level_columns(frame) == LEVEL_COLS

    def test_rejects_frames_without_quantiles(self):
        with pytest.raises(ValueError, match="quantile columns"):
            quantile_level_columns(pd.DataFrame({"y_true": [1.0], "point": [1.0]}))

    def test_forecast_metrics_multi_window_averages(self):
        rng = np.random.default_rng(12)
        y = rng.normal(size=40)
        q = np.sort(rng.normal(size=(40, 9)), axis=1)
        window_idx = np.repeat([0, 1], 20)
        combined = forecast_metrics(y, q, LEVELS, window_idx)
        w0 = forecast_metrics(y[:20], q[:20], LEVELS)
        w1 = forecast_metrics(y[20:], q[20:], LEVELS)
        assert combined["WQL"] == pytest.approx((w0["WQL"] + w1["WQL"]) / 2)
