"""Hand-computed unit tests for the quantization-specific diagnostics."""

import numpy as np
import pytest

from chronosquant.evaluation.metrics import MACE, QCR, central_interval_pairs

NINE_QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def _dummy_kwargs(y_true, q_pred, quantile_levels):
    n = y_true.shape[0]
    return dict(
        y_true=y_true,
        y_pred=np.nanmedian(q_pred, axis=-1),
        y_past=np.zeros((n * 10, y_true.shape[2])),
        y_past_lengths=np.full(n, 10),
        q_pred=q_pred,
        seasonality=1,
        quantile_levels=quantile_levels,
    )


class TestCentralIntervalPairs:
    def test_nine_quantile_grid(self):
        assert central_interval_pairs(NINE_QUANTILES) == [
            (0.1, 0.9),
            (0.2, 0.8),
            (0.3, 0.7),
            (0.4, 0.6),
        ]

    def test_three_quantiles(self):
        assert central_interval_pairs([0.1, 0.5, 0.9]) == [(0.1, 0.9)]

    def test_asymmetric_levels_have_no_pairs(self):
        assert central_interval_pairs([0.1, 0.4, 0.5]) == []


class TestMACE:
    def test_perfect_coverage_of_all_intervals(self):
        """Intervals covering every observation: coverage 1.0 for all pairs.

        MACE = mean(|1-0.8|, |1-0.6|, |1-0.4|, |1-0.2|) = 0.5
        """
        y_true = np.zeros((2, 3, 1))
        q_pred = np.linspace(-1, 1, 9).reshape(1, 1, 1, 9) * np.ones((2, 3, 1, 9))
        scores = MACE().compute_scores(**_dummy_kwargs(y_true, q_pred, NINE_QUANTILES))
        assert scores["MACE"] == pytest.approx(0.5)
        for nominal in [0.8, 0.6, 0.4, 0.2]:
            assert scores[f"coverage[{nominal}]"] == pytest.approx(1.0)

    def test_zero_coverage(self):
        """Ground truth far above all quantiles: coverage 0 everywhere.

        MACE = mean(0.8, 0.6, 0.4, 0.2) = 0.5
        """
        y_true = np.full((2, 3, 1), 100.0)
        q_pred = np.linspace(-1, 1, 9).reshape(1, 1, 1, 9) * np.ones((2, 3, 1, 9))
        scores = MACE().compute_scores(**_dummy_kwargs(y_true, q_pred, NINE_QUANTILES))
        assert scores["MACE"] == pytest.approx(0.5)
        assert scores["coverage[0.8]"] == pytest.approx(0.0)

    def test_partial_coverage_exact_fraction(self):
        """4 points: 2 inside the (0.1, 0.9) interval, 2 outside -> coverage[0.8] = 0.5."""
        y_true = np.array([0.0, 0.0, 100.0, 100.0]).reshape(4, 1, 1)
        q_pred = np.linspace(-1, 1, 9).reshape(1, 1, 1, 9) * np.ones((4, 1, 1, 9))
        scores = MACE().compute_scores(**_dummy_kwargs(y_true, q_pred, NINE_QUANTILES))
        assert scores["coverage[0.8]"] == pytest.approx(0.5)

    def test_nan_ground_truth_excluded(self):
        """NaN observations are excluded from the coverage denominator."""
        y_true = np.array([0.0, np.nan, 100.0]).reshape(3, 1, 1)
        q_pred = np.linspace(-1, 1, 9).reshape(1, 1, 1, 9) * np.ones((3, 1, 1, 9))
        scores = MACE().compute_scores(**_dummy_kwargs(y_true, q_pred, NINE_QUANTILES))
        # of the 2 finite points, 1 is covered
        assert scores["coverage[0.8]"] == pytest.approx(0.5)

    def test_requires_symmetric_pair(self):
        y_true = np.zeros((1, 1, 1))
        q_pred = np.zeros((1, 1, 1, 2))
        with pytest.raises(ValueError, match="symmetric"):
            MACE().compute(**_dummy_kwargs(y_true, q_pred, [0.1, 0.5]))


class TestQCR:
    def test_monotone_quantiles_have_zero_crossing(self):
        q_pred = np.linspace(0, 1, 9).reshape(1, 1, 1, 9) * np.ones((2, 4, 1, 9))
        y_true = np.zeros((2, 4, 1))
        assert QCR().compute(**_dummy_kwargs(y_true, q_pred, NINE_QUANTILES)) == 0.0

    def test_crossing_rate_exact_fraction(self):
        """Exactly 1 of 8 forecast points has crossed quantiles -> QCR = 0.125."""
        q_pred = np.tile(np.linspace(0, 1, 9), (2, 4, 1, 1))  # [2, 4, 1, 9] monotone
        q_pred = q_pred.reshape(2, 4, 1, 9).copy()
        q_pred[0, 0, 0, 3] = 10.0  # spike breaks monotonicity at one point
        y_true = np.zeros((2, 4, 1))
        assert QCR().compute(**_dummy_kwargs(y_true, q_pred, NINE_QUANTILES)) == pytest.approx(
            1 / 8
        )

    def test_requires_two_levels(self):
        y_true = np.zeros((1, 1, 1))
        q_pred = np.zeros((1, 1, 1, 1))
        with pytest.raises(ValueError, match="two quantile levels"):
            QCR().compute(**_dummy_kwargs(y_true, q_pred, [0.5]))
