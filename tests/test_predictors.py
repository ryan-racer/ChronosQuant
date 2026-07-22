"""Tests for predictor adapters, including an analytic end-to-end metric check."""

import numpy as np
import pytest

from chronosquant.evaluation.predictors import SeasonalNaivePredictor, build_predictor
from tests.conftest import HISTORY_LENGTH, HORIZON, NUM_SERIES, QUANTILE_LEVELS, SEASONALITY


class TestSeasonalNaiveForecastValues:
    def test_forecast_is_last_season_prefix(self, trend_task):
        """With H <= m, the forecast equals the first H values of the last season."""
        window = trend_task.get_window(0)
        predictions = SeasonalNaivePredictor().predict_window(window, trend_task)

        assert len(predictions) == NUM_SERIES
        for i, item in enumerate(predictions):
            y = np.arange(HISTORY_LENGTH, dtype=np.float64)[: HISTORY_LENGTH - HORIZON] + 1000 * i
            expected = y[-SEASONALITY:][:HORIZON]
            np.testing.assert_allclose(item["predictions"], expected)
            # deterministic forecaster: all quantiles equal the point forecast
            for q in QUANTILE_LEVELS:
                np.testing.assert_allclose(item[str(q)], expected)

    def test_horizon_longer_than_season_tiles_cyclically(self, trend_dataset_path):
        from tests.conftest import make_task

        task = make_task(trend_dataset_path, horizon=30, seasonality=24)
        window = task.get_window(0)
        predictions = SeasonalNaivePredictor().predict_window(window, task)
        forecast = np.asarray(predictions[0]["predictions"])
        # last season before the cutoff (cutoff = -30)
        y = np.arange(HISTORY_LENGTH, dtype=np.float64)[:-30]
        template = y[-24:]
        np.testing.assert_allclose(forecast, np.resize(template, 30))

    def test_format_accepted_by_fev(self, trend_task):
        window = trend_task.get_window(0)
        predictions = SeasonalNaivePredictor().predict_window(window, trend_task)
        cleaned = trend_task.clean_and_validate_predictions(predictions)
        assert set(cleaned.keys()) == {"target"}


class TestAnalyticMetricValues:
    """End-to-end check of predictor + fev metrics against closed-form values.

    For y[t] = t (+ constant offset per series), every seasonal difference equals the
    seasonality m, so the in-sample seasonal error is exactly m; the seasonal-naive
    forecast is exactly m below the ground truth at every step. Therefore:

    - MASE = m / m = 1.0 exactly.
    - Quantile loss at level q (all quantiles = point forecast, y always above it):
      2 * m * q; averaged over the 9-quantile grid -> 2m * 0.5 = m.
      WQL = m / mean(|y_true|).
    """

    def test_mase_exactly_one_and_wql_closed_form(self, trend_task):
        predictor = SeasonalNaivePredictor()
        predictions_per_window, _ = predictor.predict_task(trend_task)
        summary = trend_task.evaluation_summary(predictions_per_window, model_name="sn")

        assert summary["MASE"] == pytest.approx(1.0, abs=1e-12)

        # WQL = m / mean(|y_true|) over all (series, step) points
        y_true = np.concatenate(
            [
                np.arange(HISTORY_LENGTH - HORIZON, HISTORY_LENGTH, dtype=np.float64) + 1000 * i
                for i in range(NUM_SERIES)
            ]
        )
        expected_wql = SEASONALITY / np.abs(y_true).mean()
        assert summary["WQL"] == pytest.approx(expected_wql, rel=1e-12)
        assert summary["test_error"] == summary["MASE"]  # eval_metric is MASE

    def test_deterministic_across_runs(self, noisy_task):
        predictor = SeasonalNaivePredictor()
        preds_1, _ = predictor.predict_task(noisy_task)
        s1 = noisy_task.evaluation_summary(preds_1, model_name="sn")
        preds_2, _ = predictor.predict_task(noisy_task)
        s2 = noisy_task.evaluation_summary(preds_2, model_name="sn")
        assert s1["MASE"] == s2["MASE"]
        assert s1["WQL"] == s2["WQL"]


class TestBuildPredictor:
    def test_builds_seasonal_naive(self):
        predictor = build_predictor({"type": "seasonal_naive"})
        assert isinstance(predictor, SeasonalNaivePredictor)
        assert predictor.name == "seasonal_naive"

    def test_builds_chronos2_lazily_without_loading_model(self):
        predictor = build_predictor(
            {"type": "chronos2", "model_id": "amazon/chronos-2", "name": "chronos2-fp32"}
        )
        assert predictor.name == "chronos2-fp32"
        assert predictor._pipeline is None  # no download at construction time

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown predictor type"):
            build_predictor({"type": "nope"})
