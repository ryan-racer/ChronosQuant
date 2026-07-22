"""Tests for aggregation: leaderboard math cross-checks, retention, reference validation."""

import numpy as np
import pandas as pd
import pytest
import scipy.stats

from chronosquant.analysis import (
    leaderboard,
    retention_table,
    validate_against_reference,
)


def make_summaries(errors_by_model: dict[str, list[float]], metric: str = "WQL") -> pd.DataFrame:
    """Build a minimal fev-compatible summaries frame: one row per (model, task)."""
    rows = []
    for model, errors in errors_by_model.items():
        for task_idx, error in enumerate(errors):
            rows.append(
                {
                    "model_name": model,
                    "dataset_path": f"task_{task_idx}",
                    "task_name": f"task_{task_idx}",
                    "horizon": 12,
                    "seasonality": 24,
                    "eval_metric": "MASE",
                    "quantile_levels": "[0.1, 0.5, 0.9]",
                    "test_error": error,
                    metric: error,
                    "training_time_s": 0.0,
                    "inference_time_s": 1.0,
                    "num_forecasts": 10,
                    "trained_on_this_dataset": False,
                    "fev_version": "0.9.0",
                    "dataset_fingerprint": f"fp_{task_idx}",
                }
            )
    return pd.DataFrame(rows)


class TestLeaderboardMath:
    """Cross-check fev's skill score / win rate on hand-computable inputs."""

    def test_skill_score_is_one_minus_gmean_of_relative_errors(self):
        summaries = make_summaries(
            {
                "seasonal_naive": [1.0, 2.0, 4.0],
                "model_a": [0.5, 1.6, 4.0],  # relative errors: 0.5, 0.8, 1.0
            }
        )
        table = leaderboard(summaries, metric_column="WQL", baseline_model="seasonal_naive")
        expected_skill = 1 - scipy.stats.gmean([0.5, 0.8, 1.0])
        assert table.loc["model_a", "skill_score"] == pytest.approx(expected_skill)
        assert table.loc["seasonal_naive", "skill_score"] == pytest.approx(0.0)

    def test_win_rate_with_ties(self):
        """model_a beats baseline on 2 tasks and ties on 1: win rate = (2 + 0.5)/3."""
        summaries = make_summaries(
            {
                "seasonal_naive": [1.0, 2.0, 4.0],
                "model_a": [0.5, 1.6, 4.0],
            }
        )
        table = leaderboard(summaries, metric_column="WQL", baseline_model="seasonal_naive")
        assert table.loc["model_a", "win_rate"] == pytest.approx(2.5 / 3)
        assert table.loc["seasonal_naive", "win_rate"] == pytest.approx(0.5 / 3)

    def test_relative_error_clipping(self):
        """A catastrophic task error is clipped at 100x before the geometric mean."""
        summaries = make_summaries(
            {
                "seasonal_naive": [1.0, 1.0],
                "model_bad": [1.0, 100000.0],  # relative 1.0 and 1e5 -> clipped to 100
            }
        )
        table = leaderboard(summaries, metric_column="WQL", baseline_model="seasonal_naive")
        expected_skill = 1 - scipy.stats.gmean([1.0, 100.0])
        assert table.loc["model_bad", "skill_score"] == pytest.approx(expected_skill)

    def test_bootstrap_ci_reproducible_and_ordered(self):
        summaries = make_summaries(
            {
                "seasonal_naive": [1.0, 2.0, 4.0, 3.0, 5.0],
                "model_a": [0.5, 1.6, 4.0, 2.9, 4.0],
            }
        )
        t1 = leaderboard(
            summaries, metric_column="WQL", baseline_model="seasonal_naive", n_resamples=200
        )
        t2 = leaderboard(
            summaries, metric_column="WQL", baseline_model="seasonal_naive", n_resamples=200
        )
        pd.testing.assert_frame_equal(t1, t2)  # seeded bootstrap: fully reproducible
        row = t1.loc["model_a"]
        assert row["skill_score_lower"] <= row["skill_score"] <= row["skill_score_upper"]


class TestRetentionTable:
    def test_retention_is_gmean_of_ratios_vs_reference(self):
        summaries = make_summaries(
            {
                "chronos2-fp32": [1.0, 2.0, 4.0],
                "chronos2-w8": [1.1, 2.0, 4.4],  # ratios 1.1, 1.0, 1.1
                "chronos2-w4": [2.0, 4.0, 8.0],  # ratios 2.0, 2.0, 2.0
            }
        )
        table = retention_table(summaries, reference_model="chronos2-fp32", metrics=["WQL"])
        assert table.loc["chronos2-fp32", "WQL_rel"] == pytest.approx(1.0)
        assert table.loc["chronos2-w8", "WQL_rel"] == pytest.approx(
            scipy.stats.gmean([1.1, 1.0, 1.1])
        )
        assert table.loc["chronos2-w4", "WQL_rel"] == pytest.approx(2.0)
        # w8 loses 2 tasks and ties 1 vs the reference -> win rate (0 + 0.5)/3
        assert table.loc["chronos2-w8", "WQL_win_rate"] == pytest.approx(0.5 / 3)
        assert table.loc["chronos2-w4", "WQL_win_rate"] == 0.0
        assert np.isnan(table.loc["chronos2-fp32", "WQL_win_rate"])
        # sorted best-retention first
        assert list(table.index) == ["chronos2-fp32", "chronos2-w8", "chronos2-w4"]

    def test_missing_results_raise(self):
        summaries = make_summaries(
            {
                "chronos2-fp32": [1.0, 2.0, 4.0],
                "chronos2-w8": [1.1, 2.0, 4.4],
            }
        )
        summaries = summaries.drop(summaries.index[-1])  # drop one w8 task
        with pytest.raises(ValueError, match="Missing WQL results"):
            retention_table(summaries, reference_model="chronos2-fp32", metrics=["WQL"])


class TestValidateAgainstReference:
    def test_matching_results_pass(self):
        ours = make_summaries({"seasonal_naive": [1.0, 2.0]})
        reference = make_summaries({"seasonal_naive": [1.0, 2.0]})
        table = validate_against_reference(ours, reference, metrics=["WQL"])
        assert (table["WQL_rel_diff"].abs() < 1e-12).all()
        assert table["fingerprint_match"].all()

    def test_fingerprint_mismatch_raises_within_same_fev_version(self):
        ours = make_summaries({"seasonal_naive": [1.0, 2.0]})
        reference = make_summaries({"seasonal_naive": [1.0, 2.0]})
        reference.loc[0, "dataset_fingerprint"] = "different"
        with pytest.raises(ValueError, match="fingerprint mismatch"):
            validate_against_reference(ours, reference, metrics=["WQL"])

    def test_fingerprint_mismatch_warns_across_fev_versions(self):
        """Fingerprint algorithms changed between fev versions; mismatch only warns."""
        ours = make_summaries({"seasonal_naive": [1.0, 2.0]})
        reference = make_summaries({"seasonal_naive": [1.0, 2.0]})
        reference["fev_version"] = "0.6.0"
        reference.loc[0, "dataset_fingerprint"] = "different"
        with pytest.warns(UserWarning, match="different fev versions"):
            table = validate_against_reference(ours, reference, metrics=["WQL"])
        assert not table["fingerprint_match"].iloc[0]
        assert table["fingerprint_match"].iloc[1]

    def test_no_overlap_raises(self):
        ours = make_summaries({"seasonal_naive": [1.0]})
        reference = make_summaries({"seasonal_naive": [1.0]})
        reference["task_name"] = "other_task"
        with pytest.raises(ValueError, match="No overlapping"):
            validate_against_reference(ours, reference, metrics=["WQL"])
