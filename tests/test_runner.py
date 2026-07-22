"""Tests for the config-driven evaluation runner (offline, synthetic data)."""

import numpy as np
import pandas as pd
import pytest
import yaml

from chronosquant.evaluation.predictors import SeasonalNaivePredictor
from chronosquant.evaluation.runner import RunConfig, load_benchmark_tasks, run_evaluation


@pytest.fixture()
def benchmark_yaml(tmp_path, trend_dataset_path, noisy_dataset_path):
    """A 2-task benchmark file over the synthetic datasets."""
    config = {
        "tasks": [
            {
                "dataset_path": str(trend_dataset_path),
                "horizon": 12,
                "seasonality": 24,
                "eval_metric": "MASE",
                "extra_metrics": ["WQL"],
                "quantile_levels": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                "task_name": "trend",
            },
            {
                "dataset_path": str(noisy_dataset_path),
                "horizon": 12,
                "seasonality": 24,
                "eval_metric": "MASE",
                "extra_metrics": ["WQL"],
                "quantile_levels": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                "task_name": "noisy",
            },
        ]
    }
    path = tmp_path / "benchmark.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def make_config(benchmark_yaml, tmp_path, **overrides) -> RunConfig:
    defaults = dict(
        name="sn_test",
        benchmark=str(benchmark_yaml),
        predictor={"type": "seasonal_naive"},
        output_root=str(tmp_path / "results"),
    )
    defaults.update(overrides)
    return RunConfig(**defaults)


class TestLoadBenchmarkTasks:
    def test_loads_all_tasks(self, benchmark_yaml):
        tasks = load_benchmark_tasks(benchmark_yaml, None)
        assert [t.task_name for t in tasks] == ["trend", "noisy"]

    def test_subset_filter(self, benchmark_yaml):
        tasks = load_benchmark_tasks(benchmark_yaml, ["noisy"])
        assert [t.task_name for t in tasks] == ["noisy"]

    def test_unknown_task_name_raises(self, benchmark_yaml):
        with pytest.raises(ValueError, match="Unknown task names"):
            load_benchmark_tasks(benchmark_yaml, ["nonexistent"])

    def test_vendored_chronos_zeroshot_has_27_tasks(self):
        tasks = load_benchmark_tasks("configs/evaluation/benchmarks/chronos_zeroshot.yaml", None)
        assert len(tasks) == 27  # Chronos paper Benchmark II
        for task in tasks:
            assert task.eval_metric == "MASE"
            assert task.extra_metrics == ["WQL"]
            assert task.quantile_levels == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


class TestRunEvaluation:
    def test_produces_complete_summaries(self, benchmark_yaml, tmp_path):
        output_dir = run_evaluation(make_config(benchmark_yaml, tmp_path))

        summaries = pd.read_csv(output_dir / "summaries.csv")
        assert len(summaries) == 2
        expected_columns = {
            # fev-standard summary
            "model_name", "task_name", "test_error", "MASE", "WQL",
            "inference_time_s", "num_forecasts", "dataset_fingerprint", "fev_version",
            # full supplementary metric suite
            "SQL", "MQL", "MAE", "RMSE", "RMSSE", "WAPE", "SMAPE", "MAPE",
            # per-quantile breakdowns
            "WQL[0.1]", "WQL[0.9]", "SQL[0.1]", "SQL[0.9]",
            # calibration diagnostics
            "MACE", "QCR", "coverage[0.8]", "coverage[0.2]",
            # provenance & systems
            "run_name", "git_sha", "peak_gpu_memory_mb", "task_wall_time_s",
            "model_predictor_type",
        }  # fmt: skip
        assert expected_columns <= set(summaries.columns)
        assert (summaries["model_name"] == "seasonal_naive").all()
        # deterministic forecaster with identical quantiles: no crossings by construction
        assert (summaries["QCR"] == 0.0).all()
        assert (summaries["task_wall_time_s"] > 0).all()
        metadata = yaml.safe_load((output_dir / "run_metadata.yaml").read_text())
        assert metadata["num_completed"] == 2
        assert metadata["num_failed"] == 0
        assert metadata["model_card"]["predictor_type"] == "SeasonalNaivePredictor"
        assert len(metadata["benchmark_sha256"]) == 64
        assert metadata["task_names"] == ["trend", "noisy"]

    def test_persists_predictions_with_ground_truth(self, benchmark_yaml, tmp_path):
        """Raw predictions + y_true are stored so any per-series metric is recomputable."""
        output_dir = run_evaluation(make_config(benchmark_yaml, tmp_path, name="preds"))

        parquet_path = output_dir / "predictions" / "trend.parquet"
        assert parquet_path.exists()
        frame = pd.read_parquet(parquet_path)
        # 3 series x horizon 12 x 1 window
        assert len(frame) == 36
        expected_columns = {"window_idx", "item_id", "timestamp", "step", "y_true", "point"}
        expected_columns |= {str(q) for q in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}
        assert expected_columns <= set(frame.columns)
        # ground truth for the trend dataset is exactly known: y[t] = t + 1000*i
        series_0 = frame[frame["item_id"] == "series_0"].sort_values("step")
        np.testing.assert_allclose(series_0["y_true"], np.arange(84, 96, dtype=float))
        # seasonal naive forecast: one season back
        np.testing.assert_allclose(series_0["point"], np.arange(60, 72, dtype=float))

    def test_save_predictions_can_be_disabled(self, benchmark_yaml, tmp_path):
        config = make_config(benchmark_yaml, tmp_path, name="nopreds", save_predictions=False)
        output_dir = run_evaluation(config)
        assert not (output_dir / "predictions" / "trend.parquet").exists()

    def test_deterministic_across_runs(self, benchmark_yaml, tmp_path):
        config_1 = make_config(benchmark_yaml, tmp_path, name="run1")
        config_2 = make_config(benchmark_yaml, tmp_path, name="run2")
        df1 = pd.read_csv(run_evaluation(config_1) / "summaries.csv")
        df2 = pd.read_csv(run_evaluation(config_2) / "summaries.csv")
        for col in ["MASE", "WQL", "MACE", "QCR"]:
            np.testing.assert_array_equal(df1[col].to_numpy(), df2[col].to_numpy())

    def test_refuses_to_overwrite_by_default(self, benchmark_yaml, tmp_path):
        config = make_config(benchmark_yaml, tmp_path)
        run_evaluation(config)
        with pytest.raises(FileExistsError):
            run_evaluation(config)
        # but succeeds with overwrite=True
        run_evaluation(make_config(benchmark_yaml, tmp_path, overwrite=True))

    def test_overwrite_removes_stale_artifacts(self, benchmark_yaml, tmp_path):
        """A rerun must not inherit failure logs (failures.jsonl is append-mode)."""

        class FailsOnNoisy(SeasonalNaivePredictor):
            def predict_window(self, window, task):
                if task.task_name == "noisy":
                    raise RuntimeError("injected failure")
                return super().predict_window(window, task)

        config = make_config(benchmark_yaml, tmp_path, name="rerun")
        output_dir = run_evaluation(config, predictor=FailsOnNoisy())
        assert (output_dir / "failures.jsonl").exists()

        config = make_config(benchmark_yaml, tmp_path, name="rerun", overwrite=True)
        output_dir = run_evaluation(config)  # healthy predictor
        assert not (output_dir / "failures.jsonl").exists()
        assert len(pd.read_csv(output_dir / "summaries.csv")) == 2

    def test_failure_isolation(self, benchmark_yaml, tmp_path):
        class FailsOnNoisy(SeasonalNaivePredictor):
            def predict_window(self, window, task):
                if task.task_name == "noisy":
                    raise RuntimeError("injected failure")
                return super().predict_window(window, task)

        config = make_config(benchmark_yaml, tmp_path, name="partial")
        output_dir = run_evaluation(config, predictor=FailsOnNoisy())

        summaries = pd.read_csv(output_dir / "summaries.csv")
        assert summaries["task_name"].tolist() == ["trend"]
        failures = (output_dir / "failures.jsonl").read_text().strip().splitlines()
        assert len(failures) == 1
        assert "injected failure" in failures[0]
        metadata = yaml.safe_load((output_dir / "run_metadata.yaml").read_text())
        assert metadata["num_completed"] == 1
        assert metadata["num_failed"] == 1

    def test_all_tasks_failing_raises(self, benchmark_yaml, tmp_path):
        class AlwaysFails(SeasonalNaivePredictor):
            def predict_window(self, window, task):
                raise RuntimeError("boom")

        config = make_config(benchmark_yaml, tmp_path, name="all_fail")
        with pytest.raises(RuntimeError, match="All 2 tasks failed"):
            run_evaluation(config, predictor=AlwaysFails())
