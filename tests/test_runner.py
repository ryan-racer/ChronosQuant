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
            "model_name", "task_name", "test_error", "MASE", "WQL",
            "MACE", "QCR", "coverage[0.8]", "coverage[0.2]",
            "WQL[0.1]", "WQL[0.9]",  # per-quantile breakdown
            "inference_time_s", "num_forecasts", "dataset_fingerprint",
            "fev_version", "run_name", "git_sha",
        }  # fmt: skip
        assert expected_columns <= set(summaries.columns)
        assert (summaries["model_name"] == "seasonal_naive").all()
        # deterministic forecaster with identical quantiles: no crossings by construction
        assert (summaries["QCR"] == 0.0).all()
        assert (output_dir / "run_metadata.yaml").exists()
        metadata = yaml.safe_load((output_dir / "run_metadata.yaml").read_text())
        assert metadata["num_completed"] == 2
        assert metadata["num_failed"] == 0

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
