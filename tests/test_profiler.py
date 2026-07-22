"""Unit tests for profiler statistics (pure functions; full profiles run via scripts)."""

import pytest

from chronosquant.evaluation.profiler import ProfilerConfig, _synthetic_contexts, latency_stats


class TestLatencyStats:
    def test_hand_computed_values(self):
        stats = latency_stats([0.5, 0.1, 0.3, 0.2, 0.4])
        assert stats["median_s"] == pytest.approx(0.3)
        assert stats["mean_s"] == pytest.approx(0.3)
        assert stats["min_s"] == pytest.approx(0.1)
        assert stats["n_reps"] == 5
        assert stats["p90_s"] == pytest.approx(0.5)  # index round(0.9*4)=4 of sorted

    def test_single_rep(self):
        stats = latency_stats([0.25])
        assert stats["median_s"] == 0.25
        assert stats["std_s"] == 0.0

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            latency_stats([])


class TestSyntheticContexts:
    def test_deterministic_and_correct_shape(self):
        a = _synthetic_contexts(batch_size=4, context_length=128, seed=42)
        b = _synthetic_contexts(batch_size=4, context_length=128, seed=42)
        assert len(a) == 4
        assert all(x.shape == (128,) for x in a)
        for x, y in zip(a, b):
            assert (x == y).all()  # same seed -> identical profiling inputs

    def test_series_differ_within_batch(self):
        contexts = _synthetic_contexts(batch_size=2, context_length=64, seed=0)
        assert not (contexts[0] == contexts[1]).all()


def test_profiler_config_serializable():
    config = ProfilerConfig(batch_sizes=(1, 8))
    assert config.to_dict()["batch_sizes"] == (1, 8)
    assert config.to_dict()["context_length"] == 2048
