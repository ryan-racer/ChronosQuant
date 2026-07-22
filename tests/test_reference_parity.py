"""Integration test: our harness must reproduce published per-task numbers.

Runs Seasonal Naive on real Benchmark II tasks and compares per-task MASE/WQL with the
published results from the fev repository (vendored under `results/reference/`). This is
the reproduction gate: if this passes, our data loading, splitting, prediction format and
metric computation all match the published evaluation pipeline.

Requires network access (downloads small HF datasets on first run):
    uv run pytest -m network -q
"""

import pandas as pd
import pytest

from chronosquant.analysis import validate_against_reference
from chronosquant.evaluation.predictors import SeasonalNaivePredictor
from chronosquant.evaluation.runner import load_benchmark_tasks
from chronosquant.utils import REPO_ROOT, ensure_truststore

pytestmark = pytest.mark.network

REFERENCE_CSV = REPO_ROOT / "results/reference/chronos_zeroshot/seasonal_naive.csv"
# Small tasks for a fast gate; the full parity check is scripts/evaluate.py on all 27.
PARITY_TASKS = ["monash_cif_2016", "monash_m1_quarterly", "monash_covid_deaths"]


@pytest.fixture(scope="module")
def parity_results() -> pd.DataFrame:
    ensure_truststore()
    tasks = load_benchmark_tasks(
        "configs/evaluation/benchmarks/chronos_zeroshot.yaml", PARITY_TASKS
    )
    predictor = SeasonalNaivePredictor()
    summaries = []
    for task in tasks:
        predictions_per_window, _ = predictor.predict_task(task)
        summaries.append(
            task.evaluation_summary(predictions_per_window, model_name="seasonal_naive")
        )
    return pd.DataFrame(summaries)


def test_mase_and_wql_match_published(parity_results):
    """Per-task MASE and WQL agree with the published values to float precision.

    Note on fingerprints: the published CSVs were produced with fev 0.6, whose
    fingerprint algorithm differs from ours (0.9+), so fingerprints are not comparable
    across this version gap — `validate_against_reference` warns and metric agreement
    (identical data + identical deterministic forecaster + identical metric code)
    is the authoritative equality check.
    """
    reference = pd.read_csv(REFERENCE_CSV)
    with pytest.warns(UserWarning, match="different fev versions"):
        table = validate_against_reference(parity_results, reference, metrics=["MASE", "WQL"])
    assert len(table) == len(PARITY_TASKS)
    assert (table["MASE_rel_diff"].abs() < 1e-6).all(), table
    assert (table["WQL_rel_diff"].abs() < 1e-6).all(), table
