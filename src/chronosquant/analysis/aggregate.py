"""Aggregation of evaluation summaries into leaderboards, retention tables, and checks.

Builds directly on `fev.analysis` so that skill scores, win rates and bootstrap CIs are
computed *exactly* as in fev-bench / the Chronos-2 paper (clipped relative errors,
geometric mean, pairwise win rates with ties = 0.5, 1000-resample percentile bootstrap).

This module adds the quantization-specific views on top:

- `retention_table` — per-model accuracy retention relative to the fp32 parent model
  (the headline table of a quantization paper).
- `validate_against_reference` — cross-checks our per-task results against published
  reference CSVs (same tasks, same `dataset_fingerprint`), the reproduction gate that
  must pass before any quantization results are trusted.
"""

import warnings
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats
from fev.analysis import bootstrap, leaderboard, pairwise_comparison, pivot_table

__all__ = [
    "load_summaries",
    "leaderboard",
    "pairwise_comparison",
    "pivot_table",
    "retention_table",
    "validate_against_reference",
]

#: Number of bootstrap resamples used throughout (fev-bench convention).
N_RESAMPLES = 1000


def load_summaries(paths: Sequence[str | Path]) -> pd.DataFrame:
    """Load and concatenate summary CSVs (run outputs and/or reference results).

    Accepts files or directories (directories are searched for ``summaries.csv``
    and ``*.csv``).
    """
    frames = []
    for path in paths:
        path = Path(path)
        if path.is_dir():
            csvs = sorted(path.rglob("*.csv"))
            if not csvs:
                raise FileNotFoundError(f"No CSV files found under {path}")
            frames.extend(pd.read_csv(f) for f in csvs)
        elif path.is_file():
            frames.append(pd.read_csv(path))
        else:
            raise FileNotFoundError(path)
    return pd.concat(frames, ignore_index=True)


def retention_table(
    summaries: pd.DataFrame,
    reference_model: str,
    metrics: Sequence[str] = ("WQL", "MASE"),
    min_relative_error: float = 1e-2,
    max_relative_error: float = 100.0,
    n_resamples: int = N_RESAMPLES,
    seed: int = 123,
) -> pd.DataFrame:
    """Accuracy retention of each model relative to a reference model (e.g. fp32 parent).

    For each model and metric, reports:

    - ``<metric>_rel``: geometric mean over tasks of ``metric(model) / metric(reference)``
      (clipped to ``[min_relative_error, max_relative_error]``, fev convention).
      1.00 = parity with the reference; 1.05 = 5% worse.
    - ``<metric>_rel_lower/upper``: 95% bootstrap CI over tasks.
    - ``<metric>_win_rate``: fraction of tasks where the model beats the reference
      (ties count 0.5).

    Every model must have results for exactly the same tasks as the reference
    (missing results raise — aligned task sets are a precondition for retention claims).
    """
    results: dict[str, pd.Series | np.ndarray] = {}
    model_index = None
    for metric in metrics:
        errors = pivot_table(summaries, metric_column=metric, baseline_model=reference_model)
        if errors.isna().any().any():
            missing = errors.isna().sum()
            raise ValueError(
                f"Missing {metric} results for some (task, model) pairs:\n{missing[missing > 0]}"
            )
        errors = errors.clip(lower=min_relative_error, upper=max_relative_error)
        model_index = errors.columns

        rel, rel_lower, rel_upper = bootstrap(
            errors.to_numpy(),
            statistic=lambda x: scipy.stats.gmean(x, axis=0),
            n_resamples=n_resamples,
            seed=seed,
        )
        # Win rate vs the reference: model beats reference when relative error < 1
        rel_errors = errors.to_numpy()
        win_rate = ((rel_errors < 1.0).mean(axis=0) + 0.5 * (rel_errors == 1.0).mean(axis=0))

        results[f"{metric}_rel"] = rel
        results[f"{metric}_rel_lower"] = rel_lower
        results[f"{metric}_rel_upper"] = rel_upper
        results[f"{metric}_win_rate"] = win_rate

    table = pd.DataFrame(results, index=model_index)
    # The reference's win rate against itself is undefined, not 0.5-by-tie
    for metric in metrics:
        table.loc[reference_model, f"{metric}_win_rate"] = np.nan
    return table.sort_values(f"{metrics[0]}_rel")


def validate_against_reference(
    our_summaries: pd.DataFrame,
    reference_summaries: pd.DataFrame,
    metrics: Sequence[str] = ("MASE", "WQL"),
    check_fingerprints: bool = True,
) -> pd.DataFrame:
    """Compare our per-task results with published reference results, task by task.

    Joins on ``task_name`` and returns a table with our value, the reference value and
    the relative difference for each metric — plus a fingerprint check confirming both
    evaluations saw byte-identical data (fev's ``dataset_fingerprint``).

    Fingerprint semantics: fev's fingerprint algorithm is only stable *within* a fev
    version (it changed between 0.6 and 0.9, so e.g. the published chronos_zeroshot
    reference CSVs carry 0.6-era fingerprints). A mismatch therefore **raises** only when
    both sides were produced by the same fev version; across versions it warns and
    reports ``fingerprint_match=False``, and metric agreement is the authoritative
    equality check.
    """
    our = our_summaries.set_index("task_name")
    ref = reference_summaries.set_index("task_name")
    common = our.index.intersection(ref.index)
    if len(common) == 0:
        raise ValueError("No overlapping task_name between our results and the reference")

    rows = {}
    if check_fingerprints:
        fp_ours = our.loc[common, "dataset_fingerprint"]
        fp_ref = ref.loc[common, "dataset_fingerprint"]
        matches = (fp_ours == fp_ref).to_numpy()
        mismatched = common[~matches]
        if len(mismatched) > 0:
            same_fev_version = (
                "fev_version" in our.columns
                and "fev_version" in ref.columns
                and (
                    our.loc[common, "fev_version"].astype(str).to_numpy()
                    == ref.loc[common, "fev_version"].astype(str).to_numpy()
                ).all()
            )
            if same_fev_version:
                raise ValueError(
                    f"dataset_fingerprint mismatch for tasks {list(mismatched)}: "
                    "the loaded data differs from what the reference results were computed on"
                )
            warnings.warn(
                f"dataset_fingerprint differs for tasks {list(mismatched)}, but the two "
                "sides were produced by different fev versions (fingerprint algorithms "
                "are not comparable across versions). Relying on metric agreement instead.",
                stacklevel=2,
            )
        rows["fingerprint_match"] = pd.Series(matches, index=common)

    for metric in metrics:
        ours_m = our.loc[common, metric].astype(float)
        ref_m = ref.loc[common, metric].astype(float)
        rows[f"{metric}_ours"] = ours_m
        rows[f"{metric}_reference"] = ref_m
        rows[f"{metric}_rel_diff"] = (ours_m - ref_m) / ref_m
    return pd.DataFrame(rows)
