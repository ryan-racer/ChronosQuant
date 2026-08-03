"""Result aggregation and analysis (leaderboards, retention tables, validation)."""

from chronosquant.analysis.repair import aggregate_repair, analyze_task, conformal_repair, rearrange
from chronosquant.analysis.aggregate import (
    leaderboard,
    load_summaries,
    pairwise_comparison,
    pivot_table,
    retention_table,
    validate_against_reference,
)

__all__ = [
    "aggregate_repair",
    "analyze_task",
    "conformal_repair",
    "rearrange",
    "leaderboard",
    "load_summaries",
    "pairwise_comparison",
    "pivot_table",
    "retention_table",
    "validate_against_reference",
]
