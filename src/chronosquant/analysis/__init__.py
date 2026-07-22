"""Result aggregation and analysis (leaderboards, retention tables, validation)."""

from chronosquant.analysis.aggregate import (
    leaderboard,
    load_summaries,
    pairwise_comparison,
    pivot_table,
    retention_table,
    validate_against_reference,
)

__all__ = [
    "leaderboard",
    "load_summaries",
    "pairwise_comparison",
    "pivot_table",
    "retention_table",
    "validate_against_reference",
]
