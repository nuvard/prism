"""
Visualization of pipeline outputs: dynamics, score distribution, sparsity statistics.

Reads from output_dir (metadata and optional attention_rows), produces plots
and saves them to a configurable directory.
"""

from __future__ import annotations

from visualization.aggregate import (
    discover_request_ids,
    aggregate_request_metrics,
)
from visualization.generate import run_visualization
from visualization.plots import (
    plot_importance_and_deltas_dynamics,
    plot_score_distribution,
    plot_sparsity_stats,
)

__all__ = [
    "discover_request_ids",
    "aggregate_request_metrics",
    "run_visualization",
    "plot_importance_and_deltas_dynamics",
    "plot_score_distribution",
    "plot_sparsity_stats",
]
