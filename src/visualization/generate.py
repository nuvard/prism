"""
Orchestrate visualization: discover requests, aggregate data, plot, save.

Can be called from the pipeline after writing outputs or standalone with a config path.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from visualization.aggregate import (
    aggregate_request_metrics,
    discover_request_ids,
    load_attention_weights_for_distribution,
)
from visualization.plots import (
    plot_importance_and_deltas_dynamics,
    plot_score_distribution,
    plot_sparsity_stats,
)

if TYPE_CHECKING:
    from attention_scores.config import Config


def run_visualization(config: "Config") -> list[Path]:
    """
    Generate all visualization plots from pipeline output_dir and save to files.

    Uses config.output_dir as input and config.visualization_output_dir (or
    output_dir/visualization if empty) as output. Respects config.visualization_enabled
    and config.visualization_formats.

    Args:
        config: Pipeline config with output_dir and visualization_* fields.

    Returns:
        List of paths to saved plot files.
    """
    if not getattr(config, "visualization_enabled", True):
        return []

    out_dir = getattr(config, "visualization_output_dir", "") or ""
    if not out_dir.strip():
        out_dir = str(Path(config.output_dir) / "visualization")
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    formats = getattr(config, "visualization_formats", None) or ["png"]
    output_dir = Path(config.output_dir)
    request_ids = discover_request_ids(output_dir)
    saved: list[Path] = []

    for request_id in request_ids:
        metrics = aggregate_request_metrics(output_dir, request_id)
        base_name = request_id.replace("/", "_")

        # Importance and deltas dynamics (one multi-panel figure)
        plot_importance_and_deltas_dynamics(
            metrics,
            save_path=out_path / f"{base_name}_importance_dynamics",
            formats=formats,
        )
        for ext in formats:
            saved.append(out_path / f"{base_name}_importance_dynamics.{ext}")

        # Sparsity stats
        plot_sparsity_stats(
            metrics,
            save_path=out_path / f"{base_name}_sparsity_stats",
            formats=formats,
        )
        for ext in formats:
            saved.append(out_path / f"{base_name}_sparsity_stats.{ext}")

        # Score distribution: load attention rows for this request
        steps = metrics["steps"]
        weights = load_attention_weights_for_distribution(
            output_dir, request_id, steps, max_steps=15, max_weights_per_step=30_000
        )
        plot_score_distribution(
            weights,
            request_id=request_id,
            save_path=out_path / f"{base_name}_score_distribution",
            formats=formats,
        )
        for ext in formats:
            saved.append(out_path / f"{base_name}_score_distribution.{ext}")

    # Optional: one summary score distribution across all requests (sample)
    if request_ids:
        from attention_scores.read_outputs import load_metadata as _load_meta
        all_weights: list[float] = []
        for rid in request_ids[:5]:  # cap requests
            meta = _load_meta(output_dir, rid)
            steps = [p["step"] for p in (meta.get("per_step") or []) if "step" in p]
            all_weights.extend(
                load_attention_weights_for_distribution(
                    output_dir, rid, steps, max_steps=5, max_weights_per_step=10_000
                )
            )
        plot_score_distribution(
            all_weights,
            request_id="(all requests)",
            save_path=out_path / "score_distribution_all",
            formats=formats,
        )
        for ext in formats:
            saved.append(out_path / f"score_distribution_all.{ext}")

    return saved


def main() -> None:
    """CLI entry: load config from argv and run visualization only (no model)."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m visualization.generate <config.yaml|config.json>", file=sys.stderr)
        sys.exit(1)

    from attention_scores.config import Config

    config = Config.from_file(sys.argv[1])
    paths = run_visualization(config)
    vis_dir = getattr(config, "visualization_output_dir", "") or str(Path(config.output_dir) / "visualization")
    print(f"Saved {len(paths)} visualization file(s) to {vis_dir}")


if __name__ == "__main__":
    main()
