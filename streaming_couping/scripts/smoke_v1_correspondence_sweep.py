#!/usr/bin/env python3
"""CPU smoke for fixed V1 correspondence-radius gate selection."""

from __future__ import annotations

from pathlib import Path

from streaming_couping.scripts.run_v1_correspondence_sweep import (
    SweepRun,
    _evaluate_radius,
    _select_radius,
)
from streaming_couping.scripts.run_v1_instance_geometry import V1Run


def main() -> None:
    base = V1Run(
        source_path=Path("synthetic_sweep.yaml"),
        baseline_config=Path("synthetic_v0.yaml"),
        output_dir=Path("synthetic_output"),
        device="cpu",
        max_active_instances=5,
        geometry_excluded_prompts=("wardrobe",),
        min_history_visible_frames=4,
        min_track_score=0.5,
        min_mask_area_ratio=0.001,
        max_mask_area_ratio=0.25,
        min_current_points=256,
        point_confidence_threshold=0.3,
        erosion_radius=2,
        neighbors=16,
        min_support_frames=3,
        max_current_points_per_instance=1024,
        max_history_points_per_frame=512,
        max_history_points_total=8192,
        match_radius_scene_fraction=0.01,
        normal_variance_max=0.2,
        alpha=0.5,
        max_displacement_scene_fraction=0.0025,
        chunk_size=256,
        shifted_mask_y_fraction=0.0625,
        shifted_mask_x_fraction=0.125,
        symmetric_metric_points=2048,
    )
    sweep = SweepRun(base, (0.01, 0.025, 0.05), 2, 0.10, 2.0)

    def rows(radius: float, correct_rate: float) -> list[dict[str, object]]:
        common = {
            "radius_scene_fraction": radius,
            "radius_raw": radius * 10.0,
            "selected_track_frames": 4,
            "active_tracks": 2,
            "selected_points": 1024,
            "history_points": 4096,
        }
        return [
            {
                **common,
                "branch": "correct_id",
                "valid_surfel_rate": correct_rate,
                "mean_nearest_median": 0.1,
                "mean_normal_residual_median": 0.05,
            },
            {
                **common,
                "branch": "shuffled_id",
                "valid_surfel_rate": 0.04,
                "mean_nearest_median": 0.3,
                "mean_normal_residual_median": 0.2,
            },
            {
                **common,
                "branch": "shifted_mask",
                "valid_surfel_rate": 0.05,
                "mean_nearest_median": 0.2,
                "mean_normal_residual_median": 0.1,
            },
        ]

    gates = [
        _evaluate_radius(rows(0.01, 0.09), sweep),
        _evaluate_radius(rows(0.025, 0.11), sweep),
        _evaluate_radius(rows(0.05, 0.20), sweep),
    ]
    assert [row["radius_gate_pass"] for row in gates] == [0, 1, 1]
    decision = _select_radius(gates)
    assert decision["selected_radius_scene_fraction"] == 0.025
    assert decision["selected_pointmap_modified"] == 0
    assert decision["target_geometry_fields_read"] == 0
    assert decision["i1_run"] == 0
    print("V1 fixed correspondence-radius gate smoke passed")


if __name__ == "__main__":
    main()
