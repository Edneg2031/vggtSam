#!/usr/bin/env python3
"""Print a compact summary for the object alignment-loss ablation."""

from __future__ import annotations

import json
from pathlib import Path
import sys


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(
            "usage: print_object_pose_loss_summary.py "
            "POSE_REFINEMENT_SUMMARY COMPARISON_JSON"
        )
    summary_path = Path(sys.argv[1]).expanduser().resolve()
    comparison_path = Path(sys.argv[2]).expanduser().resolve()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    loss = summary.get("loss_statistics", {})
    change = summary.get("raw_vs_refined_pose_change", {})
    optimizer = summary.get("optimizer", {})
    print(
        "frames={frame_count} tracked_instances={tracked_instance_count} "
        "raw_obs={raw_observation_count} retained_obs={retained_observation_count} "
        "filtered_obs={filtered_observation_count} "
        "candidates={candidate_pair_count} accepted_edges={accepted_edge_count} "
        "rejected_edges={rejected_edge_count} optimized_frames={optimization_attempted_frame_count} "
        "accepted_frames={accepted_frame_count}".format(**summary)
    )
    print(
        "object_loss_initial_mean_m={0} object_loss_final_mean_m={1} "
        "relative_improvement_mean={2}".format(
            loss.get("accepted_initial_mean_m"),
            loss.get("accepted_final_mean_m"),
            loss.get("accepted_relative_improvement_mean"),
        )
    )
    print(
        "pose_correction_mean_deg={0} pose_correction_max_deg={1} "
        "translation_correction_mean_m={2} translation_correction_max_m={3}".format(
            change.get("mean_rotation_correction_deg"),
            change.get("max_rotation_correction_deg"),
            change.get("mean_translation_correction_m"),
            change.get("max_translation_correction_m"),
        )
    )
    print(
        "optimizer_backend={0} optimizer_attempted={1} optimizer_success={2}".format(
            optimizer.get("backend"),
            optimizer.get("attempted"),
            optimizer.get("success"),
        )
    )
    maps = comparison.get("maps", {})
    raw = maps.get("raw_pose", {})
    refined = maps.get("object_pose_refined", {})
    print(
        "raw_object_voxels={0} refined_object_voxels={1} "
        "raw_objects={2} refined_objects={3}".format(
            raw.get("voxel_count"),
            refined.get("voxel_count"),
            raw.get("instance_count"),
            refined.get("instance_count"),
        )
    )
    print(
        "raw_objects_dir="
        + str(Path(raw.get("semantic_ply", "")).parent / "objects")
    )
    print(
        "refined_objects_dir="
        + str(Path(refined.get("semantic_ply", "")).parent / "objects")
    )
    print("summary=" + str(summary_path))
    print("comparison=" + str(comparison_path))


if __name__ == "__main__":
    main()
