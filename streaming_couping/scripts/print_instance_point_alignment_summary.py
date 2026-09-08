#!/usr/bin/env python3
"""Print the compact result of the V3 object-only point-alignment ablation."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(
            "usage: print_instance_point_alignment_summary.py "
            "RUNTIME_OUTPUT_DIR EVALUATION_SUMMARY"
        )
    runtime_root = Path(sys.argv[1]).expanduser().resolve()
    evaluation = _read_json(Path(sys.argv[2]))
    comparison = _read_json(runtime_root / "comparison.json")

    print("===== SCANNET OBJECT POINT ALIGNMENT V3 100F =====")
    print(f"frames={evaluation.get('frame_count')} map_source={evaluation.get('map_source')}")
    print("camera_pose=raw_horizonstream_for_both_branches")

    evaluation_branches = evaluation.get("branches", {})
    comparison_branches = comparison.get("branches", {})
    for branch in ("raw", "instance_point_alignment"):
        row = evaluation_branches.get(branch, {})
        metrics = row.get("summary", {}) if isinstance(row, dict) else {}
        print(
            f"map={branch} "
            f"matched_objects={metrics.get('matched_objects')} "
            f"accuracy_m={metrics.get('object_accuracy_m')} "
            f"completeness_m={metrics.get('object_completeness_m')} "
            f"F5cm={metrics.get('fscore_5cm')} "
            f"voxelIoU5cm={metrics.get('voxel_iou_5cm')} "
            f"ghost={metrics.get('ghost_point_ratio')}"
        )

    alignment = _branch_alignment_summary(
        comparison_branches.get("instance_point_alignment", {})
    )
    print(
        "alignment_scope "
        f"camera_pose_modified={alignment.get('camera_pose_modified', False)} "
        f"full_scene_geometry_modified={alignment.get('full_scene_geometry_modified', False)} "
        f"object_points_modified={alignment.get('object_points_modified', True)}"
    )
    print(
        "alignment "
        f"decision_count={alignment.get('decision_count')} "
        f"bootstrap_count={alignment.get('bootstrap_count')} "
        f"accepted_count={alignment.get('accepted_count')} "
        f"rejected_count={alignment.get('rejected_count')} "
        f"mean_initial_rmse_m={alignment.get('mean_initial_rmse_m')} "
        f"mean_final_rmse_m={alignment.get('mean_final_rmse_m')} "
        f"mean_relative_improvement={alignment.get('mean_relative_improvement')} "
        f"mean_correction_rotation_deg={alignment.get('mean_correction_rotation_deg')} "
        f"mean_correction_translation_m={alignment.get('mean_correction_translation_m')}"
    )

    pose_branches = evaluation.get("pose_evaluation", {}).get("branches", {})
    raw_pose = pose_branches.get("raw_pose", {})
    pose_summary = raw_pose.get("summary", {}) if isinstance(raw_pose, dict) else {}
    if pose_summary:
        print(
            "pose=raw_pose "
            f"ATE_RMSE_m={pose_summary.get('ate_rmse_m')} "
            f"RPE_t_RMSE_m={pose_summary.get('rpe_translation_rmse_m')} "
            f"RPE_r_RMSE_deg={pose_summary.get('rpe_rotation_rmse_deg')}"
        )

    track_plys = evaluation.get("object_track_ply_comparison", {})
    print(f"object_tracks_directory={track_plys.get('directory')}")
    for name, path in (track_plys.get("files", {}) or {}).items():
        print(f"object_tracks_{name}={path}")
    print(f"evaluation_summary={Path(sys.argv[2]).expanduser().resolve()}")


def _branch_alignment_summary(branch: Any) -> dict[str, Any]:
    if not isinstance(branch, dict):
        return {}
    value = branch.get("instance_point_alignment", {})
    return dict(value) if isinstance(value, dict) else {}


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


if __name__ == "__main__":
    main()
