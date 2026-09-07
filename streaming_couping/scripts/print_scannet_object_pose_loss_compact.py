#!/usr/bin/env python3
"""Print the compact one-shot 100-frame object pose-loss result."""

from __future__ import annotations

import json
from pathlib import Path
import sys


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(
            "usage: print_scannet_object_pose_loss_compact.py "
            "POSE_REFINEMENT_SUMMARY EVALUATION_SUMMARY"
        )
    refinement = _read_json(Path(sys.argv[1]))
    evaluation = _read_json(Path(sys.argv[2]))
    loss = refinement.get("loss_statistics", {})
    change = refinement.get("raw_vs_refined_pose_change", {})
    print("===== SCANNET OBJECT POSE LOSS 100F =====")
    print(
        "frames={frame_count} tracked_instances={tracked_instance_count} "
        "candidates={candidate_pair_count} accepted_edges={accepted_edge_count} "
        "rejected_edges={rejected_edge_count} accepted_frames={accepted_frame_count}".format(
            **refinement
        )
    )
    print(
        "loss_initial_m={0} loss_final_m={1} "
        "pose_correction_mean_deg={2} translation_correction_mean_m={3}".format(
            loss.get("accepted_initial_mean_m"),
            loss.get("accepted_final_mean_m"),
            change.get("mean_rotation_correction_deg"),
            change.get("mean_translation_correction_m"),
        )
    )

    pose = evaluation.get("pose_evaluation", {}).get("branches", {})
    for branch in ("raw_pose", "object_pose_refined"):
        row = pose.get(branch, {})
        summary = row.get("summary", {}) if isinstance(row, dict) else {}
        print(
            f"pose={branch} "
            f"ATE_RMSE_m={summary.get('ate_rmse_m')} "
            f"RPE_t_RMSE_m={summary.get('rpe_translation_rmse_m')} "
            f"RPE_r_RMSE_deg={summary.get('rpe_rotation_rmse_deg')}"
        )

    branches = evaluation.get("branches", {})
    for branch in ("raw_pose", "object_pose_refined"):
        row = branches.get(branch, {}).get("summary", {})
        print(
            f"map={branch} "
            f"matched_objects={row.get('matched_objects')} "
            f"accuracy_m={row.get('object_accuracy_m')} "
            f"completeness_m={row.get('object_completeness_m')} "
            f"F5cm={row.get('fscore_5cm')} "
            f"voxelIoU5cm={row.get('voxel_iou_5cm')} "
            f"ghost={row.get('ghost_point_ratio')}"
        )

    plys = evaluation.get("object_ply_comparison", {})
    print(f"gt_objects={plys.get('gt_directory')}")
    print(
        "raw_objects="
        + str(plys.get("branch_directories", {}).get("raw_pose"))
    )
    print(
        "refined_objects="
        + str(plys.get("branch_directories", {}).get("object_pose_refined"))
    )
    print(f"ply_manifest={plys.get('manifest')}")
    print(f"evaluation_summary={sys.argv[2]}")


def _read_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


if __name__ == "__main__":
    main()
