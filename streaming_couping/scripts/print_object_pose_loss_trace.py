#!/usr/bin/env python3
"""Print the per-frame outer-iteration trace for object pose loss."""

from __future__ import annotations

import json
from pathlib import Path
import sys


def main() -> None:
    if len(sys.argv) not in {2, 3}:
        raise SystemExit(
            "usage: print_object_pose_loss_trace.py "
            "POSE_REFINEMENT_SUMMARY [EVALUATION_SUMMARY]"
        )
    path = Path(sys.argv[1])
    payload = json.loads(path.read_text(encoding="utf8"))
    rows = payload.get("optimization_trace", [])
    if not isinstance(rows, list):
        raise ValueError("optimization_trace must be a list")
    pose_errors = _load_pose_errors(Path(sys.argv[2])) if len(sys.argv) == 3 else {}
    print("===== OBJECT POSE LOSS OUTER-ITERATION TRACE =====")
    print(f"trace_enabled={int(bool(payload.get('optimization_trace_enabled')))} rows={len(rows)}")
    for row in rows:
        if not isinstance(row, dict):
            continue
        frame = row.get("frame_id")
        iteration = row.get("outer_iteration")
        phase = row.get("phase")
        status = row.get("status")
        gap_min = row.get("reference_gap_min")
        gap_max = row.get("reference_gap_max")
        if phase == "frame":
            gt = _format_gt_errors(pose_errors.get(int(frame)))
            print(
                "frame={0} phase=frame observations={1} pairs={2} status={3}{4}".format(
                    frame,
                    row.get("observation_count"),
                    row.get("candidate_pair_count"),
                    status,
                    gt,
                )
            )
        elif phase == "initial":
            print(
                "frame={0} phase=initial gaps={1}-{2} pairs={3} "
                "matches={4} loss_m={5} status={6}".format(
                    frame,
                    gap_min,
                    gap_max,
                    row.get("match_pair_count"),
                    row.get("match_count"),
                    row.get("loss_m"),
                    status,
                )
            )
        elif phase == "final":
            gt = _format_gt_errors(pose_errors.get(int(frame)))
            print(
                "frame={0} phase=final iter={1} gaps={2}-{3} "
                "matches={4} loss_m={5} delta_deg={6} delta_m={7} status={8}{9}".format(
                    frame,
                    iteration,
                    gap_min,
                    gap_max,
                    row.get("match_count"),
                    row.get("loss_m"),
                    row.get("delta_rotation_deg"),
                    row.get("delta_translation_m"),
                    status,
                    gt,
                )
            )
        else:
            print(
                "frame={0} iter={1} gaps={2}-{3} "
                "matches={4}->{5} loss_m={6}->{7} "
                "delta_deg={8} delta_m={9} status={10}".format(
                    frame,
                    iteration,
                    gap_min,
                    gap_max,
                    row.get("match_count"),
                    row.get("candidate_match_count"),
                    row.get("start_loss_m"),
                    row.get("candidate_loss_m"),
                    row.get("delta_rotation_deg"),
                    row.get("delta_translation_m"),
                    status,
                )
            )

def _load_pose_errors(path: Path) -> dict[int, dict[str, float]]:
    payload = json.loads(path.read_text(encoding="utf8"))
    pose = payload.get("pose_evaluation", {})
    branches = pose.get("branches", {}) if isinstance(pose, dict) else {}
    output: dict[int, dict[str, float]] = {}
    for branch_name in ("raw_pose", "object_pose_refined"):
        branch = branches.get(branch_name, {})
        rows = branch.get("frames", []) if isinstance(branch, dict) else []
        for row in rows:
            if not isinstance(row, dict) or "frame_id" not in row:
                continue
            frame_id = int(row["frame_id"])
            prefix = "raw" if branch_name == "raw_pose" else "refined"
            output.setdefault(frame_id, {})[f"{prefix}_t"] = float(
                row.get("translation_error_m", float("nan"))
            )
            output.setdefault(frame_id, {})[f"{prefix}_r"] = float(
                row.get("rotation_error_deg", float("nan"))
            )
    return output


def _format_gt_errors(row: dict[str, float] | None) -> str:
    if not row:
        return ""
    return (
        " gt_t={0}->{1} gt_r={2}->{3}".format(
            row.get("raw_t"),
            row.get("refined_t"),
            row.get("raw_r"),
            row.get("refined_r"),
        )
    )


if __name__ == "__main__":
    main()
