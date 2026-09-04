#!/usr/bin/env python3
"""Print a compact summary for the instance point consistency ablation."""

from __future__ import annotations

import json
from pathlib import Path
import sys


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(
            "usage: print_instance_point_consistency_summary.py "
            "COMPARISON_JSON OUTPUT_DIR"
        )
    comparison_path, output_dir = map(Path, sys.argv[1:])
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    branches = comparison["branches"]
    raw = branches["raw"]
    consistent = branches["instance_point_consistency"]
    raw_map = json.loads(
        (output_dir / "raw" / "map_summary.json").read_text(encoding="utf-8")
    )
    consistent_map = json.loads(
        (
            output_dir / "instance_point_consistency" / "map_summary.json"
        ).read_text(encoding="utf-8")
    )
    raw_meta = raw_map.get("metadata", {})
    consistent_meta = consistent_map.get("metadata", {})
    consistency = consistent_meta.get("instance_point_consistency", {})
    raw_track_points = sum(
        int(row.get("point_count", 0)) for row in raw_map.get("objects", [])
    )
    consistent_track_points = sum(
        int(row.get("point_count", 0))
        for row in consistent_map.get("objects", [])
    )
    print(
        "frames={0} tracked_instances={1} raw_scene_voxels={2} "
        "consistent_scene_voxels={3}".format(
            raw_meta.get("frame_count"),
            raw.get("instance_count"),
            raw.get("scene_voxel_count"),
            consistent.get("scene_voxel_count"),
        )
    )
    print(
        "raw_semantic_voxels={0} consistent_semantic_voxels={1} "
        "raw_track_points={2} consistent_track_points={3}".format(
            raw.get("voxel_count"),
            consistent.get("voxel_count"),
            raw_track_points,
            consistent_track_points,
        )
    )
    print(
        "filtered_points={0} downweighted_points={1} filtered_ratio={2:.6f} "
        "downweighted_ratio={3:.6f}".format(
            consistency.get("filtered_points", 0),
            consistency.get("downweighted_points", 0),
            float(consistency.get("filtered_point_ratio", 0.0)),
            float(consistency.get("downweighted_point_ratio", 0.0)),
        )
    )
    print("raw_semantic_ply=" + str(raw.get("semantic_ply")))
    print("consistent_semantic_ply=" + str(consistent.get("semantic_ply")))
    print("comparison=" + str(comparison_path))
    print("full_log=" + str(output_dir / "run.log"))


if __name__ == "__main__":
    main()
