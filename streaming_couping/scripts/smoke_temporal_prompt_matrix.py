#!/usr/bin/env python3
"""CPU smoke test for the frozen V0 temporal-prompt A--E matrix."""

from __future__ import annotations

import torch

from streaming_couping.scripts.run_v0_bidirectional_feedback import V0Arrays
from streaming_couping.scripts.run_v0_temporal_prompt_matrix import (
    _build_candidates,
    _summarize_box_branches,
    _summarize_point_branches,
)
from streaming_couping.src.semantic_tracking_metrics import GroundTruthInstances
from streaming_couping.scripts.run_v0_temporal_prompt_matrix import (
    _annotate_with_ground_truth,
)


def main() -> None:
    sequence, height, width, slots = 3, 32, 32, 1
    yy, xx = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )
    # A small planar object with several distinct world points.  It remains
    # stationary while the camera pose stays identity, so all causal branches
    # should project into the same image region.
    points_one = torch.stack(
        ((xx - 16.0) / 16.0, (yy - 16.0) / 16.0, torch.ones_like(xx)),
        dim=-1,
    )
    points = points_one.unsqueeze(0).repeat(sequence, 1, 1, 1)
    masks = torch.zeros(sequence, slots, height, width, dtype=torch.bool)
    masks[:, 0, 10:22, 10:22] = True
    confidence = torch.ones(sequence, height, width)
    scores = torch.ones(sequence, slots)
    pose = torch.eye(4, dtype=torch.float32)[:3].unsqueeze(0).repeat(sequence, 1, 1)
    intrinsics = torch.tensor(
        [[16.0, 0.0, 16.0], [0.0, 16.0, 16.0], [0.0, 0.0, 1.0]]
    ).unsqueeze(0).repeat(sequence, 1, 1)
    arrays = V0Arrays(
        points=points,
        confidence=confidence,
        raw_masks=masks,
        scores=scores,
        depth=torch.ones(sequence, height, width),
        intrinsics=intrinsics,
        raw_world_to_camera=pose,
        selected_world_to_camera=pose,
        image_size=(height, width),
        map_size=(height, width),
    )
    candidates = _build_candidates(
        arrays=arrays,
        frames=(0, 1, 2),
        track_ids=(7,),
        raw_world_to_camera=pose,
        max_history_points=512,
        confidence_threshold=0.3,
        track_score_threshold=0.5,
        front_fraction=0.35,
        box_quantile=0.02,
        min_box_points=8,
        absolute_depth_tolerance=0.05,
        depth_relative_tolerances=(0.1, 0.3),
    )
    assert candidates.metadata["query_count"] == 2
    branches = {str(row["branch"]) for row in candidates.query_rows}
    assert {"A_center", "B_surface_3", "C_surface_5", "D_bbox", "E_depth_gate"} <= branches
    assert all("gt_mask_hit" in row for row in candidates.point_rows)
    assert len(candidates.box_rows) == 2

    gt_masks = torch.zeros(sequence, 1, height, width, dtype=torch.bool)
    gt_masks[:, 0, 10:22, 10:22] = True
    gt = GroundTruthInstances(
        masks=gt_masks,
        instance_ids=(42,),
        labels=("object",),
        all_visible_instance_ids=(42,),
    )
    assignments = ({"slot": 0, "gt_index": 0},)
    _annotate_with_ground_truth(
        candidates.point_rows,
        candidates.box_rows,
        candidates.query_rows,
        ground_truth=gt,
        assignments=assignments,
    )
    point_summary = _summarize_point_branches(
        candidates.point_rows,
        candidates.query_rows,
        scene_id="smoke",
        clip_name="smoke",
    )
    box_summary = _summarize_box_branches(
        candidates.box_rows,
        candidates.query_rows,
        scene_id="smoke",
        clip_name="smoke",
    )
    assert point_summary
    assert box_summary and box_summary[0]["branch"] == "D_bbox"
    center = next(row for row in point_summary if row["branch"] == "A_center")
    assert float(center["object_frame_coverage"]) > 0.0
    print(
        "V0 temporal prompt matrix smoke passed "
        f"queries={candidates.metadata['query_count']} "
        f"point_rows={len(candidates.point_rows)} "
        f"box_rows={len(candidates.box_rows)}"
    )


if __name__ == "__main__":
    main()
