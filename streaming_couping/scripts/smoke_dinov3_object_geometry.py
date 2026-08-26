#!/usr/bin/env python3
"""No-model smoke test for the object-conditioned residual head."""

from __future__ import annotations

import torch

from streaming_couping.src.dinov3_object_geometry import (
    ObjectConditionedResidualHead,
    WorldSpaceGateConfig,
    apply_world_space_consistency_gate,
)
from streaming_couping.src.semantic_map_metrics import (
    SemanticMapMetricConfig,
    evaluate_semantic_object_map,
)
from streaming_couping.src.semantic_tracking_metrics import (
    GroundTruthInstances,
    TrackingMetricConfig,
    evaluate_tracking_variants,
)


def main() -> None:
    features = torch.randn(2, 2, 6, 8)
    masks = torch.zeros(2, 2, 4, 6, dtype=torch.bool)
    masks[:, 0, :, :2] = True
    masks[:, 1, :, 4:] = True
    object_features = torch.randn(2, 2, 5)
    valid = torch.ones(2, 2, dtype=torch.bool)
    head = ObjectConditionedResidualHead(
        feature_channels=8,
        level_count=2,
        object_feature_channels=5,
        projection_channels=4,
        object_projection_channels=3,
        hidden_channels=6,
    )

    # Zero initialization must preserve the raw pointmap exactly.
    output = head(
        features,
        output_size=(4, 6),
        patch_shape=(2, 3),
        object_masks=masks,
        object_features=object_features,
        object_valid=valid,
    )
    assert tuple(output.correction.shape) == (2, 4, 6, 3)
    assert tuple(output.object_union.shape) == (2, 4, 6)
    assert bool(output.object_union.any()) and not bool(output.object_union.all())
    assert float(output.correction.abs().max()) == 0.0

    # A constant output must be clipped to the object support, never to the
    # background.  This tests the deployment-time safety invariant directly.
    with torch.no_grad():
        head.residual_head.bias.fill_(1.0)
    masked = head(
        features[:1],
        output_size=(4, 6),
        patch_shape=(2, 3),
        object_masks=masks[:1],
        object_features=object_features[:1],
        object_valid=valid[:1],
    )
    object_values = masked.correction[0][masked.object_union[0]]
    background_values = masked.correction[0][~masked.object_union[0]]
    assert float(object_values.abs().min()) > 0.99
    assert float(background_values.abs().max()) == 0.0

    # Geometry-only B uses the same head and the same mask write gate.
    geometry_only = head(
        features[:1],
        output_size=(4, 6),
        patch_shape=(2, 3),
        object_masks=masks[:1],
        object_features=None,
    )
    assert tuple(geometry_only.correction.shape) == (1, 4, 6, 3)

    # V2.4 must accept a small, temporally consistent correction and fall
    # back to raw geometry when a later observation makes a large jump.
    raw_points = torch.tensor(
        [
            [
                [[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]],
                [[0.0, 0.1, 1.0], [0.1, 0.1, 1.0]],
            ],
            [
                [[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]],
                [[0.0, 0.1, 1.0], [0.1, 0.1, 1.0]],
            ],
        ]
    )
    candidate_points = raw_points.clone()
    candidate_points[0, ..., 2] += 0.01
    candidate_points[1, ..., 2] += 0.50
    gate = apply_world_space_consistency_gate(
        raw_points,
        candidate_points,
        object_masks=torch.ones(2, 1, 2, 2, dtype=torch.bool),
        point_confidence=torch.ones(2, 2, 2),
        track_scores=torch.ones(2, 1),
        config=WorldSpaceGateConfig(
            min_points=2,
            max_correction_m=0.05,
            memory_momentum=0.80,
        ),
    )
    assert gate.object_gate.tolist() == [[True], [False]]
    assert float((gate.points[0] - candidate_points[0]).abs().max()) == 0.0
    assert float((gate.points[1] - raw_points[1]).abs().max()) == 0.0

    # Exercise the actual semantic-map readout with a tiny perfect synthetic
    # object.  This guards the newly added corrected-pointmap evaluator before
    # a long multi-scene training run starts.
    points = torch.tensor(
        [
            [
                [[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]],
                [[0.0, 0.1, 1.0], [0.1, 0.1, 1.0]],
            ]
        ]
    )
    tiny_masks = torch.ones(1, 1, 2, 2, dtype=torch.bool)
    tiny_scores = torch.ones(1, 1)
    tiny_gt = torch.ones(1, 1, 2, 2, dtype=torch.bool)
    tracking = evaluate_tracking_variants(
        scene_id="smoke",
        clip_name="smoke",
        frame_indices=(0,),
        variant_masks={"raw_sam": tiny_masks},
        variant_scores={"raw_sam": tiny_scores},
        raw_variant="raw_sam",
        track_ids=(11,),
        track_prompts=("object",),
        ground_truth=GroundTruthInstances(
            masks=tiny_gt,
            instance_ids=(7,),
            labels=("object",),
            all_visible_instance_ids=(7,),
        ),
        config=TrackingMetricConfig(),
    )
    map_result = evaluate_semantic_object_map(
        scene_id="smoke",
        clip_name="smoke",
        variant="smoke",
        map_policy="smoke",
        aligned_world_points=points,
        target_world_points=points.clone(),
        confidence=torch.ones(1, 2, 2),
        predicted_masks=tiny_masks,
        track_scores=tiny_scores,
        gt_masks=tiny_gt,
        gt_instance_ids=(7,),
        gt_labels=("object",),
        assignments=tracking["assignments"],
        track_ids=(11,),
        config=SemanticMapMetricConfig(
            max_points_per_object=16,
            distance_chunk_size=16,
        ),
    )
    assert float(map_result["summary"]["voxel_iou_5cm"]) == 1.0
    assert float(map_result["summary"]["fscore_5cm"]) == 1.0
    assert float(map_result["summary"]["ghost_point_ratio"]) == 0.0
    print(
        "DINOv3 object-conditioned geometry smoke passed "
        f"features={tuple(features.shape)} output={tuple(output.correction.shape)}"
    )


if __name__ == "__main__":
    main()
