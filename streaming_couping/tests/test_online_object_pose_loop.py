"""Synthetic tests for the external online object-pose loop."""

from __future__ import annotations

import torch

from streaming_couping.src.semantic_mapping.contracts import (
    GeometryFrame,
    ObjectObservation,
    SegmentationFrame,
)
from streaming_couping.src.semantic_mapping.online_object_pose_loop import (
    OnlineObjectPoseLoopConfig,
    OnlineObjectPoseLoopRefiner,
)
from streaming_couping.src.semantic_mapping.object_pose_loss_refinement import (
    ObjectPoseLossRefinementConfig,
)


def _masks() -> tuple[torch.Tensor, torch.Tensor]:
    first = torch.zeros((8, 8), dtype=torch.bool)
    first[1:4, 1:4] = True
    second = torch.zeros((8, 8), dtype=torch.bool)
    second[4:7, 4:7] = True
    return first, second


def _points(shift: torch.Tensor | None = None) -> torch.Tensor:
    height, width = 8, 8
    yy, xx = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )
    points = torch.stack((0.10 * xx, 0.10 * yy, torch.ones_like(xx)), dim=-1)
    if shift is not None:
        first, second = _masks()
        points[first | second] += shift
    return points


def _geometry(frame_id: int, points: torch.Tensor) -> GeometryFrame:
    size = tuple(int(value) for value in points.shape[:2])
    return GeometryFrame(
        frame_id=frame_id,
        image_size=size,
        world_points=points,
        camera_to_world=torch.eye(4),
        confidence=torch.ones(size),
        valid=torch.ones(size, dtype=torch.bool),
        rgb=torch.ones(*size, 3),
        backend="synthetic_horizonstream",
    )


def _segmentation(frame_id: int, *, one_instance: bool = False) -> SegmentationFrame:
    first, second = _masks()
    observations = [
        ObjectObservation(
            category="chair",
            instance_id=7,
            mask=first,
            score=0.95,
            static_score=1.0,
        )
    ]
    if not one_instance:
        observations.append(
            ObjectObservation(
                category="table",
                instance_id=8,
                mask=second,
                score=0.95,
                static_score=1.0,
            )
        )
    return SegmentationFrame(
        frame_id=frame_id,
        image_size=(8, 8),
        observations=tuple(observations),
        backend="synthetic_sam3",
    )


def _config(**overrides) -> OnlineObjectPoseLoopConfig:
    observation = ObjectPoseLossRefinementConfig(
        anchor_frame_count=1,
        max_anchor_observations=1,
        max_history_observations=3,
        max_points_per_observation=64,
        min_points_per_observation=4,
        min_mask_pixels=4,
        max_match_distance_m=0.20,
        trim_ratio=0.90,
        min_matches_per_pair=4,
        min_total_matches=8,
        min_geometry_confidence=0.0,
        pose_prior_weight=0.0,
        device="cpu",
    )
    values = dict(
        observation_config=observation,
        window_size=3,
        max_reference_frames=2,
        max_reference_gap=3,
        anchor_frame_count=1,
        min_independent_instances=2,
        min_matches_per_instance=4,
        min_total_matches=8,
        rematch_iterations=2,
        optimizer_steps=80,
        learning_rate=0.02,
        huber_delta_m=0.05,
        pose_prior_weight=0.0,
        temporal_edge_weight=0.0,
        correction_smoothness_weight=0.0,
        max_local_correction_rotation_deg=5.0,
        max_local_correction_translation_m=0.10,
        min_relative_loss_improvement=0.02,
        max_validation_residual_m=0.10,
        device="cpu",
        trace_optimization=True,
    )
    values.update(overrides)
    return OnlineObjectPoseLoopConfig(**values)


def test_online_loop_uses_two_instances_and_carries_external_correction() -> None:
    shift = torch.tensor([0.02, -0.01, 0.005])
    geometry = (
        _geometry(0, _points()),
        _geometry(1, _points(shift)),
        _geometry(2, _points(shift)),
    )
    segmentation = (
        _segmentation(0),
        _segmentation(1),
        _segmentation(2),
    )

    result = OnlineObjectPoseLoopRefiner(_config()).refine(
        geometry,
        segmentation,
        image_paths=(),
    )

    assert result.summary["causal"] is True
    assert result.summary["horizonstream_feedback"] is False
    assert result.summary["accepted_frame_count"] >= 1
    assert len(result.accepted_edges) >= 2
    assert all(edge.reference_role == "anchor" for edge in result.accepted_edges)
    assert torch.allclose(
        result.raw_camera_to_world[1],
        torch.eye(4),
    )
    assert float(torch.linalg.vector_norm(result.refined_camera_to_world[1][:3, 3])) > 1e-3
    # The second corrected frame is produced from the external corrected state,
    # while the raw HorizonStream pose remains unchanged.
    assert float(torch.linalg.vector_norm(result.refined_camera_to_world[2][:3, 3])) > 1e-3
    assert any(
        row["phase"] == "outer_iteration"
        and row["independent_instance_count"] >= 2
        for row in result.summary["optimization_trace"]
    )


def test_online_loop_rejects_single_instance_correction() -> None:
    shift = torch.tensor([0.02, 0.0, 0.0])
    geometry = (
        _geometry(0, _points()),
        _geometry(1, _points(shift)),
    )
    segmentation = (
        _segmentation(0, one_instance=True),
        _segmentation(1, one_instance=True),
    )

    result = OnlineObjectPoseLoopRefiner(_config()).refine(
        geometry,
        segmentation,
        image_paths=(),
    )

    assert result.summary["accepted_frame_count"] == 0
    assert not result.accepted_edges
    assert torch.equal(
        result.refined_camera_to_world[1],
        result.raw_camera_to_world[1],
    )
    assert any(
        row.get("reason") == "reject:insufficient_independent_matches"
        for row in result.summary["frame_diagnostics"]
    )


def test_online_loop_window_size_counts_current_frame_once() -> None:
    refiner = OnlineObjectPoseLoopRefiner(_config(window_size=1))

    assert refiner._variable_ids(4, (0, 1, 2, 3), {0}) == (4,)
    assert OnlineObjectPoseLoopRefiner(_config(window_size=2))._variable_ids(
        4, (0, 1, 2, 3), {0}
    ) == (3, 4)


def test_temporal_edge_backpropagates_through_both_window_poses() -> None:
    refiner = OnlineObjectPoseLoopRefiner(
        _config(
            temporal_edge_weight=1.0,
            pose_prior_weight=0.0,
            correction_smoothness_weight=0.0,
        )
    )
    raw_poses = {frame_id: torch.eye(4) for frame_id in (0, 1)}
    base_poses = {frame_id: pose.clone() for frame_id, pose in raw_poses.items()}
    deltas = {
        0: torch.nn.Parameter(torch.tensor([0.0, 0.0, 0.0, 0.01, 0.0, 0.0])),
        1: torch.nn.Parameter(torch.tensor([0.0, 0.0, 0.0, 0.02, 0.0, 0.0])),
    }
    poses = refiner._poses_from_deltas(base_poses, deltas)
    loss = refiner._window_loss(poses, deltas, (), raw_poses)
    loss.backward()

    assert deltas[0].grad is not None
    assert deltas[1].grad is not None
    assert float(torch.linalg.vector_norm(deltas[0].grad)) > 0.0
    assert float(torch.linalg.vector_norm(deltas[1].grad)) > 0.0
