"""Tests for causal local rigid alignment of persistent object point clouds."""

from __future__ import annotations

import torch

from streaming_couping.src.semantic_mapping.contracts import (
    GeometryFrame,
    ObjectObservation,
    SegmentationFrame,
)
from streaming_couping.src.semantic_mapping.instance_point_alignment import (
    InstancePointAlignmentConfig,
    InstancePointAlignmentMemory,
    apply_point_alignment,
)
from streaming_couping.src.semantic_mapping.mapping import (
    SemanticMapBuilder,
    SemanticMapConfig,
)
from streaming_couping.src.semantic_mapping.pipeline import SemanticMapPipeline


def _cloud() -> torch.Tensor:
    return torch.tensor(
        [
            [0.00, 0.00, 1.00],
            [0.10, 0.00, 1.00],
            [0.00, 0.10, 1.00],
            [0.10, 0.10, 1.00],
            [0.00, 0.00, 1.10],
            [0.10, 0.00, 1.10],
            [0.00, 0.10, 1.10],
            [0.10, 0.10, 1.10],
        ]
    )


def _config() -> InstancePointAlignmentConfig:
    return InstancePointAlignmentConfig(
        history_frames=4,
        max_history_points=64,
        min_history_points=4,
        max_registration_points=64,
        max_match_distance_m=0.30,
        min_matches=4,
        trim_ratio=1.0,
        max_iterations=4,
        min_relative_improvement=0.05,
        max_rotation_deg=5.0,
        max_translation_m=0.20,
    )


def test_local_alignment_recovers_translation_and_updates_history_causally() -> None:
    memory = InstancePointAlignmentMemory(_config())
    reference = _cloud()
    weights = torch.ones(reference.shape[0])

    first = memory.decide(7, reference, weights, frame_id=0)
    assert first.bootstrap is True
    assert first.accepted is False
    memory.update(7, reference, weights, frame_id=0, decision=first)

    shift = torch.tensor([0.06, -0.04, 0.02])
    current = reference + shift
    decision = memory.decide(7, current, weights, frame_id=1)

    assert decision.accepted is True
    assert decision.update_history is True
    assert decision.correction_translation_m > 0.0
    assert decision.correction_translation_m < 0.20
    assert decision.final_rmse_m is not None
    assert decision.initial_rmse_m is not None
    assert decision.final_rmse_m < decision.initial_rmse_m

    aligned = apply_point_alignment(decision.transform, current)
    assert torch.allclose(aligned, reference, atol=1e-4)
    memory.update(7, aligned, weights, frame_id=1, decision=decision)
    assert memory.summary()["accepted_count"] == 1
    assert memory.summary()["stored_history_frames"] == 2


def test_rejected_observation_does_not_pollute_history() -> None:
    memory = InstancePointAlignmentMemory(_config())
    reference = _cloud()
    weights = torch.ones(reference.shape[0])
    first = memory.decide(3, reference, weights, frame_id=0)
    memory.update(3, reference, weights, frame_id=0, decision=first)

    rejected = memory.decide(3, reference + 2.0, weights, frame_id=1)
    assert rejected.accepted is False
    assert rejected.update_history is False
    memory.update(3, reference + 2.0, weights, frame_id=1, decision=rejected)
    summary = memory.summary()
    state = next(item for item in summary["states"] if item["instance_id"] == 3)
    assert state["history_frames"] == 1
    assert state["observation_count"] == 2


def _geometry(points: torch.Tensor, frame_id: int) -> GeometryFrame:
    height, width, _ = points.shape
    return GeometryFrame(
        frame_id=frame_id,
        image_size=(height, width),
        world_points=points,
        camera_to_world=torch.eye(4),
        confidence=torch.ones(height, width),
        valid=torch.ones(height, width, dtype=torch.bool),
        rgb=torch.ones(height, width, 3),
        backend="fake_geometry",
    )


def _segmentation(frame_id: int, size: tuple[int, int]) -> SegmentationFrame:
    return SegmentationFrame(
        frame_id=frame_id,
        image_size=size,
        observations=(
            ObjectObservation(
                category="chair",
                instance_id=7,
                mask=torch.ones(size, dtype=torch.bool),
                score=0.9,
                static_score=1.0,
            ),
        ),
        backend="fake_segmentation",
    )


class _GeometryProvider:
    backend_name = "fake_geometry"

    def __init__(self, frames: tuple[GeometryFrame, ...]) -> None:
        self.frames = frames

    def infer(self, image_paths):
        del image_paths
        return self.frames


class _SegmentationProvider:
    backend_name = "fake_segmentation"

    def __init__(self, frames: tuple[SegmentationFrame, ...]) -> None:
        self.frames = frames

    def infer(self, image_paths, prompts=None):
        del image_paths, prompts
        return self.frames


def test_raw_branch_is_unchanged_and_aligned_branch_only_moves_object_points() -> None:
    reference = _cloud().reshape(2, 4, 3)
    shifted = (reference + torch.tensor([0.06, -0.04, 0.02])).clone()
    geometries = (_geometry(reference, 0), _geometry(shifted, 1))
    segmentations = (
        _segmentation(0, (2, 4)),
        _segmentation(1, (2, 4)),
    )
    mapper = SemanticMapBuilder(
        SemanticMapConfig(
            voxel_size_m=0.01,
            max_points_per_observation=64,
            instance_point_alignment=_config(),
        )
    )
    pipeline = SemanticMapPipeline(
        geometry=_GeometryProvider(geometries),
        segmentation=_SegmentationProvider(segmentations),
        mapper=mapper,
    )
    results = pipeline.run_branches(
        ("frame0.jpg", "frame1.jpg"),
        prompts=("chair",),
        policies=("raw", "instance_point_alignment"),
    )
    raw = results["raw"]
    aligned = results["instance_point_alignment"]

    assert raw.metadata["instance_point_alignment"]["enabled"] is False
    assert aligned.metadata["instance_point_alignment"]["enabled"] is True
    assert raw.metadata["pointmap_modified"] is False
    assert aligned.metadata["pointmap_modified"] is True
    assert torch.allclose(raw.scene_voxel_points, aligned.scene_voxel_points)
    assert torch.allclose(raw.object_tracks[0].points[:8], aligned.object_tracks[0].points[:8])
    assert not torch.allclose(raw.object_tracks[0].points[8:], aligned.object_tracks[0].points[8:])
    assert torch.allclose(
        aligned.object_tracks[0].points[8:],
        reference.reshape(-1, 3),
        atol=1e-4,
    )
    assert torch.equal(geometries[0].camera_to_world, torch.eye(4))
    assert torch.equal(geometries[1].camera_to_world, torch.eye(4))
