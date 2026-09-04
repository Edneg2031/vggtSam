"""Tests for the opt-in persistent-instance point consistency branch."""

from __future__ import annotations

import torch

from streaming_couping.src.semantic_mapping.contracts import (
    GeometryFrame,
    ObjectObservation,
    SegmentationFrame,
)
from streaming_couping.src.semantic_mapping.instance_point_consistency import (
    InstancePointConsistencyConfig,
    InstancePointConsistencyMemory,
)
from streaming_couping.src.semantic_mapping.mapping import (
    SemanticMapBuilder,
    SemanticMapConfig,
)
from streaming_couping.src.semantic_mapping.pipeline import SemanticMapPipeline


def test_instance_consistency_is_causal_and_rejects_far_points() -> None:
    config = InstancePointConsistencyConfig(
        history_frames=2,
        max_history_points=16,
        min_history_points=4,
        support_radius_m=0.03,
        min_support_points=2,
        min_support_ratio=0.5,
        bounds_margin_m=0.03,
        max_novel_points=1,
        novel_weight=0.25,
    )
    memory = InstancePointConsistencyMemory(config)
    first = torch.tensor(
        [
            [0.00, 0.00, 1.00],
            [0.02, 0.00, 1.00],
            [0.04, 0.00, 1.00],
            [0.00, 0.02, 1.00],
        ]
    )
    first_weights = torch.ones(4)

    first_decision = memory.decide(3, first, first_weights, frame_id=0)
    assert first_decision.reason == "bootstrap:history_insufficient"
    assert first_decision.history_frame_count == 0
    memory.update(3, first, first_weights, frame_id=0, decision=first_decision)

    # Only the first frame is visible to this decision.  The current frame is
    # inserted only after decide() returns and update() is called explicitly.
    current = torch.cat(
        (
            first[:3],
            torch.tensor([[0.06, 0.00, 1.00]]),  # bounded novel surface
            torch.tensor([[1.00, 1.00, 1.00]]),  # clear outlier
        ),
        dim=0,
    )
    decision = memory.decide(3, current, torch.ones(5), frame_id=1)
    assert decision.history_frame_count == 1
    assert decision.history_point_count == 4
    assert decision.consistency_ready is True
    assert decision.supported_points == 3
    assert decision.novel_points == 1
    assert decision.output_points == 4
    assert bool(decision.keep_mask[-1]) is False
    assert float(decision.weight_multipliers[3]) == 0.25

    memory.update(3, current, torch.ones(5), frame_id=1, decision=decision)
    summary = memory.summary()
    assert summary["decision_count"] == 2
    assert summary["filtered_points"] == 1
    assert summary["downweighted_points"] == 1


def _geometry(points: torch.Tensor, frame_id: int) -> GeometryFrame:
    height, width, _ = points.shape
    return GeometryFrame(
        frame_id=frame_id,
        image_size=(height, width),
        world_points=points,
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


def test_consistency_changes_semantic_map_but_not_scene_or_raw_tracks() -> None:
    near = torch.tensor(
        [
            [0.00, 0.00, 1.00],
            [0.02, 0.00, 1.00],
            [0.04, 0.00, 1.00],
            [0.06, 0.00, 1.00],
            [0.00, 0.02, 1.00],
            [0.02, 0.02, 1.00],
            [0.04, 0.02, 1.00],
            [0.06, 0.02, 1.00],
        ]
    ).reshape(2, 4, 3)
    second = near.clone()
    second[-1, -1] = torch.tensor([1.00, 1.00, 1.00])
    geometries = (_geometry(near, 0), _geometry(second, 1))
    segmentations = (_segmentation(0, (2, 4)), _segmentation(1, (2, 4)))

    base = SemanticMapBuilder(
        SemanticMapConfig(
            fusion_policy="raw",
            voxel_size_m=0.01,
            max_points_per_observation=64,
        )
    )
    refined = SemanticMapBuilder(
        SemanticMapConfig(
            fusion_policy="instance_point_consistency",
            voxel_size_m=0.01,
            max_points_per_observation=64,
            instance_point_consistency=InstancePointConsistencyConfig(
                min_history_points=4,
                support_radius_m=0.03,
                min_support_points=4,
                min_support_ratio=0.5,
                bounds_margin_m=0.03,
                max_novel_points=1,
                novel_weight=0.25,
            ),
        )
    )
    for geometry, segmentation in zip(geometries, segmentations):
        base.update(geometry, segmentation)
        refined.update(geometry, segmentation)
    raw_result = base.finalize()
    refined_result = refined.finalize()

    # Full-scene geometry is deliberately untouched.  Only semantic object
    # voxel insertion uses the persistent-instance consistency decision.
    assert refined_result.scene_voxel_count == raw_result.scene_voxel_count
    assert refined_result.voxel_count < raw_result.voxel_count
    assert raw_result.object_tracks[0].points.shape[0] == 16
    assert refined_result.object_tracks[0].points.shape[0] == 16
    assert refined.last_stats is not None
    assert refined.last_stats.instance_consistency_filtered_point_count == 1
    assert refined_result.metadata["instance_point_consistency"]["filtered_points"] == 1


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


def test_pipeline_branches_keep_raw_and_consistency_metadata_separate() -> None:
    near = torch.tensor(
        [
            [0.00, 0.00, 1.00],
            [0.02, 0.00, 1.00],
            [0.04, 0.00, 1.00],
            [0.06, 0.00, 1.00],
            [0.00, 0.02, 1.00],
            [0.02, 0.02, 1.00],
            [0.04, 0.02, 1.00],
            [0.06, 0.02, 1.00],
        ]
    ).reshape(2, 4, 3)
    second = near.clone()
    second[-1, -1] = torch.tensor([1.00, 1.00, 1.00])
    geometries = (_geometry(near, 0), _geometry(second, 1))
    segmentations = (_segmentation(0, (2, 4)), _segmentation(1, (2, 4)))
    mapper = SemanticMapBuilder(
        SemanticMapConfig(
            voxel_size_m=0.01,
            max_points_per_observation=64,
            instance_point_consistency=InstancePointConsistencyConfig(
                min_history_points=4,
                support_radius_m=0.03,
                min_support_points=4,
                min_support_ratio=0.5,
                bounds_margin_m=0.03,
                max_novel_points=1,
                novel_weight=0.25,
            ),
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
        policies=("raw", "instance_point_consistency"),
    )

    assert results["raw"].metadata["instance_point_consistency"]["enabled"] is False
    assert (
        results["instance_point_consistency"].metadata[
            "instance_point_consistency"
        ]["enabled"]
        is True
    )
    assert results["raw"].metadata["instance_point_consistency_requested"] is False
    assert (
        results["instance_point_consistency"].metadata[
            "instance_point_consistency_requested"
        ]
        is True
    )
    assert results["raw"].scene_voxel_count == results["instance_point_consistency"].scene_voxel_count
    assert results["raw"].object_tracks[0].points.shape == results[
        "instance_point_consistency"
    ].object_tracks[0].points.shape
    assert results["instance_point_consistency"].voxel_count < results["raw"].voxel_count


def test_default_config_keeps_consistency_disabled() -> None:
    config = SemanticMapConfig()
    assert config.fusion_policy == "raw"
    assert config.instance_point_consistency.enabled is False
    result = SemanticMapBuilder(config).finalize()
    assert result.metadata["instance_point_consistency"]["enabled"] is False
