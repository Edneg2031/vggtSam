from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from streaming_couping.src.semantic_mapping.contracts import (
    GeometryFrame,
    ObjectObservation,
    SegmentationFrame,
)
from streaming_couping.src.semantic_mapping.object_pose_refinement import (
    FeatureMatch,
    ObjectPoseRefinementConfig,
    ObjectPoseRefiner,
    estimate_rigid_transform_ransac,
    weighted_rigid_transform,
    write_pose_refinement_debug,
)
from streaming_couping.src.semantic_mapping.mapping import (
    SemanticMapBuilder,
    SemanticMapConfig,
)
from streaming_couping.src.semantic_mapping.pipeline import SemanticMapPipeline


def test_weighted_rigid_transform_maps_camera_i_to_camera_j() -> None:
    points_i = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.5],
            [0.5, 0.2, 2.0],
            [0.2, 0.8, 1.3],
        ]
    )
    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    translation = torch.tensor([0.4, -0.2, 0.3])
    points_j = points_i @ rotation.T + translation
    transform = weighted_rigid_transform(points_i, points_j)

    assert torch.allclose(transform[:3, :3], rotation, atol=1e-5)
    assert torch.allclose(transform[:3, 3], translation, atol=1e-5)
    assert torch.allclose(
        points_i @ transform[:3, :3].T + transform[:3, 3], points_j, atol=1e-5
    )


def test_ransac_rejects_outlier_and_returns_expected_transform() -> None:
    points_i = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.5, 0.2, 2.0],
            [0.2, 0.8, 1.3],
            [0.3, 0.4, 1.7],
        ]
    )
    translation = torch.tensor([0.2, 0.0, 0.1])
    points_j = points_i + translation
    points_j[-1] = torch.tensor([8.0, -4.0, 3.0])
    config = ObjectPoseRefinementConfig(
        min_matches=3,
        min_inliers=6,
        min_inlier_ratio=0.70,
        ransac_iterations=128,
        ransac_inlier_threshold_m=0.02,
        max_registration_rmse_m=0.02,
    )
    result, reason = estimate_rigid_transform_ransac(
        points_i,
        points_j,
        config=config,
    )

    assert reason == "accept"
    assert result is not None
    assert result.num_matches == 7
    assert result.num_inliers == 6
    assert torch.allclose(result.transform[:3, 3], translation, atol=1e-5)


class _FixedFeatureMatcher:
    backend_name = "fixed_test_features"

    def __init__(self, matches: tuple[FeatureMatch, ...]):
        self.matches = matches
        self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def match(
        self,
        image_i,
        image_j,
        mask_i,
        mask_j,
        *,
        image_size_i,
        image_size_j,
    ):
        del image_i, image_j, image_size_i, image_size_j
        self.calls.append((mask_i.clone(), mask_j.clone()))
        return self.matches


def _depth_geometry(
    frame_id: int,
    *,
    camera_to_world: torch.Tensor,
    target_shift: int = 0,
) -> GeometryFrame:
    size = (16, 16)
    depth = torch.ones(size)
    for v in range(3, 6):
        for u in range(3, 6):
            target_u = u + target_shift
            depth[v, target_u] = 1.0
    return GeometryFrame(
        frame_id=frame_id,
        image_size=size,
        depth=depth,
        intrinsics=torch.eye(3),
        camera_to_world=camera_to_world,
        confidence=torch.ones(size),
        valid=torch.ones(size, dtype=torch.bool),
        rgb=torch.zeros(*size, 3),
        backend="fake_horizonstream",
    )


def test_refiner_uses_persistent_id_for_pairing_and_keeps_local_pose_direction(
    tmp_path: Path,
) -> None:
    image_paths = []
    for index in range(2):
        path = tmp_path / f"frame_{index}.png"
        Image.new("RGB", (16, 16), color=(40 + index, 50, 60)).save(path)
        image_paths.append(path)

    mask_i = torch.zeros(16, 16, dtype=torch.bool)
    mask_i[3:6, 3:6] = True
    mask_j = torch.zeros(16, 16, dtype=torch.bool)
    mask_j[3:6, 4:7] = True
    matches = tuple(
        FeatureMatch(
            u_i=float(u),
            v_i=float(v),
            u_j=float(u + 1),
            v_j=float(v),
            score=1.0,
            similarity=1.0,
        )
        for v in range(3, 6)
        for u in range(3, 6)
    )
    matcher = _FixedFeatureMatcher(matches)
    config = ObjectPoseRefinementConfig(
        min_temporal_gap=5,
        min_mask_pixels=4,
        min_geometry_points=4,
        min_matches=6,
        min_inliers=6,
        min_inlier_ratio=0.70,
        ransac_inlier_threshold_m=0.02,
        max_registration_rmse_m=0.02,
        max_pairs_per_instance=4,
    )
    refiner = ObjectPoseRefiner(config, matcher=matcher)
    geometry = (
        _depth_geometry(0, camera_to_world=torch.eye(4)),
        _depth_geometry(
            10,
            # C_j translation -1 means inv(C_j) C_i translates X_i by +1
            # into camera-j coordinates.
            camera_to_world=torch.tensor(
                [[1.0, 0.0, 0.0, -1.0],
                 [0.0, 1.0, 0.0, 0.0],
                 [0.0, 0.0, 1.0, 0.0],
                 [0.0, 0.0, 0.0, 1.0]]
            ),
            target_shift=1,
        ),
    )
    segmentation = tuple(
        SegmentationFrame(
            frame_id=frame_id,
            image_size=(16, 16),
            observations=(
                ObjectObservation(
                    category="rug",
                    instance_id=7,
                    mask=mask,
                    score=0.95,
                ),
            ),
            backend="fake_sam",
        )
        for frame_id, mask in ((0, mask_i), (10, mask_j))
    )

    result = refiner.refine(geometry, segmentation, image_paths)

    assert len(result.candidates) == 1
    assert len(result.accepted_edges) == 1
    edge = result.accepted_edges[0]
    assert edge.instance_id == 7
    assert edge.category == "rug"
    assert torch.allclose(edge.relative_pose[:3, 3], torch.tensor([1.0, 0.0, 0.0]))
    assert matcher.calls
    assert torch.equal(matcher.calls[0][0], mask_i)
    assert torch.equal(matcher.calls[0][1], mask_j)
    assert result.summary["candidate_pair_count"] == 1
    assert result.summary["accepted_edge_count"] == 1
    assert result.summary["config"]["feature_backend"] == "rgb_patch"


def test_refiner_rejects_extreme_object_pose_disagreement(tmp_path: Path) -> None:
    paths = []
    for index in range(2):
        path = tmp_path / f"frame_{index}.png"
        Image.new("RGB", (16, 16), color=(80, 80, 80)).save(path)
        paths.append(path)
    mask = torch.zeros(16, 16, dtype=torch.bool)
    mask[3:6, 3:6] = True
    shifted_mask = torch.zeros(16, 16, dtype=torch.bool)
    shifted_mask[3:6, 8:11] = True
    matches = tuple(
        FeatureMatch(float(u), float(v), float(u + 5), float(v), 1.0, 1.0)
        for v in range(3, 6)
        for u in range(3, 6)
    )
    refiner = ObjectPoseRefiner(
        ObjectPoseRefinementConfig(
            min_temporal_gap=5,
            min_mask_pixels=4,
            min_geometry_points=4,
            min_matches=6,
            min_inliers=6,
            min_inlier_ratio=0.7,
            ransac_inlier_threshold_m=0.02,
            max_registration_rmse_m=0.02,
            max_translation_disagreement_m=0.5,
            reject_extreme_disagreement_multiplier=1.5,
        ),
        matcher=_FixedFeatureMatcher(matches),
    )
    geometry = (
        _depth_geometry(0, camera_to_world=torch.eye(4)),
        _depth_geometry(10, camera_to_world=torch.eye(4), target_shift=5),
    )
    segmentation = tuple(
        SegmentationFrame(
            frame_id=frame_id,
            image_size=(16, 16),
            observations=(
                ObjectObservation("chair", 2, frame_mask, score=0.95),
            ),
            backend="fake_sam",
        )
        for frame_id, frame_mask in ((0, mask), (10, shifted_mask))
    )

    result = refiner.refine(geometry, segmentation, paths)

    assert not result.accepted_edges
    assert result.rejected_edges[0]["reason"] == "reject:extreme_pose_disagreement"
    assert torch.equal(result.raw_camera_to_world[1], result.refined_camera_to_world[1])


class _StaticGeometryProvider:
    backend_name = "fake_horizonstream"

    def __init__(self, frames):
        self.frames = tuple(frames)

    def infer(self, image_paths):
        assert len(image_paths) == len(self.frames)
        return self.frames


class _StaticSegmentationProvider:
    backend_name = "fake_sam"

    def __init__(self, frames):
        self.frames = tuple(frames)

    def infer(self, image_paths, prompts=None):
        del prompts
        assert len(image_paths) == len(self.frames)
        return self.frames


def test_pipeline_exports_raw_and_refined_results_from_shared_frames(
    tmp_path: Path,
) -> None:
    paths = []
    for index in range(2):
        path = tmp_path / f"pipeline_{index}.png"
        Image.new("RGB", (16, 16), color=(40 + index, 50, 60)).save(path)
        paths.append(path)
    mask_i = torch.zeros(16, 16, dtype=torch.bool)
    mask_i[3:6, 3:6] = True
    mask_j = torch.zeros(16, 16, dtype=torch.bool)
    mask_j[3:6, 4:7] = True
    geometry = (
        _depth_geometry(0, camera_to_world=torch.eye(4)),
        _depth_geometry(
            10,
            camera_to_world=torch.tensor(
                [[1.0, 0.0, 0.0, -1.0],
                 [0.0, 1.0, 0.0, 0.0],
                 [0.0, 0.0, 1.0, 0.0],
                 [0.0, 0.0, 0.0, 1.0]]
            ),
            target_shift=1,
        ),
    )
    segmentation = tuple(
        SegmentationFrame(
            frame_id=frame_id,
            image_size=(16, 16),
            observations=(ObjectObservation("rug", 3, frame_mask, score=0.95),),
            backend="fake_sam",
        )
        for frame_id, frame_mask in ((0, mask_i), (10, mask_j))
    )
    matches = tuple(
        FeatureMatch(float(u), float(v), float(u + 1), float(v), 1.0, 1.0)
        for v in range(3, 6)
        for u in range(3, 6)
    )
    pipeline = SemanticMapPipeline(
        geometry=_StaticGeometryProvider(geometry),
        segmentation=_StaticSegmentationProvider(segmentation),
        mapper=SemanticMapBuilder(
            SemanticMapConfig(
                voxel_size_m=0.5,
                min_geometry_confidence=0.3,
                min_track_score=0.5,
            )
        ),
    )
    run = pipeline.run_with_object_pose_refinement(
        tuple(paths),
        refiner=ObjectPoseRefiner(
            ObjectPoseRefinementConfig(
                min_temporal_gap=5,
                min_mask_pixels=4,
                min_geometry_points=4,
                min_matches=6,
                min_inliers=6,
                min_inlier_ratio=0.7,
                ransac_inlier_threshold_m=0.02,
                max_registration_rmse_m=0.02,
            ),
            matcher=_FixedFeatureMatcher(matches),
        ),
        prompts=("rug",),
    )

    assert set(run.raw_results) == {"raw"}
    assert set(run.refined_results) == {"raw"}
    assert run.refinement.summary["accepted_edge_count"] == 1
    assert run.raw_results["raw"].metadata["pose_variant"] == "raw_horizonstream"
    assert run.refined_results["raw"].metadata["pose_variant"] == "object_pose_refined"
    assert run.refined_results["raw"].voxel_count > 0
    debug_paths = write_pose_refinement_debug(
        run.refinement,
        tmp_path / "object_pose_refinement",
    )
    assert all(Path(path).is_file() for path in debug_paths.values())
