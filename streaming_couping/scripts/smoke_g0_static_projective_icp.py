#!/usr/bin/env python3
"""CPU smoke checks for G0 geometry, Jacobians, LM and GT isolation."""

from __future__ import annotations

import math
from dataclasses import replace

import torch

from streaming_couping.src.edge_directional_gt_audit import local_se3_pose
from streaming_couping.src.edge_pose_feasibility import _pose_errors
from streaming_couping.src.static_projective_icp import (
    FORBIDDEN_GT_FIELDS,
    CorrespondenceSet,
    build_branch_static_masks,
    depth_to_vertex_map,
    generate_projective_icp_candidates,
    huber_irls_weights,
    linearize_point_to_plane,
    load_static_projective_icp_config,
    objective_payload_from_cache,
    point_to_plane_energy,
    vertex_normals,
    weighted_normal_equations,
)


def main() -> None:
    torch.manual_seed(0)
    config = load_static_projective_icp_config(
        "streaming_couping/configs/g0_static_projective_icp.yaml"
    )
    _gt_isolation(config)
    _vertex_normal_geometry()
    _mask_controls()
    _jacobian_and_masked_recovery(config)
    _end_to_end_projective_recovery(config)
    print("G0 causal static projective point-to-plane ICP smoke passed")


def _gt_isolation(config) -> None:
    cache = {
        "clip_name": "smoke",
        "frame_indices": list(range(20)),
        "image_size": [16, 16],
        "baseline_depth": torch.ones(20, 16, 16),
        "baseline_depth_confidence": torch.ones(20, 16, 16),
        "scene_scale": 1.0,
        "associated_tracking_masks_output": torch.zeros(
            20, 1, 16, 16, dtype=torch.bool
        ),
        "target_pose_encoding": torch.ones(20, 9),
        "target_world_to_camera": torch.eye(4).repeat(20, 1, 1),
        "target_depth": torch.ones(20, 16, 16),
        "target_world_points": torch.ones(20, 16, 16, 3),
    }
    objective = objective_payload_from_cache(cache)
    assert not (FORBIDDEN_GT_FIELDS & set(objective))
    try:
        generate_projective_icp_candidates(
            payload={"target_world_to_camera": torch.eye(4)},
            raw_world_to_camera=torch.eye(4)[None],
            intrinsics=torch.eye(3)[None],
            config=config,
            device="cpu",
        )
    except ValueError as error:
        assert "GT fields" in str(error)
    else:
        raise AssertionError("G0 must reject GT leakage during generation.")


def _vertex_normal_geometry() -> None:
    h = w = 24
    y, x = torch.meshgrid(
        torch.arange(h, dtype=torch.float32),
        torch.arange(w, dtype=torch.float32),
        indexing="ij",
    )
    depth = (2.0 + 0.002 * x + 0.003 * y)[None]
    k = torch.tensor(
        [[[40.0, 0.0, 11.5], [0.0, 40.0, 11.5], [0.0, 0.0, 1.0]]]
    )
    vertices = depth_to_vertex_map(depth, k)
    normals, valid = vertex_normals(
        vertices,
        depth,
        min_depth=0.05,
        max_depth=25.0,
        discontinuity_abs=0.1,
        discontinuity_rel=0.1,
    )
    assert vertices.shape == (1, h, w, 3)
    assert int(valid.sum()) == (h - 2) * (w - 2)
    assert bool(torch.isfinite(normals[valid]).all())
    assert torch.allclose(
        torch.linalg.vector_norm(normals[valid], dim=-1),
        torch.ones(int(valid.sum())),
        atol=1e-5,
    )


def _mask_controls() -> None:
    exclusion = torch.zeros(4, 24, 24, dtype=torch.bool)
    exclusion[:, 6:14, 8:17] = True
    branches = build_branch_static_masks(
        exclusion,
        branches=(
            "full_image",
            "sam_object_excluded",
            "shifted_object_mask_control",
        ),
    )
    assert bool(branches["full_image"].all())
    assert not bool(branches["sam_object_excluded"][:, 6:14, 8:17].any())
    assert not torch.equal(
        branches["sam_object_excluded"],
        branches["shifted_object_mask_control"],
    )


def _jacobian_and_masked_recovery(config) -> None:
    count = 640
    target_points = torch.empty(count, 3).uniform_(-0.8, 0.8)
    target_points[:, 2] = torch.empty(count).uniform_(1.3, 3.0)
    normals = torch.randn(count, 3)
    normals = normals / normals.norm(dim=-1, keepdim=True)
    true_vertices = target_points.clone()
    source_indices = torch.zeros(count, dtype=torch.long)
    weights = torch.ones(count)
    true_pose = torch.eye(4)
    raw_pose = local_se3_pose(
        torch.tensor(
            [math.radians(0.7), math.radians(-0.5), math.radians(0.3),
             0.012, -0.009, 0.008]
        ),
        true_pose,
    )
    # Source frame is fixed at identity; target is perturbed from identity.
    raw_sequence = torch.stack((true_pose, raw_pose))
    static = CorrespondenceSet(
        target_points=target_points,
        source_vertices=true_vertices,
        source_normals=normals,
        confidence_weights=weights,
        source_index=source_indices,
        projected_in_bounds=count,
        depth_consistent=count,
    )

    # Check the analytic left-SE(3) Jacobian against central differences.
    residual, jacobian, _ = linearize_point_to_plane(
        candidate_target_w2c=raw_pose,
        raw_world_to_camera=raw_sequence,
        correspondences=static,
        target_index=1,
        scene_scale=1.0,
    )
    epsilon = 1e-4
    numeric = []
    for axis in range(6):
        step = torch.zeros(6)
        step[axis] = epsilon
        plus, _, _ = linearize_point_to_plane(
            candidate_target_w2c=local_se3_pose(step, raw_pose),
            raw_world_to_camera=raw_sequence,
            correspondences=static,
            target_index=1,
            scene_scale=1.0,
        )
        minus, _, _ = linearize_point_to_plane(
            candidate_target_w2c=local_se3_pose(-step, raw_pose),
            raw_world_to_camera=raw_sequence,
            correspondences=static,
            target_index=1,
            scene_scale=1.0,
        )
        numeric.append((plus - minus) / (2.0 * epsilon))
    numeric = torch.stack(numeric, dim=-1)
    assert torch.allclose(jacobian, numeric, atol=3e-3, rtol=3e-3)
    assert float(residual.abs().mean()) > 0.0

    solver = replace(
        config.solver,
        rotation_prior_weight=0.0,
        translation_prior_weight=0.0,
        huber_delta_scene_fraction=10.0,
    )
    smoke_config = replace(config, solver=solver)
    static_pose = _frozen_correspondence_gn(
        raw_pose, raw_sequence, static, smoke_config
    )
    static_error = sum(_pose_errors(static_pose, true_pose))
    raw_error = sum(_pose_errors(raw_pose, true_pose))
    assert static_error < 0.10 * raw_error

    # A moving prompted region creates a coherent false point-to-plane pull.
    # The SAM-excluded subset remains geometrically correct; full-image keeps
    # the displaced vertices and therefore converges to a biased compromise.
    polluted_vertices = true_vertices.clone()
    moving = torch.arange(count) < count // 2
    polluted_vertices[moving] += 0.12 * normals[moving]
    polluted = CorrespondenceSet(
        target_points=target_points,
        source_vertices=polluted_vertices,
        source_normals=normals,
        confidence_weights=weights,
        source_index=source_indices,
        projected_in_bounds=count,
        depth_consistent=count,
    )
    retained = ~moving
    excluded = CorrespondenceSet(
        target_points=target_points[retained],
        source_vertices=polluted_vertices[retained],
        source_normals=normals[retained],
        confidence_weights=weights[retained],
        source_index=source_indices[retained],
        projected_in_bounds=int(retained.sum()),
        depth_consistent=int(retained.sum()),
    )
    full_pose = _frozen_correspondence_gn(
        raw_pose, raw_sequence, polluted, smoke_config
    )
    excluded_pose = _frozen_correspondence_gn(
        raw_pose, raw_sequence, excluded, smoke_config
    )
    full_error = sum(_pose_errors(full_pose, true_pose))
    excluded_error = sum(_pose_errors(excluded_pose, true_pose))
    assert excluded_error < 0.25 * full_error


def _frozen_correspondence_gn(
    initial_pose,
    raw_sequence,
    correspondences,
    config,
):
    pose = initial_pose.clone()
    for _ in range(8):
        residual, jacobian, confidence = linearize_point_to_plane(
            candidate_target_w2c=pose,
            raw_world_to_camera=raw_sequence,
            correspondences=correspondences,
            target_index=1,
            scene_scale=1.0,
        )
        weights = confidence * huber_irls_weights(residual, delta=10.0)
        normal, gradient = weighted_normal_equations(
            jacobian, residual, weights
        )
        step = torch.linalg.solve(normal + 1e-5 * torch.eye(6), -gradient)
        before = point_to_plane_energy(
            candidate_target_w2c=pose,
            raw_world_to_camera=raw_sequence,
            target_index=1,
            correspondences=correspondences,
            scene_scale=1.0,
            raw_pose=initial_pose,
            config=config,
        )
        proposal = local_se3_pose(step, pose)
        after = point_to_plane_energy(
            candidate_target_w2c=proposal,
            raw_world_to_camera=raw_sequence,
            target_index=1,
            correspondences=correspondences,
            scene_scale=1.0,
            raw_pose=initial_pose,
            config=config,
        )
        assert after <= before + 1e-7
        pose = proposal
    return pose


def _end_to_end_projective_recovery(config) -> None:
    """Exercise dense association, pyramid and LM as one complete path."""

    from streaming_couping.src.static_projective_icp import (
        build_icp_pyramid,
        optimize_target_pose,
    )

    test_config = replace(
        config,
        pyramid=replace(
            config.pyramid,
            downsample_factors=(2, 1),
            iterations=(10, 10),
        ),
        association=replace(
            config.association,
            source_offsets=(1, 2, 4),
            min_correspondences=48,
            max_points_per_pair=1200,
            depth_abs_scene_fraction=0.20,
            depth_rel_tolerance=0.20,
            normal_depth_abs_scene_fraction=0.20,
            normal_depth_rel_tolerance=0.20,
            min_normal_cosine=0.10,
        ),
        solver=replace(
            config.solver,
            rotation_prior_weight=1e-4,
            translation_prior_weight=1e-4,
            max_design_condition=1e12,
        ),
    )
    frames, h, w = 5, 48, 64
    y, x = torch.meshgrid(
        torch.arange(h, dtype=torch.float32),
        torch.arange(w, dtype=torch.float32),
        indexing="ij",
    )
    surface = (
        2.0
        + 0.08 * torch.sin(x / 7.0)
        + 0.05 * torch.cos(y / 5.0)
        + 0.0004 * (x - 32.0) * (y - 24.0)
    )
    depth = surface.repeat(frames, 1, 1)
    confidence = torch.ones_like(depth)
    intrinsics = torch.tensor(
        [[60.0, 0.0, 31.5], [0.0, 60.0, 23.5], [0.0, 0.0, 1.0]]
    ).repeat(frames, 1, 1)
    static = {
        branch: torch.ones(frames, h, w, dtype=torch.bool)
        for branch in test_config.branches
    }
    levels = build_icp_pyramid(
        depth=depth,
        confidence=confidence,
        intrinsics=intrinsics,
        branch_static=static,
        scene_scale=1.0,
        config=test_config,
        device=torch.device("cpu"),
    )
    true_pose = torch.eye(4)
    poses = true_pose.repeat(frames, 1, 1)
    raw_pose = local_se3_pose(
        torch.tensor(
            [
                math.radians(0.7),
                math.radians(-0.5),
                math.radians(0.3),
                0.012,
                -0.009,
                0.008,
            ]
        ),
        true_pose,
    )
    poses[-1] = raw_pose
    candidate = optimize_target_pose(
        target_index=frames - 1,
        target_frame=frames - 1,
        branch="full_image",
        raw_world_to_camera=poses,
        levels=levels,
        scene_scale=1.0,
        config=test_config,
    )
    raw_rotation, raw_translation = _pose_errors(raw_pose, true_pose)
    candidate_rotation, candidate_translation = _pose_errors(
        candidate.pose, true_pose
    )
    assert candidate.active == 1
    assert candidate.source_pairs == 3
    assert candidate.correspondences >= test_config.association.min_correspondences
    assert candidate.final_geometry_energy < candidate.initial_geometry_energy
    assert candidate_rotation < raw_rotation
    assert candidate_translation < raw_translation


if __name__ == "__main__":
    main()
