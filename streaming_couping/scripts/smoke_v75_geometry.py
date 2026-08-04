#!/usr/bin/env python3
"""Small CPU tensor checks for V7.5 masks, ray solve and memory controls."""

from __future__ import annotations

import torch

from streaming_couping.src.explicit_pose_causality import (
    ExplicitPoseConfig,
    _bbox_masks,
    _random_shift_masks,
    _solve_frame_center,
)
from streaming_couping.src.instance_observations import (
    InstanceRefinementConfig,
    translation_icp,
)
from streaming_couping.src.learned_pose.config import RayPoseConfig


def main() -> None:
    mask = torch.zeros(1, 12, 16, dtype=torch.bool)
    mask[0, 3:8, 5:11] = True
    bbox = _bbox_masks(mask)
    shifted = _random_shift_masks(mask)
    _require(int(bbox.sum()) == 30, "bbox changed a rectangular mask")
    _require(int(shifted.sum()) == int(mask.sum()), "random control changed area")
    _require(not torch.equal(shifted, mask), "random control did not move mask")

    height, width = 18, 24
    intrinsics = torch.tensor(
        [[20.0, 0.0, 11.5], [0.0, 20.0, 8.5], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    rows, columns = torch.meshgrid(
        torch.arange(height, dtype=torch.float64),
        torch.arange(width, dtype=torch.float64),
        indexing="ij",
    )
    pixels = torch.stack([columns, rows, torch.ones_like(columns)], dim=-1)
    directions = pixels @ torch.linalg.inv(intrinsics).T
    directions = directions / torch.linalg.vector_norm(
        directions, dim=-1, keepdim=True
    )
    true_center = torch.tensor([0.04, -0.03, 0.02], dtype=torch.float64)
    points = true_center + 2.0 * directions
    world_to_camera = torch.eye(4, dtype=torch.float64)[:3]
    confidence = torch.ones(height, width)
    masks = torch.ones(1, height, width, dtype=torch.bool)
    config = ExplicitPoseConfig(
        point_confidence_threshold=0.3,
        track_confidence_threshold=0.5,
        matched_point_budget=height * width,
        max_center_shift=0.15,
        center_blend=1.0,
        ray=RayPoseConfig(
            min_points=64,
            max_points=height * width,
            max_center_shift=0.15,
            max_residual_rmse=0.01,
        ),
        refinement=InstanceRefinementConfig(min_instance_points=16),
    )
    center, accepted, diagnostics = _solve_frame_center(
        baseline_world_to_camera=world_to_camera,
        intrinsics=intrinsics,
        world_points=points.float(),
        confidence=confidence,
        masks=masks,
        translation=torch.zeros(3),
        matched_budget=height * width,
        config=config,
    )
    _require(accepted, f"synthetic ray fit rejected: {diagnostics}")
    _require(
        float(torch.linalg.vector_norm(center - true_center)) < 1e-6,
        f"synthetic centre mismatch: {center} vs {true_center}",
    )

    # End-to-end sign check: current pointmap drift is the inverse of the
    # persistent-map ICP correction.  Applying the returned translation before
    # the ray solve must recover the same camera center.
    drift_correction = torch.tensor([0.025, -0.015, 0.01])
    drifted = points.float() - drift_correction[None]
    proposal = translation_icp(
        drifted.reshape(-1, 3),
        points.float().reshape(-1, 3),
        instance_id=0,
        config=config.refinement,
    )
    _require(proposal.accepted, f"synthetic ICP rejected: {proposal}")
    _require(
        float(torch.linalg.vector_norm(proposal.translation - drift_correction))
        < 1e-5,
        f"synthetic ICP sign mismatch: {proposal.translation}",
    )
    corrected_center, corrected, corrected_diagnostics = _solve_frame_center(
        baseline_world_to_camera=world_to_camera,
        intrinsics=intrinsics,
        world_points=drifted,
        confidence=confidence,
        masks=masks,
        translation=proposal.translation,
        matched_budget=height * width,
        config=config,
    )
    _require(corrected, f"ICP+ray fit rejected: {corrected_diagnostics}")
    _require(
        float(torch.linalg.vector_norm(corrected_center - true_center)) < 1e-6,
        f"ICP+ray centre mismatch: {corrected_center} vs {true_center}",
    )
    print("V7.5 explicit-pose tensor smoke passed")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"V7.5 tensor smoke failed: {message}")


if __name__ == "__main__":
    main()
