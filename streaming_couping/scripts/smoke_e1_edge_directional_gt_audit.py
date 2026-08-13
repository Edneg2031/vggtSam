#!/usr/bin/env python3
"""CPU smoke checks for E1 candidate/GT isolation and scoring logic."""

from __future__ import annotations

import math
from pathlib import Path

import torch

from streaming_couping.src.edge_directional_gt_audit import (
    DirectionalCandidate,
    VALID_DIRECTIONS,
    generate_directional_candidates,
    load_edge_directional_audit_config,
    local_se3_pose,
    objective_payload_from_cache,
    score_directional_candidates,
)


def main() -> None:
    config = load_edge_directional_audit_config(
        "streaming_couping/configs/e1_edge_directional_gt_audit.yaml"
    )
    cache = {
        "clip_name": "smoke",
        "frame_indices": list(range(12)),
        "image_paths": ["unused"] * 12,
        "image_size": [16, 16],
        "baseline_depth": torch.ones(12, 16, 16),
        "baseline_depth_confidence": torch.ones(12, 16, 16),
        "scene_scale": 1.0,
        "associated_tracking_masks_output": torch.zeros(
            12, 1, 16, 16, dtype=torch.bool
        ),
        "target_pose_encoding": torch.ones(12, 9),
        "target_world_to_camera": torch.eye(4).repeat(12, 1, 1),
        "target_depth": torch.ones(12, 16, 16),
    }
    objective = objective_payload_from_cache(cache)
    assert not any(key.startswith("target_") for key in objective)
    try:
        generate_directional_candidates(
            payload={"target_world_to_camera": torch.eye(4)},
            raw_world_to_camera=torch.eye(4)[None],
            intrinsics=torch.eye(3)[None],
            config=config,
            device="cpu",
        )
    except ValueError as error:
        assert "GT fields" in str(error)
    else:
        raise AssertionError("E1 must reject GT leakage during generation.")

    # Matrix construction must preserve gradients for all six pose variables.
    tangent = torch.tensor(
        [0.02, -0.01, 0.015, 0.03, -0.02, 0.01], requires_grad=True
    )
    local_se3_pose(tangent, torch.eye(4)).square().sum().backward()
    assert tangent.grad is not None
    assert bool(torch.isfinite(tangent.grad).all())
    assert float(tangent.grad.norm()) > 0.0

    rotation_step = math.radians(config.directional.rotation_step_degrees)
    translation_step = config.directional.translation_step_scene_fraction
    negative_delta = torch.tensor(
        [rotation_step, 0.0, 0.0, translation_step, 0.0, 0.0]
    )
    positive_delta = -negative_delta
    rotation_delta = torch.tensor([rotation_step, 0.0, 0.0, 0.0, 0.0, 0.0])
    translation_delta = torch.tensor(
        [0.0, 0.0, 0.0, translation_step, 0.0, 0.0]
    )
    raw = torch.eye(4)
    poses = {
        "raw": raw,
        "negative_joint": local_se3_pose(negative_delta, raw),
        "positive_joint": local_se3_pose(positive_delta, raw),
        "negative_rotation_only": local_se3_pose(rotation_delta, raw),
        "positive_rotation_only": local_se3_pose(-rotation_delta, raw),
        "negative_translation_only": local_se3_pose(translation_delta, raw),
        "positive_translation_only": local_se3_pose(-translation_delta, raw),
    }
    assert tuple(poses) == VALID_DIRECTIONS
    losses = {
        "raw": 1.0,
        "negative_joint": 0.5,
        "positive_joint": 1.5,
        "negative_rotation_only": 0.7,
        "positive_rotation_only": 1.3,
        "negative_translation_only": 0.8,
        "positive_translation_only": 1.2,
    }
    rotation_steps = {
        name: (
            config.directional.rotation_step_degrees
            if "rotation" in name or name.endswith("joint")
            else 0.0
        )
        for name in VALID_DIRECTIONS
    }
    rotation_steps["raw"] = 0.0
    translation_steps = {
        name: (
            translation_step
            if "translation" in name or name.endswith("joint")
            else 0.0
        )
        for name in VALID_DIRECTIONS
    }
    translation_steps["raw"] = 0.0
    candidates = []
    targets = torch.eye(4).repeat(12, 1, 1)
    for index, frame in enumerate(config.evaluation_frames):
        targets[index] = poses["negative_joint"]
        for branch in config.branches:
            candidates.append(
                DirectionalCandidate(
                    target_sequence_index=index,
                    target_frame_index=frame,
                    branch=branch,
                    active=1,
                    failure_reason="ok",
                    locked_pairs=1,
                    locked_points=128,
                    rotation_gradient_norm=1.0,
                    translation_gradient_norm=1.0,
                    raw_loss=1.0,
                    poses={key: value.clone() for key, value in poses.items()},
                    losses=dict(losses),
                    rotation_steps_deg=dict(rotation_steps),
                    translation_steps_native=dict(translation_steps),
                )
            )
    scored = score_directional_candidates(
        candidates,
        target_world_to_camera=targets,
        config=config,
    )
    assert len(scored["rows"]) == 36
    assert len(scored["fold_summary"]) == 9
    assert scored["decision"]["any_edge_direction_all_fold_pass"] == 1
    assert scored["decision"]["selected_pose_modified"] == 0
    assert all(int(row["fold_pass"]) == 1 for row in scored["fold_summary"])
    assert all(
        float(row["negative_joint_rotation_error_deg"]) < float(row["raw_rotation_error_deg"])
        for row in scored["rows"]
    )
    assert all(
        float(row["negative_joint_center_error_native"]) < float(row["raw_center_error_native"])
        for row in scored["rows"]
    )
    print("E1 fixed-step directional GT-isolation smoke passed")


if __name__ == "__main__":
    main()
