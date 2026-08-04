#!/usr/bin/env python3
"""Small CPU checks for V8 tensors, gradients, GT labels and fallback."""

from __future__ import annotations

import torch

from streaming_couping.src.learned_pose.v6_camera_fusion import V6FusionConfig
from streaming_couping.src.learned_pose.v7_fusion import V7PoseFusion
from streaming_couping.src.learned_pose.v73_correspondence_fusion import (
    perturb_v73_inputs,
)
from streaming_couping.src.learned_pose.v80_supervised_correspondence import (
    V80MatchingConfig,
    V80SupervisedCorrespondenceResidual,
    compute_v80_matching_loss,
    configure_v80_training_stage,
    projection_gradient_norms,
    v80_memory_write,
)


def main() -> None:
    torch.manual_seed(80)
    config = V6FusionConfig(
        hidden_dim=16,
        num_heads=4,
        dropout=0.0,
        min_track_confidence=0.25,
        identity_gate_policy="soft_unknown_strict_memory",
    )
    values, baseline, gt = _inputs()
    base = V7PoseFusion(
        architecture="l0_camera_only",
        camera_dim=values["camera_hidden"].shape[-1],
        appearance_dim=values["appearance"].shape[-1],
        geometry_dim=values["pose_geometry"].shape[-1],
        local_feature_dim=values["local_features"].shape[-1],
        config=config,
    )
    model = V80SupervisedCorrespondenceResidual(
        base_model=base,
        architecture="sam_geometry_transport",
        sam_local_dim=values["sam_local_features"].shape[-1],
        geometry_local_dim=values["local_features"].shape[-1],
        config=config,
        value_mode="geometry_sam_dual",
    )
    output = _forward(model, values, baseline)
    assert output["world_to_camera"].shape == (1, 3, 3, 4)
    assert output["transport_probability"].shape == (1, 3, 1, 4, 4)
    assert output["transported_geometry"].shape == (1, 3, 1, 4, 16)
    assert output["transported_sam"].shape == (1, 3, 1, 4, 16)
    assert torch.equal(output["world_to_camera"], output["base_world_to_camera"])

    matching = V80MatchingConfig(max_distance=0.05, temperature=0.01)
    result = _loss(model, values, baseline, gt, matching)
    assert int(result.supervised_queries) == 8
    parameters = configure_v80_training_stage(model, "matching")
    optimizer = torch.optim.Adam(parameters, lr=5e-3)
    initial = float(result.loss)
    best = initial
    maximum_sam_grad = 0.0
    maximum_geometry_grad = 0.0
    for _ in range(5):
        optimizer.zero_grad(set_to_none=True)
        result = _loss(model, values, baseline, gt, matching)
        result.loss.backward()
        gradients = projection_gradient_norms(model)
        maximum_sam_grad = max(
            maximum_sam_grad, gradients["sam_projection_grad_norm"]
        )
        maximum_geometry_grad = max(
            maximum_geometry_grad,
            gradients["geometry_projection_grad_norm"],
        )
        optimizer.step()
        best = min(best, float(_loss(model, values, baseline, gt, matching).loss))
    final = float(_loss(model, values, baseline, gt, matching).loss)
    assert maximum_sam_grad > 0.0
    assert maximum_geometry_grad > 0.0
    assert best < initial
    assert torch.isfinite(torch.tensor(final))

    probability_before_pose = _forward(
        model, values, baseline
    )["transport_probability"].detach().clone()
    pose_parameters = configure_v80_training_stage(model, "pose")
    pose_optimizer = torch.optim.Adam(pose_parameters, lr=5e-3)
    pose_target = baseline.clone()
    pose_target[:, 1:, 0, 3] = 0.05
    for _ in range(2):
        pose_optimizer.zero_grad(set_to_none=True)
        pose_output = _forward(model, values, baseline)
        pose_loss = (pose_output["world_to_camera"] - pose_target).square().mean()
        pose_loss.backward()
        pose_optimizer.step()
    probability_after_pose = _forward(
        model, values, baseline
    )["transport_probability"].detach()
    assert torch.equal(probability_before_pose, probability_after_pose)

    off, uniform = perturb_v73_inputs(values, "instance_off")
    assert not uniform
    with torch.no_grad():
        model.se3_head[-1].bias.fill_(0.3)
    fallback = _forward(model, off, baseline)
    assert not bool(fallback["active_frames"].any())
    assert torch.equal(fallback["world_to_camera"], fallback["base_world_to_camera"])
    zero = compute_v80_matching_loss(
        fallback,
        current_gt_world=gt,
        current_gt_valid=torch.ones(gt.shape[:-1], dtype=torch.bool),
        source_local_valid=off["local_valid"],
        memory_write=v80_memory_write(off, min_confidence=0.25),
        config=matching,
        sequence_indices=[1, 2],
    )
    assert float(zero.loss) == 0.0
    print("V8 supervised-correspondence smoke passed")


def _inputs():
    batch, sequence, instances, points = 1, 3, 1, 4
    local = torch.randn(batch, sequence, instances, points, 7)
    uv = torch.tensor(
        [[-0.8, -0.8], [0.8, -0.8], [-0.8, 0.8], [0.8, 0.8]]
    )
    local[..., 3:5] = uv
    local[..., -1] = 1.0
    sam_reference = torch.randn(batch, 1, instances, points, 10)
    values = {
        "camera_hidden": torch.randn(batch, sequence, 12),
        "appearance": torch.randn(batch, sequence, instances, 6),
        "pose_geometry": torch.randn(batch, sequence, instances, 7),
        "quality": torch.ones(batch, sequence, instances, 3),
        "observed": torch.ones(batch, sequence, instances, dtype=torch.bool),
        "identity_valid": torch.ones(batch, sequence, instances, dtype=torch.bool),
        "identity_unknown": torch.zeros(batch, sequence, instances, dtype=torch.bool),
        "local_features": local,
        "local_valid": torch.ones(batch, sequence, instances, points, dtype=torch.bool),
        "sam_local_features": sam_reference.repeat(1, sequence, 1, 1, 1)
        + 0.01 * torch.randn(batch, sequence, instances, points, 10),
        "sam_local_uv": uv.reshape(1, 1, 1, points, 2).repeat(
            batch, sequence, instances, 1, 1
        ),
        "sam_local_valid": torch.ones(batch, sequence, instances, points, dtype=torch.bool),
    }
    baseline = torch.eye(4).reshape(1, 1, 4, 4).repeat(
        batch, sequence, 1, 1
    )[..., :3, :4]
    points_world = torch.tensor(
        [[0.0, 0.0, 1.0], [0.2, 0.0, 1.0], [0.0, 0.2, 1.0], [0.2, 0.2, 1.0]]
    )
    gt = points_world.reshape(1, 1, 1, points, 3).repeat(
        batch, sequence, instances, 1, 1
    )
    return values, baseline, gt


def _forward(model, values, baseline):
    return model(
        camera_hidden=values["camera_hidden"],
        baseline_world_to_camera=baseline,
        appearance=values["appearance"],
        geometry=values["pose_geometry"],
        quality=values["quality"],
        observed=values["observed"],
        identity_valid=values["identity_valid"],
        identity_unknown=values["identity_unknown"],
        local_features=values["local_features"],
        local_valid=values["local_valid"],
        sam_local_features=values["sam_local_features"],
        sam_local_uv=values["sam_local_uv"],
        sam_local_valid=values["sam_local_valid"],
        reference_index=0,
    )


def _loss(model, values, baseline, gt, matching):
    output = _forward(model, values, baseline)
    return compute_v80_matching_loss(
        output,
        current_gt_world=gt,
        current_gt_valid=torch.ones(gt.shape[:-1], dtype=torch.bool),
        source_local_valid=values["local_valid"],
        memory_write=v80_memory_write(values, min_confidence=0.25),
        config=matching,
        sequence_indices=[1, 2],
    )


if __name__ == "__main__":
    main()
