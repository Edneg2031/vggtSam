#!/usr/bin/env python3
"""CPU smoke checks for the retained causal baseline."""

from __future__ import annotations

import copy

import torch

from streaming_couping.src.learned_pose.dynamic_instance_baseline import (
    BaselineModelConfig,
    CameraPoseBaseline,
    DynamicInstanceGeometryRefiner,
    se3_exp,
)
from streaming_couping.src.learned_pose.baseline_runtime import (
    OptimizerConfig,
    train_pose_model,
)
from streaming_couping.src.geometry_segmentation import (
    GeometrySegmentationPrompt,
    V6GeometrySegmentationConfig,
    causal_prompts_after_birth,
    select_adaptive_correction,
)


def main() -> None:
    torch.manual_seed(0)
    config = BaselineModelConfig(
        hidden_dim=16,
        min_track_confidence=0.5,
        min_geometry_confidence=0.2,
        min_static_score=0.2,
    )
    # The successful V7.1 path starts at an exact identity update. Verify that
    # the exponential map remains differentiable there and that an actual
    # camera L0 fit cannot silently become a no-op.
    zero_twist = torch.zeros(1, 6, requires_grad=True)
    se3_exp(zero_twist).sum().backward()
    assert zero_twist.grad is not None
    assert bool(torch.isfinite(zero_twist.grad).all())
    assert float(zero_twist.grad.abs().sum()) > 0.0
    train_base = CameraPoseBaseline(camera_dim=8, config=config)
    train_batch = {"camera_hidden": torch.randn(1, 5, 8)}
    train_baseline = _identity_poses(sequence=5)
    train_target = train_baseline.clone()
    train_target[0, 1:, 0, 3] = torch.tensor([0.03, -0.05, 0.08, -0.10])
    training = train_pose_model(
        train_base,
        batch=train_batch,
        baseline=train_baseline,
        target=train_target,
        reference_index=0,
        training_indices=[1, 2, 3, 4],
        steps=80,
        config=OptimizerConfig(
            base_steps=80,
            refiner_steps=80,
            learning_rate=1e-3,
            log_every=20,
            min_train_loss_drop_percent=0.1,
        ),
    )
    assert training["best_step"] > 0
    assert training["loss_drop_percent"] >= 0.1
    assert training["parameter_update_norm"] > 0.0
    base = CameraPoseBaseline(camera_dim=8, config=config)
    model = DynamicInstanceGeometryRefiner(
        base_model=base,
        geometry_dim=7,
        config=config,
    ).eval()
    batch = _dummy_batch()
    baseline = _identity_poses(sequence=5)
    with torch.no_grad():
        output = _forward(model, batch, baseline)

    # Slot 0 is born at frame 1. It can write memory after that frame and may
    # first affect pose at its next reliable observation, frame 2.
    assert not bool(output["active_frames"][0, 1])
    assert bool(output["active_frames"][0, 2])
    assert bool(output["active_frames"][0, 3])
    # Slot 1 is observed but marked non-static, so it never enters pose.
    assert not bool(output["eligible_instances"][0, :, 1].any())
    # Missing support is an exact camera-baseline fallback.
    assert not bool(output["active_frames"][0, 4])
    assert torch.equal(
        output["world_to_camera"][:, 4],
        output["base_world_to_camera"][:, 4],
    )
    # Future input cannot change any earlier result.
    changed = copy.deepcopy(batch)
    changed["local_features"][:, 4] = torch.randn_like(
        changed["local_features"][:, 4]
    ) * 100
    with torch.no_grad():
        changed_output = _forward(model, changed, baseline)
    assert torch.equal(
        output["active_frames"][:, :4], changed_output["active_frames"][:, :4]
    )
    assert torch.allclose(
        output["world_to_camera"][:, :4],
        changed_output["world_to_camera"][:, :4],
        rtol=0,
        atol=1e-7,
    )
    # The validated mask competition always has an explicit raw fallback.
    selected, reason = select_adaptive_correction(
        raw_row={
            "mask_pixels": 16,
            "support_recall": 0.5,
            "box_precision": 0.5,
            "geometry_score": 0.5,
        },
        prompted_row=None,
        config=V6GeometrySegmentationConfig(),
    )
    assert selected is None and reason == "keep_raw:no_prompt_candidate"
    prompt = GeometrySegmentationPrompt(
        box_mask=torch.ones(2, 2, dtype=torch.bool),
        positive_mask=torch.ones(2, 2, dtype=torch.bool),
    )
    causal = causal_prompts_after_birth(
        (prompt, prompt, prompt, prompt), birth_index=2
    )
    assert causal[:3] == (None, None, None) and causal[3] is prompt
    print("dynamic-instance baseline smoke passed")


def _dummy_batch() -> dict[str, torch.Tensor]:
    sequence, instances, points = 5, 2, 4
    quality = torch.ones(1, sequence, instances, 3)
    quality[:, :, 1, 2] = 0.05
    observed = torch.zeros(1, sequence, instances, dtype=torch.bool)
    identity = torch.zeros_like(observed)
    observed[:, 1:4, 0] = True
    identity[:, 1:4, 0] = True
    observed[:, 2:4, 1] = True
    identity[:, 2:4, 1] = True
    local_valid = observed[..., None].expand(-1, -1, -1, points).clone()
    features = torch.randn(1, sequence, instances, points, 7)
    features[..., -1] = 1.0
    return {
        "camera_hidden": torch.randn(1, sequence, 8),
        "quality": quality,
        "observed": observed,
        "identity_valid": identity,
        "local_features": features,
        "local_valid": local_valid,
    }


def _identity_poses(*, sequence: int) -> torch.Tensor:
    pose = torch.zeros(1, sequence, 3, 4)
    pose[..., :3, :3] = torch.eye(3)
    return pose


def _forward(model, batch, baseline):
    return model(
        camera_hidden=batch["camera_hidden"],
        baseline_world_to_camera=baseline,
        quality=batch["quality"],
        observed=batch["observed"],
        identity_valid=batch["identity_valid"],
        local_features=batch["local_features"],
        local_valid=batch["local_valid"],
        reference_index=0,
    )


if __name__ == "__main__":
    main()
