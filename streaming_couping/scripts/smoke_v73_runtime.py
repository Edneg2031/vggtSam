#!/usr/bin/env python3
"""Dependency-free V7.3 tensor smoke checks for servers without pytest."""

from __future__ import annotations

import torch

from streaming_couping.src.instance_observations import InstanceRefinementConfig
from streaming_couping.src.learned_pose.observations import (
    build_geometry_observations,
)
from streaming_couping.src.learned_pose.v6_camera_fusion import V6FusionConfig
from streaming_couping.src.learned_pose.v7_fusion import V7PoseFusion
from streaming_couping.src.learned_pose.v73_correspondence_fusion import (
    V73_ARCHITECTURES,
    V73FrozenCorrespondenceResidual,
    _masked_transport_probability,
    perturb_v73_inputs,
)


def main() -> None:
    values, baseline = _inputs()
    _check_zero_initialization(values, baseline)
    _check_instance_off(values, baseline)
    _check_sam_fallback(values, baseline)
    _check_causal_dynamic_birth(values, baseline)
    _check_dynamic_geometry_birth()
    _check_masked_transport()
    _check_frozen_backward(values, baseline)
    print("V7.3 dependency-free tensor smoke passed")


def _config() -> V6FusionConfig:
    return V6FusionConfig(
        hidden_dim=16,
        num_heads=4,
        dropout=0.0,
        min_track_confidence=0.25,
        identity_gate_policy="soft_unknown_strict_memory",
    )


def _inputs() -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    torch.manual_seed(73)
    batch, sequence, instances, points = 1, 5, 3, 8
    local = torch.randn(batch, sequence, instances, points, 7)
    local[..., 3:5] = torch.rand_like(local[..., 3:5]) * 2 - 1
    local[..., -1] = torch.rand_like(local[..., -1]) + 0.25
    values = {
        "camera_hidden": torch.randn(batch, sequence, 12),
        "appearance": torch.randn(batch, sequence, instances, 6),
        "pose_geometry": torch.randn(batch, sequence, instances, 7),
        "quality": torch.ones(batch, sequence, instances, 3),
        "observed": torch.ones(
            batch, sequence, instances, dtype=torch.bool
        ),
        "identity_valid": torch.ones(
            batch, sequence, instances, dtype=torch.bool
        ),
        "identity_unknown": torch.zeros(
            batch, sequence, instances, dtype=torch.bool
        ),
        "local_features": local,
        "local_valid": torch.ones(
            batch, sequence, instances, points, dtype=torch.bool
        ),
        "sam_local_features": torch.randn(
            batch, sequence, instances, points, 10
        ),
        "sam_local_uv": local[..., 3:5].clone(),
        "sam_local_valid": torch.ones(
            batch, sequence, instances, points, dtype=torch.bool
        ),
    }
    baseline = torch.eye(4).reshape(1, 1, 4, 4).repeat(
        batch, sequence, 1, 1
    )[..., :3, :4]
    return values, baseline


def _base(values: dict[str, torch.Tensor]) -> V7PoseFusion:
    return V7PoseFusion(
        architecture="l0_camera_only",
        camera_dim=int(values["camera_hidden"].shape[-1]),
        appearance_dim=int(values["appearance"].shape[-1]),
        geometry_dim=int(values["pose_geometry"].shape[-1]),
        local_feature_dim=int(values["local_features"].shape[-1]),
        config=_config(),
    )


def _model(
    values: dict[str, torch.Tensor], architecture: str, *,
    memory_mode: str = "fixed_reference",
) -> V73FrozenCorrespondenceResidual:
    return V73FrozenCorrespondenceResidual(
        base_model=_base(values),
        architecture=architecture,
        sam_local_dim=int(values["sam_local_features"].shape[-1]),
        geometry_local_dim=int(values["local_features"].shape[-1]),
        config=_config(),
        memory_mode=memory_mode,
    )


def _forward(
    model: V73FrozenCorrespondenceResidual,
    values: dict[str, torch.Tensor],
    baseline: torch.Tensor,
) -> dict[str, torch.Tensor]:
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


def _base_forward(
    model: V7PoseFusion,
    values: dict[str, torch.Tensor],
    baseline: torch.Tensor,
) -> dict[str, torch.Tensor]:
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
        reference_index=0,
    )


def _check_zero_initialization(values, baseline) -> None:
    shared_base = _base(values)
    expected = _base_forward(shared_base, values, baseline)
    for architecture in V73_ARCHITECTURES:
        model = _model(values, architecture)
        model.base_model.load_state_dict(shared_base.state_dict())
        output = _forward(model, values, baseline)
        _require(
            torch.equal(
                output["world_to_camera"], expected["world_to_camera"]
            ),
            f"{architecture} zero initialization does not reproduce L0",
        )
        _require(
            torch.equal(output["world_to_camera"][:, 0], baseline[:, 0]),
            f"{architecture} changed the reference pose",
        )
        _require(
            bool(torch.isfinite(output["transport_entropy"]).all()),
            f"{architecture} produced non-finite transport statistics",
        )


def _check_instance_off(values, baseline) -> None:
    off, _ = perturb_v73_inputs(values, "instance_off")
    for architecture in V73_ARCHITECTURES:
        model = _model(values, architecture)
        with torch.no_grad():
            model.se3_head[-1].bias.fill_(0.2)
        output = _forward(model, off, baseline)
        expected = _base_forward(model.base_model, off, baseline)
        _require(
            not bool(output["active_frames"].any()),
            f"{architecture} remains active under instance_off",
        )
        _require(
            torch.equal(
                output["world_to_camera"], expected["world_to_camera"]
            ),
            f"{architecture} does not return exactly to L0",
        )


def _check_sam_fallback(values, baseline) -> None:
    off, _ = perturb_v73_inputs(values, "sam_off")
    pure = _forward(_model(values, "sam_transport"), off, baseline)
    combined = _forward(
        _model(values, "sam_geometry_transport"), off, baseline
    )
    _require(
        not bool(pure["active_frames"].any()),
        "pure SAM transport remains active with no SAM token",
    )
    _require(
        bool(combined["active_frames"][:, 1:].all()),
        "combined transport failed to retain geometry fallback",
    )
    _require(
        not bool(combined["sam_used"].any()),
        "combined transport reports SAM use after sam_off",
    )


def _check_causal_dynamic_birth(values, baseline) -> None:
    dynamic = {name: value.clone() for name, value in values.items()}
    # Keep only instance 0 and make it first appear at sequence index 2.
    for name in ("observed", "identity_valid"):
        dynamic[name][:] = False
        dynamic[name][:, 2:, 0] = True
    dynamic["identity_unknown"][:] = False
    dynamic["local_valid"][:] = False
    dynamic["local_valid"][:, 2:, 0] = True
    dynamic["sam_local_valid"][:] = False
    dynamic["sam_local_valid"][:, 2:, 0] = True
    model = _model(
        dynamic,
        "sam_geometry_transport",
        memory_mode="causal_last_observation",
    )
    output = _forward(model, dynamic, baseline)
    _require(
        not bool(output["active_frames"][0, 2]),
        "dynamic instance changed pose on its birth frame",
    )
    _require(
        bool(output["active_frames"][0, 3]),
        "dynamic instance was not active on its second observation",
    )
    _require(
        not bool(output["memory_mature"][0, 2, 0])
        and bool(output["memory_mature"][0, 3, 0]),
        "causal matcher memory maturity is incorrect",
    )

    prefix_values = {name: value[:, :4] for name, value in dynamic.items()}
    prefix_output = _forward(model, prefix_values, baseline[:, :4])
    _require(
        torch.equal(
            prefix_output["world_to_camera"],
            output["world_to_camera"][:, :4],
        ),
        "future observations changed a causal prefix",
    )


def _check_dynamic_geometry_birth() -> None:
    sequence, height, width = 4, 4, 4
    y, x = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )
    points = torch.stack([x * 0.01, y * 0.01, torch.ones_like(x)], dim=-1)
    world = points[None].repeat(sequence, 1, 1, 1)
    confidence = torch.ones(sequence, height, width)
    masks = torch.zeros(sequence, 2, height, width, dtype=torch.bool)
    masks[:, 0] = True
    masks[2:, 1, :2, :2] = True
    observations = build_geometry_observations(
        world_points=world,
        confidence=confidence,
        masks=masks,
        scores=torch.ones(sequence, 2),
        instance_ids=(0, 1),
        frame_indices=(90, 105, 120, 135),
        reference_index=0,
        confidence_threshold=0.1,
        refinement=InstanceRefinementConfig(
            min_instance_points=4,
            icp_max_points=16,
            map_max_points=32,
            min_participating_instances=1,
            compute_device="cpu",
        ),
        sampled_instance_points=8,
        causal_instance_memory=True,
    )
    _require(
        observations["instance_birth_indices"].tolist() == [0, 2],
        "late geometry-supported instance birth was not recorded",
    )
    _require(
        bool(observations["identity_valid"][2, 1]),
        "late birth did not initialize its persistent identity",
    )
    _require(
        not bool(observations["observed"][1, 1]),
        "late instance leaked into a frame before birth",
    )


def _check_masked_transport() -> None:
    logits = torch.randn(2, 3, 4)
    probability = _masked_transport_probability(
        logits,
        query_valid=torch.tensor(
            [[True, False, True], [True, True, True]]
        ),
        key_valid=torch.tensor(
            [[True, False, True, False], [False, False, False, False]]
        ),
    )
    _require(
        bool(torch.isfinite(probability).all()),
        "masked transport produced non-finite probability",
    )
    _require(
        torch.equal(probability[1], torch.zeros(3, 4)),
        "all-invalid transport is not exactly zero",
    )
    _require(
        torch.equal(probability[0, 1], torch.zeros(4)),
        "invalid query transport is not exactly zero",
    )


def _check_frozen_backward(values, baseline) -> None:
    model = _model(values, "sam_geometry_transport")
    output = _forward(model, values, baseline)
    target = baseline.clone()
    target[:, 1:, 0, 3] = 0.1
    (output["world_to_camera"] - target).square().mean().backward()
    _require(
        all(parameter.grad is None for parameter in model.base_model.parameters()),
        "backward propagated into frozen L0",
    )
    _require(
        any(
            parameter.grad is not None
            for name, parameter in model.named_parameters()
            if not name.startswith("base_model.")
        ),
        "V7.3 residual received no gradient",
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"V7.3 smoke failed: {message}")


if __name__ == "__main__":
    main()
