import torch

from streaming_couping.scripts.run_v71_instance_causality import (
    _load_checkpoint,
    _slice_batch_prefix,
    _validate_temporal_partition,
)
from streaming_couping.src.learned_pose.v6_camera_fusion import (
    V6FusionConfig,
)
from streaming_couping.src.learned_pose.v7_fusion import (
    V7PoseFusion,
    perturb_v7_inputs,
)
from streaming_couping.src.learned_pose.v71_causal_fusion import (
    V71_RESIDUAL_ARCHITECTURES,
    V71FrozenResidualFusion,
    build_common_instance_state,
)


def _inputs():
    torch.manual_seed(4)
    batch, sequence, instances = 1, 4, 2
    values = {
        "camera_hidden": torch.randn(batch, sequence, 8),
        "appearance": torch.randn(batch, sequence, instances, 5),
        "pose_geometry": torch.randn(batch, sequence, instances, 6),
        "quality": torch.ones(batch, sequence, instances, 3),
        "observed": torch.ones(
            batch,
            sequence,
            instances,
            dtype=torch.bool,
        ),
        "identity_valid": torch.ones(
            batch,
            sequence,
            instances,
            dtype=torch.bool,
        ),
        "identity_unknown": torch.zeros(
            batch,
            sequence,
            instances,
            dtype=torch.bool,
        ),
        "local_features": torch.randn(
            batch,
            sequence,
            instances,
            4,
            7,
        ),
        "local_valid": torch.ones(
            batch,
            sequence,
            instances,
            4,
            dtype=torch.bool,
        ),
    }
    baseline = torch.eye(4).reshape(1, 1, 4, 4).repeat(
        batch,
        sequence,
        1,
        1,
    )[..., :3, :4]
    return values, baseline


def _config():
    return V6FusionConfig(
        hidden_dim=16,
        num_heads=4,
        min_track_confidence=0.25,
        identity_gate_policy="soft_unknown_strict_memory",
    )


def _base(inputs, baseline):
    return V7PoseFusion(
        architecture="l0_camera_only",
        camera_dim=inputs["camera_hidden"].shape[-1],
        appearance_dim=inputs["appearance"].shape[-1],
        geometry_dim=inputs["pose_geometry"].shape[-1],
        local_feature_dim=inputs["local_features"].shape[-1],
        config=_config(),
    )


def _forward(model, inputs, baseline):
    return model(
        camera_hidden=inputs["camera_hidden"],
        baseline_world_to_camera=baseline,
        appearance=inputs["appearance"],
        geometry=inputs["pose_geometry"],
        quality=inputs["quality"],
        observed=inputs["observed"],
        identity_valid=inputs["identity_valid"],
        identity_unknown=inputs["identity_unknown"],
        local_features=inputs["local_features"],
        local_valid=inputs["local_valid"],
        reference_index=0,
    )


def test_common_gate_is_unchanged_by_content_perturbations() -> None:
    inputs, _ = _inputs()
    states = []
    for variant in (
        "normal",
        "wrong_geometry",
        "shuffle_time",
        "appearance_only",
        "geometry_only",
    ):
        current = perturb_v7_inputs(inputs, variant)
        states.append(
            build_common_instance_state(
                appearance=current["appearance"],
                geometry=current["pose_geometry"],
                quality=current["quality"],
                observed=current["observed"],
                identity_valid=current["identity_valid"],
                identity_unknown=current["identity_unknown"],
                config=_config(),
            )
        )

    expected = states[0]
    for current in states[1:]:
        assert torch.equal(current.valid, expected.valid)
        assert torch.equal(current.reliability, expected.reliability)
        assert torch.equal(current.gate_features, expected.gate_features)


def test_all_residuals_are_zero_initialized_over_the_frozen_l0() -> None:
    inputs, baseline = _inputs()
    base = _base(inputs, baseline)
    base_output = _forward(base, inputs, baseline)

    shared_active = None
    for architecture in V71_RESIDUAL_ARCHITECTURES:
        model = V71FrozenResidualFusion(
            base_model=_base(inputs, baseline),
            architecture=architecture,
            appearance_dim=inputs["appearance"].shape[-1],
            geometry_dim=inputs["pose_geometry"].shape[-1],
            local_feature_dim=inputs["local_features"].shape[-1],
            config=_config(),
        )
        output = _forward(model, inputs, baseline)

        assert torch.equal(
            output["world_to_camera"],
            base_output["world_to_camera"],
        )
        assert torch.equal(
            output["world_to_camera"][:, 0],
            baseline[:, 0],
        )
        assert all(
            not parameter.requires_grad
            for parameter in model.base_model.parameters()
        )
        if architecture != "camera_extra_all":
            if shared_active is None:
                shared_active = output["active_frames"]
            else:
                assert torch.equal(output["active_frames"], shared_active)


def test_instance_off_returns_exactly_to_frozen_l0() -> None:
    inputs, baseline = _inputs()
    instance_off = perturb_v7_inputs(inputs, "instance_off")

    for architecture in V71_RESIDUAL_ARCHITECTURES[1:]:
        base = _base(inputs, baseline)
        base_output = _forward(base, instance_off, baseline)
        model = V71FrozenResidualFusion(
            base_model=base,
            architecture=architecture,
            appearance_dim=inputs["appearance"].shape[-1],
            geometry_dim=inputs["pose_geometry"].shape[-1],
            local_feature_dim=inputs["local_features"].shape[-1],
            config=_config(),
        )
        with torch.no_grad():
            model.se3_head[-1].bias.fill_(0.1)
        output = _forward(model, instance_off, baseline)

        assert torch.equal(
            output["world_to_camera"],
            base_output["world_to_camera"],
        )
        assert not bool(output["active_frames"].any())


def test_residual_backward_does_not_touch_frozen_camera_model() -> None:
    inputs, baseline = _inputs()
    model = V71FrozenResidualFusion(
        base_model=_base(inputs, baseline),
        architecture="decoupled_global",
        appearance_dim=inputs["appearance"].shape[-1],
        geometry_dim=inputs["pose_geometry"].shape[-1],
        local_feature_dim=inputs["local_features"].shape[-1],
        config=_config(),
    )
    target = baseline.clone()
    target[:, 1:, 0, 3] = 0.05
    output = _forward(model, inputs, baseline)
    loss = (output["world_to_camera"] - target).square().mean()
    loss.backward()

    assert all(
        parameter.grad is None
        for parameter in model.base_model.parameters()
    )
    residual_gradients = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if not name.startswith("base_model.") and parameter.grad is not None
    ]
    assert residual_gradients
    assert all(
        bool(torch.isfinite(value).all()) for value in residual_gradients
    )


def test_temporal_partition_must_be_complete_and_chronological() -> None:
    groups = {
        "base_train": (105, 120),
        "residual_train": (135,),
        "development": (150,),
        "future": (165, 180),
    }
    _validate_temporal_partition(
        groups,
        frame_indices=(90, 105, 120, 135, 150, 165, 180),
        reference_index=0,
    )

    bad = dict(groups)
    bad["future"] = (150, 180)
    try:
        _validate_temporal_partition(
            bad,
            frame_indices=(90, 105, 120, 135, 150, 165, 180),
            reference_index=0,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("overlapping V7.1 temporal groups were accepted")


def test_training_prefix_and_checkpoint_signature(tmp_path) -> None:
    inputs, _ = _inputs()
    prefix = _slice_batch_prefix(inputs, length=3)
    assert all(value.shape[1] == 3 for value in prefix.values())

    path = tmp_path / "checkpoint.pt"
    torch.save(
        {"signature": "expected", "model": {"weight": torch.ones(1)}},
        path,
    )
    loaded = _load_checkpoint(path, expected_signature="expected")
    assert torch.equal(loaded["model"]["weight"], torch.ones(1))
    try:
        _load_checkpoint(path, expected_signature="different")
    except ValueError:
        pass
    else:
        raise AssertionError("mismatched V7.1 checkpoint was accepted")
