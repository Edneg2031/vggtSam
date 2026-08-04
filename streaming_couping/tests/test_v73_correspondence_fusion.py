import pytest
import torch

from streaming_couping.scripts.run_v73_correspondence_ablation import (
    V73CheckpointSignatureMismatch,
    _load_checkpoint,
)

from streaming_couping.src.learned_pose.v6_camera_fusion import V6FusionConfig
from streaming_couping.src.learned_pose.v7_fusion import V7PoseFusion
from streaming_couping.src.learned_pose.v73_correspondence_fusion import (
    V73_ARCHITECTURES,
    SAMWeightedGeometryMatcher,
    V73FrozenCorrespondenceResidual,
    _masked_transport_probability,
    perturb_v73_inputs,
)


def _config():
    return V6FusionConfig(
        hidden_dim=16,
        num_heads=4,
        dropout=0.0,
        min_track_confidence=0.25,
        identity_gate_policy="soft_unknown_strict_memory",
    )


def _inputs():
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
        "observed": torch.ones(batch, sequence, instances, dtype=torch.bool),
        "identity_valid": torch.ones(batch, sequence, instances, dtype=torch.bool),
        "identity_unknown": torch.zeros(batch, sequence, instances, dtype=torch.bool),
        "local_features": local,
        "local_valid": torch.ones(batch, sequence, instances, points, dtype=torch.bool),
        "sam_local_features": torch.randn(batch, sequence, instances, points, 10),
        "sam_local_uv": local[..., 3:5].clone(),
        "sam_local_valid": torch.ones(batch, sequence, instances, points, dtype=torch.bool),
    }
    baseline = torch.eye(4).reshape(1, 1, 4, 4).repeat(batch, sequence, 1, 1)[
        ..., :3, :4
    ]
    return values, baseline


def _base(values):
    return V7PoseFusion(
        architecture="l0_camera_only",
        camera_dim=values["camera_hidden"].shape[-1],
        appearance_dim=values["appearance"].shape[-1],
        geometry_dim=values["pose_geometry"].shape[-1],
        local_feature_dim=values["local_features"].shape[-1],
        config=_config(),
    )


def _model(values, architecture):
    return V73FrozenCorrespondenceResidual(
        base_model=_base(values),
        architecture=architecture,
        sam_local_dim=values["sam_local_features"].shape[-1],
        geometry_local_dim=values["local_features"].shape[-1],
        config=_config(),
    )


def _forward(model, values, baseline, *, uniform_sam=False):
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
        uniform_sam=uniform_sam,
    )


def test_all_v73_heads_start_as_exact_frozen_l0():
    values, baseline = _inputs()
    base = _base(values)
    base_output = base(
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
    for architecture in V73_ARCHITECTURES:
        model = _model(values, architecture)
        model.base_model.load_state_dict(base.state_dict())
        output = _forward(model, values, baseline)
        assert torch.equal(output["world_to_camera"], base_output["world_to_camera"])
        assert torch.equal(output["world_to_camera"][:, 0], baseline[:, 0])
        assert bool(
            (
                (output["transport_entropy"] >= 0)
                & (output["transport_entropy"] <= 1.0001)
            ).all()
        )
        assert bool(((output["transport_max"] >= 0) & (output["transport_max"] <= 1.0001)).all())
        assert all(not p.requires_grad for p in model.base_model.parameters())


def test_instance_off_exactly_returns_to_frozen_l0_after_nonzero_head():
    values, baseline = _inputs()
    off, uniform = perturb_v73_inputs(values, "instance_off")
    assert not uniform
    for architecture in V73_ARCHITECTURES:
        model = _model(values, architecture)
        with torch.no_grad():
            model.se3_head[-1].bias.fill_(0.2)
        output = _forward(model, off, baseline)
        base = model.base_model(
            camera_hidden=off["camera_hidden"],
            baseline_world_to_camera=baseline,
            appearance=off["appearance"],
            geometry=off["pose_geometry"],
            quality=off["quality"],
            observed=off["observed"],
            identity_valid=off["identity_valid"],
            identity_unknown=off["identity_unknown"],
            local_features=off["local_features"],
            local_valid=off["local_valid"],
            reference_index=0,
        )
        assert not bool(output["active_frames"].any())
        assert torch.equal(output["world_to_camera"], base["world_to_camera"])


def test_sam_geometry_transport_reduces_exactly_to_geometry_when_sam_is_off():
    values, _ = _inputs()
    geometry = SAMWeightedGeometryMatcher(
        architecture="geometry_transport",
        sam_feature_dim=values["sam_local_features"].shape[-1],
        geometry_feature_dim=values["local_features"].shape[-1],
        config=_config(),
    )
    combined = SAMWeightedGeometryMatcher(
        architecture="sam_geometry_transport",
        sam_feature_dim=values["sam_local_features"].shape[-1],
        geometry_feature_dim=values["local_features"].shape[-1],
        config=_config(),
    )
    for name in (
        "geometry_encoder",
        "geometry_query",
        "geometry_key",
        "residual_encoder",
    ):
        getattr(combined, name).load_state_dict(getattr(geometry, name).state_dict())
    common = {
        "local_features": values["local_features"],
        "local_valid": values["local_valid"],
        "sam_local_features": values["sam_local_features"],
        "sam_local_uv": values["sam_local_uv"],
        "reference_index": 0,
    }
    geometry_output = geometry(
        **common,
        sam_local_valid=torch.zeros_like(values["sam_local_valid"]),
    )
    combined_output = combined(
        **common,
        sam_local_valid=torch.zeros_like(values["sam_local_valid"]),
    )
    assert torch.equal(combined_output["available"], geometry_output["available"])
    assert torch.equal(
        combined_output["instance_evidence"],
        geometry_output["instance_evidence"],
    )
    assert torch.equal(combined_output["transport_entropy"], geometry_output["transport_entropy"])
    assert not bool(combined_output["sam_used"].any())
    assert not bool(combined_output["sam_affinity_delta"].any())


def test_pure_sam_transport_is_inactive_without_sam_but_combined_is_not():
    values, baseline = _inputs()
    off, _ = perturb_v73_inputs(values, "sam_off")
    pure = _forward(_model(values, "sam_transport"), off, baseline)
    combined = _forward(_model(values, "sam_geometry_transport"), off, baseline)
    assert not bool(pure["active_frames"].any())
    assert bool(combined["active_frames"][:, 1:].all())
    assert not bool(combined["sam_used"].any())


def test_causal_memory_accepts_late_birth_only_on_second_observation():
    values, baseline = _inputs()
    for name in ("observed", "identity_valid"):
        values[name][:] = False
        values[name][:, 2:, 0] = True
    values["identity_unknown"][:] = False
    values["local_valid"][:] = False
    values["local_valid"][:, 2:, 0] = True
    values["sam_local_valid"][:] = False
    values["sam_local_valid"][:, 2:, 0] = True
    model = V73FrozenCorrespondenceResidual(
        base_model=_base(values),
        architecture="sam_geometry_transport",
        sam_local_dim=values["sam_local_features"].shape[-1],
        geometry_local_dim=values["local_features"].shape[-1],
        config=_config(),
        memory_mode="causal_last_observation",
    )
    output = _forward(model, values, baseline)
    assert not bool(output["active_frames"][0, 2])
    assert bool(output["active_frames"][0, 3])
    assert not bool(output["memory_mature"][0, 2, 0])
    assert bool(output["memory_mature"][0, 3, 0])

    prefix = {name: value[:, :4] for name, value in values.items()}
    prefix_output = _forward(model, prefix, baseline[:, :4])
    assert torch.equal(
        prefix_output["world_to_camera"],
        output["world_to_camera"][:, :4],
    )


def test_identity_perturbations_preserve_common_gate_and_change_sam_evidence():
    values, baseline = _inputs()
    model = _model(values, "sam_geometry_transport")
    normal = _forward(model, values, baseline)
    wrong, uniform = perturb_v73_inputs(values, "wrong_sam_identity")
    assert not uniform
    changed = _forward(model, wrong, baseline)
    assert torch.equal(normal["active_frames"], changed["active_frames"])
    assert not torch.equal(normal["evidence"], changed["evidence"])


def test_backward_never_updates_frozen_l0():
    values, baseline = _inputs()
    model = _model(values, "sam_geometry_transport")
    output = _forward(model, values, baseline)
    target = baseline.clone()
    target[:, 1:, 0, 3] = 0.1
    (output["world_to_camera"] - target).square().mean().backward()
    assert all(p.grad is None for p in model.base_model.parameters())
    assert any(
        p.grad is not None
        for name, p in model.named_parameters()
        if not name.startswith("base_model.")
    )


def test_masked_transport_is_finite_and_zero_for_all_invalid_rows():
    logits = torch.randn(2, 3, 4)
    probability = _masked_transport_probability(
        logits,
        query_valid=torch.tensor([[True, False, True], [True, True, True]]),
        key_valid=torch.tensor([[True, False, True, False], [False] * 4]),
    )
    assert bool(torch.isfinite(probability).all())
    assert torch.equal(probability[0, 1], torch.zeros(4))
    assert torch.equal(probability[1], torch.zeros(3, 4))
    assert torch.allclose(probability[0, 0].sum(), torch.tensor(1.0))


def test_v73_checkpoint_provenance_mismatch_is_distinct_from_corruption(tmp_path):
    path = tmp_path / "checkpoint.pt"
    torch.save({"signature": "old", "model": {"weight": torch.ones(1)}}, path)
    with pytest.raises(V73CheckpointSignatureMismatch):
        _load_checkpoint(path, "new")

    torch.save({"signature": "new"}, path)
    with pytest.raises(ValueError, match="Invalid V7.3 checkpoint"):
        _load_checkpoint(path, "new")
