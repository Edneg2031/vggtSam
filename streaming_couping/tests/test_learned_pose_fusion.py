from pathlib import Path

import torch

from streaming_couping.src.learned_pose.config import FusionConfig
from streaming_couping.src.learned_pose.model import (
    InstancePoseAdapter,
    ZeroInitializedCrossAttention,
)
from streaming_couping.src.learned_pose.pipeline import (
    _compose_camera_update,
    _so3_exp,
)
from vggtsam.utils.imports import maybe_add_repo_to_path


def _inputs():
    torch.manual_seed(3)
    return {
        "camera": torch.randn(1, 4, 16),
        "appearance": torch.randn(1, 4, 3, 4),
        "geometry": torch.randn(1, 4, 3, 5),
        "quality": torch.ones(1, 4, 3, 3),
        "observed": torch.ones(1, 4, 3, dtype=torch.bool),
    }


def _adapter():
    config = FusionConfig(
        instance_dim=8,
        attention_dim=8,
        num_heads=2,
        dpt_layer_indices=(4, 11, 17, 23),
    )
    return InstancePoseAdapter(
        appearance_dim=4,
        geometry_dim=5,
        token_dim=16,
        config=config,
    )


def test_camera_zero_initialization_is_exact_and_projection_gets_gradient():
    values = _inputs()
    adapter = _adapter()
    refined, logs = adapter.forward_camera(
        values["camera"],
        appearance=values["appearance"],
        pose_geometry=values["geometry"],
        quality=values["quality"],
        observed=values["observed"],
    )
    assert torch.equal(refined, values["camera"])
    assert float(logs["residual_rms"]) == 0.0
    (refined - torch.randn_like(refined)).square().mean().backward()
    gradient = adapter.camera_fusion.zero_proj.weight.grad
    assert gradient is not None
    assert bool(torch.isfinite(gradient).all())
    assert float(gradient.abs().sum()) > 0.0


def test_module_off_is_exact_after_parameters_change():
    values = _inputs()
    adapter = _adapter()
    with torch.no_grad():
        adapter.camera_fusion.zero_proj.weight.normal_()
        for fusion in adapter.patch_token_fusions.values():
            fusion.zero_proj.weight.normal_()
    refined, _ = adapter.forward_camera(
        values["camera"],
        appearance=values["appearance"],
        pose_geometry=values["geometry"],
        quality=values["quality"],
        observed=values["observed"],
        module_off=True,
    )
    levels = {layer: torch.randn(1, 4, 6, 16) for layer in (4, 11, 17, 23)}
    updated, _ = adapter.forward_patch_tokens(
        levels,
        patch_start_idx=2,
        appearance=values["appearance"],
        geometry=values["geometry"],
        quality=values["quality"],
        observed=values["observed"],
        module_off=True,
    )
    assert torch.equal(refined, values["camera"])
    assert all(torch.equal(updated[layer], levels[layer]) for layer in levels)


def test_no_valid_instance_is_exact_after_parameters_change():
    values = _inputs()
    adapter = _adapter()
    with torch.no_grad():
        adapter.camera_fusion.zero_proj.weight.normal_()
    refined, logs = adapter.forward_camera(
        values["camera"],
        appearance=values["appearance"],
        pose_geometry=values["geometry"],
        quality=torch.zeros_like(values["quality"]),
        observed=values["observed"],
    )
    assert torch.equal(refined, values["camera"])
    assert float(logs["active_frame_fraction"]) == 0.0


def test_pose_tokenizer_is_causal_and_ignores_geometry():
    values = _inputs()
    adapter = _adapter()
    original, valid, _ = adapter.tokenizer(
        values["appearance"],
        values["geometry"],
        values["quality"],
        values["observed"],
        branch="pose",
    )
    changed_appearance = values["appearance"].clone()
    changed_geometry = values["geometry"].clone()
    changed_appearance[:, 3] += 1000.0
    changed_geometry[:, :3] -= 1000.0
    changed, changed_valid, _ = adapter.tokenizer(
        changed_appearance,
        changed_geometry,
        values["quality"],
        values["observed"],
        branch="pose",
    )
    assert torch.equal(original[:, :3], changed[:, :3])
    assert torch.equal(valid, changed_valid)
    assert not torch.equal(original[:, 3], changed[:, 3])


def test_reference_frame_initializes_memory_without_active_token():
    values = _inputs()
    adapter = _adapter()
    _, valid, _ = adapter.tokenizer(
        values["appearance"],
        values["geometry"],
        values["quality"],
        values["observed"],
        branch="geometry",
    )
    assert not bool(valid[:, 0].any())
    assert bool(valid[:, 1:].all())


def test_patch_prefix_is_preserved_at_zero_initialization():
    values = _inputs()
    adapter = _adapter()
    levels = {layer: torch.randn(1, 4, 6, 16) for layer in (4, 11, 17, 23)}
    updated, logs = adapter.forward_patch_tokens(
        levels,
        patch_start_idx=2,
        appearance=values["appearance"],
        geometry=values["geometry"],
        quality=values["quality"],
        observed=values["observed"],
    )
    for layer in levels:
        assert torch.equal(updated[layer], levels[layer])
        assert float(logs[f"layer_{layer}_residual_rms"]) == 0.0


def test_strict_identity_rejection_neither_uses_nor_updates_memory():
    values = _inputs()
    config = FusionConfig(
        instance_dim=8,
        attention_dim=8,
        num_heads=2,
        strict_identity_gate=True,
        dpt_layer_indices=(4, 11, 17, 23),
    )
    adapter = InstancePoseAdapter(
        appearance_dim=4,
        geometry_dim=5,
        token_dim=16,
        config=config,
    )
    identity_valid = torch.ones_like(values["observed"])
    identity_valid[:, 1] = False

    _, valid, logs = adapter.tokenizer(
        values["appearance"],
        values["geometry"],
        values["quality"],
        values["observed"],
        identity_valid,
        branch="geometry",
    )

    assert not bool(valid[:, 0].any())
    assert not bool(valid[:, 1].any())
    assert bool(valid[:, 2:].all())
    assert float(logs["memory_updates"][:, 1].sum()) == 0.0


def test_strict_patch_fusion_writes_only_inside_validated_spatial_mask():
    values = _inputs()
    config = FusionConfig(
        instance_dim=8,
        attention_dim=8,
        num_heads=2,
        strict_identity_gate=True,
        patch_mask_dilation=0,
        dpt_layer_indices=(4, 11, 17, 23),
    )
    adapter = InstancePoseAdapter(
        appearance_dim=4,
        geometry_dim=5,
        token_dim=16,
        config=config,
    )
    with torch.no_grad():
        for fusion in adapter.patch_token_fusions.values():
            fusion.zero_proj.weight.normal_()
    levels = {layer: torch.randn(1, 4, 6, 16) for layer in (4, 11, 17, 23)}
    spatial_mask = torch.zeros(1, 4, 1, 4, dtype=torch.bool)
    spatial_mask[:, :, :, 2:] = True

    updated, _ = adapter.forward_patch_tokens(
        levels,
        patch_start_idx=2,
        appearance=values["appearance"],
        geometry=values["geometry"],
        quality=values["quality"],
        observed=values["observed"],
        identity_valid=torch.ones_like(values["observed"]),
        spatial_mask=spatial_mask,
        patch_shape=(1, 4),
    )

    for layer in levels:
        assert torch.equal(updated[layer][:, :, :4], levels[layer][:, :, :4])
        assert not torch.equal(updated[layer][:, 1:, 4:], levels[layer][:, 1:, 4:])


def test_unknown_identity_is_camera_only_and_does_not_update_memory():
    values = _inputs()
    config = FusionConfig(
        instance_dim=8,
        attention_dim=8,
        num_heads=2,
        strict_identity_gate=True,
        unknown_camera_weight=0.25,
        dpt_layer_indices=(4, 11, 17, 23),
    )
    adapter = InstancePoseAdapter(
        appearance_dim=4,
        geometry_dim=5,
        token_dim=16,
        config=config,
    )
    matches = torch.ones_like(values["observed"])
    unknown = torch.zeros_like(values["observed"])
    matches[:, 1] = False
    unknown[:, 1] = True

    _, pose_valid, pose_logs = adapter.tokenizer(
        values["appearance"],
        values["geometry"],
        values["quality"],
        values["observed"],
        matches,
        unknown,
        branch="pose",
    )
    _, geometry_valid, geometry_logs = adapter.geometry_tokenizer(
        values["appearance"],
        values["geometry"],
        values["quality"],
        values["observed"],
        matches,
        unknown,
        branch="geometry",
    )

    assert bool(pose_valid[:, 1].all())
    assert not bool(geometry_valid[:, 1].any())
    assert torch.all(
        pose_logs["_instance_reliability"][:, 1] == 0.25
    )
    assert float(pose_logs["memory_updates"][:, 1].sum()) == 0.0
    assert float(geometry_logs["memory_updates"][:, 1].sum()) == 0.0


def test_mismatch_identity_is_fully_isolated():
    values = _inputs()
    config = FusionConfig(
        instance_dim=8,
        attention_dim=8,
        num_heads=2,
        strict_identity_gate=True,
        dpt_layer_indices=(4, 11, 17, 23),
    )
    adapter = InstancePoseAdapter(
        appearance_dim=4,
        geometry_dim=5,
        token_dim=16,
        config=config,
    )
    matches = torch.ones_like(values["observed"])
    matches[:, 1] = False
    unknown = torch.zeros_like(values["observed"])

    _, valid, logs = adapter.tokenizer(
        values["appearance"],
        values["geometry"],
        values["quality"],
        values["observed"],
        matches,
        unknown,
        branch="pose",
    )

    assert not bool(valid[:, 1].any())
    assert float(logs["memory_updates"][:, 1].sum()) == 0.0
    assert float(logs["_instance_reliability"][:, 1].sum()) == 0.0


def test_unknown_reliability_scales_single_instance_residual():
    config = FusionConfig(
        instance_dim=8,
        attention_dim=8,
        num_heads=2,
    )
    fusion = ZeroInitializedCrossAttention(8, 8, config)
    with torch.no_grad():
        fusion.zero_proj.weight.copy_(torch.eye(8))
    queries = torch.randn(1, 1, 1, 8)
    tokens = torch.randn(1, 1, 1, 8)
    valid = torch.ones(1, 1, 1, dtype=torch.bool)
    full, _ = fusion(
        queries,
        tokens,
        valid,
        instance_reliability=torch.ones(1, 1, 1),
    )
    weak, _ = fusion(
        queries,
        tokens,
        valid,
        instance_reliability=torch.full((1, 1, 1), 0.25),
    )

    assert torch.allclose(
        weak - queries,
        0.25 * (full - queries),
        atol=1e-6,
        rtol=1e-5,
    )


def test_per_instance_spatial_attention_preserves_token_mask_binding():
    values = _inputs()
    config = FusionConfig(
        instance_dim=8,
        attention_dim=8,
        num_heads=2,
        strict_identity_gate=True,
        spatial_attention_mode="per_instance",
        patch_mask_dilation=0,
        dpt_layer_indices=(4, 11, 17, 23),
    )
    adapter = InstancePoseAdapter(
        appearance_dim=4,
        geometry_dim=5,
        token_dim=16,
        config=config,
    )
    with torch.no_grad():
        for fusion in adapter.patch_token_fusions.values():
            fusion.zero_proj.weight.normal_()
    levels = {
        layer: torch.randn(1, 4, 6, 16)
        for layer in (4, 11, 17, 23)
    }
    masks = torch.zeros(1, 4, 3, 1, 4, dtype=torch.bool)
    masks[:, :, 0, :, :2] = True
    masks[:, :, 1, :, 2:3] = True
    masks[:, :, 2, :, 3:] = True
    common = dict(
        patch_start_idx=2,
        appearance=values["appearance"],
        geometry=values["geometry"],
        quality=values["quality"],
        observed=values["observed"],
        identity_valid=torch.ones_like(values["observed"]),
        spatial_mask=masks,
        patch_shape=(1, 4),
    )
    aligned, _ = adapter.forward_patch_tokens(
        levels,
        shuffle_instance_tokens=False,
        **common,
    )
    shuffled, _ = adapter.forward_patch_tokens(
        levels,
        shuffle_instance_tokens=True,
        **common,
    )

    assert any(
        not torch.equal(aligned[layer][:, 1:], shuffled[layer][:, 1:])
        for layer in levels
    )


def test_so3_exponential_is_valid_and_has_gradient_at_zero():
    omega = torch.zeros(2, 3, requires_grad=True)
    rotation = _so3_exp(omega)
    identity = torch.eye(3).expand_as(rotation)
    assert torch.allclose(rotation, identity, atol=1e-7)
    assert torch.allclose(
        rotation.transpose(-1, -2) @ rotation,
        identity,
        atol=1e-6,
    )
    assert torch.allclose(
        torch.linalg.det(rotation),
        torch.ones(2),
        atol=1e-6,
    )
    rotation[..., 2, 1].sum().backward()
    assert omega.grad is not None
    assert float(omega.grad.abs().sum()) > 0.0


def test_bounded_so3_camera_update_is_exact_trainable_and_bounded():
    maybe_add_repo_to_path(
        Path(__file__).resolve().parents[2] / "externals/streamvggt"
    )

    class CameraHead:
        trunk_depth = 1

        def __call__(
            self,
            token_levels,
            *,
            past_key_values_camera,
            use_cache,
        ):
            del use_cache
            hidden = token_levels[-1][:, :, 0]
            quaternion_x = 0.1 * hidden[..., 0]
            zero = torch.zeros_like(quaternion_x)
            pose = torch.stack(
                [
                    zero,
                    zero,
                    zero,
                    quaternion_x,
                    zero,
                    zero,
                    torch.ones_like(quaternion_x),
                    torch.ones_like(quaternion_x),
                    torch.ones_like(quaternion_x),
                ],
                dim=-1,
            )
            return [pose], past_key_values_camera

    raw = torch.zeros(1, 2, 4)
    perturbation = torch.zeros_like(raw, requires_grad=True)
    refined = raw + perturbation
    baseline = torch.zeros(1, 2, 9)
    baseline[..., 6] = 1.0
    baseline[..., 7:] = 1.0
    exact, correction = _compose_camera_update(
        CameraHead(),
        raw,
        refined,
        baseline,
        image_size=(32, 48),
        update_mode="bounded_so3",
        max_rotation_update_degrees=5.0,
    )

    assert torch.equal(exact, baseline)
    assert torch.equal(correction, torch.zeros_like(correction))
    exact[..., 3].sum().backward()
    assert perturbation.grad is not None
    assert float(perturbation.grad.abs().sum()) > 0.0

    large = raw.clone()
    large[..., 0] = 100.0
    bounded, correction = _compose_camera_update(
        CameraHead(),
        raw,
        large,
        baseline,
        image_size=(32, 48),
        update_mode="bounded_so3",
        max_rotation_update_degrees=5.0,
    )
    from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri

    extrinsics, _ = pose_encoding_to_extri_intri(
        bounded,
        image_size_hw=(32, 48),
    )
    rotation = extrinsics[..., :3, :3]
    identity = torch.eye(3).expand_as(rotation)
    assert float(correction.max()) <= 5.001
    assert torch.allclose(
        rotation.transpose(-1, -2) @ rotation,
        identity,
        atol=1e-5,
    )
    assert torch.allclose(
        torch.linalg.det(rotation),
        torch.ones_like(torch.linalg.det(rotation)),
        atol=1e-5,
    )
    assert torch.equal(bounded[..., :3], baseline[..., :3])
    assert torch.equal(bounded[..., 7:], baseline[..., 7:])
