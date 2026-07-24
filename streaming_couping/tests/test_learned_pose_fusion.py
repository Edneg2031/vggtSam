import torch

from streaming_couping.src.learned_pose.config import FusionConfig
from streaming_couping.src.learned_pose.model import InstancePoseAdapter


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
        geometry=values["geometry"],
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
        geometry=values["geometry"],
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
        geometry=values["geometry"],
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
