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


def test_zero_initialization_is_exact_and_zero_projection_gets_gradient():
    values = _inputs()
    adapter = _adapter()
    refined, logs = adapter.forward_camera(
        values["camera"],
        appearance=values["appearance"],
        geometry=values["geometry"],
        quality=values["quality"],
        observed=values["observed"],
        mode="camera_token_fusion",
    )
    assert torch.equal(refined, values["camera"])
    assert float(logs["residual_rms"]) == 0.0
    target = torch.randn_like(refined)
    (refined - target).square().mean().backward()
    gradient = adapter.camera_fusion.zero_proj.weight.grad
    assert gradient is not None
    assert bool(torch.isfinite(gradient).all())
    assert float(gradient.abs().sum()) > 0.0


def test_module_off_is_exact_after_parameters_change():
    values = _inputs()
    adapter = _adapter()
    with torch.no_grad():
        adapter.camera_fusion.zero_proj.weight.normal_()
    refined, _ = adapter.forward_camera(
        values["camera"],
        appearance=values["appearance"],
        geometry=values["geometry"],
        quality=values["quality"],
        observed=values["observed"],
        mode="camera_token_fusion",
        perturbation="module_off",
    )
    assert torch.equal(refined, values["camera"])


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
        mode="camera_token_fusion",
    )
    assert torch.equal(refined, values["camera"])
    assert float(logs["active_frame_fraction"]) == 0.0


def test_tokenizer_is_causal():
    values = _inputs()
    adapter = _adapter()
    original, valid, _ = adapter.tokenizer(
        values["appearance"],
        values["geometry"],
        values["quality"],
        values["observed"],
        mode="camera_token_fusion",
    )
    changed_appearance = values["appearance"].clone()
    changed_geometry = values["geometry"].clone()
    changed_appearance[:, 3] += 1000.0
    changed_geometry[:, 3] -= 1000.0
    changed, changed_valid, _ = adapter.tokenizer(
        changed_appearance,
        changed_geometry,
        values["quality"],
        values["observed"],
        mode="camera_token_fusion",
    )
    assert torch.equal(original[:, :3], changed[:, :3])
    assert torch.equal(valid[:, :3], changed_valid[:, :3])
    assert not torch.equal(original[:, 3], changed[:, 3])


def test_reference_frame_has_no_active_instance_token():
    values = _inputs()
    adapter = _adapter()
    _, valid, _ = adapter.tokenizer(
        values["appearance"],
        values["geometry"],
        values["quality"],
        values["observed"],
        mode="camera_token_fusion",
    )
    assert not bool(valid[:, 0].any())
    assert bool(valid[:, 1:].all())


def test_zero_geometry_preserves_aligned_trust_mask_but_changes_tokens():
    values = _inputs()
    values["quality"][:, 2, 1, 1:] = 0.0
    adapter = _adapter()
    aligned, aligned_valid, _ = adapter.tokenizer(
        values["appearance"],
        values["geometry"],
        values["quality"],
        values["observed"],
        mode="camera_token_fusion",
    )
    zeroed, zeroed_valid, _ = adapter.tokenizer(
        values["appearance"],
        values["geometry"],
        values["quality"],
        values["observed"],
        mode="camera_token_fusion",
        perturbation="zero_geometry",
    )
    assert torch.equal(aligned_valid, zeroed_valid)
    assert not torch.equal(aligned, zeroed)


def test_all_token_fusion_zero_initialization_is_exact():
    values = _inputs()
    adapter = _adapter()
    levels = {
        layer: torch.randn(1, 4, 6, 16)
        for layer in (4, 11, 17, 23)
    }
    updated, logs = adapter.forward_all_tokens(
        levels,
        appearance=values["appearance"],
        geometry=values["geometry"],
        quality=values["quality"],
        observed=values["observed"],
    )
    for layer in levels:
        assert torch.equal(updated[layer], levels[layer])
        assert float(logs[f"layer_{layer}_residual_rms"]) == 0.0
