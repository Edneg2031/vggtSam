from pathlib import Path

import torch

from streaming_couping.scripts.run_v6_camera_overfit import load_v6_config
from streaming_couping.src.learned_pose.v6_camera_fusion import (
    V6CameraFusion,
    V6FusionConfig,
    perturb_instance_inputs,
    se3_exp,
)


def _inputs() -> dict[str, torch.Tensor]:
    torch.manual_seed(4)
    return {
        "camera_hidden": torch.randn(1, 3, 8),
        "baseline_world_to_camera": torch.cat(
            [
                torch.eye(3).reshape(1, 1, 3, 3).expand(1, 3, 3, 3),
                torch.randn(1, 3, 3, 1),
            ],
            dim=-1,
        ),
        "appearance": torch.randn(1, 3, 2, 4),
        "pose_geometry": torch.randn(1, 3, 2, 3),
        "quality": torch.ones(1, 3, 2, 3),
        "observed": torch.ones(1, 3, 2, dtype=torch.bool),
        "identity_valid": torch.ones(1, 3, 2, dtype=torch.bool),
        "identity_unknown": torch.zeros(1, 3, 2, dtype=torch.bool),
    }


def _model() -> V6CameraFusion:
    return V6CameraFusion(
        camera_dim=8,
        appearance_dim=4,
        geometry_dim=3,
        config=V6FusionConfig(
            hidden_dim=8,
            num_heads=2,
            max_rotation_degrees=10.0,
            max_translation_native=0.25,
        ),
    )


def _forward(model: V6CameraFusion, inputs: dict[str, torch.Tensor]):
    return model(
        camera_hidden=inputs["camera_hidden"],
        baseline_world_to_camera=inputs["baseline_world_to_camera"],
        appearance=inputs["appearance"],
        geometry=inputs["pose_geometry"],
        quality=inputs["quality"],
        observed=inputs["observed"],
        identity_valid=inputs["identity_valid"],
        identity_unknown=inputs["identity_unknown"],
        reference_index=0,
    )


def test_se3_zero_is_exact_identity_and_trainable() -> None:
    twist = torch.zeros(2, 6, requires_grad=True)
    transform = se3_exp(twist)
    identity = torch.eye(4).expand_as(transform)
    assert torch.equal(transform, identity)
    transform[..., 0, 3].sum().backward()
    assert twist.grad is not None
    assert float(twist.grad.abs().sum()) > 0.0


def test_v6_zero_initialization_and_reference_are_exact() -> None:
    inputs = _inputs()
    output = _forward(_model(), inputs)
    assert torch.equal(
        output["world_to_camera"],
        inputs["baseline_world_to_camera"],
    )
    assert torch.equal(output["twist"][:, 0], torch.zeros(1, 6))
    assert not bool(output["active_frames"][:, 0].any())
    assert bool(output["active_frames"][:, 1:].all())


def test_v6_gradients_reach_merger_and_instance_encoder() -> None:
    inputs = _inputs()
    model = _model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    target = torch.full((1, 3, 6), 0.05)
    target[:, 0] = 0.0
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        output = _forward(model, inputs)
        loss = (output["twist"] - target).square().mean()
        loss.backward()
        optimizer.step()
    assert model.feature_merger[1].weight.grad is not None
    assert float(model.feature_merger[1].weight.grad.abs().sum()) > 0.0
    first_instance_linear = model.instance_encoder.encoder[1]
    assert first_instance_linear.weight.grad is not None
    assert float(first_instance_linear.weight.grad.abs().sum()) > 0.0


def test_same_checkpoint_ablations_change_only_requested_inputs() -> None:
    inputs = _inputs()
    off = perturb_instance_inputs(inputs, "instance_off")
    camera_off = perturb_instance_inputs(inputs, "camera_off")
    shuffled = perturb_instance_inputs(inputs, "shuffle_time")
    wrong_geometry = perturb_instance_inputs(inputs, "wrong_geometry")
    appearance_only = perturb_instance_inputs(inputs, "appearance_only")
    geometry_only = perturb_instance_inputs(inputs, "geometry_only")

    assert not bool(off["observed"].any())
    assert torch.count_nonzero(camera_off["camera_hidden"]) == 0
    assert torch.equal(shuffled["appearance"][:, 0], inputs["appearance"][:, -1])
    assert torch.equal(shuffled["observed"], inputs["observed"])
    assert torch.equal(shuffled["identity_valid"], inputs["identity_valid"])
    assert torch.equal(wrong_geometry["appearance"], inputs["appearance"])
    assert torch.equal(
        wrong_geometry["pose_geometry"][:, 0],
        inputs["pose_geometry"][:, -1],
    )
    assert torch.count_nonzero(appearance_only["pose_geometry"]) == 0
    assert torch.count_nonzero(geometry_only["appearance"]) == 0


def test_v6_command_and_config_are_retained() -> None:
    root = Path(__file__).resolve().parents[2]
    command = (root / "streaming_couping/commands_v6_camera_overfit.txt").read_text()
    config = (root / "streaming_couping/configs/v6_camera_overfit.yaml").read_text()
    assert "run_v6_camera_overfit" in command
    assert "v6_summary.csv" in command
    assert "00a231a370_90_240_37_68_54" in config
    assert "steps: 1200" in config

    loaded = load_v6_config(
        root / "streaming_couping/configs/v6_camera_overfit.yaml"
    )
    assert loaded.clip_name == "00a231a370_90_240_37_68_54"
    assert loaded.fusion.hidden_dim == 256
    assert loaded.training.steps == 1200
    assert loaded.success.camera_loss_ratio == 0.80
