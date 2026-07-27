from pathlib import Path

import torch

from streaming_couping.scripts.run_v6_camera_overfit import (
    _camera_centers,
    _compose_decoupled_output,
    _frame_diagnostic_rows,
    _pose_metrics,
    load_v6_config,
)
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


def _forward(
    model: V6CameraFusion,
    inputs: dict[str, torch.Tensor],
    *,
    mode: str = "fusion",
):
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
        mode=mode,
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


def test_independently_trained_modes_have_fair_active_paths() -> None:
    inputs = _inputs()
    missing = {**inputs, "observed": torch.zeros_like(inputs["observed"])}

    camera = _forward(_model(), missing, mode="camera_only")
    assert bool(camera["active_frames"][:, 1:].all())
    assert not bool(camera["active_frames"][:, 0].any())

    instance = _forward(_model(), inputs, mode="instance_only")
    assert bool(instance["active_frames"][:, 1:].all())
    assert torch.equal(
        instance["world_to_camera"],
        inputs["baseline_world_to_camera"],
    )

    target = torch.full((1, 3, 6), 0.03)
    target[:, 0] = 0.0
    for mode in ("camera_only", "instance_only"):
        model = _model()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            output = _forward(model, inputs, mode=mode)
            (output["twist"] - target).square().mean().backward()
            optimizer.step()
        assert float(_forward(model, inputs, mode=mode)["twist"].abs().sum()) > 0


def test_unknown_training_mode_is_rejected() -> None:
    try:
        _forward(_model(), _inputs(), mode="not_a_mode")
    except ValueError as error:
        assert "training mode" in str(error)
    else:
        raise AssertionError("Expected an invalid V6 mode to be rejected.")


def test_pose_metrics_can_select_only_heldout_frames() -> None:
    target = torch.cat(
        [
            torch.eye(3).reshape(1, 1, 3, 3).expand(1, 3, 3, 3),
            torch.zeros(1, 3, 3, 1),
        ],
        dim=-1,
    )
    predicted = target.clone()
    predicted[:, 1, 0, 3] = 1.0
    predicted[:, 2, 0, 3] = 2.0

    first = _pose_metrics(
        predicted,
        target,
        reference_index=0,
        translation_weight=1.0,
        evaluation_indices=[1],
    )
    second = _pose_metrics(
        predicted,
        target,
        reference_index=0,
        translation_weight=1.0,
        evaluation_indices=[2],
    )
    assert first["translation_native"] == 1.0
    assert second["translation_native"] == 2.0


def test_decoupled_pose_uses_requested_rotation_and_camera_center() -> None:
    baseline = torch.cat(
        [
            torch.eye(3).reshape(1, 1, 3, 3).expand(1, 3, 3, 3),
            torch.zeros(1, 3, 3, 1),
        ],
        dim=-1,
    ).clone()
    rotation_pose = baseline.clone()
    quarter_turn = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    rotation_pose[:, 1:, :3, :3] = quarter_turn

    center_pose = baseline.clone()
    requested_centers = torch.tensor(
        [[[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [-2.0, 0.5, 4.0]]]
    )
    center_pose[..., :3, 3] = -requested_centers
    output = _compose_decoupled_output(
        {
            "world_to_camera": rotation_pose,
            "active_frames": torch.tensor([[False, True, True]]),
        },
        {
            "world_to_camera": center_pose,
            "active_frames": torch.tensor([[False, True, False]]),
        },
        baseline_w2c=baseline,
        reference_index=0,
    )

    assert torch.equal(output["world_to_camera"][:, 0], baseline[:, 0])
    assert torch.equal(
        output["world_to_camera"][:, 1:, :3, :3],
        rotation_pose[:, 1:, :3, :3],
    )
    assert torch.allclose(
        _camera_centers(output["world_to_camera"]),
        requested_centers,
    )
    assert torch.equal(
        output["active_frames"],
        torch.tensor([[False, True, False]]),
    )


def test_frame_diagnostics_report_support_and_error_deltas() -> None:
    inputs = _inputs()
    baseline = inputs["baseline_world_to_camera"].clone()
    baseline[..., :3, 3] = 0.0
    target = baseline.clone()
    camera_pose = baseline.clone()
    instance_pose = baseline.clone()
    camera_pose[:, 1:, 0, 3] = -1.0
    instance_pose[:, 1:, 0, 3] = -2.0
    rows = _frame_diagnostic_rows(
        split="test",
        frame_indices=[10, 20, 30],
        evaluation_indices=[1, 2],
        batch=inputs,
        baseline_w2c=baseline,
        target_w2c=target,
        camera_output={"world_to_camera": camera_pose},
        instance_output={"world_to_camera": instance_pose},
        instance_model=_model(),
    )

    assert [row["frame"] for row in rows] == [20, 30]
    assert [row["usable_instances"] for row in rows] == [2, 2]
    assert [row["geometry_confidence"] for row in rows] == ["1", "1"]
    assert [row["camera_center_delta"] for row in rows] == ["1", "1"]
    assert [row["instance_center_delta"] for row in rows] == ["2", "2"]


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
    assert "v6_cross_clip_summary.csv" in command
    assert "v6_frame_diagnostics.csv" in command
    assert "00a231a370_90_240_37_68_54" in config
    assert "steps: 1200" in config

    loaded = load_v6_config(
        root / "streaming_couping/configs/v6_camera_overfit.yaml"
    )
    assert loaded.clip_name == "00a231a370_90_240_37_68_54"
    assert loaded.test_clip_name == "00a231a370_492_589_37_68_54"
    assert loaded.fusion.hidden_dim == 256
    assert loaded.training.steps == 1200
    assert loaded.success.camera_loss_ratio == 0.80
