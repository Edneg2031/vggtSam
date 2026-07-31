from pathlib import Path

import torch

from streaming_couping.scripts.run_v7_fusion_ablation import (
    _compact_rows,
    _label_rows,
    _local_geometry_features,
    load_v7_config,
)
from streaming_couping.src.learned_pose.config import (
    load_learned_pose_config,
)
from streaming_couping.src.learned_pose.v6_camera_fusion import (
    V6FusionConfig,
)
from streaming_couping.src.learned_pose.v7_fusion import (
    V7_ARCHITECTURES,
    V7PoseFusion,
    perturb_v7_inputs,
)


def _inputs():
    torch.manual_seed(0)
    batch = 1
    sequence = 3
    instances = 2
    camera_dim = 8
    appearance_dim = 4
    geometry_dim = 6
    points = 4
    local_dim = 7
    camera = torch.randn(batch, sequence, camera_dim)
    appearance = torch.randn(
        batch,
        sequence,
        instances,
        appearance_dim,
    )
    geometry = torch.randn(
        batch,
        sequence,
        instances,
        geometry_dim,
    )
    quality = torch.ones(batch, sequence, instances, 3)
    observed = torch.ones(
        batch,
        sequence,
        instances,
        dtype=torch.bool,
    )
    matched = observed.clone()
    unknown = torch.zeros_like(observed)
    local_features = torch.randn(
        batch,
        sequence,
        instances,
        points,
        local_dim,
    )
    local_valid = torch.ones(
        batch,
        sequence,
        instances,
        points,
        dtype=torch.bool,
    )
    baseline = torch.eye(4).reshape(1, 1, 4, 4).repeat(
        batch,
        sequence,
        1,
        1,
    )[..., :3, :4]
    values = {
        "camera_hidden": camera,
        "appearance": appearance,
        "pose_geometry": geometry,
        "quality": quality,
        "observed": observed,
        "identity_valid": matched,
        "identity_unknown": unknown,
        "local_features": local_features,
        "local_valid": local_valid,
    }
    return values, baseline


def test_all_v7_levels_are_zero_initialized_and_reference_exact() -> None:
    inputs, baseline = _inputs()
    config = V6FusionConfig(hidden_dim=16, num_heads=4)

    for architecture in V7_ARCHITECTURES:
        model = V7PoseFusion(
            architecture=architecture,
            camera_dim=inputs["camera_hidden"].shape[-1],
            appearance_dim=inputs["appearance"].shape[-1],
            geometry_dim=inputs["pose_geometry"].shape[-1],
            local_feature_dim=inputs["local_features"].shape[-1],
            config=config,
        )
        output = model(
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

        assert torch.equal(output["world_to_camera"], baseline)
        assert not bool(output["active_frames"][:, 0].any())
        assert bool(output["active_frames"][:, 1:].all())


def test_instance_off_falls_back_to_raw_for_instance_levels() -> None:
    inputs, baseline = _inputs()
    config = V6FusionConfig(hidden_dim=16, num_heads=4)
    perturbed = perturb_v7_inputs(inputs, "instance_off")

    for architecture in V7_ARCHITECTURES[1:]:
        model = V7PoseFusion(
            architecture=architecture,
            camera_dim=inputs["camera_hidden"].shape[-1],
            appearance_dim=inputs["appearance"].shape[-1],
            geometry_dim=inputs["pose_geometry"].shape[-1],
            local_feature_dim=inputs["local_features"].shape[-1],
            config=config,
        )
        output = model(
            camera_hidden=perturbed["camera_hidden"],
            baseline_world_to_camera=baseline,
            appearance=perturbed["appearance"],
            geometry=perturbed["pose_geometry"],
            quality=perturbed["quality"],
            observed=perturbed["observed"],
            identity_valid=perturbed["identity_valid"],
            identity_unknown=perturbed["identity_unknown"],
            local_features=perturbed["local_features"],
            local_valid=perturbed["local_valid"],
            reference_index=0,
        )

        assert torch.equal(output["world_to_camera"], baseline)
        assert not bool(output["active_frames"].any())


def test_all_v7_levels_support_a_finite_training_step() -> None:
    inputs, baseline = _inputs()
    config = V6FusionConfig(hidden_dim=16, num_heads=4)
    target = baseline.clone()
    target[:, 1:, 0, 3] = 0.05

    for architecture in V7_ARCHITECTURES:
        model = V7PoseFusion(
            architecture=architecture,
            camera_dim=inputs["camera_hidden"].shape[-1],
            appearance_dim=inputs["appearance"].shape[-1],
            geometry_dim=inputs["pose_geometry"].shape[-1],
            local_feature_dim=inputs["local_features"].shape[-1],
            config=config,
        )
        output = model(
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
        loss = (output["world_to_camera"] - target).square().mean()
        loss.backward()

        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
        assert gradients
        assert all(bool(torch.isfinite(value).all()) for value in gradients)


def test_appearance_only_removes_geometry_evidence_from_decoupled_levels() -> None:
    inputs, baseline = _inputs()
    config = V6FusionConfig(hidden_dim=16, num_heads=4)
    perturbed = perturb_v7_inputs(inputs, "appearance_only")

    for architecture in V7_ARCHITECTURES[3:]:
        model = V7PoseFusion(
            architecture=architecture,
            camera_dim=inputs["camera_hidden"].shape[-1],
            appearance_dim=inputs["appearance"].shape[-1],
            geometry_dim=inputs["pose_geometry"].shape[-1],
            local_feature_dim=inputs["local_features"].shape[-1],
            config=config,
        )
        output = model(
            camera_hidden=perturbed["camera_hidden"],
            baseline_world_to_camera=baseline,
            appearance=perturbed["appearance"],
            geometry=perturbed["pose_geometry"],
            quality=perturbed["quality"],
            observed=perturbed["observed"],
            identity_valid=perturbed["identity_valid"],
            identity_unknown=perturbed["identity_unknown"],
            local_features=perturbed["local_features"],
            local_valid=perturbed["local_valid"],
            reference_index=0,
        )

        assert torch.equal(output["world_to_camera"], baseline)
        assert not bool(output["active_frames"].any())


def test_local_geometry_features_gather_cached_world_points() -> None:
    world = torch.arange(2 * 3 * 4 * 3).reshape(2, 3, 4, 3).float()
    confidence = torch.ones(2, 3, 4)
    uvd = torch.zeros(2, 1, 5, 3)
    valid = torch.zeros(2, 1, 5, dtype=torch.bool)
    for frame in range(2):
        uvd[frame, 0, :, 0] = torch.tensor([0, 1, 2, 3, 0])
        uvd[frame, 0, :, 1] = torch.tensor([0, 1, 2, 0, 2])
        uvd[frame, 0, :, 2] = 2.0 + frame
        valid[frame, 0] = True
    payload = {
        "baseline_world_points": world,
        "baseline_world_confidence": confidence,
        "instance_uvd": uvd,
        "instance_uvd_valid": valid,
        "scene_origin": torch.zeros(3),
        "scene_scale": 2.0,
    }

    features, selected = _local_geometry_features(
        payload,
        max_points=3,
    )

    assert features.shape == (2, 1, 3, 7)
    assert selected.shape == (2, 1, 3)
    assert bool(selected.all())
    assert bool(torch.isfinite(features).all())


def test_v7_selection_can_retain_raw_when_all_models_are_worse() -> None:
    rows = []
    for split in (
        "temporal_holdout",
        "validation",
        "cross_clip",
        "long30",
    ):
        rows.append(
            {
                "split": split,
                "architecture": "raw_streamvggt",
                "loss": "1.0",
                "split_best": 0,
                "development_score": "",
                "development_best": 0,
            }
        )
        rows.append(
            {
                "split": split,
                "architecture": "l0_camera_only",
                "loss": "1.1",
                "split_best": 0,
                "development_score": "",
                "development_best": 0,
            }
        )

    _label_rows(rows, architectures=("l0_camera_only",))

    assert all(
        int(row["development_best"]) == 1
        for row in rows
        if row["architecture"] == "raw_streamvggt"
    )
    assert all(
        int(row["split_best"]) == 1
        for row in rows
        if row["architecture"] == "raw_streamvggt"
    )


def test_v7_compact_output_has_one_row_per_architecture() -> None:
    rows = []
    splits = (
        "train_capacity",
        "temporal_holdout",
        "validation",
        "cross_clip",
        "long30",
    )
    for architecture, level in (
        ("raw_streamvggt", -1),
        ("l0_camera_only", 0),
    ):
        for split in splits:
            rows.append(
                {
                    "architecture_level": level,
                    "architecture": architecture,
                    "mechanism": "test",
                    "key_source": "test",
                    "value_source": "test",
                    "parameters": 0,
                    "spatial_tokens": 0,
                    "training_seconds": 0,
                    "model_overfit_pass": 0,
                    "development_score": 1,
                    "development_best": int(architecture == "raw_streamvggt"),
                    "split": split,
                    "rotation_deg": 1,
                    "translation_native": 1,
                    "loss": 1,
                    "loss_drop_percent": 0,
                    "active_frames": 0,
                    "split_best": int(architecture == "raw_streamvggt"),
                    "instance_off_loss": "",
                    "wrong_geometry_loss": "",
                    "shuffle_time_loss": "",
                    "appearance_only_loss": "",
                    "geometry_only_loss": "",
                    "wrong_geometry_delta_percent": "",
                    "shuffle_time_delta_percent": "",
                    "reference_exact": 1,
                }
            )

    compact = _compact_rows(
        rows,
        architectures=("l0_camera_only",),
    )

    assert len(compact) == 2
    assert compact[0]["architecture"] == "raw_streamvggt"
    assert compact[1]["architecture"] == "l0_camera_only"
    assert compact[0]["reference_exact_all_splits"] == 1
    assert "long30_wrong_geometry_loss" in compact[1]


def test_v7_data_stage_has_an_independent_four_clip_cache() -> None:
    root = Path(__file__).resolve().parents[2]
    experiment = load_v7_config(
        root / "streaming_couping/configs/v7_fusion_ablation.yaml"
    )
    data = load_learned_pose_config(experiment.data_config)

    assert experiment.data_config.name == "v7_fusion_data.yaml"
    assert len(data.clips) == 4
    assert data.recovery_config.name == "recovery_v7_sam31.yaml"
    assert (
        data.features.cache_dir
        == root / "outputs/streaming_couping_v7_fusion_ablation/cache"
    )
    assert "streaming_couping_v6" not in str(data.output_dir)
    assert "streaming_couping_v6" not in str(data.features.cache_dir)
