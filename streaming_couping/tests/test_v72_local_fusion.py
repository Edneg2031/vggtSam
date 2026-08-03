import csv

import pytest
import torch

from streaming_couping.scripts.run_v72_local_token_ablation import (
    V72CheckpointSignatureMismatch,
    _assert_v71_baseline_metrics,
    _load_checkpoint,
)
from streaming_couping.src.learned_pose.config import load_learned_pose_config
from streaming_couping.src.learned_pose.observations import (
    sample_sam_instance_tokens,
)
from streaming_couping.src.learned_pose.v6_camera_fusion import V6FusionConfig
from streaming_couping.src.learned_pose.v7_fusion import V7PoseFusion
from streaming_couping.src.learned_pose.v72_local_fusion import (
    V72_ARCHITECTURES,
    V72FrozenLocalResidual,
    perturb_v72_inputs,
)


def _inputs():
    torch.manual_seed(19)
    batch, sequence, instances, points = 1, 5, 3, 8
    values = {
        "camera_hidden": torch.randn(batch, sequence, 12),
        "appearance": torch.randn(batch, sequence, instances, 6),
        "pose_geometry": torch.randn(batch, sequence, instances, 7),
        "quality": torch.ones(batch, sequence, instances, 3),
        "observed": torch.ones(batch, sequence, instances, dtype=torch.bool),
        "identity_valid": torch.ones(batch, sequence, instances, dtype=torch.bool),
        "identity_unknown": torch.zeros(batch, sequence, instances, dtype=torch.bool),
        "local_features": torch.randn(batch, sequence, instances, points, 7),
        "local_valid": torch.ones(
            batch, sequence, instances, points, dtype=torch.bool
        ),
        "sam_local_features": torch.randn(
            batch, sequence, instances, points, 10
        ),
        "sam_local_uv": torch.rand(batch, sequence, instances, points, 2) * 2 - 1,
        "sam_local_valid": torch.ones(
            batch, sequence, instances, points, dtype=torch.bool
        ),
    }
    baseline = torch.eye(4).reshape(1, 1, 4, 4).repeat(
        batch, sequence, 1, 1
    )[..., :3, :4]
    return values, baseline


def _config():
    return V6FusionConfig(
        hidden_dim=16,
        num_heads=4,
        min_track_confidence=0.25,
        identity_gate_policy="soft_unknown_strict_memory",
    )


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
    return V72FrozenLocalResidual(
        base_model=_base(values),
        architecture=architecture,
        sam_local_dim=values["sam_local_features"].shape[-1],
        geometry_dim=values["pose_geometry"].shape[-1],
        geometry_local_dim=values["local_features"].shape[-1],
        config=_config(),
    )


def _forward(model, values, baseline):
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


def test_local_sampler_is_deterministic_spatial_and_mask_bounded() -> None:
    features = torch.arange(2 * 4 * 6 * 8, dtype=torch.float32).reshape(
        2, 4, 6, 8
    )
    masks = torch.zeros(2, 2, 12, 16, dtype=torch.bool)
    masks[:, 0, 2:10, 3:13] = True
    masks[0, 1, 4:8, 5:10] = True

    first = sample_sam_instance_tokens(features, masks, max_tokens=8)
    second = sample_sam_instance_tokens(features, masks, max_tokens=8)
    for left, right in zip(first, second):
        assert torch.equal(left, right)
    descriptors, uv, valid = first
    assert descriptors.shape == (2, 2, 8, 4)
    assert uv.shape == (2, 2, 8, 2)
    assert valid.shape == (2, 2, 8)
    assert not bool(valid[1, 1].any())
    assert bool(((uv[valid] >= -1) & (uv[valid] <= 1)).all())
    # Farthest sampling should cover more than one spatial coordinate.
    assert torch.unique(uv[0, 0, valid[0, 0]], dim=0).shape[0] == 8


def test_all_v72_heads_start_as_exact_frozen_l0() -> None:
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
    for architecture in V72_ARCHITECTURES:
        model = _model(values, architecture)
        model.base_model.load_state_dict(base.state_dict())
        output = _forward(model, values, baseline)
        assert torch.equal(output["world_to_camera"], base_output["world_to_camera"])
        assert torch.equal(output["world_to_camera"][:, 0], baseline[:, 0])
        assert bool(
            (
                (output["sam_attention_entropy"] >= 0)
                & (output["sam_attention_entropy"] <= 1.0001)
            ).all()
        )
        assert bool(
            (
                (output["sam_attention_max"] >= 0)
                & (output["sam_attention_max"] <= 1.0001)
            ).all()
        )
        assert all(
            not parameter.requires_grad for parameter in model.base_model.parameters()
        )


def test_instance_off_exactly_returns_to_l0_after_nonzero_head() -> None:
    values, baseline = _inputs()
    off = perturb_v72_inputs(values, "instance_off")
    for architecture in V72_ARCHITECTURES:
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


def test_wrong_local_identity_changes_evidence_not_shared_gate() -> None:
    values, baseline = _inputs()
    model = _model(values, "sam_local_match")
    normal = _forward(model, values, baseline)
    wrong_values = perturb_v72_inputs(values, "wrong_local_identity")
    wrong = _forward(model, wrong_values, baseline)
    assert torch.equal(normal["active_frames"], wrong["active_frames"])
    assert not torch.equal(normal["evidence"], wrong["evidence"])


def test_backward_never_updates_frozen_l0() -> None:
    values, baseline = _inputs()
    model = _model(values, "dual_key_global_geometry")
    output = _forward(model, values, baseline)
    target = baseline.clone()
    target[:, 1:, 0, 3] = 0.1
    (output["world_to_camera"] - target).square().mean().backward()
    assert all(parameter.grad is None for parameter in model.base_model.parameters())
    assert any(
        parameter.grad is not None
        for name, parameter in model.named_parameters()
        if not name.startswith("base_model.")
    )


def test_v72_data_config_explicitly_enables_local_cache() -> None:
    config = load_learned_pose_config(
        "streaming_couping/configs/v72_local_token_data.yaml"
    )
    assert config.features.cache_sam_local_tokens
    assert config.features.sam_local_token_count == 32
    assert config.features.sam_local_sampling == "farthest_uv"


def test_v72_requires_metrics_from_the_exact_v71_frozen_l0(tmp_path) -> None:
    splits = (
        "base_train",
        "residual_train",
        "development",
        "future",
        "validation",
        "cross",
    )
    metrics = {
        split: {
            "rotation_degrees": 1.0 + index,
            "translation_native": 0.01 + index,
            "loss": 0.1 + index,
        }
        for index, split in enumerate(splits)
    }
    row = {"architecture": "frozen_l0"}
    for split, values in metrics.items():
        row[f"{split}_rotation_deg"] = values["rotation_degrees"]
        row[f"{split}_translation"] = values["translation_native"]
        row[f"{split}_loss"] = values["loss"]
    path = tmp_path / "v71_instance_causality.csv"
    with path.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)

    _assert_v71_baseline_metrics(tmp_path, metrics)

    mismatched = {
        split: dict(values) for split, values in metrics.items()
    }
    mismatched["cross"]["loss"] += 0.01
    with pytest.raises(RuntimeError, match="cross.loss"):
        _assert_v71_baseline_metrics(tmp_path, mismatched)


def test_v72_checkpoint_provenance_mismatch_is_distinct_from_corruption(
    tmp_path,
) -> None:
    path = tmp_path / "checkpoint.pt"
    torch.save(
        {"signature": "old", "model": {"weight": torch.ones(1)}},
        path,
    )
    with pytest.raises(V72CheckpointSignatureMismatch):
        _load_checkpoint(path, "new")

    torch.save({"signature": "new"}, path)
    with pytest.raises(ValueError, match="Invalid V7.2 checkpoint"):
        _load_checkpoint(path, "new")
