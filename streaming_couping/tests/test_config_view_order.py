from dataclasses import replace
from pathlib import Path

import pytest

from streaming_couping.src.instance_observations import InstanceRefinementConfig
from streaming_couping.src.learned_pose.config import (
    _validate,
    load_learned_pose_config,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "streaming_couping/configs/final_joint_pointcloud_pose_test.yaml"


def test_final_observation_config_keeps_checkpoint_temporal_scale() -> None:
    assert InstanceRefinementConfig().temporal_max_frame_gap == 15


def test_configured_view_order_may_be_non_monotonic() -> None:
    config = load_learned_pose_config(CONFIG)
    clip = replace(
        config.clips[0],
        frame_indices=(100, 500, 200, 400, 300),
    )

    _validate(replace(config, clips=(clip,)))


def test_configured_view_order_rejects_duplicate_frames() -> None:
    config = load_learned_pose_config(CONFIG)
    clip = replace(
        config.clips[0],
        frame_indices=(100, 500, 200, 500, 300),
    )

    with pytest.raises(ValueError, match="contains duplicates"):
        _validate(replace(config, clips=(clip,)))


def test_temporal_holdout_uses_configured_order_not_frame_number() -> None:
    config = load_learned_pose_config(CONFIG)
    clip = replace(
        config.clips[0],
        split="train",
        frame_indices=(100, 500, 200, 400, 300),
        training_frame_indices=(100, 500, 200),
        evaluation_frame_indices=(400, 300),
    )

    _validate(replace(config, clips=(clip,)))


def test_ray_pose_reference_blends_reject_duplicates() -> None:
    config = load_learned_pose_config(CONFIG)
    ray_pose = replace(
        config.evaluation.ray_pose,
        reference_blend_values=(0.5, 0.5),
    )

    with pytest.raises(ValueError, match="must not contain duplicates"):
        _validate(
            replace(
                config,
                evaluation=replace(config.evaluation, ray_pose=ray_pose),
            )
        )


@pytest.mark.parametrize("blend", [0.0, -0.25, 1.25])
def test_ray_pose_reference_blends_must_be_bounded(blend: float) -> None:
    config = load_learned_pose_config(CONFIG)
    ray_pose = replace(
        config.evaluation.ray_pose,
        reference_blend_values=(blend,),
    )

    with pytest.raises(ValueError, match=r"must be in \(0,1\]"):
        _validate(
            replace(
                config,
                evaluation=replace(config.evaluation, ray_pose=ray_pose),
            )
        )
