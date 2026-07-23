from dataclasses import replace
from pathlib import Path

import pytest

from streaming_couping.src.learned_pose.config import (
    _validate,
    load_learned_pose_config,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "streaming_couping/configs/final_joint_pointcloud_pose_test.yaml"


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
