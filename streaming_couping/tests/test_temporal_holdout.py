from pathlib import Path

import torch

from streaming_couping.src.learned_pose.config import ClipConfig
from streaming_couping.src.learned_pose.pipeline import (
    _evaluation_metadata,
    _evaluation_sequence_indices,
    _slice_training_payload,
)


def _clip() -> ClipConfig:
    return ClipConfig(
        name="temporal",
        scene_id="scene",
        frame_indices=(90, 105, 119, 130, 140, 210, 240),
        instance_ids=(37, 68, 54),
        reference_sequence_index=0,
        tracking_cache=Path("tracking.npz"),
        training_frame_indices=(90, 105, 119, 130, 140),
        evaluation_frame_indices=(210, 240),
    )


def test_training_payload_physically_excludes_temporal_holdout() -> None:
    sequence = 7
    payload = {
        "clip_name": "temporal",
        "scene_id": "scene",
        "frame_indices": [90, 105, 119, 130, 140, 210, 240],
        "reference_sequence_index": 0,
        "image_paths": [f"{index}.jpg" for index in range(sequence)],
        "camera_hidden": torch.arange(sequence * 2).reshape(sequence, 2),
        "appearance": torch.arange(sequence * 3).reshape(sequence, 1, 3),
        "token_levels": torch.arange(4 * sequence * 2).reshape(4, sequence, 1, 2),
        "stream_images": torch.arange(sequence).reshape(sequence, 1, 1, 1),
    }
    sliced = _slice_training_payload(payload, _clip())

    assert sliced["frame_indices"] == [90, 105, 119, 130, 140]
    assert sliced["supervision_frame_indices"] == [90, 105, 119, 130, 140]
    assert sliced["cache_context_frame_indices"] == [90, 105, 119, 130, 140, 210, 240]
    assert sliced["reference_sequence_index"] == 0
    assert sliced["camera_hidden"].shape[0] == 5
    assert sliced["appearance"].shape[0] == 5
    assert sliced["token_levels"].shape[1] == 5
    assert sliced["stream_images"].shape[0] == 5
    assert int(sliced["camera_hidden"][-1, 0]) == 8


def test_temporal_evaluation_uses_suffix_with_full_context_metadata() -> None:
    clip = _clip()
    assert _evaluation_sequence_indices(clip) == [5, 6]
    assert _evaluation_metadata(clip) == {
        "evaluation_protocol": "causal_temporal_holdout",
        "context_frames": 7,
        "context_frame_indices": "90 105 119 130 140 210 240",
        "training_frame_indices": "90 105 119 130 140",
        "evaluated_frame_indices": "210 240",
        "alignment_reference_frame_index": 90,
    }
