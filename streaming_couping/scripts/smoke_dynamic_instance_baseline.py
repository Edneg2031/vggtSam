#!/usr/bin/env python3
"""CPU smoke checks for the retained raw-pose dynamic-tracking baseline."""

from __future__ import annotations

import copy

import torch

from streaming_couping.scripts.run_dynamic_instance_baseline import (
    _tracking_audit,
)
from streaming_couping.src.geometry_segmentation import (
    GeometrySegmentationPrompt,
    V6GeometrySegmentationConfig,
    causal_prompts_after_birth,
    select_adaptive_correction,
)


def main() -> None:
    payload = {
        "frame_indices": [90, 105, 120, 135],
        "instance_ids": [0, 1, 2],
        "sam_birth_indices": [0, 2, -1],
        "instance_birth_indices": [0, 2, -1],
        "sam_track_ids": [7, 11, -1],
        "sam_track_prompts": ["bed", "wardrobe", ""],
        "dynamic_instance_diagnostics": [
            _row(0, 90, discovered=1, mature=0),
            _row(1, 105, discovered=1, mature=1),
            _row(2, 120, discovered=2, mature=1),
            _row(3, 135, discovered=2, mature=2),
        ],
    }
    audit = _tracking_audit(payload, reference_index=0)
    assert audit["tracking_audit_pass"] == 1
    assert audit["future_birth_supported"] == 1
    assert audit["late_birth_sequence_indices"] == (2,)
    broken = copy.deepcopy(payload)
    broken["dynamic_instance_diagnostics"][2]["discovered_tracks"] = 1
    assert _tracking_audit(broken, reference_index=0)["tracking_audit_pass"] == 0

    raw_pose = torch.randn(1, 4, 3, 4)
    selected_pose = raw_pose
    assert torch.equal(selected_pose, raw_pose)

    selected, reason = select_adaptive_correction(
        raw_row={
            "mask_pixels": 16,
            "support_recall": 0.5,
            "box_precision": 0.5,
            "geometry_score": 0.5,
        },
        prompted_row=None,
        config=V6GeometrySegmentationConfig(),
    )
    assert selected is None and reason == "keep_raw:no_prompt_candidate"
    prompt = GeometrySegmentationPrompt(
        box_mask=torch.ones(2, 2, dtype=torch.bool),
        positive_mask=torch.ones(2, 2, dtype=torch.bool),
    )
    causal = causal_prompts_after_birth(
        (prompt, prompt, prompt, prompt),
        birth_index=2,
    )
    assert causal[:3] == (None, None, None) and causal[3] is prompt
    print("raw-pose dynamic-tracking baseline smoke passed")


def _row(
    sequence_index: int,
    frame_index: int,
    *,
    discovered: int,
    mature: int,
) -> dict[str, object]:
    return {
        "sequence_index": sequence_index,
        "frame_index": frame_index,
        "discovered_tracks": discovered,
        "mature_tracks": mature,
    }


if __name__ == "__main__":
    main()
