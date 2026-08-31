#!/usr/bin/env python3
"""CPU-only smoke checks for class-agnostic SAM3.1 proposal plumbing."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile

import numpy as np
import torch
from PIL import Image

from streaming_couping.scripts.evaluate_sam31_auto_proposals import (
    ALL_INSTANCE_SCOPE,
    AUTO_LABEL,
    PROMPT_LABEL,
    _decision,
    _scope_variants,
)
from streaming_couping.src.backbones.sam3_video import (
    SAM3VideoTrackerAdapter,
    assess_auto_proposal_mask,
    auto_proposal_point_grid,
    auto_proposal_points_for_discovery,
)


class _FakeAutoPredictor:
    """Small CPU predictor double for the adapter/session lifecycle."""

    def __init__(self) -> None:
        self.model = SimpleNamespace(
            hotstart_delay=15,
            masklet_confirmation_consecutive_det_thresh=3,
            postprocess_batch_size=16,
        )
        self.active: dict[int, tuple[int, float, float]] = {}
        self.add_calls: list[dict] = []
        self.removed: list[int] = []
        self.closed = False
        self.num_frames = 0

    def start_session(self, *, resource_path):
        self.num_frames = len(tuple(Path(resource_path).glob("*.jpg")))
        self.active.clear()
        self.closed = False
        return {"session_id": "smoke-session"}

    def add_prompt(self, **kwargs):
        self.add_calls.append(dict(kwargs))
        obj_id = int(kwargs["obj_id"])
        frame_idx = int(kwargs["frame_idx"])
        point = kwargs["points"][0]
        point_x = float(point[0])
        point_y = float(point[1])
        self.active[obj_id] = (frame_idx, point_x, point_y)

        # Deliberately reject one proposal so the smoke covers remove_object
        # and confirms that rejected IDs do not consume a retained slot.
        if len(self.add_calls) == 2:
            mask = np.ones((1, 10, 10), dtype=bool)
        else:
            mask = np.zeros((1, 10, 10), dtype=bool)
            column = min(9, max(0, int(point_x * 10.0)))
            row = min(9, max(0, int(point_y * 10.0)))
            mask[0, row, column] = True
        return self._result(frame_idx, [obj_id], mask)

    def remove_object(self, *, session_id, frame_idx, obj_id, is_user_action):
        assert session_id == "smoke-session"
        assert is_user_action is False
        self.removed.append(int(obj_id))
        self.active.pop(int(obj_id), None)
        return self._result(
            int(frame_idx),
            [],
            np.zeros((0, 10, 10), dtype=bool),
        )

    def propagate_in_video(
        self,
        *,
        session_id,
        propagation_direction,
        start_frame_idx,
        max_frame_num_to_track,
        output_prob_thresh,
    ):
        assert session_id == "smoke-session"
        assert propagation_direction == "forward"
        start = int(start_frame_idx)
        end = min(
            self.num_frames - 1,
            start + int(max_frame_num_to_track),
        )
        for frame_idx in range(start, end + 1):
            obj_ids = []
            masks = []
            for obj_id, (birth, point_x, point_y) in sorted(self.active.items()):
                if frame_idx < birth:
                    continue
                mask = np.zeros((10, 10), dtype=bool)
                column = min(9, max(0, int(point_x * 10.0)))
                row = min(9, max(0, int(point_y * 10.0)))
                mask[row, column] = True
                obj_ids.append(obj_id)
                masks.append(mask)
            stacked = (
                np.stack(masks, axis=0)
                if masks
                else np.zeros((0, 10, 10), dtype=bool)
            )
            yield self._result(frame_idx, obj_ids, stacked)

    def close_session(self, session_id):
        assert session_id == "smoke-session"
        self.closed = True

    @staticmethod
    def _result(frame_idx, obj_ids, masks):
        return {
            "frame_index": int(frame_idx),
            "outputs": {
                "out_obj_ids": np.asarray(obj_ids, dtype=np.int64),
                "out_probs": np.ones(len(obj_ids), dtype=np.float32),
                "out_binary_masks": masks,
            },
        }


def main() -> None:
    grid = auto_proposal_point_grid(2, 3)
    assert len(grid) == 6
    assert len({index for index, _, _ in grid}) == 6
    first = auto_proposal_points_for_discovery(
        grid,
        discovery_index=0,
        limit=2,
    )
    second = auto_proposal_points_for_discovery(
        grid,
        discovery_index=1,
        limit=2,
    )
    assert {value[0] for value in first}.isdisjoint(
        {value[0] for value in second}
    )

    mask = torch.zeros(10, 10, dtype=torch.bool)
    mask[2:6, 2:6] = True
    accepted = assess_auto_proposal_mask(
        mask,
        retained_masks=(),
        point_x=0.35,
        point_y=0.35,
        output_size=(10, 10),
        min_mask_pixels=4,
        max_mask_area_ratio=0.50,
        duplicate_iou=0.80,
        duplicate_intersection_over_smaller=0.90,
    )
    assert accepted["accepted"] == 1, accepted

    outside = assess_auto_proposal_mask(
        mask,
        retained_masks=(),
        point_x=0.85,
        point_y=0.85,
        output_size=(10, 10),
        min_mask_pixels=4,
        max_mask_area_ratio=0.50,
        duplicate_iou=0.80,
        duplicate_intersection_over_smaller=0.90,
    )
    assert outside["reason"] == "prompt_point_outside_mask"
    duplicate = assess_auto_proposal_mask(
        mask,
        retained_masks=(mask.clone(),),
        point_x=0.35,
        point_y=0.35,
        output_size=(10, 10),
        min_mask_pixels=4,
        max_mask_area_ratio=0.50,
        duplicate_iou=0.80,
        duplicate_intersection_over_smaller=0.90,
    )
    assert duplicate["reason"] == "duplicate_iou"
    too_large = assess_auto_proposal_mask(
        torch.ones(10, 10, dtype=torch.bool),
        retained_masks=(),
        point_x=0.5,
        point_y=0.5,
        output_size=(10, 10),
        min_mask_pixels=4,
        max_mask_area_ratio=0.50,
        duplicate_iou=0.80,
        duplicate_intersection_over_smaller=0.90,
    )
    assert too_large["reason"] == "mask_too_large"

    fake = _FakeAutoPredictor()
    adapter = SAM3VideoTrackerAdapter(
        fake,
        output_prob_thresh=0.5,
        prompt_with_box=False,
    )
    with tempfile.TemporaryDirectory(prefix="sam31_auto_smoke_images_") as tmp:
        image_paths = []
        for index in range(3):
            image_path = Path(tmp) / f"frame-{index}.jpg"
            Image.new("RGB", (20, 20), "black").save(image_path)
            image_paths.append(image_path)
        tracked = adapter.track_auto_points_forward_from_paths(
            image_paths,
            output_size=(10, 10),
            max_objects=2,
            discovery_stride=1,
            grid_rows=2,
            grid_columns=3,
            points_per_discovery=2,
            min_mask_pixels=1,
            max_mask_area_ratio=0.50,
            duplicate_iou=0.80,
            duplicate_intersection_over_smaller=0.90,
            quiet=False,
        )
    assert tracked.masks.shape == (3, 2, 10, 10)
    assert len(tracked.obj_ids) == 2
    assert len(set(tracked.obj_ids)) == 2
    assert len(fake.removed) >= 1
    assert fake.closed
    assert all(call["text"] is None for call in fake.add_calls)
    assert all(call["point_labels"].dtype == torch.int32 for call in fake.add_calls)
    assert fake.model.hotstart_delay == 15
    assert fake.model.masklet_confirmation_consecutive_det_thresh == 3
    assert fake.model.postprocess_batch_size == 16

    clip = {
        "clip": "synthetic_clip",
        "scene_id": "synthetic_scene",
        "split": "validation",
        "variants": {
            PROMPT_LABEL: {
                "tracking": {"mean_frame_iou": 0.20, "frame_idf1": 0.25}
            },
            AUTO_LABEL: {
                "tracking": {"mean_frame_iou": 0.30, "frame_idf1": 0.35}
            },
        },
    }
    aggregate = {
        "comparison": {
            "delta_tracking_iou": 0.10,
            "delta_frame_idf1": 0.10,
        }
    }
    assert _decision(aggregate, [clip])["status"] == "GO"
    clip["variants"][AUTO_LABEL]["tracking"]["frame_idf1"] = 0.10
    assert _decision(aggregate, [clip])["status"] == "NO_GO"

    clip["all_instance_scope"] = {
        "variants": {
            PROMPT_LABEL: clip["variants"][PROMPT_LABEL],
            AUTO_LABEL: clip["variants"][AUTO_LABEL],
        }
    }
    assert _scope_variants(clip, ALL_INSTANCE_SCOPE)[AUTO_LABEL]
    print(
        "SAM3.1 auto-proposal smoke passed "
        "grid=6 admission=covered session_lifecycle=1 "
        "dual_gt_scope=1 decision=GO/NO_GO"
    )


if __name__ == "__main__":
    main()
