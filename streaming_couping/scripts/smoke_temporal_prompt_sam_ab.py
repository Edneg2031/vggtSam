#!/usr/bin/env python3
"""CPU smoke test for the real-SAM A/B bookkeeping and safety gates.

No SAM model, image, cache, or annotation is loaded.  A tiny fake proposer is
used only to exercise the same runner-side mask/score/fallback path.
"""

from __future__ import annotations

from pathlib import Path

import torch

from streaming_couping.scripts.run_v0_bidirectional_feedback import V0Arrays
from streaming_couping.scripts.run_v0_temporal_prompt_sam_ab import (
    _prepare_causal_events,
    _run_real_sam_branches,
)
from streaming_couping.src.temporal_prompt_sam_ab import (
    REAL_SAM_BRANCHES,
    build_prompt_mask,
    decide_score_fallback,
    deterministic_query_subset,
    summarize_prompt_events,
    summarize_worsened_frames,
)
from streaming_couping.src.types import SAM3MaskCandidate


class _FakeSAM:
    """Return deterministic masks; C deliberately triggers score fallback."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def propose_geometry_prompted_masks(
        self,
        image_path: Path,
        *,
        prompt: str,
        output_size: tuple[int, int],
        geometry_prompt: torch.Tensor,
        positive_prompt: torch.Tensor,
        max_positive_points: int,
        use_box: bool,
        use_points: bool,
    ) -> list[SAM3MaskCandidate]:
        del image_path, prompt, geometry_prompt, use_box, use_points
        count = int(positive_prompt.sum())
        self.calls.append(("points", count))
        mask = positive_prompt.clone()
        # Make a non-empty, deterministic prompted result.
        if not bool(mask.any()):
            mask[3, 3] = True
        score = 0.50 if max_positive_points == 5 else 0.95
        return [SAM3MaskCandidate(obj_id=17, mask=mask, score=score)]


def main() -> None:
    height, width = 16, 20
    sequence, slots = 2, 1
    points = torch.zeros(sequence, height, width, 3)
    points[..., 2] = 1.0
    confidence = torch.ones(sequence, height, width)
    raw_masks = torch.zeros(sequence, slots, height, width, dtype=torch.bool)
    raw_masks[:, 0, 4:8, 5:9] = True
    scores = torch.full((sequence, slots), 0.90)
    pose = torch.eye(4, dtype=torch.float32)[:3].unsqueeze(0).repeat(sequence, 1, 1)
    intrinsics = torch.tensor(
        [[10.0, 0.0, 10.0], [0.0, 10.0, 8.0], [0.0, 0.0, 1.0]]
    ).unsqueeze(0).repeat(sequence, 1, 1)
    arrays = V0Arrays(
        points=points,
        confidence=confidence,
        raw_masks=raw_masks,
        scores=scores,
        depth=torch.ones(sequence, height, width),
        intrinsics=intrinsics,
        raw_world_to_camera=pose,
        selected_world_to_camera=pose,
        image_size=(height, width),
        map_size=(height, width),
    )
    selected_keys = ((1, 0),)
    point_rows = []
    for spec in REAL_SAM_BRANCHES:
        tolerance = spec.tolerance
        for index in range(spec.max_points):
            point_rows.append(
                {
                    "branch": spec.candidate_branch,
                    "tolerance": tolerance,
                    "sequence_index": 1,
                    "slot": 0,
                    "candidate_index": index,
                    "accepted_prompt": 1,
                    "projected_u": 6.0 + index,
                    "projected_v": 5.0 + index,
                }
            )
            if spec.max_points == 1:
                break
    query_rows = [
        {
            "branch": "A_center",
            "tolerance": "",
            "sequence_index": 1,
            "frame_index": 1,
            "slot": 0,
            "sam_track_id": 7,
            "history_sequence_index": 0,
            "history_frame_index": 0,
            "history_gap": 1,
        }
    ]
    causal_events = _prepare_causal_events(
        point_rows,
        query_rows,
        selected_keys=selected_keys,
        source_size=(height, width),
        target_size=(height, width),
    )
    assert len(causal_events) == len(REAL_SAM_BRANCHES)

    fake = _FakeSAM()
    variant_masks = {"raw_v0": raw_masks.clone()}
    variant_scores = {"raw_v0": scores.clone()}
    events = _run_real_sam_branches(
        sam3=fake,
        image_paths=(Path("smoke.jpg"), Path("smoke.jpg")),
        arrays=arrays,
        track_prompts=("chair",),
        point_rows=point_rows,
        selected_keys=selected_keys,
        causal_events=causal_events,
        variant_masks=variant_masks,
        variant_scores=variant_scores,
        score_fallback_margin=0.10,
        score_fallback_ratio=0.80,
        max_positive_points=5,
        progress_interval=100,
    )
    assert len(events) == len(REAL_SAM_BRANCHES)
    assert len(fake.calls) == len(REAL_SAM_BRANCHES)
    assert summarize_prompt_events(events, branch="A_center")["prompt_applied_count"] == 1
    assert (
        summarize_prompt_events(events, branch="C_surface_5")["score_fallback_count"]
        == 1
    )
    assert torch.equal(variant_masks["C_surface_5"], raw_masks)
    assert not torch.equal(variant_masks["A_center"], raw_masks)

    prompt_mask, coordinates = build_prompt_mask(
        [{"projected_u": 2.0, "projected_v": 3.0}],
        source_size=(10, 10),
        target_size=(20, 20),
    )
    assert int(prompt_mask.sum()) == 1 and len(coordinates) == 1
    assert decide_score_fallback(0.9, 0.5)[0]
    assert deterministic_query_subset(((2, 0), (0, 1), (1, 0)), 2) == (
        (0, 1),
        (2, 0),
    )

    frame_rows = [
        {
            "variant": "demo",
            "slot": 0,
            "sequence_index": 0,
            "gt_visible": 1,
            "raw_iou": 0.80,
            "iou": 0.70,
        },
        {
            "variant": "demo",
            "slot": 0,
            "sequence_index": 1,
            "gt_visible": 1,
            "raw_iou": 0.40,
            "iou": 0.50,
        },
    ]
    safety = summarize_worsened_frames(frame_rows, branch="demo")
    assert safety["worsened_frame_ratio"] == 0.5
    print(
        "V0 real-SAM temporal prompt A/B smoke passed "
        f"events={len(events)} calls={len(fake.calls)} "
        f"score_fallback={sum(int(row['score_fallback']) for row in events)}"
    )


if __name__ == "__main__":
    main()
