"""What each prompt's tracks became, and why the ones that died did.

A prompt can return masks on every frame and still contribute nothing: dropped
at birth for size, dropped as a duplicate of an earlier-born track, or left
behind the object cap.  All three are silent, and the observations that survive
cannot tell them apart from "the prompt returned no masks at all" -- which are
opposite diagnoses.  So the ledger has to name the outcome AND, for a duplicate,
the track it lost to, because which of two overlapping tracks survives is
decided by birth frame rather than by the prompt.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from streaming_couping.src.semantic_mapping.adapters import (
    SAM31SegmentationAdapter,
    ledger_summary,
)

SIZE = 8
MIN_BIRTH_PIXELS = 4


@dataclass
class _Tracked:
    masks: torch.Tensor
    scores: torch.Tensor
    obj_ids: list[int]
    birth_indices: list[int]


def _block(row: int, column: int, height: int = 2, width: int = 2) -> torch.Tensor:
    mask = torch.zeros(SIZE, SIZE, dtype=torch.bool)
    mask[row : row + height, column : column + width] = True
    return mask


class _StubWrapper:
    """One entry per prompt: ``(obj_id, birth_frame, mask)`` per returned track.

    The mask is the object's mask; it is absent before its birth frame, which
    is what makes the birth index meaningful.
    """

    def __init__(
        self, per_prompt: dict[str, list[tuple[int, int, torch.Tensor]]]
    ) -> None:
        self.per_prompt = per_prompt

    def track_all_forward(
        self, paths: Sequence[Any], *, prompt: str, output_size, max_objects: int
    ) -> _Tracked:
        entries = self.per_prompt.get(prompt, [])
        frame_count = len(paths)
        if not entries:
            empty = torch.zeros(frame_count, 0, SIZE, SIZE, dtype=torch.bool)
            return _Tracked(empty, torch.ones(frame_count, 0), [], [])
        per_object = []
        for _, birth, mask in entries:
            frames = torch.zeros(frame_count, SIZE, SIZE, dtype=torch.bool)
            frames[birth:] = mask
            per_object.append(frames)
        masks = torch.stack(per_object, dim=1)
        return _Tracked(
            masks=masks,
            scores=torch.ones(frame_count, len(entries)),
            obj_ids=[obj_id for obj_id, _, _ in entries],
            birth_indices=[birth for _, birth, _ in entries],
        )


def _adapter(**kwargs: Any) -> SAM31SegmentationAdapter:
    defaults = dict(
        output_size=(SIZE, SIZE),
        max_objects_per_prompt=16,
        max_total_objects=16,
        min_birth_pixels=MIN_BIRTH_PIXELS,
        duplicate_iou=0.5,
    )
    defaults.update(kwargs)
    return SAM31SegmentationAdapter(_StubWrapper({}), **defaults)


def _run(per_prompt: dict[str, list[tuple[int, int, torch.Tensor]]], **kwargs: Any):
    adapter = _adapter(**kwargs)
    adapter.wrapper = _StubWrapper(per_prompt)
    adapter.infer(["a.png", "b.png"], ["bed", "wardrobe"])
    return adapter


def test_a_prompt_that_returns_nothing_is_named() -> None:
    adapter = _run({"bed": [(0, 0, _block(0, 0))], "wardrobe": []})
    summary = adapter.candidate_ledger_summary
    assert summary["prompts_with_no_track"] == ["wardrobe"]
    # it is present with no outcomes, which is not the same as absent: absence
    # would mean the prompt was never asked for
    assert summary["per_prompt"]["wardrobe"] == {}
    assert summary["prompt_count"] == 2


def test_a_track_born_too_small_is_recorded_as_such() -> None:
    tiny = torch.zeros(SIZE, SIZE, dtype=torch.bool)
    tiny[0, 0] = True  # 1 px, below the floor
    adapter = _run({"bed": [(0, 0, tiny)], "wardrobe": []})
    assert adapter.candidate_ledger[0]["outcome"] == "birth_mask_rejected"
    assert adapter.candidate_ledger[0]["birth_pixels"] == 1


def test_a_track_dropped_as_a_duplicate_names_what_it_lost_to() -> None:
    """The earlier-born track wins, so the later one is the one that vanishes."""

    overlapping = _block(0, 0)
    adapter = _run(
        {
            # cabinet is first in the prompt list and so is examined first when
            # the birth frames tie
            "bed": [(0, 0, overlapping)],
            "wardrobe": [(0, 0, overlapping.clone())],
        }
    )
    by_prompt = {entry["prompt"]: entry for entry in adapter.candidate_ledger}
    assert by_prompt["bed"]["outcome"] == "accepted"
    assert by_prompt["wardrobe"]["outcome"] == "duplicate"
    assert by_prompt["wardrobe"]["duplicate_of_prompt"] == "bed"
    assert by_prompt["wardrobe"]["duplicate_of_birth"] == 0


def test_an_earlier_birth_beats_an_earlier_prompt() -> None:
    """Ordering is by birth frame first, so a later prompt can win on age."""

    overlapping = _block(0, 0)
    adapter = _run(
        {
            # bed's track is born on frame 1; wardrobe's on frame 0
            "bed": [(0, 1, overlapping)],
            "wardrobe": [(0, 0, overlapping.clone())],
        }
    )
    by_prompt = {entry["prompt"]: entry for entry in adapter.candidate_ledger}
    assert by_prompt["wardrobe"]["outcome"] == "accepted"
    assert by_prompt["bed"]["outcome"] == "duplicate"
    assert by_prompt["bed"]["duplicate_of_prompt"] == "wardrobe"


def test_tracks_behind_the_cap_are_recorded_rather_than_left_silent() -> None:
    """The cap breaks out of the loop, so nothing else would ever mark them."""

    adapter = _run(
        {
            "bed": [(0, 0, _block(0, 0))],
            "wardrobe": [(0, 0, _block(4, 4))],
        },
        max_total_objects=1,
    )
    outcomes = [entry["outcome"] for entry in adapter.candidate_ledger]
    assert outcomes.count("accepted") == 1
    assert outcomes.count("over_object_cap") == 1


def test_ledger_summary_counts_each_outcome_per_prompt() -> None:
    summary = ledger_summary(
        [
            {"prompt": "bed", "outcome": "accepted"},
            {"prompt": "bed", "outcome": "duplicate"},
            {"prompt": "bed", "outcome": "duplicate"},
            {"prompt": "table", "outcome": "birth_mask_rejected"},
            {"prompt": "rug", "outcome": "over_object_cap"},
        ],
        ["bed", "table", "rug", "door"],
    )
    assert summary["total"] == 5
    assert summary["per_prompt"]["bed"] == {"accepted": 1, "duplicate": 2}
    # a prompt whose only track was dropped as a duplicate DID produce a track;
    # one whose track was too small did not, and neither did one never seen
    assert summary["prompts_with_no_track"] == ["door", "table"]
