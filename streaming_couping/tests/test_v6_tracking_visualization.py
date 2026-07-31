import csv

import torch
from PIL import Image

from streaming_couping.src.learned_pose.tracking_visualization import (
    evaluate_cached_tracking,
    write_csv,
    write_tracking_comparisons,
)


def test_tracking_success_excludes_reference_and_gt_absence(tmp_path) -> None:
    predicted = torch.zeros(3, 2, 4, 5, dtype=torch.bool)
    target = torch.zeros_like(predicted)
    target[0, :, :2, :2] = True
    predicted[0] = target[0]
    target[1, 0, :2, :2] = True
    predicted[1, 0, :2, :2] = True
    target[1, 1, :2, :2] = True
    predicted[1, 1, 0, 0] = True
    predicted[2, 1, 0, 0] = True
    shape = (3, 2)
    matched = torch.ones(shape, dtype=torch.bool)
    empty = torch.zeros(shape, dtype=torch.bool)

    rows, frame_rows, summary = evaluate_cached_tracking(
        clip_name="clip",
        split="test",
        frame_indices=(90, 105, 120),
        instance_ids=(37, 68),
        reference_sequence_index=0,
        predicted_masks=predicted,
        target_masks=target,
        tracking_scores=torch.ones(shape),
        identity_valid=matched,
        identity_unknown=empty,
        identity_mismatch=empty,
        associated_valid=matched,
        trusted_valid=matched,
        success_iou_threshold=0.5,
    )

    clip = summary[0]
    assert clip["gt_visible_frames"] == 2
    assert clip["tracking_success_frames"] == 1
    assert clip["false_positive_absent_frames"] == 1
    assert clip["tracking_success_rate_visible"] == 0.5
    assert frame_rows[1]["successful_visible_instances_iou50"] == 1
    assert frame_rows[2]["false_positive_absent_instances"] == 1
    assert sum(int(row["is_reference"]) for row in rows) == 2


def test_tracking_comparison_and_csv_are_written(tmp_path) -> None:
    image_paths = []
    for index in range(2):
        path = tmp_path / f"{index}.png"
        Image.new("RGB", (12, 8), (30, 40, 50)).save(path)
        image_paths.append(path)
    predicted = torch.zeros(2, 1, 4, 6, dtype=torch.bool)
    target = torch.zeros_like(predicted)
    predicted[:, 0, 1:3, 2:5] = True
    target.copy_(predicted)
    rows, _, _ = evaluate_cached_tracking(
        clip_name="clip",
        split="test",
        frame_indices=(90, 105),
        instance_ids=(37,),
        reference_sequence_index=0,
        predicted_masks=predicted,
        target_masks=target,
        tracking_scores=torch.ones(2, 1),
        identity_valid=torch.ones(2, 1, dtype=torch.bool),
        identity_unknown=torch.zeros(2, 1, dtype=torch.bool),
        identity_mismatch=torch.zeros(2, 1, dtype=torch.bool),
        associated_valid=torch.ones(2, 1, dtype=torch.bool),
        trusted_valid=torch.ones(2, 1, dtype=torch.bool),
    )

    root = tmp_path / "visual"
    write_tracking_comparisons(
        root,
        image_paths=image_paths,
        frame_indices=(90, 105),
        instance_ids=(37,),
        predicted_masks=predicted,
        target_masks=target,
        instance_frame_rows=rows,
    )
    write_csv(tmp_path / "rows.csv", rows)

    assert (root / "seq_000_frame_000090.png").is_file()
    assert (root / "seq_001_frame_000105.png").is_file()
    assert (root / "sequence_overview.png").is_file()
    with (tmp_path / "rows.csv").open(newline="", encoding="utf8") as handle:
        loaded = list(csv.DictReader(handle))
    assert loaded[1]["tracking_success_iou50"] == "1"
