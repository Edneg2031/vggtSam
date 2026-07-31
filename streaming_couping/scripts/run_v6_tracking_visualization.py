#!/usr/bin/env python3
"""Render cached V6 tracking once; never run SAM3.1 or StreamVGGT."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from streaming_couping.src.config import load_config
from streaming_couping.src.instance_observations import load_instance_sequences
from streaming_couping.src.learned_pose.cache import (
    cache_path,
    load_feature_cache,
)
from streaming_couping.src.learned_pose.config import (
    ClipConfig,
    load_learned_pose_config,
)
from streaming_couping.src.learned_pose.tracking_visualization import (
    evaluate_cached_tracking,
    write_csv,
    write_tracking_comparisons,
)


def main() -> None:
    args = _parse_args()
    config = load_learned_pose_config(args.config)
    clips = _select_clips(config.clips, names=args.clip)
    output_dir = Path(args.output_dir).expanduser()
    completion = output_dir / "COMPLETE"
    if completion.is_file() and not args.overwrite:
        print(f"tracking visualization already complete: {output_dir}")
        print(output_dir / "tracking_success_summary.csv")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    all_instance_rows: list[dict[str, object]] = []
    all_frame_rows: list[dict[str, object]] = []
    all_summary_rows: list[dict[str, object]] = []
    for clip in clips:
        print(f"tracking visualization from cache clip={clip.name}")
        payload = load_feature_cache(cache_path(config, clip))
        predicted = torch.as_tensor(
            payload["tracking_masks_output"]
        ).bool().cpu()
        recovery = load_config(
            config.recovery_config,
            {
                "manifest": config.manifest,
                "scene_id": clip.scene_id,
                "frame_indices": clip.frame_indices,
                "output_dir": output_dir / "runtime" / clip.name,
            },
        )
        _, target_by_id = load_instance_sequences(
            recovery,
            instance_ids=clip.instance_ids,
            reference_sequence_index=clip.reference_sequence_index,
            allow_missing_reference=clip.allow_missing_reference_instances,
        )
        target = torch.stack(
            [target_by_id[int(value)] for value in clip.instance_ids],
            dim=1,
        ).bool()
        instance_rows, frame_rows, summary_rows = evaluate_cached_tracking(
            clip_name=clip.name,
            split=clip.split,
            frame_indices=clip.frame_indices,
            instance_ids=clip.instance_ids,
            reference_sequence_index=clip.reference_sequence_index,
            predicted_masks=predicted,
            target_masks=target,
            tracking_scores=payload["tracking_scores"],
            identity_valid=payload["identity_valid"],
            identity_unknown=payload["identity_unknown"],
            identity_mismatch=payload["identity_mismatch"],
            associated_valid=payload["associated_instance_valid"],
            trusted_valid=payload["trusted_instance_valid"],
            segmentation_diagnostics=payload.get(
                "segmentation_diagnostics",
                (),
            ),
            success_iou_threshold=args.success_iou,
        )
        write_tracking_comparisons(
            output_dir / clip.name / "rgb_gt_v6_tracking",
            image_paths=payload["image_paths"],
            frame_indices=clip.frame_indices,
            instance_ids=clip.instance_ids,
            predicted_masks=predicted,
            target_masks=target,
            instance_frame_rows=instance_rows,
        )
        all_instance_rows.extend(instance_rows)
        all_frame_rows.extend(
            {"clip": clip.name, "split": clip.split, **row}
            for row in frame_rows
        )
        all_summary_rows.extend(summary_rows)

    write_csv(
        output_dir / "tracking_instance_frames.csv",
        all_instance_rows,
    )
    write_csv(output_dir / "tracking_frames.csv", all_frame_rows)
    write_csv(
        output_dir / "tracking_success_summary.csv",
        all_summary_rows,
    )
    (output_dir / "README.txt").write_text(
        "One-shot cached V6 tracking visualization\n"
        "========================================\n\n"
        "No SAM3.1, StreamVGGT, or learned model was run here. Each PNG shows "
        "RGB | GT instance masks | the final SAM3.1 mask stored in the V6 "
        "feature cache. A visible-object tracking success is a non-reference "
        f"instance-frame with GT visibility and IoU >= {args.success_iou:.3f}."
        " GT-absent frames and their false positives are counted separately.\n"
        "tracking_success_summary.csv is the compact upload table; "
        "tracking_frames.csv is the per-frame count; "
        "tracking_instance_frames.csv contains the full diagnostics.\n",
        encoding="utf8",
    )
    completion.write_text("complete\n", encoding="utf8")
    print("V6 TRACKING SUCCESS SUMMARY")
    _print_csv(output_dir / "tracking_success_summary.csv")
    print(
        "overview="
        + str(
            output_dir
            / clips[-1].name
            / "rgb_gt_v6_tracking"
            / "sequence_overview.png"
        )
    )


def _select_clips(
    clips: tuple[ClipConfig, ...],
    *,
    names: list[str],
) -> tuple[ClipConfig, ...]:
    if not names:
        return clips
    requested = set(names)
    selected = tuple(clip for clip in clips if clip.name in requested)
    missing = requested - {clip.name for clip in selected}
    if missing:
        raise ValueError(
            "Unknown --clip value(s): " + ", ".join(sorted(missing))
        )
    return selected


def _print_csv(path: Path) -> None:
    with path.open("r", encoding="utf8") as handle:
        print(handle.read().rstrip())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="streaming_couping/configs/v6_sam31_long30_base.yaml",
    )
    parser.add_argument(
        "--clip",
        action="append",
        default=[],
        help="Clip name to render; repeat to select multiple clips.",
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "outputs/streaming_couping_v6_sam31_long30/"
            "tracking_visualization_once"
        ),
    )
    parser.add_argument(
        "--success-iou",
        type=float,
        default=0.50,
        help="GT-visible mask IoU threshold counted as tracking success.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate files even when the COMPLETE marker exists.",
    )
    args = parser.parse_args()
    if not 0.0 <= args.success_iou <= 1.0:
        parser.error("--success-iou must be in [0, 1].")
    return args


if __name__ == "__main__":
    main()
