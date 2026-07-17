"""Causal ablation for geometry recovery and same-ID SAM3 memory writeback.

The experiment keeps candidate generation, the selected dense recovery mask,
and the causal split fixed between the aligned no-memory and memory branches.
GT masks after the reference frame are metrics-only.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw

from test_sam.data import load_mask_tracking_sequence

from .backbones.sam3_wrapper import SAM3Wrapper
from .backbones.streamvggt_wrapper import StreamVGGTWrapper
from .bridge.gating import binary_iou
from .config import ExperimentConfig, load_config
from .pipeline import (
    _mine_recovery,
    _resize_target_masks,
    summarize_masks,
    summarize_visible_after,
)
from .types import GeometrySequence, SAM3MaskCandidate, TrackingSequence


MODES = (
    "original",
    "geometry_recovery_no_memory",
    "geometry_recovery_same_id_memory",
    "shuffled_geometry_same_id_memory",
)


def main() -> None:
    args = _parse_args()
    instance_ids = _unique_ids(args.instance_ids)
    overrides = {
        key: value
        for key, value in {
            "manifest": args.manifest,
            "scene_id": args.scene_id,
            "instance_id": instance_ids[0],
            "frame_indices": args.frame_indices,
            "sam3_device": args.sam3_device,
            "geometry_device": args.geometry_device,
            "output_dir": args.output_dir,
        }.items()
        if value is not None
    }
    config = load_config(args.config, overrides)
    run_experiment(
        config,
        instance_ids=instance_ids,
        reference_sequence_index=args.reference_sequence_index,
    )


def run_experiment(
    config: ExperimentConfig,
    *,
    instance_ids: Sequence[int],
    reference_sequence_index: int | None,
) -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    sequences = {}
    target_masks = {}
    for instance_id in instance_ids:
        sequence = load_mask_tracking_sequence(
            config.manifest,
            scene_id=config.scene_id,
            frame_indices=config.frame_indices,
            sequence_length=len(config.frame_indices),
            frame_stride=1,
            window_index=0,
            instance_id=int(instance_id),
            min_pixels=config.min_pixels,
            max_area_ratio=config.max_area_ratio,
            min_visible_frames=1,
            excluded_labels=config.excluded_labels,
            seed=0,
        )
        resized = _resize_target_masks(sequence.target_masks, config.output_size)
        if reference_sequence_index is None:
            visible_indices = resized.flatten(1).any(dim=1).nonzero(
                as_tuple=False
            ).flatten()
            if not len(visible_indices):
                raise ValueError(
                    f"Instance {instance_id} is not visible in the selected sequence."
                )
            reference_index = int(visible_indices[0])
        else:
            reference_index = int(reference_sequence_index)
            if not 0 <= reference_index < len(sequence.frame_indices):
                raise ValueError(
                    f"Reference sequence index {reference_index} is outside the "
                    f"{len(sequence.frame_indices)} selected frames."
                )
            if not resized[reference_index].any():
                raise ValueError(
                    f"Instance {instance_id} is absent at requested reference "
                    f"frame {sequence.frame_indices[reference_index]}."
                )
        sequence = replace(sequence, reference_frame_idx=reference_index)
        sequences[int(instance_id)] = sequence
        target_masks[int(instance_id)] = resized

    _validate_shared_sequence(sequences)
    shared = sequences[int(instance_ids[0])]
    print(
        f"causal recovery ablation scene={shared.scene_id} "
        f"frames={shared.frame_indices} instances={list(instance_ids)}"
    )

    print("extracting frozen StreamVGGT geometry once...")
    geometry = StreamVGGTWrapper(
        repo_path=config.streamvggt_repo,
        checkpoint_path=config.streamvggt_checkpoint,
        device=config.geometry_device,
        image_mode=config.image_mode,
        streaming_cache=config.streaming_cache,
    ).load().extract(shared.image_paths)

    print("loading frozen SAM3...")
    sam3 = SAM3Wrapper(
        repo_path=config.sam3_repo,
        checkpoint_path=config.sam3_checkpoint,
        device=config.sam3_device,
        output_threshold=config.sam3_output_threshold,
        prompt_with_box=config.prompt_with_box,
    ).load()

    all_summary_rows = []
    all_frame_rows = []
    all_candidate_rows = []
    instance_metadata = {}
    for instance_id in instance_ids:
        sequence = sequences[int(instance_id)]
        print(
            f"running instance={instance_id} label={sequence.label!r} "
            f"reference={sequence.frame_indices[sequence.reference_frame_idx]}"
        )
        result = _run_instance(
            replace(config, instance_id=int(instance_id)),
            sequence=sequence,
            target_masks=target_masks[int(instance_id)],
            geometry=geometry,
            sam3=sam3,
        )
        all_summary_rows.extend(result["summary_rows"])
        all_frame_rows.extend(result["frame_rows"])
        all_candidate_rows.extend(result["candidate_rows"])
        instance_metadata[str(instance_id)] = result["metadata"]
        instance_dir = config.output_dir / f"instance_{instance_id}"
        instance_dir.mkdir(parents=True, exist_ok=True)
        _write_csv(instance_dir / "summary.csv", result["summary_rows"])
        _write_csv(instance_dir / "frame_metrics.csv", result["frame_rows"])
        _write_csv(
            instance_dir / "candidate_diagnostics.csv",
            result["candidate_rows"],
        )
        _save_report(
            instance_dir / "recovery_writeback_report.png",
            image_paths=sequence.image_paths,
            frame_indices=sequence.frame_indices,
            target_masks=target_masks[int(instance_id)],
            predictions=result["predictions"],
            aligned_recovery_index=result["aligned_recovery_index"],
            shuffled_recovery_index=result["shuffled_recovery_index"],
            output_size=config.output_size,
        )

    _write_csv(config.output_dir / "summary.csv", all_summary_rows)
    _write_csv(config.output_dir / "frame_metrics.csv", all_frame_rows)
    _write_csv(
        config.output_dir / "candidate_diagnostics.csv",
        all_candidate_rows,
    )
    metadata = {
        "experiment": "causal_geometry_recovery_memory_writeback_ablation",
        "scene_id": config.scene_id,
        "frame_indices": list(config.frame_indices),
        "instance_ids": [int(value) for value in instance_ids],
        "modes": list(MODES),
        "reference_policy": (
            "explicit_sequence_index"
            if reference_sequence_index is not None
            else "earliest_visible_frame_per_instance"
        ),
        "aligned_pair_shares_exact_recovery_mask": True,
        "later_gt_used_for_selection": False,
        "shuffled_control": (
            "reference geometry fixed; non-reference StreamVGGT outputs are "
            "cyclically permuted while RGB and SAM3 candidates stay fixed"
        ),
        "instances": instance_metadata,
    }
    with (config.output_dir / "metadata.json").open("w", encoding="utf8") as handle:
        json.dump(metadata, handle, indent=2, allow_nan=True)
    print(f"summary: {config.output_dir / 'summary.csv'}")
    print(f"frames: {config.output_dir / 'frame_metrics.csv'}")
    print(f"candidates: {config.output_dir / 'candidate_diagnostics.csv'}")


def _run_instance(
    config: ExperimentConfig,
    *,
    sequence,
    target_masks: torch.Tensor,
    geometry: GeometrySequence,
    sam3: SAM3Wrapper,
) -> dict:
    original = sam3.track(
        sequence.image_paths,
        prompt=sequence.label,
        output_size=config.output_size,
        reference_frame_idx=sequence.reference_frame_idx,
        reference_mask=target_masks[sequence.reference_frame_idx],
    )
    candidate_cache: dict[int, list[SAM3MaskCandidate]] = {}
    aligned = _prepare_recovery(
        config,
        sequence=sequence,
        target_masks=target_masks,
        original=original,
        geometry=geometry,
        geometry_alignment="aligned",
        geometry_permutation=tuple(range(len(sequence.frame_indices))),
        sam3=sam3,
        candidate_cache=candidate_cache,
    )

    predictions: dict[str, TrackingSequence] = {"original": original}
    if aligned["recovery_mask"] is None:
        aligned_no_memory = original
        aligned_memory = original
    else:
        recovery_index = int(aligned["recovery_index"])
        split = sam3.track_split_without_memory(
            sequence.image_paths,
            prompt=sequence.label,
            output_size=config.output_size,
            reference_frame_idx=sequence.reference_frame_idx,
            reference_mask=target_masks[sequence.reference_frame_idx],
            split_frame_idx=recovery_index,
        )
        if split.selected_obj_id != original.selected_obj_id:
            raise RuntimeError(
                "Aligned no-memory branch changed the persistent obj_id: "
                f"{original.selected_obj_id} -> {split.selected_obj_id}."
            )
        for index in range(len(sequence.frame_indices)):
            if not torch.equal(split.masks[index], original.masks[index]):
                raise RuntimeError(
                    "The causal no-memory split changed the original SAM3 "
                    f"prediction at sequence index {index}; the paired control "
                    "is therefore invalid."
                )
        no_memory_masks = split.masks.clone()
        no_memory_scores = split.scores.clone()
        no_memory_masks[recovery_index] = aligned["recovery_mask"]
        no_memory_scores[recovery_index] = 1.0
        aligned_no_memory = TrackingSequence(
            masks=no_memory_masks,
            scores=no_memory_scores,
            selected_obj_id=split.selected_obj_id,
        )
        aligned_memory = sam3.track_with_recovery_mask_memory(
            sequence.image_paths,
            prompt=sequence.label,
            output_size=config.output_size,
            reference_frame_idx=sequence.reference_frame_idx,
            reference_mask=target_masks[sequence.reference_frame_idx],
            recovery_frame_idx=recovery_index,
            recovery_mask=aligned["recovery_mask"],
        )
        if aligned_memory.selected_obj_id != original.selected_obj_id:
            raise RuntimeError(
                "Aligned memory branch changed the persistent obj_id: "
                f"{original.selected_obj_id} -> "
                f"{aligned_memory.selected_obj_id}."
            )
        if not torch.equal(
            aligned_no_memory.masks[recovery_index],
            aligned_memory.masks[recovery_index],
        ):
            raise RuntimeError(
                "Aligned no-memory and memory branches do not share the exact "
                "same recovery mask."
            )
        for index in range(recovery_index):
            if not torch.equal(
                aligned_no_memory.masks[index],
                aligned_memory.masks[index],
            ):
                raise RuntimeError(
                    "Aligned branches diverged before recovery at sequence "
                    f"index {index}."
                )
    predictions["geometry_recovery_no_memory"] = aligned_no_memory
    predictions["geometry_recovery_same_id_memory"] = aligned_memory

    permutation = _shuffled_permutation(
        len(sequence.frame_indices),
        reference_index=sequence.reference_frame_idx,
    )
    shuffled_geometry = _permute_geometry(geometry, permutation)
    shuffled = _prepare_recovery(
        config,
        sequence=sequence,
        target_masks=target_masks,
        original=original,
        geometry=shuffled_geometry,
        geometry_alignment="shuffled",
        geometry_permutation=permutation,
        sam3=sam3,
        candidate_cache=candidate_cache,
    )
    if shuffled["recovery_mask"] is None:
        shuffled_memory = original
    else:
        shuffled_index = int(shuffled["recovery_index"])
        shuffled_memory = sam3.track_with_recovery_mask_memory(
            sequence.image_paths,
            prompt=sequence.label,
            output_size=config.output_size,
            reference_frame_idx=sequence.reference_frame_idx,
            reference_mask=target_masks[sequence.reference_frame_idx],
            recovery_frame_idx=shuffled_index,
            recovery_mask=shuffled["recovery_mask"],
        )
        if shuffled_memory.selected_obj_id != original.selected_obj_id:
            raise RuntimeError(
                "Shuffled memory branch changed the persistent obj_id: "
                f"{original.selected_obj_id} -> "
                f"{shuffled_memory.selected_obj_id}."
            )
    predictions["shuffled_geometry_same_id_memory"] = shuffled_memory

    original_metrics = summarize_masks(
        original.masks,
        target_masks,
        reference_frame_idx=sequence.reference_frame_idx,
    )
    aligned_index = aligned["recovery_index"]
    summary_rows = []
    for mode in MODES:
        event = shuffled if mode == "shuffled_geometry_same_id_memory" else aligned
        if mode == "original":
            event = aligned
        tracking = predictions[mode]
        metrics = summarize_masks(
            tracking.masks,
            target_masks,
            reference_frame_idx=sequence.reference_frame_idx,
        )
        post = summarize_visible_after(
            tracking.masks,
            target_masks,
            recovery_frame_idx=event["recovery_index"],
        )
        summary_rows.append(
            {
                "scene_id": sequence.scene_id,
                "instance_id": int(sequence.instance_id),
                "instance_label": sequence.label,
                "frame_indices": " ".join(
                    str(value) for value in sequence.frame_indices
                ),
                "mode": mode,
                "geometry_alignment": event["geometry_alignment"],
                "geometry_permutation": " ".join(
                    str(value) for value in event["geometry_permutation"]
                ),
                "reference_sequence_index": sequence.reference_frame_idx,
                "reference_frame_index": sequence.frame_indices[
                    sequence.reference_frame_idx
                ],
                **metrics,
                "cross_iou_gain_over_original": (
                    metrics["cross_view_iou"]
                    - original_metrics["cross_view_iou"]
                ),
                "visible_miss_rate": _visible_miss_rate(
                    tracking.masks, target_masks
                ),
                "recovery_requested": int(event["recovery_requested"]),
                "recovery_applied": int(
                    event["recovery_mask"] is not None
                    and mode != "original"
                ),
                "recovery_sequence_index": event["recovery_index"],
                "recovery_frame_index": event["recovery_frame_index"],
                "recovery_reason": event["reason"],
                "candidate_count": event["candidate_count"],
                "selected_support_coverage": event[
                    "selected_support_coverage"
                ],
                "selected_candidate_gt_iou": event[
                    "selected_candidate_gt_iou"
                ],
                "oracle_candidate_gt_iou": event[
                    "oracle_candidate_gt_iou"
                ],
                "geometry_selected_oracle": event[
                    "geometry_selected_oracle"
                ],
                "recovery_mask_iou": event["selected_candidate_gt_iou"],
                "post_recovery_iou": post["iou"],
                "post_recovery_recall": post["recall"],
                "post_recovery_visible_frames": post["visible_frames"],
                "post_recovery_miss_rate": _visible_miss_rate(
                    tracking.masks,
                    target_masks,
                    after_index=event["recovery_index"],
                ),
                "persistent_obj_id": tracking.selected_obj_id,
                "same_obj_id_as_original": int(
                    tracking.selected_obj_id == original.selected_obj_id
                ),
                "aligned_pair_exact_same_recovery_mask": int(
                    mode
                    in {
                        "geometry_recovery_no_memory",
                        "geometry_recovery_same_id_memory",
                    }
                    and aligned["recovery_mask"] is not None
                ),
            }
        )

    no_memory_row = next(
        row
        for row in summary_rows
        if row["mode"] == "geometry_recovery_no_memory"
    )
    for row in summary_rows:
        row["memory_cross_iou_gain_over_no_memory"] = (
            row["cross_view_iou"] - no_memory_row["cross_view_iou"]
            if row["mode"] == "geometry_recovery_same_id_memory"
            else 0.0
        )
        row["memory_post_iou_gain_over_no_memory"] = (
            row["post_recovery_iou"] - no_memory_row["post_recovery_iou"]
            if row["mode"] == "geometry_recovery_same_id_memory"
            else 0.0
        )

    frame_rows = []
    for mode, tracking in predictions.items():
        event = shuffled if mode == "shuffled_geometry_same_id_memory" else aligned
        for index, frame_index in enumerate(sequence.frame_indices):
            frame_rows.append(
                {
                    "scene_id": sequence.scene_id,
                    "instance_id": int(sequence.instance_id),
                    "instance_label": sequence.label,
                    "mode": mode,
                    "sequence_index": index,
                    "frame_index": frame_index,
                    "gt_visible": int(target_masks[index].any()),
                    "prediction_pixels": int(tracking.masks[index].sum()),
                    "target_pixels": int(target_masks[index].sum()),
                    "iou": binary_iou(
                        tracking.masks[index], target_masks[index]
                    ),
                    "missed_visible_instance": int(
                        target_masks[index].any()
                        and not tracking.masks[index].any()
                    ),
                    "is_recovery_frame": int(
                        event["recovery_index"] == index
                    ),
                    "after_recovery": int(
                        event["recovery_index"] is not None
                        and index > int(event["recovery_index"])
                    ),
                    "persistent_obj_id": tracking.selected_obj_id,
                }
            )

    return {
        "summary_rows": summary_rows,
        "frame_rows": frame_rows,
        "candidate_rows": [
            *aligned["candidate_rows"],
            *shuffled["candidate_rows"],
        ],
        "predictions": predictions,
        "aligned_recovery_index": aligned_index,
        "shuffled_recovery_index": shuffled["recovery_index"],
        "metadata": {
            "label": sequence.label,
            "reference_sequence_index": sequence.reference_frame_idx,
            "reference_frame_index": sequence.frame_indices[
                sequence.reference_frame_idx
            ],
            "original_obj_id": original.selected_obj_id,
            "aligned": _event_metadata(aligned),
            "shuffled": _event_metadata(shuffled),
        },
    }


def _prepare_recovery(
    config: ExperimentConfig,
    *,
    sequence,
    target_masks: torch.Tensor,
    original: TrackingSequence,
    geometry: GeometrySequence,
    geometry_alignment: str,
    geometry_permutation: tuple[int, ...],
    sam3: SAM3Wrapper,
    candidate_cache: dict[int, list[SAM3MaskCandidate]],
) -> dict:
    mined = _mine_recovery(
        config,
        sequence=sequence,
        target_masks=target_masks,
        original_masks=original.masks,
        original_scores=original.scores,
        geometry=geometry,
    )
    recovery_index = next(
        (
            index
            for index, row in enumerate(mined["rows"])
            if index > int(sequence.reference_frame_idx)
            and row["use_correction"]
            and mined["candidates"][index].supported_mask.any()
        ),
        None,
    )
    base = {
        "geometry_alignment": geometry_alignment,
        "geometry_permutation": geometry_permutation,
        "recovery_requested": recovery_index is not None,
        "recovery_index": recovery_index,
        "recovery_frame_index": (
            sequence.frame_indices[recovery_index]
            if recovery_index is not None
            else None
        ),
        "recovery_mask": None,
        "candidate_count": 0,
        "selected_support_coverage": float("nan"),
        "selected_candidate_gt_iou": float("nan"),
        "oracle_candidate_gt_iou": float("nan"),
        "geometry_selected_oracle": 0,
        "candidate_rows": [],
        "reason": "no weak-tracker frame had an accepted geometry candidate",
    }
    if recovery_index is None:
        return base

    candidates = candidate_cache.get(recovery_index)
    if candidates is None:
        candidates = sam3.propose_text_masks(
            sequence.image_paths[recovery_index],
            prompt=sequence.label,
            output_size=config.output_size,
        )
        candidate_cache[recovery_index] = candidates
    base["candidate_count"] = len(candidates)
    if not candidates:
        base["reason"] = "SAM3 global-text query produced no candidate mask"
        return base

    geometry_candidate = mined["candidates"][recovery_index]
    supported = geometry_candidate.supported_mask.bool()
    projected = geometry_candidate.projected_mask.bool()
    coarse_box = geometry_candidate.mask.bool()
    rows = []
    ranking = []
    gt_ious = []
    for candidate_index, candidate in enumerate(candidates):
        mask = candidate.mask.bool()
        support_coverage = _coverage(supported, mask)
        projected_coverage = _coverage(projected, mask)
        box_iou = binary_iou(mask, coarse_box)
        gt_iou = binary_iou(mask, target_masks[recovery_index])
        gt_ious.append(gt_iou)
        ranking.append(
            (
                support_coverage,
                projected_coverage,
                box_iou,
                float(candidate.score),
                -int(candidate.obj_id),
                candidate_index,
            )
        )
        rows.append(
            {
                "scene_id": sequence.scene_id,
                "instance_id": int(sequence.instance_id),
                "instance_label": sequence.label,
                "geometry_alignment": geometry_alignment,
                "geometry_permutation": " ".join(
                    str(value) for value in geometry_permutation
                ),
                "sequence_index": recovery_index,
                "frame_index": sequence.frame_indices[recovery_index],
                "candidate_index": candidate_index,
                "temporary_obj_id": candidate.obj_id,
                "sam3_score": candidate.score,
                "candidate_pixels": int(mask.sum()),
                "support_coverage": support_coverage,
                "projected_coverage": projected_coverage,
                "coarse_box_iou": box_iou,
                "candidate_gt_iou": gt_iou,
                "selected": 0,
                "oracle": 0,
            }
        )
    selected_index = int(max(ranking)[-1])
    oracle_iou = max(gt_ious)
    oracle_indices = {
        index
        for index, value in enumerate(gt_ious)
        if abs(value - oracle_iou) <= 1e-8
    }
    rows[selected_index]["selected"] = 1
    for index in oracle_indices:
        rows[index]["oracle"] = 1
    base["candidate_rows"] = rows
    selected = candidates[selected_index]
    support_coverage = float(rows[selected_index]["support_coverage"])
    base.update(
        {
            "selected_support_coverage": support_coverage,
            "selected_candidate_gt_iou": gt_ious[selected_index],
            "oracle_candidate_gt_iou": oracle_iou,
            "geometry_selected_oracle": int(selected_index in oracle_indices),
        }
    )
    if not selected.mask.any():
        base["reason"] = "selected SAM3 candidate mask is empty"
        return base
    if support_coverage < float(
        config.hard_memory_min_recovery_support_recall
    ):
        base["reason"] = (
            "selected full mask failed geometry support coverage: "
            f"{support_coverage:.4f} < "
            f"{config.hard_memory_min_recovery_support_recall:.4f}"
        )
        return base
    base["recovery_mask"] = selected.mask.detach().cpu().bool()
    base["reason"] = "accepted full SAM3 candidate selected by aligned geometry"
    if geometry_alignment == "shuffled":
        base["reason"] = (
            "accepted full SAM3 candidate selected by shuffled geometry control"
        )
    return base


def _shuffled_permutation(
    count: int,
    *,
    reference_index: int,
) -> tuple[int, ...]:
    if count < 3:
        raise ValueError("Shuffled geometry control requires at least three frames.")
    movable = [index for index in range(count) if index != int(reference_index)]
    rotated = movable[1:] + movable[:1]
    permutation = list(range(count))
    for destination, source in zip(movable, rotated):
        permutation[destination] = source
    return tuple(permutation)


def _permute_geometry(
    geometry: GeometrySequence,
    permutation: tuple[int, ...],
) -> GeometrySequence:
    def permute_tensor(value):
        if value is None:
            return None
        indices = torch.tensor(permutation, dtype=torch.long, device=value.device)
        return value.index_select(0, indices)

    return GeometrySequence(
        world_points=permute_tensor(geometry.world_points),
        confidence=permute_tensor(geometry.confidence),
        world_to_camera=permute_tensor(geometry.world_to_camera),
        intrinsics=permute_tensor(geometry.intrinsics),
        processed_size=geometry.processed_size,
        source_sizes=tuple(geometry.source_sizes[index] for index in permutation),
        depth=permute_tensor(geometry.depth),
        depth_confidence=permute_tensor(geometry.depth_confidence),
        camera_world_points=permute_tensor(geometry.camera_world_points),
    )


def _coverage(evidence: torch.Tensor, mask: torch.Tensor) -> float:
    evidence = evidence.detach().cpu().bool()
    mask = mask.detach().cpu().bool()
    denominator = int(evidence.sum())
    if denominator == 0:
        return 0.0
    return float((evidence & mask).sum()) / denominator


def _visible_miss_rate(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    after_index: int | None = None,
) -> float:
    visible = target.flatten(1).any(dim=1)
    if after_index is not None:
        visible &= torch.arange(len(target)) > int(after_index)
    if not visible.any():
        return 0.0
    predicted = prediction.flatten(1).any(dim=1)
    return float((~predicted[visible]).float().mean())


def _event_metadata(event: dict) -> dict:
    return {
        key: value
        for key, value in event.items()
        if key not in {"recovery_mask", "candidate_rows"}
    }


def _save_report(
    path: Path,
    *,
    image_paths,
    frame_indices,
    target_masks: torch.Tensor,
    predictions: dict[str, TrackingSequence],
    aligned_recovery_index: int | None,
    shuffled_recovery_index: int | None,
    output_size: tuple[int, int],
) -> None:
    height, width = output_size
    header = 34
    columns = 2 + len(MODES)
    canvas = Image.new(
        "RGB",
        (columns * width, len(image_paths) * (height + header)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    labels = ("RGB", "GT", *MODES)
    colors = {
        "GT": (20, 220, 70),
        "original": (220, 60, 60),
        "geometry_recovery_no_memory": (245, 170, 30),
        "geometry_recovery_same_id_memory": (45, 110, 255),
        "shuffled_geometry_same_id_memory": (170, 70, 210),
    }
    for row, image_path in enumerate(image_paths):
        with Image.open(image_path) as source:
            rgb = source.convert("RGB").resize(
                (width, height), Image.Resampling.BILINEAR
            )
        panels = [rgb, _overlay(rgb, target_masks[row], colors["GT"])]
        panels.extend(
            _overlay(rgb, predictions[mode].masks[row], colors[mode])
            for mode in MODES
        )
        top = row * (height + header)
        for column, (label, panel) in enumerate(zip(labels, panels)):
            canvas.paste(panel, (column * width, top + header))
            suffix = f" frame={frame_indices[row]}"
            if label in predictions:
                suffix += (
                    f" IoU={binary_iou(predictions[label].masks[row], target_masks[row]):.3f}"
                )
            if row == aligned_recovery_index and label in {
                "geometry_recovery_no_memory",
                "geometry_recovery_same_id_memory",
            }:
                suffix += " RECOVERY"
            if (
                row == shuffled_recovery_index
                and label == "shuffled_geometry_same_id_memory"
            ):
                suffix += " SHUFFLED-RECOVERY"
            draw.text(
                (column * width + 5, top + 7),
                label + suffix,
                fill=colors.get(label, (0, 0, 0)),
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _overlay(
    image: Image.Image,
    mask: torch.Tensor,
    color: tuple[int, int, int],
) -> Image.Image:
    array = np.asarray(image).copy()
    selected = mask.detach().cpu().numpy().astype(bool)
    if selected.any():
        array[selected] = (
            0.45 * array[selected] + 0.55 * np.asarray(color)
        ).astype(np.uint8)
    return Image.fromarray(array)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def _validate_shared_sequence(sequences: dict[int, object]) -> None:
    values = list(sequences.values())
    first = values[0]
    for sequence in values[1:]:
        if sequence.frame_indices != first.frame_indices:
            raise RuntimeError("All instances must use the same frame indices.")
        if [str(path) for path in sequence.image_paths] != [
            str(path) for path in first.image_paths
        ]:
            raise RuntimeError("All instances must use the same RGB sequence.")


def _unique_ids(values: Sequence[int]) -> tuple[int, ...]:
    result = []
    for value in values:
        value = int(value)
        if value not in result:
            result.append(value)
    if not result:
        raise ValueError("At least one instance ID is required.")
    return tuple(result)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="streaming_couping/configs/default.yaml",
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--scene-id")
    parser.add_argument("--instance-ids", type=int, nargs="+", required=True)
    parser.add_argument("--frame-indices", type=int, nargs="+")
    parser.add_argument(
        "--reference-sequence-index",
        type=int,
        help=(
            "Use one explicit reference position for every instance. By default "
            "each instance uses its earliest visible selected frame."
        ),
    )
    parser.add_argument("--sam3-device")
    parser.add_argument("--geometry-device")
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    main()
