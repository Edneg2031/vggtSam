"""Paired same-instance test for geometry correction with/without SAM3 memory.

The reference GT mask initializes SAM3 and the object point map. Later GT masks
are metrics-only. Both branches share one correction mask; only memory writeback
differs after recovery.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from test_sam.coordinates import streamvggt_label_to_grid
from test_sam.data import load_mask_tracking_sequence

from .aggregation.mine_revisit_segments import mine_revisit_candidate
from .aggregation.point_map_fusion import ObjectPointMap, sample_masked_observation
from .backbones.sam3_wrapper import SAM3Wrapper
from .backbones.streamvggt_wrapper import StreamVGGTWrapper
from .bridge.gating import binary_iou, decide_correction
from .config import ExperimentConfig, load_config
from .types import GeometrySequence, RevisitCandidate


def main() -> None:
    args = _parse_args()
    overrides = {
        key: value
        for key, value in {
            "manifest": args.manifest,
            "scene_id": args.scene_id,
            "instance_id": args.instance_id,
            "frame_indices": args.frame_indices,
            "sam3_device": args.sam3_device,
            "geometry_device": args.geometry_device,
            "output_dir": args.output_dir,
        }.items()
        if value is not None
    }
    run_experiment(load_config(args.config, overrides))


def run_experiment(config: ExperimentConfig) -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    sequence = load_mask_tracking_sequence(
        config.manifest,
        scene_id=config.scene_id,
        frame_indices=config.frame_indices,
        sequence_length=len(config.frame_indices),
        frame_stride=1,
        window_index=0,
        instance_id=config.instance_id,
        min_pixels=config.min_pixels,
        max_area_ratio=config.max_area_ratio,
        min_visible_frames=1,
        excluded_labels=config.excluded_labels,
        seed=0,
    )
    target_masks = _resize_target_masks(sequence.target_masks, config.output_size)
    visible = [bool(mask.any()) for mask in target_masks]
    print(
        f"target scene={sequence.scene_id} frames={sequence.frame_indices} "
        f"instance={sequence.instance_id} label={sequence.label!r} "
        f"reference={sequence.reference_frame_idx} visible={visible}"
    )

    print("loading and running frozen SAM3 video tracker...")
    sam3 = SAM3Wrapper(
        repo_path=config.sam3_repo,
        checkpoint_path=config.sam3_checkpoint,
        device=config.sam3_device,
        output_threshold=config.sam3_output_threshold,
        prompt_with_box=config.prompt_with_box,
    ).load()
    tracking = sam3.track(
        sequence.image_paths,
        prompt=sequence.label,
        output_size=config.output_size,
        reference_frame_idx=sequence.reference_frame_idx,
        reference_mask=target_masks[sequence.reference_frame_idx],
    )
    print(f"recovery-selection SAM3 obj_id={tracking.selected_obj_id}")

    print("loading and running frozen StreamVGGT with causal caches...")
    geometry = StreamVGGTWrapper(
        repo_path=config.streamvggt_repo,
        checkpoint_path=config.streamvggt_checkpoint,
        device=config.geometry_device,
        image_mode=config.image_mode,
        streaming_cache=config.streaming_cache,
    ).load().extract(sequence.image_paths)

    result = _mine_recovery(
        config,
        sequence=sequence,
        target_masks=target_masks,
        original_masks=tracking.masks,
        original_scores=tracking.scores,
        geometry=geometry,
    )
    recovery_idx = next(
        (
            index
            for index, row in enumerate(result["rows"])
            if row["use_correction"]
            and result["candidates"][index].supported_mask.any()
        ),
        None,
    )
    if recovery_idx is None:
        raise RuntimeError(
            "No accepted aligned-geometry recovery was found for the paired memory test."
        )

    recovery_candidate = result["candidates"][recovery_idx]
    recovery_mask, recovery_score = sam3.recover_mask_with_text_geometry(
        sequence.image_paths[recovery_idx],
        prompt=sequence.label,
        output_size=config.output_size,
        candidate_mask=recovery_candidate.mask,
        supported_mask=recovery_candidate.supported_mask,
    )
    if not recovery_mask.any():
        raise RuntimeError("Text-guided SAM3 recovery returned an empty mask.")

    no_memory_tracking = sam3.track_split_without_memory(
        sequence.image_paths,
        prompt=sequence.label,
        output_size=config.output_size,
        reference_frame_idx=sequence.reference_frame_idx,
        reference_mask=target_masks[sequence.reference_frame_idx],
        split_frame_idx=recovery_idx,
    )
    memory_tracking = sam3.track_with_recovery_mask_memory(
        sequence.image_paths,
        prompt=sequence.label,
        output_size=config.output_size,
        reference_frame_idx=sequence.reference_frame_idx,
        reference_mask=target_masks[sequence.reference_frame_idx],
        recovery_frame_idx=recovery_idx,
        recovery_mask=recovery_mask,
    )
    if memory_tracking.selected_obj_id != no_memory_tracking.selected_obj_id:
        raise RuntimeError(
            "SAM3 obj_id changed between paired branches: "
            f"{no_memory_tracking.selected_obj_id} != "
            f"{memory_tracking.selected_obj_id}."
        )
    for index in range(recovery_idx):
        if not torch.equal(
            no_memory_tracking.masks[index], memory_tracking.masks[index]
        ):
            raise RuntimeError(
                f"Paired SAM3 sessions diverged before recovery at frame {index}."
            )
    corrected_mask = recovery_mask.clone()
    no_memory_masks = no_memory_tracking.masks.clone()
    no_memory_scores = no_memory_tracking.scores.clone()
    no_memory_masks[recovery_idx] = corrected_mask
    no_memory_scores[recovery_idx] = 1.0
    if not torch.equal(no_memory_masks[recovery_idx], memory_tracking.masks[recovery_idx]):
        raise RuntimeError("Paired branches do not share the same recovery mask.")

    no_memory_metrics = summarize_masks(
        no_memory_masks,
        target_masks,
        reference_frame_idx=sequence.reference_frame_idx,
    )
    memory_metrics = summarize_masks(
        memory_tracking.masks,
        target_masks,
        reference_frame_idx=sequence.reference_frame_idx,
    )
    no_memory_future = summarize_visible_after(
        no_memory_masks,
        target_masks,
        recovery_frame_idx=recovery_idx,
    )
    memory_future = summarize_visible_after(
        memory_tracking.masks,
        target_masks,
        recovery_frame_idx=recovery_idx,
    )
    for index, row in enumerate(result["rows"]):
        row["is_recovery_frame"] = int(index == recovery_idx)
        row["after_recovery"] = int(index > recovery_idx)
        row["no_memory_score"] = float(no_memory_scores[index])
        row["memory_score"] = float(memory_tracking.scores[index])
        row["no_memory_iou"] = binary_iou(
            no_memory_masks[index], target_masks[index]
        )
        row["memory_iou"] = binary_iou(
            memory_tracking.masks[index], target_masks[index]
        )

    _write_csv(config.output_dir / "frame_metrics.csv", result["rows"])
    _save_paired_report(
        config.output_dir / "paired_memory_report.png",
        image_paths=sequence.image_paths,
        frame_indices=sequence.frame_indices,
        target_masks=target_masks,
        candidates=result["candidates"],
        no_memory_masks=no_memory_masks,
        memory_masks=memory_tracking.masks,
        rows=result["rows"],
        output_size=config.output_size,
    )
    summary = {
        **{f"no_memory_{key}": value for key, value in no_memory_metrics.items()},
        **{f"memory_{key}": value for key, value in memory_metrics.items()},
        **{
            f"no_memory_post_recovery_{key}": value
            for key, value in no_memory_future.items()
        },
        **{
            f"memory_post_recovery_{key}": value
            for key, value in memory_future.items()
        },
        "recovery_sequence_index": recovery_idx,
        "recovery_frame_index": sequence.frame_indices[recovery_idx],
        "sam3_obj_id": memory_tracking.selected_obj_id,
        "same_obj_id": 1,
        "recovery_source": "text_box_points_full_mask",
        "recovery_reacquisition_used": 1,
        "paired_branch_redetection_used": 0,
        "memory_existing_object_reactivated": 1,
        "recovery_mask_score": recovery_score,
        "recovery_mask_iou": binary_iou(
            recovery_mask,
            target_masks[recovery_idx],
        ),
        "paired_causal_split": 1,
        "geometry_prompt_points": min(
            3,
            result["candidates"][recovery_idx].supported_points,
        ),
        "pre_recovery_masks_equal": 1,
        "recovery_masks_equal": 1,
    }
    _write_csv(config.output_dir / "summary.csv", [summary])
    with (config.output_dir / "resolved_config.json").open("w", encoding="utf8") as handle:
        json.dump(_config_json(config), handle, indent=2)
    print(
        f"paired memory test recovery_frame={summary['recovery_frame_index']} "
        f"obj_id={summary['sam3_obj_id']} post_recovery_iou="
        f"{no_memory_future['iou']:.4f}/{memory_future['iou']:.4f}"
    )
    print(f"summary: {config.output_dir / 'summary.csv'}")


def _mine_recovery(
    config: ExperimentConfig,
    *,
    sequence,
    target_masks: torch.Tensor,
    original_masks: torch.Tensor,
    original_scores: torch.Tensor,
    geometry: GeometrySequence,
) -> dict:
    num_frames = len(sequence.frame_indices)
    object_map = ObjectPointMap(max_points_per_object=config.max_points_per_object)
    candidates: list[RevisitCandidate] = []
    rows: list[dict] = []

    ref = sequence.reference_frame_idx
    ref_stream_mask = _output_mask_to_stream(
        target_masks[ref],
        source_size=geometry.source_sizes[ref],
        processed_size=geometry.processed_size,
        image_mode=config.image_mode,
    )
    points, weights = sample_masked_observation(
        geometry.world_points[ref],
        geometry.confidence[ref],
        ref_stream_mask,
        max_points=config.max_points_per_observation,
    )
    object_map.update(
        instance_id=sequence.instance_id,
        label=sequence.label,
        points=points,
        weights=weights,
        frame_idx=ref,
    )

    for frame_idx in range(num_frames):
        original_mask = original_masks[frame_idx]
        original_score = float(original_scores[frame_idx])
        if frame_idx == ref:
            candidate = _empty_candidate(config.output_size, "reference frame")
        else:
            entry = object_map.get(sequence.instance_id)
            if entry is None:
                candidate = _empty_candidate(config.output_size, "object map is unavailable")
            else:
                candidate = mine_revisit_candidate(
                    entry.points,
                    current_world_points=geometry.world_points[frame_idx],
                    world_to_camera=geometry.world_to_camera[frame_idx],
                    intrinsics=geometry.intrinsics[frame_idx],
                    source_size=geometry.source_sizes[frame_idx],
                    processed_size=geometry.processed_size,
                    output_size=config.output_size,
                    image_mode=config.image_mode,
                    box_quantile=config.box_quantile,
                    box_padding_ratio=config.box_padding_ratio,
                    min_projected_points=config.min_projected_points,
                    min_projected_fraction=config.min_projected_fraction,
                    min_supported_points=config.min_supported_points,
                    min_support_ratio=config.min_support_ratio,
                    support_abs_distance=config.support_abs_distance,
                    support_relative_distance=config.support_relative_distance,
                )
        candidates.append(candidate)
        decision = decide_correction(
            tracker_mask=original_mask,
            tracker_score=original_score,
            candidate=candidate,
            tracker_low_score=config.tracker_low_score,
            fallback_on_missing_mask=config.fallback_on_missing_mask,
        )

        rows.append(
            {
                "sequence_index": frame_idx,
                "frame_index": sequence.frame_indices[frame_idx],
                "geometry_index": frame_idx,
                "gt_visible": int(target_masks[frame_idx].any()),
                "sam3_score": original_score,
                "sam3_iou": binary_iou(original_mask, target_masks[frame_idx]),
                "candidate_iou": binary_iou(candidate.mask, target_masks[frame_idx]),
                "candidate_area_ratio": float(candidate.mask.float().mean()),
                "candidate_centroid_error": _centroid_error(
                    candidate.mask,
                    target_masks[frame_idx],
                ),
                "projected_points": candidate.projected_points,
                "supported_points": candidate.supported_points,
                "projected_fraction": candidate.projected_fraction,
                "support_ratio": candidate.support_ratio,
                "candidate_accepted": int(candidate.accepted),
                "use_correction": int(decision.use_correction),
                "gate_reason": decision.reason,
            }
        )
    return {
        "rows": rows,
        "candidates": candidates,
    }


def summarize_masks(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    reference_frame_idx: int,
) -> dict[str, float]:
    # Metrics are scalar diagnostics. Keeping both masks on CPU avoids mixing
    # CPU IoU tensors with GPU visibility indices during training-time eval.
    prediction = prediction.detach().cpu()
    target = target.detach().cpu()
    ious = torch.tensor(
        [binary_iou(pred, gt) for pred, gt in zip(prediction, target)],
        dtype=torch.float32,
    )
    visible = target.flatten(1).any(dim=1)
    cross = visible.clone()
    cross[int(reference_frame_idx)] = False
    absent = ~visible
    return {
        "mean_iou": float(ious.mean()),
        "positive_iou": float(ious[visible].mean()) if visible.any() else 0.0,
        "cross_view_iou": float(ious[cross].mean()) if cross.any() else 0.0,
        "cross_view_recall": (
            float((ious[cross] >= 0.5).float().mean()) if cross.any() else 0.0
        ),
        "absent_fp_ratio": (
            float(prediction[absent].float().mean()) if absent.any() else 0.0
        ),
    }


def summarize_visible_after(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    recovery_frame_idx: int | None,
) -> dict[str, float | int]:
    """Measure future visible frames without counting the recovery mask itself."""

    if recovery_frame_idx is None:
        return {"iou": 0.0, "recall": 0.0, "visible_frames": 0}
    frame_indices = torch.arange(len(target))
    visible = target.flatten(1).any(dim=1)
    selected = visible & (frame_indices > int(recovery_frame_idx))
    if not selected.any():
        return {"iou": 0.0, "recall": 0.0, "visible_frames": 0}
    ious = torch.tensor(
        [
            binary_iou(prediction[index], target[index])
            for index in selected.nonzero(as_tuple=False).flatten().tolist()
        ],
        dtype=torch.float32,
    )
    return {
        "iou": float(ious.mean()),
        "recall": float((ious >= 0.5).float().mean()),
        "visible_frames": int(selected.sum()),
    }


def _resize_target_masks(
    masks: list[np.ndarray],
    output_size: tuple[int, int],
) -> torch.Tensor:
    tensor = torch.from_numpy(np.stack(masks)).float()[:, None]
    return F.interpolate(tensor, size=output_size, mode="nearest")[:, 0].bool()


def _output_mask_to_stream(
    mask: torch.Tensor,
    *,
    source_size: tuple[int, int],
    processed_size: tuple[int, int],
    image_mode: str,
) -> torch.Tensor:
    source = F.interpolate(mask.float()[None, None], size=source_size, mode="nearest")[0, 0]
    labels = streamvggt_label_to_grid(
        source.cpu().numpy().astype(np.uint8),
        processed_size,
        mode=image_mode,
    )
    return torch.from_numpy(labels > 0)


def _empty_candidate(output_size: tuple[int, int], reason: str) -> RevisitCandidate:
    empty = torch.zeros(output_size, dtype=torch.bool)
    return RevisitCandidate(
        mask=empty,
        projected_mask=empty.clone(),
        supported_mask=empty.clone(),
        box_xyxy=None,
        projected_points=0,
        supported_points=0,
        projected_fraction=0.0,
        support_ratio=0.0,
        accepted=False,
        reason=reason,
    )


def _save_paired_report(
    path: Path,
    *,
    image_paths,
    frame_indices,
    target_masks: torch.Tensor,
    candidates: list[RevisitCandidate],
    no_memory_masks: torch.Tensor,
    memory_masks: torch.Tensor,
    rows: list[dict],
    output_size: tuple[int, int],
) -> None:
    height, width = output_size
    header = 30
    columns = 5
    canvas = Image.new("RGB", (columns * width, len(image_paths) * (height + header)), "white")
    labels = ("RGB", "GT", "geometry candidate", "no memory", "memory")
    colors = ((0, 0, 0), (0, 220, 70), (230, 55, 55), (255, 190, 0), (45, 110, 255))
    for row_idx, image_path in enumerate(image_paths):
        with Image.open(image_path) as source:
            rgb = source.convert("RGB").resize((width, height), Image.Resampling.BILINEAR)
        panels = [
            rgb,
            _overlay(rgb, target_masks[row_idx], colors[1]),
            _draw_candidate(rgb, candidates[row_idx]),
            _overlay(rgb, no_memory_masks[row_idx], colors[3]),
            _overlay(rgb, memory_masks[row_idx], colors[4]),
        ]
        y = row_idx * (height + header)
        for column, (panel, label) in enumerate(zip(panels, labels)):
            canvas.paste(panel, (column * width, y + header))
            draw = ImageDraw.Draw(canvas)
            suffix = ""
            if column == 0:
                suffix = f" frame={frame_indices[row_idx]}"
            if column == 2:
                suffix = (
                    f" support={rows[row_idx]['support_ratio']:.3f} "
                    f"accepted={rows[row_idx]['candidate_accepted']}"
                )
            if column == 3:
                suffix = f" IoU={rows[row_idx]['no_memory_iou']:.3f}"
            if column == 4:
                suffix = f" IoU={rows[row_idx]['memory_iou']:.3f}"
            draw.text((column * width + 5, y + 7), label + suffix, fill=colors[column])
    canvas.save(path)


def _overlay(image: Image.Image, mask: torch.Tensor, color: tuple[int, int, int]) -> Image.Image:
    array = np.asarray(image).copy()
    selected = mask.cpu().numpy().astype(bool)
    if selected.any():
        array[selected] = (0.45 * array[selected] + 0.55 * np.asarray(color)).astype(np.uint8)
    return Image.fromarray(array)


def _draw_candidate(image: Image.Image, candidate: RevisitCandidate) -> Image.Image:
    output = _overlay(image, candidate.projected_mask, (255, 190, 0))
    output = _overlay(output, candidate.supported_mask, (30, 220, 90))
    if candidate.box_xyxy is not None:
        color = (30, 220, 90) if candidate.accepted else (255, 190, 0)
        ImageDraw.Draw(output).rectangle(candidate.box_xyxy, outline=color, width=3)
    return output


def _centroid_error(prediction: torch.Tensor, target: torch.Tensor) -> float:
    if not prediction.any() or not target.any():
        return float("nan")
    pred_y, pred_x = prediction.nonzero(as_tuple=True)
    target_y, target_x = target.nonzero(as_tuple=True)
    height, width = prediction.shape
    dx = (pred_x.float().mean() - target_x.float().mean()) / max(width, 1)
    dy = (pred_y.float().mean() - target_y.float().mean()) / max(height, 1)
    return float(torch.sqrt(dx * dx + dy * dy))


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _config_json(config: ExperimentConfig) -> dict:
    return {
        key: str(value) if isinstance(value, Path) else list(value) if isinstance(value, tuple) else value
        for key, value in config.__dict__.items()
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="streaming_couping/configs/default.yaml")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--scene-id")
    parser.add_argument("--instance-id", type=int)
    parser.add_argument("--frame-indices", type=int, nargs="+")
    parser.add_argument("--sam3-device")
    parser.add_argument("--geometry-device")
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    main()
