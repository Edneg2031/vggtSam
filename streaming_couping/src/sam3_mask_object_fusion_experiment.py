"""Compare mask sources for object-local StreamVGGT point-cloud fusion.

The reference GT mask is shared by every branch. Later-frame corrections only
move points selected by that branch's instance mask; camera poses and the full
scene pointmaps remain unchanged.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from test_sam.data import load_mask_tracking_sequence

from .backbones.sam3_wrapper import SAM3Wrapper
from .backbones.streamvggt_wrapper import StreamVGGTWrapper
from .bridge.gating import binary_iou
from .config import ExperimentConfig, load_config
from .geometry.export import save_aggregate_ply, save_pointmap_ply
from .geometry.gt_data import load_gt_geometry_sequence
from .geometry.registration import (
    ICPResult,
    apply_rigid,
    apply_similarity,
    estimate_similarity,
    robust_icp,
    rotation_angle_degrees,
    symmetric_chamfer,
)
from .gt_mask_pose_experiment import select_causal_reference
from .pipeline import (
    _mine_recovery,
    _output_mask_to_stream,
    _resize_target_masks,
)


MASK_SOURCES = ("gt_oracle", "sam3_original", "sam3_geometry_memory")


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
    run_experiment(
        load_config(args.config, overrides),
        confidence_threshold=args.confidence_threshold,
        alignment_trim_fraction=args.alignment_trim_fraction,
        icp_max_points=args.icp_max_points,
        icp_iterations=args.icp_iterations,
        icp_trim_fraction=args.icp_trim_fraction,
        icp_max_correspondence=args.icp_max_correspondence,
        icp_min_inliers=args.icp_min_inliers,
        icp_min_fitness=args.icp_min_fitness,
        icp_max_rmse=args.icp_max_rmse,
        icp_mode=args.icp_mode,
        reference_sequence_index=args.reference_sequence_index,
    )


def run_experiment(
    config: ExperimentConfig,
    *,
    confidence_threshold: float,
    alignment_trim_fraction: float,
    icp_max_points: int,
    icp_iterations: int,
    icp_trim_fraction: float,
    icp_max_correspondence: float,
    icp_min_inliers: int,
    icp_min_fitness: float,
    icp_max_rmse: float,
    icp_mode: str,
    reference_sequence_index: int | None,
) -> None:
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
    reference = select_causal_reference(sequence, reference_sequence_index)
    sequence = replace(sequence, reference_frame_idx=reference)
    target_output_masks = _resize_target_masks(
        sequence.target_masks,
        config.output_size,
    )
    print(
        f"target scene={sequence.scene_id} frames={sequence.frame_indices} "
        f"instance={sequence.instance_id} label={sequence.label!r} "
        f"reference={reference} icp_mode={icp_mode}"
    )

    print("running frozen SAM3 original tracker...")
    sam3 = SAM3Wrapper(
        repo_path=config.sam3_repo,
        checkpoint_path=config.sam3_checkpoint,
        device=config.sam3_device,
        output_threshold=config.sam3_output_threshold,
        prompt_with_box=config.prompt_with_box,
    ).load()
    original_tracking = sam3.track(
        sequence.image_paths,
        prompt=sequence.label,
        output_size=config.output_size,
        reference_frame_idx=reference,
        reference_mask=target_output_masks[reference],
    )

    print("running frozen StreamVGGT with causal caches...")
    geometry = StreamVGGTWrapper(
        repo_path=config.streamvggt_repo,
        checkpoint_path=config.streamvggt_checkpoint,
        device=config.geometry_device,
        image_mode=config.image_mode,
        streaming_cache=config.streaming_cache,
    ).load().extract(sequence.image_paths)
    gt = load_gt_geometry_sequence(
        config.manifest,
        scene_id=sequence.scene_id,
        frame_indices=sequence.frame_indices,
        instance_id=sequence.instance_id,
        processed_size=geometry.processed_size,
        image_mode=config.image_mode,
    )

    recovery = _mine_recovery(
        config,
        sequence=sequence,
        target_masks=target_output_masks,
        original_masks=original_tracking.masks,
        original_scores=original_tracking.scores,
        geometry=geometry,
    )
    recovery_index = next(
        (
            index
            for index, row in enumerate(recovery["rows"])
            if row["use_correction"]
            and recovery["candidates"][index].supported_mask.any()
        ),
        None,
    )
    if recovery_index is None:
        print(
            "no accepted geometry recovery: keeping geometry-memory identical "
            "to the original SAM3 control"
        )
        recovery_score = None
        no_memory_tracking = original_tracking
        memory_tracking = original_tracking
    else:
        candidate = recovery["candidates"][recovery_index]
        recovery_mask, recovery_score = sam3.recover_mask_with_text_geometry(
            sequence.image_paths[recovery_index],
            prompt=sequence.label,
            output_size=config.output_size,
            candidate_mask=candidate.mask,
            supported_mask=candidate.supported_mask,
        )
        if not recovery_mask.any():
            raise RuntimeError("Geometry-guided SAM3 recovery returned an empty mask.")
        no_memory_tracking = sam3.track_split_without_memory(
            sequence.image_paths,
            prompt=sequence.label,
            output_size=config.output_size,
            reference_frame_idx=reference,
            reference_mask=target_output_masks[reference],
            split_frame_idx=recovery_index,
        )
        memory_tracking = sam3.track_with_recovery_mask_memory(
            sequence.image_paths,
            prompt=sequence.label,
            output_size=config.output_size,
            reference_frame_idx=reference,
            reference_mask=target_output_masks[reference],
            recovery_frame_idx=recovery_index,
            recovery_mask=recovery_mask,
        )
        if no_memory_tracking.selected_obj_id != memory_tracking.selected_obj_id:
            raise RuntimeError(
                "SAM3 persistent obj_id differs between paired branches: "
                f"{no_memory_tracking.selected_obj_id} != "
                f"{memory_tracking.selected_obj_id}."
            )
        for index in range(recovery_index):
            if not torch.equal(
                no_memory_tracking.masks[index], memory_tracking.masks[index]
            ):
                raise RuntimeError(
                    f"Paired SAM3 branches diverged before recovery at frame {index}."
                )

    similarity = _reference_similarity(
        geometry.world_points[reference],
        geometry.confidence[reference],
        gt.pointmaps[reference],
        confidence_threshold=confidence_threshold,
        trim_fraction=alignment_trim_fraction,
    )
    raw_points = apply_similarity(
        geometry.world_points,
        similarity.scale,
        similarity.rotation,
        similarity.translation,
    )
    print(
        f"reference Sim3 scale={similarity.scale:.6f} "
        f"inliers={similarity.inliers} rmse={similarity.rmse:.6f}"
    )

    masks = {
        "gt_oracle": gt.instance_masks.clone(),
        "sam3_original": _sam_masks_to_stream(
            no_memory_tracking.masks,
            geometry=geometry,
            image_mode=config.image_mode,
        ),
        "sam3_geometry_memory": _sam_masks_to_stream(
            memory_tracking.masks,
            geometry=geometry,
            image_mode=config.image_mode,
        ),
    }
    # All branches share the exact same reference observation and object map.
    for branch_masks in masks.values():
        branch_masks[reference] = gt.instance_masks[reference]

    reference_valid = (
        gt.instance_masks[reference]
        & torch.isfinite(raw_points[reference]).all(dim=-1)
        & (geometry.confidence[reference] >= float(confidence_threshold))
    )
    reference_points = raw_points[reference][reference_valid]
    if reference_points.shape[0] < int(icp_min_inliers):
        raise RuntimeError(
            "Reference object contains too few confident StreamVGGT points: "
            f"{reference_points.shape[0]}."
        )

    all_rows: list[dict] = []
    summaries: list[dict] = []
    transforms: dict[str, list[dict]] = {}
    exported_maps: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for source in MASK_SOURCES:
        branch_rows, raw_object_maps, refined_object_maps, corrections = _run_branch(
            source,
            masks=masks[source],
            reference=reference,
            reference_points=reference_points,
            raw_points=raw_points,
            confidence=geometry.confidence,
            gt_masks=gt.instance_masks,
            gt_points=gt.pointmaps,
            frame_indices=sequence.frame_indices,
            confidence_threshold=confidence_threshold,
            icp_max_points=icp_max_points,
            icp_iterations=icp_iterations,
            icp_trim_fraction=icp_trim_fraction,
            icp_max_correspondence=icp_max_correspondence,
            icp_min_inliers=icp_min_inliers,
            icp_min_fitness=icp_min_fitness,
            icp_max_rmse=icp_max_rmse,
            translation_only=icp_mode == "translation_only",
        )
        all_rows.extend(branch_rows)
        transforms[source] = corrections
        exported_maps[source] = (raw_object_maps, refined_object_maps)
        summaries.append(
            _summarize_branch(
                source,
                branch_rows,
                raw_object_maps=raw_object_maps,
                refined_object_maps=refined_object_maps,
                gt_points=gt.pointmaps,
                gt_masks=gt.instance_masks,
                reference=reference,
            )
        )
        summaries[-1].update(
            {
                "recovery_triggered": int(recovery_index is not None),
                "recovery_sequence_index": recovery_index,
                "recovery_frame_index": (
                    sequence.frame_indices[recovery_index]
                    if recovery_index is not None
                    else None
                ),
            }
        )

    _export_results(
        config.output_dir,
        frame_indices=sequence.frame_indices,
        raw_points=raw_points,
        gt_points=gt.pointmaps,
        gt_masks=gt.instance_masks,
        colors=gt.colors,
        confidence=geometry.confidence,
        confidence_threshold=confidence_threshold,
        branches=exported_maps,
    )
    _save_mask_report(
        config.output_dir / "mask_sources.png",
        image_paths=sequence.image_paths,
        frame_indices=sequence.frame_indices,
        gt_masks=target_output_masks,
        original_masks=no_memory_tracking.masks,
        memory_masks=memory_tracking.masks,
        recovery_index=recovery_index,
        output_size=config.output_size,
    )
    _write_csv(config.output_dir / "frame_metrics.csv", all_rows)
    _write_csv(config.output_dir / "summary.csv", summaries)
    with (config.output_dir / "transforms.json").open("w", encoding="utf8") as handle:
        json.dump(
            {
                "settings": {
                    "mask_sources": MASK_SOURCES,
                    "reference_sequence_index": reference,
                    "reference_frame_index": sequence.frame_indices[reference],
                    "reference_mask_shared": True,
                    "camera_pose_modified": False,
                    "full_scene_pointmap_modified": False,
                    "icp_mode": icp_mode,
                    "confidence_threshold": confidence_threshold,
                    "icp_max_points": icp_max_points,
                    "icp_iterations": icp_iterations,
                    "icp_trim_fraction": icp_trim_fraction,
                    "icp_max_correspondence": icp_max_correspondence,
                    "icp_min_inliers": icp_min_inliers,
                    "icp_min_fitness": icp_min_fitness,
                    "icp_max_rmse": icp_max_rmse,
                },
                "similarity": {
                    "scale": similarity.scale,
                    "rotation": similarity.rotation.tolist(),
                    "translation": similarity.translation.tolist(),
                    "inliers": similarity.inliers,
                    "rmse": similarity.rmse,
                },
                "recovery": {
                    "triggered": recovery_index is not None,
                    "sequence_index": recovery_index,
                    "frame_index": (
                        sequence.frame_indices[recovery_index]
                        if recovery_index is not None
                        else None
                    ),
                    "score": recovery_score,
                    "persistent_obj_id": memory_tracking.selected_obj_id,
                    "paired_original_session": True,
                },
                "local_icp_corrections": transforms,
            },
            handle,
            indent=2,
        )
    print(f"summary: {config.output_dir / 'summary.csv'}")
    print(f"mask report: {config.output_dir / 'mask_sources.png'}")
    print(f"object point clouds: {config.output_dir / 'pointmaps'}")


def _run_branch(
    source: str,
    *,
    masks: torch.Tensor,
    reference: int,
    reference_points: torch.Tensor,
    raw_points: torch.Tensor,
    confidence: torch.Tensor,
    gt_masks: torch.Tensor,
    gt_points: torch.Tensor,
    frame_indices,
    confidence_threshold: float,
    icp_max_points: int,
    icp_iterations: int,
    icp_trim_fraction: float,
    icp_max_correspondence: float,
    icp_min_inliers: int,
    icp_min_fitness: float,
    icp_max_rmse: float,
    translation_only: bool,
) -> tuple[list[dict], torch.Tensor, torch.Tensor, list[dict]]:
    raw_object_maps = torch.full_like(raw_points, float("nan"))
    refined_object_maps = torch.full_like(raw_points, float("nan"))
    rows = []
    corrections = []
    for sequence_index, frame_index in enumerate(frame_indices):
        selected = (
            masks[sequence_index]
            & torch.isfinite(raw_points[sequence_index]).all(dim=-1)
            & (confidence[sequence_index] >= float(confidence_threshold))
        )
        moving = raw_points[sequence_index][selected]
        raw_object_maps[sequence_index][selected] = moving
        if sequence_index == reference:
            icp = _identity_icp(raw_points.dtype, "shared reference observation")
            corrected = moving
        elif moving.shape[0] < int(icp_min_inliers):
            icp = _identity_icp(raw_points.dtype, "too few mask-selected points")
            corrected = moving
        else:
            icp = robust_icp(
                moving,
                reference_points,
                moving_weights=confidence[sequence_index][selected],
                max_points=icp_max_points,
                iterations=icp_iterations,
                trim_fraction=icp_trim_fraction,
                max_correspondence_distance=icp_max_correspondence,
                min_inliers=icp_min_inliers,
                min_fitness=icp_min_fitness,
                max_rmse=icp_max_rmse,
                translation_only=translation_only,
            )
            corrected = (
                apply_rigid(moving, icp.rotation, icp.translation)
                if icp.accepted
                else moving
            )
        refined_object_maps[sequence_index][selected] = corrected

        target = gt_points[sequence_index][gt_masks[sequence_index]]
        raw_chamfer = symmetric_chamfer(moving, target)
        refined_chamfer = symmetric_chamfer(corrected, target)
        row = {
            "mask_source": source,
            "sequence_index": sequence_index,
            "frame_index": frame_index,
            "is_reference": int(sequence_index == reference),
            "gt_visible": int(gt_masks[sequence_index].any()),
            "mask_pixels": int(masks[sequence_index].sum()),
            "selected_object_points": int(selected.sum()),
            "mask_iou": binary_iou(masks[sequence_index], gt_masks[sequence_index]),
            "icp_accepted": int(icp.accepted),
            "icp_reason": icp.reason,
            "icp_inliers": icp.inliers,
            "icp_fitness": icp.fitness,
            "icp_rmse": icp.rmse,
            "icp_rotation_degrees": rotation_angle_degrees(icp.rotation),
            "icp_translation": float(torch.linalg.vector_norm(icp.translation)),
            "raw_object_chamfer": raw_chamfer,
            "refined_object_chamfer": refined_chamfer,
            "object_chamfer_improvement": raw_chamfer - refined_chamfer,
        }
        rows.append(row)
        corrections.append(
            {
                "sequence_index": sequence_index,
                "frame_index": frame_index,
                "accepted": icp.accepted,
                "reason": icp.reason,
                "rotation": icp.rotation.tolist(),
                "translation": icp.translation.tolist(),
            }
        )
        print(
            f"source={source:<20} frame={frame_index} "
            f"visible={row['gt_visible']} mask_iou={row['mask_iou']:.4f} "
            f"points={row['selected_object_points']} icp={icp.accepted} "
            f"chamfer={raw_chamfer:.4f}->{refined_chamfer:.4f}"
        )
    return rows, raw_object_maps, refined_object_maps, corrections


def _summarize_branch(
    source: str,
    rows: list[dict],
    *,
    raw_object_maps: torch.Tensor,
    refined_object_maps: torch.Tensor,
    gt_points: torch.Tensor,
    gt_masks: torch.Tensor,
    reference: int,
) -> dict:
    visible = [
        row for row in rows if row["gt_visible"] and not row["is_reference"]
    ]
    absent = [row for row in rows if not row["gt_visible"]]
    aggregate_gt = gt_points[gt_masks]
    return {
        "mask_source": source,
        "visible_evaluation_frames": len(visible),
        "accepted_icp_frames": sum(row["icp_accepted"] for row in visible),
        "mean_cross_view_mask_iou": _finite_mean(row["mask_iou"] for row in visible),
        "cross_view_mask_recall": _finite_mean(
            float(row["mask_iou"] >= 0.5) for row in visible
        ),
        "absent_selected_points": sum(row["selected_object_points"] for row in absent),
        "mean_raw_object_chamfer": _finite_mean(
            row["raw_object_chamfer"] for row in visible
        ),
        "mean_refined_object_chamfer": _finite_mean(
            row["refined_object_chamfer"] for row in visible
        ),
        "mean_object_chamfer_improvement": _finite_mean(
            row["object_chamfer_improvement"] for row in visible
        ),
        "aggregate_raw_object_chamfer": symmetric_chamfer(
            raw_object_maps.flatten(0, 2), aggregate_gt
        ),
        "aggregate_refined_object_chamfer": symmetric_chamfer(
            refined_object_maps.flatten(0, 2), aggregate_gt
        ),
        "reference_sequence_index": reference,
        "camera_pose_modified": 0,
        "full_scene_pointmap_modified": 0,
    }


def _reference_similarity(
    predicted: torch.Tensor,
    confidence: torch.Tensor,
    target: torch.Tensor,
    *,
    confidence_threshold: float,
    trim_fraction: float,
):
    valid = (
        torch.isfinite(predicted).all(dim=-1)
        & torch.isfinite(target).all(dim=-1)
        & (confidence >= float(confidence_threshold))
    )
    source = predicted[valid]
    fixed = target[valid]
    if source.shape[0] > 30_000:
        indices = torch.linspace(0, source.shape[0] - 1, 30_000).long()
        source = source[indices]
        fixed = fixed[indices]
    return estimate_similarity(source, fixed, trim_fraction=trim_fraction)


def _sam_masks_to_stream(
    masks: torch.Tensor,
    *,
    geometry,
    image_mode: str,
) -> torch.Tensor:
    converted = [
        _output_mask_to_stream(
            mask,
            source_size=geometry.source_sizes[index],
            processed_size=geometry.processed_size,
            image_mode=image_mode,
        )
        for index, mask in enumerate(masks)
    ]
    return torch.stack(converted).bool()


def _identity_icp(dtype: torch.dtype, reason: str) -> ICPResult:
    return ICPResult(
        rotation=torch.eye(3, dtype=dtype),
        translation=torch.zeros(3, dtype=dtype),
        inliers=0,
        fitness=0.0,
        rmse=float("nan"),
        iterations=0,
        accepted=False,
        reason=reason,
    )


def _export_results(
    output_dir: Path,
    *,
    frame_indices,
    raw_points: torch.Tensor,
    gt_points: torch.Tensor,
    gt_masks: torch.Tensor,
    colors: np.ndarray,
    confidence: torch.Tensor,
    confidence_threshold: float,
    branches: dict[str, tuple[torch.Tensor, torch.Tensor]],
) -> None:
    root = output_dir / "pointmaps"
    save_aggregate_ply(
        root / "sequence_streamvggt_raw.ply",
        raw_points,
        colors,
        confidence=confidence,
        confidence_threshold=confidence_threshold,
    )
    save_aggregate_ply(root / "object_gt.ply", gt_points, colors, masks=gt_masks)
    for source, (raw_object_maps, refined_object_maps) in branches.items():
        save_aggregate_ply(
            root / f"object_{source}_raw.ply",
            raw_object_maps,
            colors,
        )
        save_aggregate_ply(
            root / f"object_{source}_refined.ply",
            refined_object_maps,
            colors,
        )
        for sequence_index, frame_index in enumerate(frame_indices):
            prefix = root / f"frame_{sequence_index:02d}_{frame_index}_{source}"
            save_pointmap_ply(
                prefix.with_name(prefix.name + "_raw_object.ply"),
                raw_object_maps[sequence_index],
                colors[sequence_index],
            )
            save_pointmap_ply(
                prefix.with_name(prefix.name + "_refined_object.ply"),
                refined_object_maps[sequence_index],
                colors[sequence_index],
            )


def _save_mask_report(
    path: Path,
    *,
    image_paths,
    frame_indices,
    gt_masks: torch.Tensor,
    original_masks: torch.Tensor,
    memory_masks: torch.Tensor,
    recovery_index: int | None,
    output_size: tuple[int, int],
) -> None:
    height, width = output_size
    header = 28
    labels = ("RGB", "GT", "SAM3 original", "geometry + memory")
    colors = ((0, 0, 0), (0, 210, 70), (255, 180, 0), (45, 105, 255))
    canvas = Image.new(
        "RGB",
        (len(labels) * width, len(image_paths) * (height + header)),
        "white",
    )
    for row, image_path in enumerate(image_paths):
        with Image.open(image_path) as source:
            rgb = source.convert("RGB").resize(
                (width, height), Image.Resampling.BILINEAR
            )
        panels = (
            rgb,
            _overlay(rgb, gt_masks[row], colors[1]),
            _overlay(rgb, original_masks[row], colors[2]),
            _overlay(rgb, memory_masks[row], colors[3]),
        )
        y = row * (height + header)
        for column, (label, panel) in enumerate(zip(labels, panels)):
            canvas.paste(panel, (column * width, y + header))
            suffix = f" frame={frame_indices[row]}" if column == 0 else ""
            if column == 2:
                suffix = f" IoU={binary_iou(original_masks[row], gt_masks[row]):.3f}"
            if column == 3:
                suffix = (
                    f" IoU={binary_iou(memory_masks[row], gt_masks[row]):.3f}"
                    + (" recovery" if row == recovery_index else "")
                )
            ImageDraw.Draw(canvas).text(
                (column * width + 5, y + 6),
                label + suffix,
                fill=colors[column],
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


def _finite_mean(values) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else float("nan")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--scene-id")
    parser.add_argument("--instance-id", type=int)
    parser.add_argument("--frame-indices", type=int, nargs="+")
    parser.add_argument("--sam3-device")
    parser.add_argument("--geometry-device")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--confidence-threshold", type=float, default=0.30)
    parser.add_argument("--alignment-trim-fraction", type=float, default=0.70)
    parser.add_argument("--icp-max-points", type=int, default=2048)
    parser.add_argument("--icp-iterations", type=int, default=30)
    parser.add_argument("--icp-trim-fraction", type=float, default=0.70)
    parser.add_argument("--icp-max-correspondence", type=float, default=0.20)
    parser.add_argument("--icp-min-inliers", type=int, default=64)
    parser.add_argument("--icp-min-fitness", type=float, default=0.10)
    parser.add_argument("--icp-max-rmse", type=float, default=0.15)
    parser.add_argument(
        "--icp-mode",
        choices=("translation_only", "full_se3"),
        default="translation_only",
    )
    parser.add_argument("--reference-sequence-index", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    main()
