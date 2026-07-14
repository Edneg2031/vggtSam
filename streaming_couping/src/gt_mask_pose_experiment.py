"""Evaluate whether oracle instance masks improve frozen StreamVGGT geometry."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

from test_sam.data import load_mask_tracking_sequence

from .backbones.streamvggt_wrapper import StreamVGGTWrapper
from .config import load_config
from .geometry.export import save_aggregate_ply, save_pointmap_ply
from .geometry.gt_data import load_gt_geometry_sequence
from .geometry.registration import (
    ICPResult,
    align_world_to_camera,
    apply_rigid,
    apply_similarity,
    apply_world_correction_to_pose,
    estimate_similarity,
    pose_errors,
    robust_icp,
    rotation_angle_degrees,
    symmetric_chamfer,
)


def main() -> None:
    args = _parse_args()
    overrides = {
        key: value
        for key, value in {
            "manifest": args.manifest,
            "scene_id": args.scene_id,
            "instance_id": args.instance_id,
            "frame_indices": args.frame_indices,
            "geometry_device": args.geometry_device,
            "output_dir": args.output_dir,
        }.items()
        if value is not None
    }
    config = load_config(args.config, overrides)
    run_experiment(
        config,
        confidence_threshold=args.confidence_threshold,
        alignment_trim_fraction=args.alignment_trim_fraction,
        icp_max_points=args.icp_max_points,
        icp_iterations=args.icp_iterations,
        icp_trim_fraction=args.icp_trim_fraction,
        icp_max_correspondence=args.icp_max_correspondence,
        icp_min_inliers=args.icp_min_inliers,
        icp_min_fitness=args.icp_min_fitness,
        icp_max_rmse=args.icp_max_rmse,
        reference_sequence_index=args.reference_sequence_index,
    )


def run_experiment(
    config,
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
    reference_mode = "explicit" if reference_sequence_index is not None else "first_visible"
    print(
        f"target scene={sequence.scene_id} frames={sequence.frame_indices} "
        f"instance={sequence.instance_id} label={sequence.label!r} "
        f"reference={reference} reference_mode={reference_mode}"
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
    if tuple(gt.pointmaps.shape) != tuple(geometry.world_points.shape):
        raise RuntimeError(
            "GT and StreamVGGT pointmap shapes disagree: "
            f"{tuple(gt.pointmaps.shape)} != {tuple(geometry.world_points.shape)}"
        )

    correspondence_mask = (
        torch.isfinite(geometry.world_points[reference]).all(dim=-1)
        & torch.isfinite(gt.pointmaps[reference]).all(dim=-1)
        & (geometry.confidence[reference] >= float(confidence_threshold))
    )
    source_alignment = geometry.world_points[reference][correspondence_mask]
    target_alignment = gt.pointmaps[reference][correspondence_mask]
    source_alignment, target_alignment = _paired_subsample(
        source_alignment,
        target_alignment,
        max_points=30_000,
    )
    similarity = estimate_similarity(
        source_alignment,
        target_alignment,
        trim_fraction=alignment_trim_fraction,
    )
    print(
        f"reference Sim3 scale={similarity.scale:.6f} "
        f"inliers={similarity.inliers} rmse={similarity.rmse:.6f}"
    )

    raw_points = apply_similarity(
        geometry.world_points,
        similarity.scale,
        similarity.rotation,
        similarity.translation,
    )
    raw_poses = torch.stack(
        [
            align_world_to_camera(pose, similarity)
            for pose in geometry.world_to_camera
        ]
    )
    refined_points = raw_points.clone()
    refined_poses = raw_poses.clone()

    reference_object_mask = (
        gt.instance_masks[reference]
        & torch.isfinite(raw_points[reference]).all(dim=-1)
        & (geometry.confidence[reference] >= float(confidence_threshold))
    )
    reference_object_points = raw_points[reference][reference_object_mask]
    if reference_object_points.shape[0] < icp_min_inliers:
        raise RuntimeError(
            "The reference GT instance mask contains too few confident StreamVGGT points: "
            f"{reference_object_points.shape[0]}."
        )

    rows = []
    correction_records = []
    for sequence_index, frame_index in enumerate(sequence.frame_indices):
        object_mask = (
            gt.instance_masks[sequence_index]
            & torch.isfinite(raw_points[sequence_index]).all(dim=-1)
            & (geometry.confidence[sequence_index] >= float(confidence_threshold))
        )
        if sequence_index == reference:
            icp = _identity_icp(raw_points.dtype, reason="reference frame")
        elif not gt.instance_masks[sequence_index].any():
            icp = _identity_icp(raw_points.dtype, reason="GT instance absent")
        else:
            icp = robust_icp(
                raw_points[sequence_index][object_mask],
                reference_object_points,
                moving_weights=geometry.confidence[sequence_index][object_mask],
                max_points=icp_max_points,
                iterations=icp_iterations,
                trim_fraction=icp_trim_fraction,
                max_correspondence_distance=icp_max_correspondence,
                min_inliers=icp_min_inliers,
                min_fitness=icp_min_fitness,
                max_rmse=icp_max_rmse,
            )
            if icp.accepted:
                refined_points[sequence_index] = apply_rigid(
                    raw_points[sequence_index],
                    icp.rotation,
                    icp.translation,
                )
                refined_poses[sequence_index] = apply_world_correction_to_pose(
                    raw_poses[sequence_index],
                    icp.rotation,
                    icp.translation,
                )
        estimated_correction = torch.eye(4, dtype=raw_points.dtype)
        estimated_correction[:3, :3] = icp.rotation
        estimated_correction[:3, 3] = icp.translation
        applied_correction = (
            estimated_correction
            if icp.accepted
            else torch.eye(4, dtype=raw_points.dtype)
        )
        correction_records.append(
            {
                "sequence_index": sequence_index,
                "frame_index": frame_index,
                "accepted": icp.accepted,
                "reason": icp.reason,
                "estimated": estimated_correction.tolist(),
                "applied": applied_correction.tolist(),
            }
        )

        raw_rotation, raw_translation = pose_errors(
            raw_poses[sequence_index], gt.world_to_camera[sequence_index]
        )
        refined_rotation, refined_translation = pose_errors(
            refined_poses[sequence_index], gt.world_to_camera[sequence_index]
        )
        raw_full = pointmap_errors(
            raw_points[sequence_index], gt.pointmaps[sequence_index]
        )
        refined_full = pointmap_errors(
            refined_points[sequence_index], gt.pointmaps[sequence_index]
        )
        raw_object = pointmap_errors(
            raw_points[sequence_index],
            gt.pointmaps[sequence_index],
            mask=gt.instance_masks[sequence_index],
        )
        refined_object = pointmap_errors(
            refined_points[sequence_index],
            gt.pointmaps[sequence_index],
            mask=gt.instance_masks[sequence_index],
        )
        target_object = gt.pointmaps[sequence_index][gt.instance_masks[sequence_index]]
        raw_object_cloud = raw_points[sequence_index][object_mask]
        refined_object_cloud = refined_points[sequence_index][object_mask]
        row = {
            "sequence_index": sequence_index,
            "frame_index": frame_index,
            "gt_visible": int(gt.instance_masks[sequence_index].any()),
            "gt_mask_pixels": int(gt.instance_masks[sequence_index].sum()),
            "pred_object_points": int(object_mask.sum()),
            "icp_accepted": int(icp.accepted),
            "icp_reason": icp.reason,
            "icp_iterations": icp.iterations,
            "icp_inliers": icp.inliers,
            "icp_fitness": icp.fitness,
            "icp_rmse": icp.rmse,
            "icp_rotation_degrees": rotation_angle_degrees(icp.rotation),
            "icp_translation": float(torch.linalg.vector_norm(icp.translation)),
            "raw_pose_rotation_degrees": raw_rotation,
            "refined_pose_rotation_degrees": refined_rotation,
            "raw_pose_translation": raw_translation,
            "refined_pose_translation": refined_translation,
            "raw_full_point_rmse": raw_full["rmse"],
            "refined_full_point_rmse": refined_full["rmse"],
            "raw_full_point_mae": raw_full["mae"],
            "refined_full_point_mae": refined_full["mae"],
            "raw_object_point_rmse": raw_object["rmse"],
            "refined_object_point_rmse": refined_object["rmse"],
            "raw_object_point_mae": raw_object["mae"],
            "refined_object_point_mae": refined_object["mae"],
            "raw_object_chamfer": symmetric_chamfer(
                raw_object_cloud, target_object
            ),
            "refined_object_chamfer": symmetric_chamfer(
                refined_object_cloud, target_object
            ),
        }
        rows.append(row)
        print(
            f"frame={frame_index} visible={row['gt_visible']} "
            f"icp={icp.accepted} fitness={icp.fitness:.3f} rmse={icp.rmse:.4f} "
            f"pose_t={raw_translation:.4f}->{refined_translation:.4f} "
            f"object_chamfer={row['raw_object_chamfer']:.4f}->"
            f"{row['refined_object_chamfer']:.4f}"
        )

    _export_pointmaps(
        config.output_dir,
        frame_indices=sequence.frame_indices,
        native_points=geometry.world_points,
        raw_points=raw_points,
        refined_points=refined_points,
        gt_points=gt.pointmaps,
        masks=gt.instance_masks,
        confidence=geometry.confidence,
        colors=gt.colors,
        confidence_threshold=confidence_threshold,
    )
    _write_csv(config.output_dir / "frame_metrics.csv", rows)
    selected = [
        row
        for row in rows
        if row["gt_visible"] and row["sequence_index"] != reference
    ]
    summary = {
        "scene_id": sequence.scene_id,
        "instance_id": sequence.instance_id,
        "label": sequence.label,
        "reference_sequence_index": reference,
        "reference_frame_index": sequence.frame_indices[reference],
        "visible_evaluation_frames": len(selected),
        "accepted_icp_frames": sum(row["icp_accepted"] for row in selected),
        "sim3_scale": similarity.scale,
        "sim3_inliers": similarity.inliers,
        "sim3_rmse": similarity.rmse,
    }
    for key in (
        "raw_pose_rotation_degrees",
        "refined_pose_rotation_degrees",
        "raw_pose_translation",
        "refined_pose_translation",
        "raw_full_point_rmse",
        "refined_full_point_rmse",
        "raw_object_point_rmse",
        "refined_object_point_rmse",
        "raw_object_chamfer",
        "refined_object_chamfer",
    ):
        summary[f"mean_{key}"] = _finite_mean(row[key] for row in selected)
    for metric in (
        "pose_rotation_degrees",
        "pose_translation",
        "full_point_rmse",
        "object_point_rmse",
        "object_chamfer",
    ):
        summary[f"mean_{metric}_improvement"] = (
            summary[f"mean_raw_{metric}"] - summary[f"mean_refined_{metric}"]
        )
    _write_csv(config.output_dir / "summary.csv", [summary])
    with (config.output_dir / "transforms.json").open("w", encoding="utf8") as handle:
        json.dump(
            {
                "settings": {
                    "confidence_threshold": confidence_threshold,
                    "alignment_trim_fraction": alignment_trim_fraction,
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
                "icp_corrections": correction_records,
                "streamvggt_native_world_to_camera": (
                    geometry.world_to_camera.tolist()
                ),
                "raw_world_to_camera": raw_poses.tolist(),
                "refined_world_to_camera": refined_poses.tolist(),
                "gt_world_to_camera": gt.world_to_camera.tolist(),
            },
            handle,
            indent=2,
        )
    print(f"summary: {config.output_dir / 'summary.csv'}")
    print(f"pointmaps: {config.output_dir / 'pointmaps'}")


def pointmap_errors(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
) -> dict[str, float | int]:
    valid = torch.isfinite(prediction).all(dim=-1) & torch.isfinite(target).all(dim=-1)
    if mask is not None:
        valid &= mask.bool()
    if not valid.any():
        return {"rmse": float("nan"), "mae": float("nan"), "points": 0}
    distance = torch.linalg.vector_norm(prediction[valid] - target[valid], dim=-1)
    return {
        "rmse": float(torch.sqrt((distance**2).mean())),
        "mae": float(distance.mean()),
        "points": int(valid.sum()),
    }


def _export_pointmaps(
    output_dir: Path,
    *,
    frame_indices,
    native_points: torch.Tensor,
    raw_points: torch.Tensor,
    refined_points: torch.Tensor,
    gt_points: torch.Tensor,
    masks: torch.Tensor,
    confidence: torch.Tensor,
    colors: np.ndarray,
    confidence_threshold: float,
) -> None:
    root = output_dir / "pointmaps"
    for index, frame_index in enumerate(frame_indices):
        prefix = root / f"frame_{index:02d}_{frame_index}"
        for name, points, point_confidence in (
            ("streamvggt_native", native_points[index], confidence[index]),
            ("raw", raw_points[index], confidence[index]),
            ("refined", refined_points[index], confidence[index]),
            ("gt", gt_points[index], None),
        ):
            save_pointmap_ply(
                prefix.with_name(prefix.name + f"_{name}.ply"),
                points,
                colors[index],
                confidence=point_confidence,
                confidence_threshold=confidence_threshold,
            )
            save_pointmap_ply(
                prefix.with_name(prefix.name + f"_{name}_object.ply"),
                points,
                colors[index],
                mask=masks[index],
                confidence=point_confidence,
                confidence_threshold=confidence_threshold,
            )
    save_aggregate_ply(
        root / "sequence_streamvggt_native.ply",
        native_points,
        colors,
        confidence=confidence,
        confidence_threshold=confidence_threshold,
    )
    save_aggregate_ply(
        root / "sequence_raw.ply",
        raw_points,
        colors,
        confidence=confidence,
        confidence_threshold=confidence_threshold,
    )
    save_aggregate_ply(
        root / "sequence_refined.ply",
        refined_points,
        colors,
        confidence=confidence,
        confidence_threshold=confidence_threshold,
    )
    save_aggregate_ply(root / "sequence_gt.ply", gt_points, colors)
    save_aggregate_ply(
        root / "object_streamvggt_native.ply",
        native_points,
        colors,
        masks=masks,
        confidence=confidence,
        confidence_threshold=confidence_threshold,
    )
    save_aggregate_ply(
        root / "object_raw.ply",
        raw_points,
        colors,
        masks=masks,
        confidence=confidence,
        confidence_threshold=confidence_threshold,
    )
    save_aggregate_ply(
        root / "object_refined.ply",
        refined_points,
        colors,
        masks=masks,
        confidence=confidence,
        confidence_threshold=confidence_threshold,
    )
    save_aggregate_ply(root / "object_gt.ply", gt_points, colors, masks=masks)


def _identity_icp(dtype: torch.dtype, *, reason: str) -> ICPResult:
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


def select_causal_reference(
    sequence,
    requested_index: int | None,
) -> int:
    if requested_index is None:
        visible = [
            index
            for index, mask in enumerate(sequence.target_masks)
            if np.asarray(mask).any()
        ]
        if not visible:
            raise ValueError("The selected instance is absent from the whole sequence.")
        return visible[0]

    index = int(requested_index)
    if index < 0 or index >= len(sequence.frame_indices):
        raise ValueError(
            f"reference_sequence_index={index} is outside "
            f"[0, {len(sequence.frame_indices) - 1}]."
        )
    if not np.asarray(sequence.target_masks[index]).any():
        raise ValueError(
            f"Instance {sequence.instance_id} is absent from reference sequence "
            f"index {index} (frame {sequence.frame_indices[index]})."
        )
    return index


def _paired_subsample(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    max_points: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if source.shape[0] <= int(max_points):
        return source, target
    indices = torch.linspace(0, source.shape[0] - 1, int(max_points)).long()
    return source[indices], target[indices]


def _finite_mean(values) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else float("nan")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Oracle GT-instance-mask pose refinement for frozen StreamVGGT."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--scene-id")
    parser.add_argument("--instance-id", type=int)
    parser.add_argument("--frame-indices", type=int, nargs="+")
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
        "--reference-sequence-index",
        type=int,
        help="Reference position in frame-indices; defaults to the earliest visible frame.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
