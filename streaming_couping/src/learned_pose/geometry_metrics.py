"""Evaluation-only depth and pointmap metrics for learned instance fusion."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch


@torch.no_grad()
def append_geometry_metrics(
    pointmap_frame_rows: list[dict],
    depth_frame_rows: list[dict],
    *,
    batch: dict,
    outputs: dict,
    mode: str,
    perturbation: str,
) -> None:
    """Evaluate direct point-head output and depth+pose reconstruction.

    The reference-frame Sim(3) cached before adapter training is reused for
    every mode and perturbation.  It is never refit to a refined prediction.
    """

    _require_single_clip_batch(batch)
    pose_encoding = outputs["pose_encoding"].detach().float()
    depth = outputs.get("depth", batch["baseline_depth"]).detach().float()
    point_head = outputs.get(
        "world_points",
        batch["baseline_world_points"],
    ).detach().float()
    target_world = batch["target_world_points"].detach().float()
    target_depth = batch["target_depth"].detach().float()
    scale = float(batch["point_alignment_scale"])
    rotation = batch["point_alignment_rotation"].detach().float()
    translation = batch["point_alignment_translation"].detach().float()

    reconstructed = _depth_pose_world_points(
        pose_encoding,
        depth,
        image_size=tuple(int(value) for value in batch["image_size"]),
    )
    prefix = {
        "clip": batch["clip_name"],
        "scene_id": batch["scene_id"],
        "mode": mode,
        "perturbation": perturbation,
        "alignment": "fixed_reference_point_sim3",
    }
    frame_indices = [int(value) for value in batch["frame_indices"]]
    reference_index = int(batch["reference_sequence_index"])

    for geometry_source, predicted in (
        ("point_head", point_head),
        ("depth_pose_backprojection", reconstructed),
    ):
        aligned = scale * (predicted @ rotation.transpose(-1, -2)) + translation
        for sequence_index, frame_index in enumerate(frame_indices):
            current = _paired_point_metrics(
                aligned[0, sequence_index],
                target_world[0, sequence_index],
            )
            pointmap_frame_rows.append(
                {
                    **prefix,
                    "geometry_source": geometry_source,
                    "sequence_index": sequence_index,
                    "frame_index": frame_index,
                    "is_reference": int(sequence_index == reference_index),
                    **current,
                }
            )

    for sequence_index, frame_index in enumerate(frame_indices):
        current = _depth_metrics(
            depth[0, sequence_index],
            target_depth[0, sequence_index],
            fixed_scale=scale,
        )
        depth_frame_rows.append(
            {
                **prefix,
                "sequence_index": sequence_index,
                "frame_index": frame_index,
                "is_reference": int(sequence_index == reference_index),
                **current,
            }
        )


def summarize_pointmap_metrics(rows: Iterable[dict]) -> list[dict]:
    keys = ("clip", "scene_id", "mode", "perturbation", "alignment", "geometry_source")
    return _summarize_frame_rows(
        rows,
        keys=keys,
        count_name="paired_points",
        metric_names=(
            "paired_distance_mean",
            "paired_distance_median",
            "paired_distance_rmse",
            "paired_distance_p90",
        ),
        max_metric="paired_distance_rmse",
    )


def summarize_depth_metrics(rows: Iterable[dict]) -> list[dict]:
    keys = ("clip", "scene_id", "mode", "perturbation", "alignment")
    return _summarize_frame_rows(
        rows,
        keys=keys,
        count_name="valid_pixels",
        metric_names=(
            "fixed_scale_mae_meters",
            "fixed_scale_rmse_meters",
            "fixed_scale_abs_rel",
            "median_scaled_abs_rel",
            "scale_invariant_log_rmse",
            "median_scale_ratio",
        ),
        max_metric="fixed_scale_rmse_meters",
    )


def _paired_point_metrics(predicted: torch.Tensor, target: torch.Tensor) -> dict:
    valid = torch.isfinite(predicted).all(dim=-1) & torch.isfinite(target).all(dim=-1)
    distances = torch.linalg.vector_norm(predicted[valid] - target[valid], dim=-1)
    if distances.numel() == 0:
        return {
            "paired_points": 0,
            "paired_distance_mean": float("nan"),
            "paired_distance_median": float("nan"),
            "paired_distance_rmse": float("nan"),
            "paired_distance_p90": float("nan"),
        }
    values = distances.float()
    return {
        "paired_points": int(values.numel()),
        "paired_distance_mean": float(values.mean().cpu()),
        "paired_distance_median": float(values.median().cpu()),
        "paired_distance_rmse": float(values.square().mean().sqrt().cpu()),
        "paired_distance_p90": float(torch.quantile(values, 0.90).cpu()),
    }


def _depth_metrics(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    fixed_scale: float,
) -> dict:
    predicted = predicted.squeeze(-1)
    target = target.squeeze(-1)
    valid = (
        torch.isfinite(predicted)
        & torch.isfinite(target)
        & (predicted > 1e-6)
        & (target > 1e-6)
    )
    predicted = predicted[valid].float()
    target = target[valid].float()
    if predicted.numel() == 0:
        return {
            "valid_pixels": 0,
            "fixed_scale_mae_meters": float("nan"),
            "fixed_scale_rmse_meters": float("nan"),
            "fixed_scale_abs_rel": float("nan"),
            "median_scale_ratio": float("nan"),
            "median_scaled_abs_rel": float("nan"),
            "scale_invariant_log_rmse": float("nan"),
        }

    fixed = predicted * float(fixed_scale)
    fixed_error = fixed - target
    ratio = target / predicted.clamp_min(1e-6)
    median_scale = ratio.median()
    median_error = predicted * median_scale - target
    log_difference = predicted.log() - target.log()
    log_difference = log_difference - log_difference.mean()
    return {
        "valid_pixels": int(predicted.numel()),
        "fixed_scale_mae_meters": float(fixed_error.abs().mean().cpu()),
        "fixed_scale_rmse_meters": float(fixed_error.square().mean().sqrt().cpu()),
        "fixed_scale_abs_rel": float(
            (fixed_error.abs() / target.clamp_min(1e-6)).mean().cpu()
        ),
        "median_scale_ratio": float(median_scale.cpu()),
        "median_scaled_abs_rel": float(
            (median_error.abs() / target.clamp_min(1e-6)).mean().cpu()
        ),
        "scale_invariant_log_rmse": float(
            log_difference.square().mean().sqrt().cpu()
        ),
    }


def _depth_pose_world_points(
    pose_encoding: torch.Tensor,
    depth: torch.Tensor,
    *,
    image_size: tuple[int, int],
) -> torch.Tensor:
    from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri

    extrinsics, intrinsics = pose_encoding_to_extri_intri(
        pose_encoding.float(),
        image_size_hw=image_size,
    )
    depth = depth.squeeze(-1)
    height, width = depth.shape[-2:]
    rows, columns = torch.meshgrid(
        torch.arange(height, dtype=depth.dtype, device=depth.device),
        torch.arange(width, dtype=depth.dtype, device=depth.device),
        indexing="ij",
    )
    fx = intrinsics[..., 0, 0][..., None, None].clamp_min(1e-6)
    fy = intrinsics[..., 1, 1][..., None, None].clamp_min(1e-6)
    cx = intrinsics[..., 0, 2][..., None, None]
    cy = intrinsics[..., 1, 2][..., None, None]
    camera_points = torch.stack(
        [
            (columns - cx) * depth / fx,
            (rows - cy) * depth / fy,
            depth,
        ],
        dim=-1,
    )
    rotation = extrinsics[..., :3, :3]
    translation = extrinsics[..., :3, 3]
    return torch.einsum(
        "bsji,bshwj->bshwi",
        rotation,
        camera_points - translation[..., None, None, :],
    )


def _summarize_frame_rows(
    rows: Iterable[dict],
    *,
    keys: tuple[str, ...],
    count_name: str,
    metric_names: tuple[str, ...],
    max_metric: str,
) -> list[dict]:
    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[key] for key in keys), []).append(row)
    output = []
    for identity, current_rows in grouped.items():
        reference = [row for row in current_rows if int(row["is_reference"]) == 1]
        nonreference = [row for row in current_rows if int(row["is_reference"]) == 0]
        for group_name, group_rows in (
            ("all_frames", current_rows),
            ("reference_frame", reference),
            ("nonreference_frames", nonreference),
        ):
            valid_rows = [row for row in group_rows if int(row[count_name]) > 0]
            result = {
                **dict(zip(keys, identity)),
                "group": group_name,
                "frames": len(group_rows),
                "valid_frames": len(valid_rows),
                count_name: sum(int(row[count_name]) for row in valid_rows),
            }
            for metric in metric_names:
                values = np.asarray(
                    [float(row[metric]) for row in valid_rows],
                    dtype=np.float64,
                )
                result[f"mean_frame_{metric}"] = (
                    float(values.mean()) if values.size else float("nan")
                )
            maximum = np.asarray(
                [float(row[max_metric]) for row in valid_rows],
                dtype=np.float64,
            )
            result[f"max_frame_{max_metric}"] = (
                float(maximum.max()) if maximum.size else float("nan")
            )
            output.append(result)
    return output


def _require_single_clip_batch(batch: dict) -> None:
    pose = batch["baseline_pose_encoding"]
    if pose.ndim != 3 or pose.shape[0] != 1:
        raise ValueError("Geometry evaluation expects one clip per batch.")

