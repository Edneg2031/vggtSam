"""Controlled ablation for StreamVGGT-aware SAM3 memory position warping."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw

from test_sam.data import load_mask_tracking_sequence

from .backbones.sam3_wrapper import SAM3Wrapper
from .backbones.streamvggt_wrapper import StreamVGGTWrapper
from .bridge.gating import binary_iou
from .bridge.memory_warp import GeometryMemoryPositionWarper
from .config import ExperimentConfig, load_config
from .pipeline import _resize_target_masks, summarize_masks
from .types import SAM3SoftSequence, TrackingSequence


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
        modes=tuple(args.modes),
        point_source=args.geometry_point_source,
        min_geometry_confidence=args.min_geometry_confidence,
        strict_identity=not args.allow_identity_mismatch,
    )


def run_experiment(
    config: ExperimentConfig,
    *,
    modes: tuple[str, ...],
    point_source: str,
    min_geometry_confidence: float,
    strict_identity: bool,
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
    target_masks = _resize_target_masks(sequence.target_masks, config.output_size)
    visible = [bool(mask.any()) for mask in target_masks]
    print(
        f"target scene={sequence.scene_id} frames={sequence.frame_indices} "
        f"instance={sequence.instance_id} label={sequence.label!r} "
        f"reference={sequence.reference_frame_idx} visible={visible}"
    )

    print("running frozen StreamVGGT once with causal caches...")
    geometry = StreamVGGTWrapper(
        repo_path=config.streamvggt_repo,
        checkpoint_path=config.streamvggt_checkpoint,
        device=config.geometry_device,
        image_mode=config.image_mode,
        streaming_cache=config.streaming_cache,
    ).load().extract(sequence.image_paths)
    print(
        f"geometry point_source={point_source} processed_size={geometry.processed_size}"
    )

    print("loading frozen SAM3 and running original memory baseline...")
    sam3 = SAM3Wrapper(
        repo_path=config.sam3_repo,
        checkpoint_path=config.sam3_checkpoint,
        device=config.sam3_device,
        output_threshold=config.sam3_output_threshold,
        prompt_with_box=config.prompt_with_box,
    ).load()
    original, original_soft = _track_original(
        sam3, sequence, target_masks, config.output_size
    )
    results: dict[str, TrackingSequence] = {"original": original}
    soft_results: dict[str, SAM3SoftSequence] = {"original": original_soft}
    warpers: dict[str, GeometryMemoryPositionWarper] = {}
    permutation = _cyclic_permutation(len(sequence.frame_indices))

    for mode in modes:
        torch.manual_seed(0)
        warper = GeometryMemoryPositionWarper(
            geometry,
            mode=mode,
            image_mode=config.image_mode,
            frame_permutation=permutation,
            min_geometry_confidence=min_geometry_confidence,
            point_source=point_source,
        )
        print(
            f"running SAM3 memory mode={mode} "
            f"geometry_permutation={list(permutation) if mode == 'shuffled' else 'aligned'}"
        )
        tracking, soft = sam3.track_with_soft_diagnostics(
            sequence.image_paths,
            prompt=sequence.label,
            output_size=config.output_size,
            reference_frame_idx=sequence.reference_frame_idx,
            reference_mask=target_masks[sequence.reference_frame_idx],
            warper=warper,
        )
        results[mode] = tracking
        soft_results[mode] = soft
        warpers[mode] = warper

    identity_matches = True
    identity_soft_matches = True
    if "identity" in results:
        identity_matches = bool(
            torch.equal(original.masks, results["identity"].masks)
            and original.selected_obj_id == results["identity"].selected_obj_id
        )
        identity_soft_matches = bool(
            torch.equal(
                original_soft.probabilities,
                soft_results["identity"].probabilities,
            )
            and torch.allclose(
                original_soft.presence_logits,
                soft_results["identity"].presence_logits,
                equal_nan=True,
            )
        )
        print(
            f"identity_matches_original={identity_matches} "
            f"identity_soft_matches_original={identity_soft_matches}"
        )

    projection_masks, projection_point_masks, projection_rows = (
        _geometry_projection_diagnostics(
            warpers,
            reference_mask=target_masks[sequence.reference_frame_idx],
            target_masks=target_masks,
            reference_frame_idx=sequence.reference_frame_idx,
            frame_indices=sequence.frame_indices,
            output_size=config.output_size,
        )
    )
    projection_summary = _projection_summary(
        projection_rows,
        reference_frame_idx=sequence.reference_frame_idx,
    )

    summary_rows = _summary_rows(
        results,
        soft_results=soft_results,
        warpers=warpers,
        projection_summary=projection_summary,
        target_masks=target_masks,
        reference_frame_idx=sequence.reference_frame_idx,
        identity_matches=identity_matches,
        identity_soft_matches=identity_soft_matches,
    )
    frame_rows = _frame_rows(
        results,
        soft_results=soft_results,
        warpers=warpers,
        target_masks=target_masks,
        frame_indices=sequence.frame_indices,
    )
    _write_csv(config.output_dir / "summary.csv", summary_rows)
    _write_csv(config.output_dir / "frame_metrics.csv", frame_rows)
    _write_csv(
        config.output_dir / "geometry_projection_metrics.csv", projection_rows
    )
    pair_rows = []
    for mode, warper in warpers.items():
        pair_rows.extend({"mode": mode, **row} for row in warper.observation_rows())
    _write_csv(config.output_dir / "memory_warp_pairs.csv", pair_rows)
    _save_visualization(
        config.output_dir / "memory_warp_report.png",
        image_paths=sequence.image_paths,
        frame_indices=sequence.frame_indices,
        target_masks=target_masks,
        results=results,
        output_size=config.output_size,
    )
    _save_soft_visualization(
        config.output_dir / "soft_response_report.png",
        image_paths=sequence.image_paths,
        frame_indices=sequence.frame_indices,
        target_masks=target_masks,
        soft_results=soft_results,
        output_size=config.output_size,
    )
    _save_projection_visualization(
        config.output_dir / "geometry_projection_report.png",
        image_paths=sequence.image_paths,
        frame_indices=sequence.frame_indices,
        target_masks=target_masks,
        projection_masks=projection_masks,
        projection_point_masks=projection_point_masks,
        output_size=config.output_size,
    )
    metadata = {
        "experiment": "sam3_memory_position_warp",
        "scene_id": sequence.scene_id,
        "frame_indices": sequence.frame_indices,
        "instance_id": sequence.instance_id,
        "label": sequence.label,
        "reference_sequence_index": sequence.reference_frame_idx,
        "modes": ["original", *modes],
        "point_source": point_source,
        "min_geometry_confidence": float(min_geometry_confidence),
        "shuffled_geometry_permutation": list(permutation),
        "identity_matches_original": identity_matches,
        "identity_soft_matches_original": identity_soft_matches,
        "scientific_controls": {
            "reference_gt_only": True,
            "later_gt_metrics_only": True,
            "geometry_fallback_disabled": True,
            "memory_writeback_disabled": True,
            "redetection_disabled": True,
            "persistent_obj_id_owned_by_sam3": True,
            "only_memory_spatial_position_encoding_changes": True,
            "soft_outputs_captured_before_presence_gate": True,
            "reference_object_projection_is_metrics_only": True,
        },
    }
    with (config.output_dir / "resolved_experiment.json").open(
        "w", encoding="utf8"
    ) as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)

    for row in summary_rows:
        print(
            f"mode={row['mode']:<9} cross_iou={row['cross_view_iou']:.4f} "
            f"recall={row['cross_view_recall']:.4f} "
            f"absent_fp={row['absent_fp_ratio']:.6f} "
            f"soft_margin={row['cross_soft_margin']:.4f} "
            f"projection_iou={row['cross_projection_iou']:.4f} "
            f"valid_warp={row['valid_warp_ratio']:.4f}"
        )
    print(f"summary: {config.output_dir / 'summary.csv'}")
    print(f"visualization: {config.output_dir / 'memory_warp_report.png'}")
    if strict_identity and not (identity_matches and identity_soft_matches):
        raise RuntimeError(
            "Identity memory hook changed SAM3 output. The aligned/shuffled comparison "
            "is not a valid single-variable ablation; inspect frame_metrics.csv."
        )


def _track_original(sam3, sequence, target_masks, output_size):
    torch.manual_seed(0)
    return sam3.track_with_soft_diagnostics(
        sequence.image_paths,
        prompt=sequence.label,
        output_size=output_size,
        reference_frame_idx=sequence.reference_frame_idx,
        reference_mask=target_masks[sequence.reference_frame_idx],
    )


def _summary_rows(
    results: dict[str, TrackingSequence],
    *,
    soft_results: dict[str, SAM3SoftSequence],
    warpers: dict[str, GeometryMemoryPositionWarper],
    projection_summary: dict[str, dict[str, float]],
    target_masks: torch.Tensor,
    reference_frame_idx: int,
    identity_matches: bool,
    identity_soft_matches: bool,
) -> list[dict[str, Any]]:
    rows = []
    original_id = results["original"].selected_obj_id
    for mode, tracking in results.items():
        metrics = summarize_masks(
            tracking.masks,
            target_masks,
            reference_frame_idx=reference_frame_idx,
        )
        warp = warpers[mode].summary() if mode in warpers else {}
        soft = _soft_summary(
            soft_results[mode],
            target_masks,
            reference_frame_idx=reference_frame_idx,
        )
        projection = projection_summary.get(mode, {})
        rows.append(
            {
                "mode": mode,
                **metrics,
                "selected_obj_id": tracking.selected_obj_id,
                "same_obj_id_as_original": int(
                    tracking.selected_obj_id == original_id
                ),
                "identity_matches_original": int(identity_matches),
                "identity_soft_matches_original": int(identity_soft_matches),
                **soft,
                "cross_projection_iou": float(
                    projection.get("cross_projection_iou", 0.0)
                ),
                "cross_projection_point_hit_ratio": float(
                    projection.get("cross_projection_point_hit_ratio", 0.0)
                ),
                "cross_projection_target_coverage": float(
                    projection.get("cross_projection_target_coverage", 0.0)
                ),
                "hook_calls": int(warp.get("hook_calls", 0)),
                "memory_pairs": int(warp.get("memory_pairs", 0)),
                "valid_warp_ratio": float(warp.get("valid_warp_ratio", 0.0)),
                "mean_warp_displacement_pixels": float(
                    warp.get("mean_warp_displacement_pixels", 0.0)
                ),
            }
        )
    return rows


def _frame_rows(
    results: dict[str, TrackingSequence],
    *,
    soft_results: dict[str, SAM3SoftSequence],
    warpers: dict[str, GeometryMemoryPositionWarper],
    target_masks: torch.Tensor,
    frame_indices: Sequence[int],
) -> list[dict[str, Any]]:
    per_frame_warp: dict[tuple[str, int], list] = {}
    for mode, warper in warpers.items():
        for observation in warper.observations:
            per_frame_warp.setdefault((mode, observation.current_frame), []).append(
                observation
            )
    rows = []
    for mode, tracking in results.items():
        for sequence_index, (prediction, target, score) in enumerate(
            zip(tracking.masks, target_masks, tracking.scores)
        ):
            observations = per_frame_warp.get((mode, sequence_index), [])
            soft = _soft_metrics(
                soft_results[mode].probabilities[sequence_index],
                target,
                presence_logit=float(
                    soft_results[mode].presence_logits[sequence_index]
                ),
            )
            total = sum(item.total_tokens for item in observations)
            valid = sum(item.valid_tokens for item in observations)
            rows.append(
                {
                    "mode": mode,
                    "sequence_index": sequence_index,
                    "frame_index": int(frame_indices[sequence_index]),
                    "gt_visible": int(target.any()),
                    "score": float(score),
                    "iou": binary_iou(prediction, target),
                    "prediction_pixels": int(prediction.sum()),
                    "target_pixels": int(target.sum()),
                    "soft_capture_count": int(
                        soft_results[mode].captures_per_frame[sequence_index]
                    ),
                    **soft,
                    "memory_pairs": len(observations),
                    "valid_warp_ratio": float(valid / max(total, 1)),
                    "mean_warp_displacement_pixels": float(
                        sum(
                            item.mean_displacement_pixels * item.valid_tokens
                            for item in observations
                        )
                        / max(valid, 1)
                    ),
                }
            )
    return rows


def _soft_metrics(
    probability: torch.Tensor,
    target: torch.Tensor,
    *,
    presence_logit: float,
) -> dict[str, float | int]:
    probability = probability.detach().float().cpu()
    target = target.detach().bool().cpu()
    finite_presence = np.isfinite(presence_logit)
    presence_probability = (
        float(torch.sigmoid(torch.tensor(presence_logit)))
        if finite_presence
        else float("nan")
    )
    result: dict[str, float | int] = {
        "presence_logit": float(presence_logit),
        "presence_probability": presence_probability,
        "presence_gate_pass": int(finite_presence and presence_logit > 0.0),
        "soft_max_probability": float(probability.max()),
        "soft_background_mean": float(probability[~target].mean()),
    }
    if not target.any():
        result.update(
            {
                "soft_gt_mean": float("nan"),
                "soft_margin": float("nan"),
                "soft_iou": float("nan"),
                "soft_mass_inside_gt": float("nan"),
                "soft_peak_inside_gt": 0,
            }
        )
        return result
    gt_mean = float(probability[target].mean())
    background_mean = float(probability[~target].mean())
    intersection = float(probability[target].sum())
    union = float(probability.sum() + target.sum() - intersection)
    peak_index = int(probability.reshape(-1).argmax())
    result.update(
        {
            "soft_gt_mean": gt_mean,
            "soft_margin": gt_mean - background_mean,
            "soft_iou": intersection / max(union, 1e-8),
            "soft_mass_inside_gt": intersection / max(float(probability.sum()), 1e-8),
            "soft_peak_inside_gt": int(target.reshape(-1)[peak_index]),
        }
    )
    return result


def _soft_summary(
    soft: SAM3SoftSequence,
    target_masks: torch.Tensor,
    *,
    reference_frame_idx: int,
) -> dict[str, float]:
    cross_rows = []
    absent_rows = []
    for frame_idx, target in enumerate(target_masks):
        metrics = _soft_metrics(
            soft.probabilities[frame_idx],
            target,
            presence_logit=float(soft.presence_logits[frame_idx]),
        )
        if target.any() and frame_idx != int(reference_frame_idx):
            cross_rows.append(metrics)
        elif not target.any():
            absent_rows.append(metrics)
    return {
        "cross_presence_probability": _mean_metric(
            cross_rows, "presence_probability"
        ),
        "cross_soft_gt_mean": _mean_metric(cross_rows, "soft_gt_mean"),
        "cross_soft_background_mean": _mean_metric(
            cross_rows, "soft_background_mean"
        ),
        "cross_soft_margin": _mean_metric(cross_rows, "soft_margin"),
        "cross_soft_iou": _mean_metric(cross_rows, "soft_iou"),
        "cross_soft_mass_inside_gt": _mean_metric(
            cross_rows, "soft_mass_inside_gt"
        ),
        "cross_soft_peak_inside_gt_rate": _mean_metric(
            cross_rows, "soft_peak_inside_gt"
        ),
        "absent_presence_probability": _mean_metric(
            absent_rows, "presence_probability"
        ),
        "absent_soft_max_probability": _mean_metric(
            absent_rows, "soft_max_probability"
        ),
    }


def _geometry_projection_diagnostics(
    warpers: dict[str, GeometryMemoryPositionWarper],
    *,
    reference_mask: torch.Tensor,
    target_masks: torch.Tensor,
    reference_frame_idx: int,
    frame_indices: Sequence[int],
    output_size: tuple[int, int],
):
    projection_masks: dict[str, torch.Tensor] = {}
    point_masks: dict[str, torch.Tensor] = {}
    rows: list[dict[str, Any]] = []
    for mode in ("aligned", "shuffled"):
        if mode not in warpers:
            continue
        dense_per_frame = []
        points_per_frame = []
        for sequence_index, target in enumerate(target_masks):
            dense, points, stats = warpers[mode].project_reference_mask(
                reference_mask,
                memory_frame=reference_frame_idx,
                current_frame=sequence_index,
                output_size=output_size,
            )
            dense_per_frame.append(dense)
            points_per_frame.append(points)
            intersection = int((dense & target).sum())
            point_hits = int((points & target).sum())
            rows.append(
                {
                    "mode": mode,
                    "sequence_index": sequence_index,
                    "frame_index": int(frame_indices[sequence_index]),
                    "gt_visible": int(target.any()),
                    **stats,
                    "projection_iou": binary_iou(dense, target),
                    "projection_target_coverage": float(
                        intersection / max(int(target.sum()), 1)
                    ),
                    "projection_point_hit_ratio": float(
                        point_hits / max(int(points.sum()), 1)
                    ),
                    "projection_centroid_error": _centroid_error(points, target),
                }
            )
        projection_masks[mode] = torch.stack(dense_per_frame)
        point_masks[mode] = torch.stack(points_per_frame)
    return projection_masks, point_masks, rows


def _projection_summary(
    rows: list[dict[str, Any]],
    *,
    reference_frame_idx: int,
) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for mode in {str(row["mode"]) for row in rows}:
        selected = [
            row
            for row in rows
            if row["mode"] == mode
            and row["gt_visible"]
            and row["sequence_index"] != int(reference_frame_idx)
        ]
        output[mode] = {
            "cross_projection_iou": _mean_metric(selected, "projection_iou"),
            "cross_projection_point_hit_ratio": _mean_metric(
                selected, "projection_point_hit_ratio"
            ),
            "cross_projection_target_coverage": _mean_metric(
                selected, "projection_target_coverage"
            ),
        }
    return output


def _centroid_error(left: torch.Tensor, right: torch.Tensor) -> float:
    if not left.any() or not right.any():
        return float("nan")
    left_yx = left.nonzero(as_tuple=False).float().mean(dim=0)
    right_yx = right.nonzero(as_tuple=False).float().mean(dim=0)
    diagonal = float((left.shape[0] ** 2 + left.shape[1] ** 2) ** 0.5)
    return float(torch.linalg.vector_norm(left_yx - right_yx) / max(diagonal, 1.0))


def _mean_metric(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if np.isfinite(float(row[key]))]
    return float(np.mean(values)) if values else 0.0


def _save_visualization(
    path: Path,
    *,
    image_paths: Sequence[Path],
    frame_indices: Sequence[int],
    target_masks: torch.Tensor,
    results: dict[str, TrackingSequence],
    output_size: tuple[int, int],
) -> None:
    height, width = output_size
    labels = ["RGB", "GT", *results.keys()]
    header_height = 24
    canvas = Image.new(
        "RGB",
        (width * len(labels), (height + header_height) * len(image_paths)),
        "white",
    )
    colors = {
        "GT": (255, 64, 64),
        "original": (64, 180, 255),
        "identity": (255, 190, 64),
        "aligned": (64, 220, 120),
        "shuffled": (190, 80, 255),
    }
    for row, image_path in enumerate(image_paths):
        with Image.open(image_path) as source:
            rgb = source.convert("RGB").resize(
                (width, height), Image.Resampling.BILINEAR
            )
        masks = {"GT": target_masks[row]}
        masks.update({mode: tracking.masks[row] for mode, tracking in results.items()})
        cells = {"RGB": rgb}
        for mode, mask in masks.items():
            cells[mode] = _overlay_mask(rgb, mask, colors[mode])
        for column, label in enumerate(labels):
            x = column * width
            y = row * (height + header_height)
            canvas.paste(cells[label], (x, y + header_height))
            draw = ImageDraw.Draw(canvas)
            title = f"{label} | frame={frame_indices[row]}"
            if label in results:
                title += f" | IoU={binary_iou(masks[label], target_masks[row]):.3f}"
            draw.text((x + 5, y + 5), title, fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _save_soft_visualization(
    path: Path,
    *,
    image_paths: Sequence[Path],
    frame_indices: Sequence[int],
    target_masks: torch.Tensor,
    soft_results: dict[str, SAM3SoftSequence],
    output_size: tuple[int, int],
) -> None:
    height, width = output_size
    labels = ["RGB", "GT", *soft_results.keys()]
    header_height = 38
    canvas = Image.new(
        "RGB",
        (width * len(labels), (height + header_height) * len(image_paths)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for row, image_path in enumerate(image_paths):
        with Image.open(image_path) as source:
            rgb = source.convert("RGB").resize(
                (width, height), Image.Resampling.BILINEAR
            )
        cells = {"RGB": rgb, "GT": _overlay_mask(rgb, target_masks[row], (255, 64, 64))}
        for mode, soft in soft_results.items():
            cells[mode] = _overlay_probability(rgb, soft.probabilities[row])
        for column, label in enumerate(labels):
            x = column * width
            y = row * (height + header_height)
            canvas.paste(cells[label], (x, y + header_height))
            title = f"{label} | frame={frame_indices[row]}"
            if label in soft_results:
                soft = soft_results[label]
                metrics = _soft_metrics(
                    soft.probabilities[row],
                    target_masks[row],
                    presence_logit=float(soft.presence_logits[row]),
                )
                title += (
                    f"\npres={metrics['presence_logit']:.3f} "
                    f"gt-bg={metrics['soft_margin']:.3f}"
                )
            draw.text((x + 5, y + 3), title, fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _save_projection_visualization(
    path: Path,
    *,
    image_paths: Sequence[Path],
    frame_indices: Sequence[int],
    target_masks: torch.Tensor,
    projection_masks: dict[str, torch.Tensor],
    projection_point_masks: dict[str, torch.Tensor],
    output_size: tuple[int, int],
) -> None:
    if not projection_masks:
        return
    height, width = output_size
    modes = list(projection_masks.keys())
    labels = ["RGB", "GT", *modes]
    header_height = 30
    canvas = Image.new(
        "RGB",
        (width * len(labels), (height + header_height) * len(image_paths)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for row, image_path in enumerate(image_paths):
        with Image.open(image_path) as source:
            rgb = source.convert("RGB").resize(
                (width, height), Image.Resampling.BILINEAR
            )
        cells = {"RGB": rgb, "GT": _overlay_mask(rgb, target_masks[row], (255, 64, 64))}
        for mode in modes:
            cells[mode] = _overlay_projection(
                rgb,
                projection_masks[mode][row],
                projection_point_masks[mode][row],
            )
        for column, label in enumerate(labels):
            x = column * width
            y = row * (height + header_height)
            canvas.paste(cells[label], (x, y + header_height))
            title = f"{label} | frame={frame_indices[row]}"
            if label in projection_masks:
                title += (
                    f" | IoU={binary_iou(projection_masks[label][row], target_masks[row]):.3f}"
                )
            draw.text((x + 5, y + 6), title, fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _overlay_mask(image: Image.Image, mask: torch.Tensor, color) -> Image.Image:
    mask_image = Image.fromarray((mask.cpu().numpy().astype(np.uint8) * 150), mode="L")
    overlay = Image.new("RGB", image.size, color)
    return Image.composite(overlay, image, mask_image)


def _overlay_probability(image: Image.Image, probability: torch.Tensor) -> Image.Image:
    values = probability.detach().float().clamp(0.0, 1.0).cpu().numpy()
    heat = np.zeros((*values.shape, 3), dtype=np.uint8)
    heat[..., 0] = (values * 255).astype(np.uint8)
    heat[..., 1] = (np.sqrt(values) * 180).astype(np.uint8)
    heat_image = Image.fromarray(heat, mode="RGB")
    alpha = Image.fromarray((values * 190).astype(np.uint8), mode="L")
    return Image.composite(heat_image, image, alpha)


def _overlay_projection(
    image: Image.Image,
    projected_mask: torch.Tensor,
    point_mask: torch.Tensor,
) -> Image.Image:
    output = _overlay_mask(image, projected_mask, (64, 220, 120))
    array = np.asarray(output).copy()
    points = point_mask.cpu().numpy().astype(bool)
    array[points] = np.array([255, 220, 32], dtype=np.uint8)
    return Image.fromarray(array, mode="RGB")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _cyclic_permutation(num_frames: int) -> tuple[int, ...]:
    if num_frames < 2:
        raise ValueError("Shuffled geometry control requires at least two frames.")
    return tuple([*range(1, num_frames), 0])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ablate geometry-aware SAM3 memory position encoding."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--scene-id")
    parser.add_argument("--instance-id", type=int)
    parser.add_argument("--frame-indices", type=int, nargs="+")
    parser.add_argument("--sam3-device")
    parser.add_argument("--geometry-device")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("identity", "aligned", "shuffled"),
        default=("identity", "aligned", "shuffled"),
    )
    parser.add_argument(
        "--geometry-point-source",
        choices=("depth_camera", "point_head"),
        default="depth_camera",
        help="Use camera-consistent depth unprojection by default.",
    )
    parser.add_argument("--min-geometry-confidence", type=float, default=0.20)
    parser.add_argument("--allow-identity-mismatch", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
