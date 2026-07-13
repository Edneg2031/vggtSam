"""Sequence-level explicit geometry bridge evaluation.

The first-frame GT mask initializes SAM3 and the object point map. Later GT
masks are used for metrics only. Geometry produces coarse candidate boxes;
SAM3 remains responsible for dense segmentation.
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
from .bridge.gating import binary_iou, decide_bridge_action
from .config import ExperimentConfig, load_config
from .types import GeometrySequence, RevisitCandidate


def main() -> None:
    args = _parse_args()
    overrides = {
        key: value
        for key, value in {
            "scene_id": args.scene_id,
            "instance_id": args.instance_id,
            "frame_indices": args.frame_indices,
            "sam3_device": args.sam3_device,
            "geometry_device": args.geometry_device,
            "geometry_modes": args.geometry_modes,
            "fallback_prompt_mode": args.fallback_prompt_mode,
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
    original_metrics = summarize_masks(
        tracking.masks,
        target_masks,
        reference_frame_idx=sequence.reference_frame_idx,
    )
    print(
        f"SAM3 original obj_id={tracking.selected_obj_id} "
        f"cross_iou={original_metrics['cross_view_iou']:.4f} "
        f"recall={original_metrics['cross_view_recall']:.4f}"
    )

    geometry = None
    if any(mode != "zero" for mode in config.geometry_modes):
        print("loading and running frozen StreamVGGT with causal caches...")
        geometry = StreamVGGTWrapper(
            repo_path=config.streamvggt_repo,
            checkpoint_path=config.streamvggt_checkpoint,
            device=config.geometry_device,
            image_mode=config.image_mode,
            streaming_cache=config.streaming_cache,
        ).load().extract(sequence.image_paths)

    summaries = []
    for mode in config.geometry_modes:
        result = _evaluate_mode(
            config,
            sequence=sequence,
            target_masks=target_masks,
            original_masks=tracking.masks,
            original_scores=tracking.scores,
            geometry=geometry,
            sam3=sam3,
            mode=mode,
        )
        mode_dir = config.output_dir / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        _write_csv(mode_dir / "frame_metrics.csv", result["rows"])
        _save_report(
            mode_dir / "tracking_report.png",
            image_paths=sequence.image_paths,
            frame_indices=sequence.frame_indices,
            target_masks=target_masks,
            original_masks=tracking.masks,
            candidates=result["candidates"],
            final_masks=result["final_masks"],
            rows=result["rows"],
            output_size=config.output_size,
            mode=mode,
        )
        summary = {
            "mode": mode,
            "fallback_prompt_mode": config.fallback_prompt_mode,
            **{f"sam3_{key}": value for key, value in original_metrics.items()},
            **{f"bridge_{key}": value for key, value in result["metrics"].items()},
            "accepted_candidates": sum(
                int(candidate.accepted) for candidate in result["candidates"]
            ),
            "fallback_frames": sum(int(row["use_fallback"]) for row in result["rows"]),
            "map_updates": sum(int(row["update_map"]) for row in result["rows"]),
        }
        summaries.append(summary)
        print(
            f"mode={mode:<8} bridge_cross_iou={result['metrics']['cross_view_iou']:.4f} "
            f"recall={result['metrics']['cross_view_recall']:.4f} "
            f"candidates={summary['accepted_candidates']} "
            f"fallbacks={summary['fallback_frames']}"
        )

    _write_csv(config.output_dir / "summary.csv", summaries)
    with (config.output_dir / "resolved_config.json").open("w", encoding="utf8") as handle:
        json.dump(_config_json(config), handle, indent=2)
    print(f"summary: {config.output_dir / 'summary.csv'}")


def _evaluate_mode(
    config: ExperimentConfig,
    *,
    sequence,
    target_masks: torch.Tensor,
    original_masks: torch.Tensor,
    original_scores: torch.Tensor,
    geometry: GeometrySequence | None,
    sam3: SAM3Wrapper,
    mode: str,
) -> dict:
    num_frames = len(sequence.frame_indices)
    permutation = _geometry_permutation(
        num_frames,
        reference_frame_idx=sequence.reference_frame_idx,
        mode=mode,
    )
    object_map = ObjectPointMap(max_points_per_object=config.max_points_per_object)
    final_masks = original_masks.clone()
    candidates: list[RevisitCandidate] = []
    rows: list[dict] = []

    if mode != "zero":
        if geometry is None:
            raise RuntimeError(f"Geometry mode {mode!r} requires StreamVGGT output.")
        ref = sequence.reference_frame_idx
        ref_geometry_idx = int(permutation[ref])
        ref_stream_mask = _output_mask_to_stream(
            target_masks[ref],
            source_size=geometry.source_sizes[ref_geometry_idx],
            processed_size=geometry.processed_size,
            image_mode=config.image_mode,
        )
        points, weights = sample_masked_observation(
            geometry.world_points[ref_geometry_idx],
            geometry.confidence[ref_geometry_idx],
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
        geometry_idx = int(permutation[frame_idx])
        original_mask = original_masks[frame_idx]
        original_score = float(original_scores[frame_idx])
        if mode == "zero" or frame_idx == sequence.reference_frame_idx:
            candidate = _empty_candidate(config.output_size, "geometry disabled or reference frame")
        else:
            assert geometry is not None
            entry = object_map.get(sequence.instance_id)
            if entry is None:
                candidate = _empty_candidate(config.output_size, "object map is unavailable")
            else:
                candidate = mine_revisit_candidate(
                    entry.points,
                    current_world_points=geometry.world_points[geometry_idx],
                    world_to_camera=geometry.world_to_camera[geometry_idx],
                    intrinsics=geometry.intrinsics[geometry_idx],
                    source_size=geometry.source_sizes[geometry_idx],
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
        tracker_candidate_iou = binary_iou(original_mask, candidate.mask)
        decision = decide_bridge_action(
            tracker_mask=original_mask,
            tracker_score=original_score,
            candidate=candidate,
            tracker_low_score=config.tracker_low_score,
            fallback_on_missing_mask=config.fallback_on_missing_mask,
            allow_map_update=config.map_update_enabled,
            tracker_candidate_iou=tracker_candidate_iou,
            map_update_min_iou=config.map_update_min_iou,
        )

        fallback_score = 0.0
        fallback_raw_iou = 0.0
        fallback_clipped_iou = 0.0
        if decision.use_fallback:
            refined, fallback_score = sam3.segment_candidate(
                sequence.image_paths[frame_idx],
                prompt=sequence.label,
                output_size=config.output_size,
                candidate_mask=candidate.mask,
                supported_mask=candidate.supported_mask,
                prompt_mode=config.fallback_prompt_mode,
            )
            clipped = refined & candidate.mask
            fallback_raw_iou = binary_iou(refined, target_masks[frame_idx])
            fallback_clipped_iou = binary_iou(clipped, target_masks[frame_idx])
            final_masks[frame_idx] = (
                clipped if config.clip_refined_to_candidate else refined
            )

        updated_map = False
        if decision.update_map and mode != "zero":
            assert geometry is not None
            stream_mask = _output_mask_to_stream(
                final_masks[frame_idx],
                source_size=geometry.source_sizes[geometry_idx],
                processed_size=geometry.processed_size,
                image_mode=config.image_mode,
            )
            points, weights = sample_masked_observation(
                geometry.world_points[geometry_idx],
                geometry.confidence[geometry_idx],
                stream_mask,
                max_points=config.max_points_per_observation,
            )
            updated_map = object_map.update(
                instance_id=sequence.instance_id,
                label=sequence.label,
                points=points,
                weights=weights,
                frame_idx=frame_idx,
            ) is not None

        rows.append(
            {
                "sequence_index": frame_idx,
                "frame_index": sequence.frame_indices[frame_idx],
                "geometry_index": geometry_idx,
                "fallback_prompt_mode": config.fallback_prompt_mode,
                "gt_visible": int(target_masks[frame_idx].any()),
                "sam3_score": original_score,
                "sam3_iou": binary_iou(original_mask, target_masks[frame_idx]),
                "candidate_iou": binary_iou(candidate.mask, target_masks[frame_idx]),
                "final_iou": binary_iou(final_masks[frame_idx], target_masks[frame_idx]),
                "fallback_raw_iou": fallback_raw_iou,
                "fallback_clipped_iou": fallback_clipped_iou,
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
                "fallback_score": fallback_score,
                "use_fallback": int(decision.use_fallback),
                "update_map": int(updated_map),
                "gate_reason": decision.reason,
            }
        )
    return {
        "rows": rows,
        "candidates": candidates,
        "final_masks": final_masks,
        "metrics": summarize_masks(
            final_masks,
            target_masks,
            reference_frame_idx=sequence.reference_frame_idx,
        ),
    }


def summarize_masks(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    reference_frame_idx: int,
) -> dict[str, float]:
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


def _geometry_permutation(
    num_frames: int,
    *,
    reference_frame_idx: int,
    mode: str,
) -> torch.Tensor:
    identity = torch.arange(num_frames)
    if mode in {"zero", "aligned"} or num_frames <= 2:
        return identity
    if mode != "shuffled":
        raise ValueError(f"Unknown geometry mode: {mode}")
    movable = [index for index in range(num_frames) if index != reference_frame_idx]
    rotated = movable[-1:] + movable[:-1]
    output = identity.clone()
    for destination, source in zip(movable, rotated):
        output[destination] = source
    return output


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


def _save_report(
    path: Path,
    *,
    image_paths,
    frame_indices,
    target_masks: torch.Tensor,
    original_masks: torch.Tensor,
    candidates: list[RevisitCandidate],
    final_masks: torch.Tensor,
    rows: list[dict],
    output_size: tuple[int, int],
    mode: str,
) -> None:
    height, width = output_size
    header = 30
    columns = 5
    canvas = Image.new("RGB", (columns * width, len(image_paths) * (height + header)), "white")
    labels = ("RGB", "GT", "SAM3 original", "geometry candidate", "bridge final")
    colors = ((0, 0, 0), (0, 220, 70), (230, 55, 55), (255, 190, 0), (45, 110, 255))
    for row_idx, image_path in enumerate(image_paths):
        with Image.open(image_path) as source:
            rgb = source.convert("RGB").resize((width, height), Image.Resampling.BILINEAR)
        panels = [
            rgb,
            _overlay(rgb, target_masks[row_idx], colors[1]),
            _overlay(rgb, original_masks[row_idx], colors[2]),
            _draw_candidate(rgb, candidates[row_idx]),
            _overlay(rgb, final_masks[row_idx], colors[4]),
        ]
        y = row_idx * (height + header)
        for column, (panel, label) in enumerate(zip(panels, labels)):
            canvas.paste(panel, (column * width, y + header))
            draw = ImageDraw.Draw(canvas)
            suffix = ""
            if column == 0:
                suffix = f" frame={frame_indices[row_idx]}"
            if column == 3:
                suffix = (
                    f" support={rows[row_idx]['support_ratio']:.3f} "
                    f"accepted={rows[row_idx]['candidate_accepted']}"
                )
            if column == 4:
                suffix = f" IoU={rows[row_idx]['final_iou']:.3f}"
            draw.text((column * width + 5, y + 7), label + suffix, fill=colors[column])
    ImageDraw.Draw(canvas).text(
        (5, 2),
        f"mode={mode} prompt={rows[0]['fallback_prompt_mode']}",
        fill=(0, 0, 0),
    )
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
    parser.add_argument("--scene-id")
    parser.add_argument("--instance-id", type=int)
    parser.add_argument("--frame-indices", type=int, nargs="+")
    parser.add_argument("--sam3-device")
    parser.add_argument("--geometry-device")
    parser.add_argument("--geometry-modes", nargs="+", choices=("zero", "aligned", "shuffled"))
    parser.add_argument(
        "--fallback-prompt-mode",
        choices=("box", "point", "box_point"),
    )
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    main()
