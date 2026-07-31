"""Evaluate StreamVGGT geometry prompts before SAM3.1 mask prediction."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
from pathlib import Path

import torch
import yaml
from PIL import Image, ImageDraw

from streaming_couping.src.backbones.sam3_wrapper import SAM3Wrapper
from streaming_couping.src.config import load_config
from streaming_couping.src.external_repos import maybe_add_repo_to_path
from streaming_couping.src.instance_observations import (
    InstanceRefinementConfig,
    load_instance_sequences,
)
from streaming_couping.src.learned_pose.cache import cache_path, load_feature_cache
from streaming_couping.src.learned_pose.config import (
    load_learned_pose_config,
)
from streaming_couping.src.types import TrackingSequence
from streaming_couping.src.v6_geometry_segmentation import (
    V6_SEGMENTATION_VARIANTS,
    V6GeometrySegmentationConfig,
    segment_instance_with_geometry_prompts,
)


@dataclass(frozen=True)
class SAMRuntimeConfig:
    version: str
    checkpoint: Path
    use_fa3: bool
    max_num_objects: int
    multiplex_count: int


@dataclass(frozen=True)
class V6SegmentationExperiment:
    base_config: Path
    output_dir: Path
    sam: SAMRuntimeConfig
    segmentation: V6GeometrySegmentationConfig
    refinement: InstanceRefinementConfig


def main() -> None:
    args = _parse_args()
    experiment = _load_experiment(args.config)
    base = load_learned_pose_config(experiment.base_config)
    recovery_base = load_config(base.recovery_config)
    maybe_add_repo_to_path(recovery_base.streamvggt_repo)
    from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri

    output_dir = args.output_dir or experiment.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        "V6 segmentation backbone="
        f"{experiment.sam.version} checkpoint={experiment.sam.checkpoint}"
    )
    sam3 = SAM3Wrapper(
        repo_path=recovery_base.sam3_repo,
        checkpoint_path=experiment.sam.checkpoint,
        device=args.sam3_device or base.sam3_device,
        output_threshold=recovery_base.sam3_output_threshold,
        prompt_with_box=recovery_base.prompt_with_box,
        version=experiment.sam.version,
        use_fa3=experiment.sam.use_fa3,
        max_num_objects=experiment.sam.max_num_objects,
        multiplex_count=experiment.sam.multiplex_count,
    ).load()
    metric_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []

    for clip in base.clips:
        print(f"V6 geometry-prompt segmentation clip={clip.name}")
        payload = load_feature_cache(cache_path(base, clip))
        recovery = load_config(
            base.recovery_config,
            {
                "manifest": base.manifest,
                "scene_id": clip.scene_id,
                "frame_indices": clip.frame_indices,
                "sam3_device": args.sam3_device or base.sam3_device,
                "geometry_device": args.geometry_device or base.geometry_device,
                "output_dir": output_dir / "runtime" / clip.name,
            },
        )
        sequences, target_by_id = load_instance_sequences(
            recovery,
            instance_ids=clip.instance_ids,
            reference_sequence_index=clip.reference_sequence_index,
            allow_missing_reference=clip.allow_missing_reference_instances,
        )
        image_paths = [Path(value) for value in payload["image_paths"]]
        source_sizes = tuple(_image_size_hw(path) for path in image_paths)
        image_size = tuple(int(value) for value in payload["image_size"])
        baseline_w2c, intrinsics = pose_encoding_to_extri_intri(
            payload["baseline_pose_encoding"][None].float(),
            image_size_hw=image_size,
        )
        all_masks = {
            variant: torch.zeros(
                len(clip.frame_indices),
                len(clip.instance_ids),
                *recovery.output_size,
                dtype=torch.bool,
            )
            for variant in V6_SEGMENTATION_VARIANTS
        }
        target_masks = torch.stack(
            [target_by_id[int(value)] for value in clip.instance_ids],
            dim=1,
        )

        for slot, instance_id in enumerate(clip.instance_ids):
            instance_id = int(instance_id)
            sequence = sequences[instance_id]
            reference_mask = target_by_id[instance_id][
                clip.reference_sequence_index
            ]
            raw = _raw_tracking(
                sam3,
                sequence=sequence,
                reference_mask=reference_mask,
                output_size=recovery.output_size,
            )
            current = TrackingSequence(
                masks=payload["tracking_masks_output"][:, slot].bool(),
                scores=payload["tracking_scores"][:, slot].float(),
                selected_obj_id=None,
            )
            result = segment_instance_with_geometry_prompts(
                recovery=recovery,
                sequence=sequence,
                reference_mask=reference_mask,
                raw_tracking=raw,
                current_late_tracking=current,
                world_points=payload["baseline_world_points"].float(),
                confidence=payload["baseline_world_confidence"].float(),
                world_to_camera=baseline_w2c[0].detach().float().cpu(),
                intrinsics=intrinsics[0].detach().float().cpu(),
                source_sizes=source_sizes,
                processed_size=image_size,
                sam3=sam3,
                refinement=replace(
                    experiment.refinement,
                    compute_device=(
                        args.geometry_device or base.geometry_device
                    ),
                ),
                config=experiment.segmentation,
            )
            for variant, masks in result["masks"].items():
                all_masks[variant][:, slot] = masks
                metric_rows.extend(
                    _mask_metric_rows(
                        clip_name=clip.name,
                        split=clip.split,
                        variant=variant,
                        instance_id=instance_id,
                        frame_indices=clip.frame_indices,
                        reference_index=clip.reference_sequence_index,
                        predicted=masks,
                        target=target_by_id[instance_id],
                    )
                )
            for row in result["diagnostics"]:
                diagnostic_rows.append(
                    {
                        "clip": clip.name,
                        "split": clip.split,
                        "instance_id": instance_id,
                        **row,
                    }
                )

        _write_visualizations(
            output_dir / "visualizations" / clip.name,
            image_paths=image_paths,
            frame_indices=clip.frame_indices,
            instance_ids=clip.instance_ids,
            target_masks=target_masks,
            masks_by_variant=all_masks,
        )
        mask_dir = output_dir / "masks"
        mask_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "clip": clip.name,
                "frame_indices": list(clip.frame_indices),
                "instance_ids": list(clip.instance_ids),
                "variants": all_masks,
                "gt_evaluation_only": target_masks,
            },
            mask_dir / f"{clip.name}.pt",
        )

    summary_rows = _summarize_metrics(metric_rows)
    summary_path = output_dir / "v6_segmentation_summary.csv"
    _write_csv(summary_path, summary_rows)
    _write_csv(output_dir / "v6_segmentation_frames.csv", metric_rows)
    _write_csv(
        output_dir / "v6_geometry_prompt_diagnostics.csv",
        diagnostic_rows,
    )
    print(summary_path)
    with summary_path.open("r", encoding="utf8") as handle:
        print(handle.read().rstrip())
    print("GT masks are used only for the CSV metrics and visualizations.")


def _raw_tracking(
    sam3: SAM3Wrapper,
    *,
    sequence,
    reference_mask: torch.Tensor,
    output_size: tuple[int, int],
) -> TrackingSequence:
    if not bool(reference_mask.any()):
        frames = len(sequence.frame_indices)
        return TrackingSequence(
            masks=torch.zeros(frames, *output_size, dtype=torch.bool),
            scores=torch.zeros(frames),
            selected_obj_id=None,
        )
    return sam3.track(
        sequence.image_paths,
        prompt=sequence.label,
        output_size=output_size,
        reference_frame_idx=sequence.reference_frame_idx,
        reference_mask=reference_mask,
    )


def _mask_metric_rows(
    *,
    clip_name: str,
    split: str,
    variant: str,
    instance_id: int,
    frame_indices: tuple[int, ...],
    reference_index: int,
    predicted: torch.Tensor,
    target: torch.Tensor,
) -> list[dict[str, object]]:
    rows = []
    reference_available = int(bool(target[reference_index].any()))
    for sequence_index, frame_index in enumerate(frame_indices):
        pred = predicted[sequence_index].bool()
        gt = target[sequence_index].bool()
        intersection = int((pred & gt).sum())
        union = int((pred | gt).sum())
        predicted_pixels = int(pred.sum())
        gt_pixels = int(gt.sum())
        iou = (
            float(intersection) / union
            if union
            else 1.0
        )
        precision = (
            float(intersection) / predicted_pixels
            if predicted_pixels
            else float(gt_pixels == 0)
        )
        recall = (
            float(intersection) / gt_pixels
            if gt_pixels
            else float(predicted_pixels == 0)
        )
        rows.append(
            {
                "clip": clip_name,
                "split": split,
                "variant": variant,
                "instance_id": instance_id,
                "sequence_index": sequence_index,
                "frame_index": int(frame_index),
                "is_reference": int(sequence_index == reference_index),
                "reference_available": reference_available,
                "gt_visible": int(gt_pixels > 0),
                "predicted_visible": int(predicted_pixels > 0),
                "gt_pixels": gt_pixels,
                "predicted_pixels": predicted_pixels,
                "iou": _short(iou),
                "precision": _short(precision),
                "recall": _short(recall),
            }
        )
    return rows


def _summarize_metrics(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    raw_by_frame = {
        (
            str(row["clip"]),
            int(row["instance_id"]),
            int(row["sequence_index"]),
        ): float(row["iou"])
        for row in rows
        if row["variant"] == "raw_sam31"
    }
    for row in rows:
        if int(row["is_reference"]):
            continue
        key = (
            str(row["split"]),
            str(row["clip"]),
            str(row["variant"]),
        )
        groups.setdefault(key, []).append(row)
    output = []
    for (split, clip, variant), current in groups.items():
        evaluable = [
            row for row in current if int(row["reference_available"])
        ]
        visible = [row for row in evaluable if int(row["gt_visible"])]
        absent = [row for row in evaluable if not int(row["gt_visible"])]
        deltas = [
            float(row["iou"])
            - raw_by_frame[
                (
                    str(row["clip"]),
                    int(row["instance_id"]),
                    int(row["sequence_index"]),
                )
            ]
            for row in evaluable
        ]
        output.append(
            {
                "split": split,
                "clip": clip,
                "variant": variant,
                "reference_instances": len(
                    {
                        int(row["instance_id"])
                        for row in evaluable
                    }
                ),
                "evaluated_instance_frames": len(evaluable),
                "skipped_no_reference_frames": (
                    len(current) - len(evaluable)
                ),
                "gt_visible_frames": len(visible),
                "mean_iou_visible": _short(_mean(visible, "iou")),
                "mean_precision_visible": _short(
                    _mean(visible, "precision")
                ),
                "mean_recall_visible": _short(_mean(visible, "recall")),
                "false_negative_visible_frames": sum(
                    not int(row["predicted_visible"]) for row in visible
                ),
                "false_positive_absent_frames": sum(
                    int(row["predicted_visible"]) for row in absent
                ),
                "mean_iou_delta_from_raw": _short(
                    sum(deltas) / len(deltas)
                    if deltas
                    else float("nan")
                ),
                "improved_frames": sum(value > 1e-8 for value in deltas),
                "worse_frames": sum(value < -1e-8 for value in deltas),
            }
        )
    return sorted(
        output,
        key=lambda row: (
            str(row["split"]),
            str(row["clip"]),
            V6_SEGMENTATION_VARIANTS.index(str(row["variant"])),
        ),
    )


def _write_visualizations(
    output_dir: Path,
    *,
    image_paths: list[Path],
    frame_indices: tuple[int, ...],
    instance_ids: tuple[int, ...],
    target_masks: torch.Tensor,
    masks_by_variant: dict[str, torch.Tensor],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    titles = ("RGB", "GT", *V6_SEGMENTATION_VARIANTS)
    for sequence_index, (image_path, frame_index) in enumerate(
        zip(image_paths, frame_indices)
    ):
        rgb = Image.open(image_path).convert("RGB").resize(
            (
                int(target_masks.shape[-1]),
                int(target_masks.shape[-2]),
            ),
            Image.Resampling.BILINEAR,
        )
        panels = [rgb]
        panels.append(
            _overlay_instances(
                rgb,
                target_masks[sequence_index],
                instance_ids=instance_ids,
            )
        )
        for variant in V6_SEGMENTATION_VARIANTS:
            panels.append(
                _overlay_instances(
                    rgb,
                    masks_by_variant[variant][sequence_index],
                    instance_ids=instance_ids,
                )
            )
        header = 24
        canvas = Image.new(
            "RGB",
            (rgb.width * len(panels), rgb.height + header),
            "white",
        )
        draw = ImageDraw.Draw(canvas)
        for index, (title, panel) in enumerate(zip(titles, panels)):
            x = index * rgb.width
            canvas.paste(panel, (x, header))
            draw.text((x + 4, 4), title, fill="black")
        canvas.save(output_dir / f"{int(frame_index):06d}.png")


def _overlay_instances(
    rgb: Image.Image,
    masks: torch.Tensor,
    *,
    instance_ids: tuple[int, ...],
) -> Image.Image:
    colors = (
        (255, 64, 64),
        (64, 220, 96),
        (64, 128, 255),
        (240, 180, 32),
    )
    output = rgb.copy()
    color_layer = Image.new("RGB", rgb.size)
    color_pixels = color_layer.load()
    alpha = Image.new("L", rgb.size, 0)
    alpha_pixels = alpha.load()
    labels: list[tuple[int, int, int, tuple[int, int, int]]] = []
    for slot, instance_id in enumerate(instance_ids):
        mask = masks[slot].detach().cpu().bool()
        color = colors[slot % len(colors)]
        ys, xs = mask.nonzero(as_tuple=True)
        for y, x in zip(ys.tolist(), xs.tolist()):
            color_pixels[x, y] = color
            alpha_pixels[x, y] = 110
        if int(mask.sum()):
            labels.append(
                (
                    int(xs.min()),
                    int(ys.min()),
                    int(instance_id),
                    color,
                )
            )
    output.paste(color_layer, mask=alpha)
    draw = ImageDraw.Draw(output)
    for x, y, instance_id, color in labels:
        draw.text((x, y), str(instance_id), fill=color)
    return output


def _mean(rows: list[dict[str, object]], key: str) -> float:
    if not rows:
        return float("nan")
    return sum(float(row[key]) for row in rows) / len(rows)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty V6 table: {path}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _image_size_hw(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return int(image.height), int(image.width)


def _load_experiment(path: str | Path) -> V6SegmentationExperiment:
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf8") as handle:
        raw = yaml.safe_load(handle) or {}
    model = raw.get("model", {})
    refinement = raw.get("refinement", {})
    sam = raw.get("sam", {})
    return V6SegmentationExperiment(
        base_config=_resolve(source, raw["base_config"]),
        output_dir=_resolve(source, raw["output_dir"]),
        sam=SAMRuntimeConfig(
            version=str(sam.get("version", "sam3.1")),
            checkpoint=_resolve(source, sam["checkpoint"]),
            use_fa3=bool(sam.get("use_fa3", False)),
            max_num_objects=int(sam.get("max_num_objects", 16)),
            multiplex_count=int(sam.get("multiplex_count", 16)),
        ),
        segmentation=V6GeometrySegmentationConfig(
            point_confidence_threshold=float(
                model.get("point_confidence_threshold", 0.30)
            ),
            min_support_recall=float(
                model.get("min_support_recall", 0.25)
            ),
            min_box_precision=float(
                model.get("min_box_precision", 0.50)
            ),
            support_dilation=int(model.get("support_dilation", 5)),
            point_positive_samples=int(
                model.get("point_positive_samples", 6)
            ),
            point_negative_samples=int(
                model.get("point_negative_samples", 4)
            ),
            point_negative_exclusion_radius=int(
                model.get("point_negative_exclusion_radius", 5)
            ),
        ),
        refinement=InstanceRefinementConfig(
            min_instance_points=int(
                refinement.get("min_instance_points", 128)
            ),
            icp_max_points=int(refinement.get("icp_max_points", 1024)),
            map_max_points=int(refinement.get("map_max_points", 4096)),
            icp_iterations=int(refinement.get("icp_iterations", 4)),
            icp_trim_quantile=float(
                refinement.get("icp_trim_quantile", 0.70)
            ),
            min_icp_fitness=float(
                refinement.get("min_icp_fitness", 0.25)
            ),
            max_icp_rmse=float(refinement.get("max_icp_rmse", 0.03)),
            correspondence_min_distance=float(
                refinement.get("correspondence_min_distance", 0.02)
            ),
            correspondence_object_ratio=float(
                refinement.get("correspondence_object_ratio", 0.05)
            ),
            max_proposal_translation=float(
                refinement.get("max_proposal_translation", 0.15)
            ),
            compute_device="cpu",
        ),
    )


def _resolve(source: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (source.parent / path).resolve()


def _short(value: float) -> str:
    return f"{float(value):.8g}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="streaming_couping/configs/v6_geometry_segmentation.yaml",
    )
    parser.add_argument("--sam3-device")
    parser.add_argument("--geometry-device")
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    main()
