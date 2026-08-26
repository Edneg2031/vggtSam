#!/usr/bin/env python3
"""Evaluate a causal map-safe gate on frozen DINOv3 geometry predictions.

V2.4 is intentionally evaluation-only.  It reuses the validation-selected
residual heads from ``run_dinov3_object_geometry.py`` and applies a causal
world-space consistency gate to every learned branch.  No backbone, SAM3,
DINOv3, point head, or residual-head parameter is rerun or updated.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from streaming_couping.src.dinov3_object_geometry import (
    ObjectConditionedResidualHead,
    WorldSpaceGateConfig,
    apply_world_space_consistency_gate,
    apply_similarity,
)
from streaming_couping.src.learned_pose.config import load_learned_pose_config
from streaming_couping.src.semantic_map_metrics import (
    SemanticMapMetricConfig,
    evaluate_semantic_object_map,
)
from streaming_couping.src.semantic_tracking_metrics import (
    TrackingMetricConfig,
    evaluate_tracking_variants,
)

from streaming_couping.scripts.run_dinov3_object_geometry import (
    BRANCHES,
    LEARNED_BRANCHES,
    _load_clip_data,
    _load_ground_truth_bundle,
    _predict_clip_native,
    _score_predictions,
    _validate_model_compatibility,
)


BASE_REVISION_PREFIX = "dinov3_object_conditioned_geometry_cross_scene_"
REVISION = "dinov3_object_conditioned_geometry_v24_world_space_gate_r1"
GATED_BRANCHES = tuple(f"{branch}_v24_gated" for branch in LEARNED_BRANCHES)
ALL_BRANCHES = (*BRANCHES, *GATED_BRANCHES)
GATED_TO_BASE = dict(zip(GATED_BRANCHES, LEARNED_BRANCHES))


def main() -> None:
    args = _parse_args()
    protocol = Path(args.protocol).expanduser().resolve()
    feature_dir = Path(args.feature_dir).expanduser().resolve()
    model_path = Path(args.model_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not protocol.is_file():
        raise FileNotFoundError(f"Protocol is missing: {protocol}")
    if not feature_dir.is_dir():
        raise FileNotFoundError(f"DINOv3 feature cache directory is missing: {feature_dir}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Frozen object-geometry model is missing: {model_path}")

    config = load_learned_pose_config(protocol)
    validation_clips = tuple(
        clip for clip in config.clips if clip.split == "validation"
    )
    test_clips = tuple(clip for clip in config.clips if clip.split == "test")
    if not validation_clips or not test_clips:
        raise ValueError(
            "V2.4 requires validation and test clips; got "
            f"validation={len(validation_clips)} test={len(test_clips)}."
        )

    payload = torch.load(model_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Frozen model payload is not a mapping: {model_path}")
    base_revision = str(payload.get("revision", ""))
    if not base_revision.startswith(BASE_REVISION_PREFIX):
        raise ValueError(
            "Frozen model revision is incompatible with V2.4: "
            f"{base_revision!r}"
        )
    model_spec = payload.get("model")
    states = payload.get("branches")
    if not isinstance(model_spec, Mapping) or not isinstance(states, Mapping):
        raise ValueError("Frozen model payload lacks model specification or branches.")
    for branch in LEARNED_BRANCHES:
        if branch not in states or not isinstance(states[branch], Mapping):
            raise ValueError(f"Frozen model payload lacks branch {branch!r}.")

    device = _resolve_device(args.device)
    gate_config = WorldSpaceGateConfig(
        confidence_threshold=float(args.confidence_threshold),
        track_score_threshold=float(args.track_score_threshold),
        min_points=int(args.gate_min_points),
        max_correction_m=float(args.gate_max_correction_m),
        consistency_margin_m=float(args.gate_consistency_margin_m),
        shape_weight=float(args.gate_shape_weight),
        memory_momentum=float(args.gate_memory_momentum),
        shape_scale_m=float(args.gate_shape_scale_m),
    )
    print("DINOv3 OBJECT-CONDITIONED GEOMETRY V2.4 WORLD-SPACE GATE")
    print(f"protocol={protocol}")
    print(f"feature_dir={feature_dir}")
    print(f"frozen_model={model_path}")
    print(
        "validation-selected residual heads are frozen; no parameters are "
        "retrained or updated; V2.4 adds causal inference-time gating only"
    )
    print(
        f"device={device} confidence_threshold={gate_config.confidence_threshold} "
        f"track_score_threshold={gate_config.track_score_threshold} "
        f"gate_min_points={gate_config.min_points} "
        f"max_correction_m={gate_config.max_correction_m}"
    )
    print(
        "branches="
        + ",".join(ALL_BRANCHES)
        + " | gated branches use the same frozen base prediction"
    )

    validation_data = [
        _load_clip_data(
            config,
            clip,
            feature_dir,
            confidence_threshold=float(args.confidence_threshold),
        )
        for clip in validation_clips
    ]
    test_data = [
        _load_clip_data(
            config,
            clip,
            feature_dir,
            confidence_threshold=float(args.confidence_threshold),
        )
        for clip in test_clips
    ]
    _validate_model_compatibility((*validation_data, *test_data))
    print(
        f"loaded validation_clips={len(validation_data)} test_clips={len(test_data)} "
        f"feature_channels={validation_data[0].features.shape[-1]} "
        f"dino_dim={validation_data[0].single_features.shape[-1]}"
    )

    model_states = {
        branch: states[branch]["state_dict"] for branch in LEARNED_BRANCHES
    }
    metric_rows: list[dict[str, Any]] = []
    object_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    map_metric_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    for split, split_data in (
        ("validation", validation_data),
        ("test", test_data),
    ):
        for data in split_data:
            result = _evaluate_clip(
                data=data,
                model_states=model_states,
                model_spec=model_spec,
                device=device,
                args=args,
                gate_config=gate_config,
                split=split,
            )
            metric_rows.extend(result["metrics"])
            object_rows.extend(result["objects"])
            frame_rows.extend(result["frames"])
            map_metric_rows.extend(result["map_metrics"])
            gate_rows.extend(result["gate_rows"])
            _print_clip_metrics(result, split)

    summary = _build_summary(
        protocol=protocol,
        feature_dir=feature_dir,
        model_path=model_path,
        base_revision=base_revision,
        model_spec=model_spec,
        metric_rows=metric_rows,
        map_metric_rows=map_metric_rows,
        gate_rows=gate_rows,
        validation_clips=validation_clips,
        test_clips=test_clips,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "summary.json", summary)
    _write_csv(output_dir / "branch_summary.csv", metric_rows)
    _write_csv(output_dir / "semantic_map_metrics.csv", map_metric_rows)
    _write_csv(output_dir / "gate_diagnostics.csv", gate_rows)
    _write_csv(output_dir / "object_metrics.csv", object_rows)
    _write_csv(output_dir / "frame_metrics.csv", frame_rows)
    _write_copyable(output_dir / "copyable_result.txt", summary)
    print(
        f"DINOv3 V2.4 result={output_dir / 'summary.json'} "
        f"decision={summary['decision']['overall']}"
    )


@torch.inference_mode()
def _evaluate_clip(
    *,
    data,
    model_states: Mapping[str, Mapping[str, torch.Tensor]],
    model_spec: Mapping[str, Any],
    device: torch.device,
    args: argparse.Namespace,
    gate_config: WorldSpaceGateConfig,
    split: str,
) -> dict[str, Any]:
    aligned_predictions: dict[str, torch.Tensor] = {
        "raw": apply_similarity(
            data.raw_native,
            scale=data.scale,
            rotation=data.rotation,
            translation=data.translation,
        ).float()
    }
    for branch in LEARNED_BRANCHES:
        model = ObjectConditionedResidualHead(**dict(model_spec)).to(device)
        model.load_state_dict(model_states[branch], strict=True)
        model.eval()
        native = _predict_clip_native(model, data, branch, device)
        aligned_predictions[branch] = apply_similarity(
            native,
            scale=data.scale,
            rotation=data.rotation,
            translation=data.translation,
        ).float()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    gate_rows: list[dict[str, Any]] = []
    for gated_branch, base_branch in GATED_TO_BASE.items():
        gate = apply_world_space_consistency_gate(
            aligned_predictions["raw"],
            aligned_predictions[base_branch],
            object_masks=data.masks,
            point_confidence=data.confidence,
            track_scores=data.scores,
            config=gate_config,
        )
        aligned_predictions[gated_branch] = gate.points
        gate_rows.append(
            {
                "clip": data.name,
                "scene_id": data.scene_id,
                "split": split,
                "branch": gated_branch,
                "base_branch": base_branch,
                **dict(gate.stats),
            }
        )

    ground_truth = _load_ground_truth_bundle(data)
    tracking = evaluate_tracking_variants(
        scene_id=data.scene_id,
        clip_name=data.name,
        frame_indices=data.frame_indices,
        variant_masks={"raw_sam": data.masks},
        variant_scores={"raw_sam": data.scores},
        raw_variant="raw_sam",
        track_ids=data.track_ids,
        track_prompts=data.track_prompts,
        ground_truth=ground_truth.instances,
        config=TrackingMetricConfig(),
        prompt_label_aliases=None,
    )
    map_config = SemanticMapMetricConfig(
        confidence_threshold=float(args.confidence_threshold),
        track_score_threshold=float(args.track_score_threshold),
        max_points_per_object=int(args.maximum_object_points),
        distance_chunk_size=256,
        fscore_thresholds_m=(0.05, 0.10),
        voxel_size_m=0.05,
        ghost_distance_m=0.10,
        duplicate_voxel_iou=0.25,
    )
    metrics: list[dict[str, Any]] = []
    objects: list[dict[str, Any]] = []
    frames: list[dict[str, Any]] = []
    map_metrics: list[dict[str, Any]] = []
    raw_object_lookup: dict[int, float] = {}
    score_args = argparse.Namespace(
        maximum_points_per_frame=int(args.maximum_points_per_frame),
        maximum_object_points=int(args.maximum_object_points),
    )
    for branch in ALL_BRANCHES:
        result = _score_predictions(
            data,
            branch=branch,
            points=aligned_predictions[branch],
            gt_masks=ground_truth.stream_masks,
            gt_instances=ground_truth.instances,
            args=score_args,
            split=split,
        )
        metrics.append(result["summary"])
        objects.extend(result["objects"])
        frames.extend(result["frames"])
        map_result = evaluate_semantic_object_map(
            scene_id=data.scene_id,
            clip_name=data.name,
            variant=branch,
            map_policy="v24_causal_world_space_gate_same_raw_masks",
            aligned_world_points=aligned_predictions[branch],
            target_world_points=data.target_metric,
            confidence=data.confidence,
            predicted_masks=data.masks,
            track_scores=data.scores,
            gt_masks=ground_truth.stream_masks,
            gt_instance_ids=ground_truth.instances.instance_ids,
            gt_labels=ground_truth.instances.labels,
            assignments=tracking["assignments"],
            map_write_mask=data.masks.flatten(2).any(dim=2),
            track_ids=data.track_ids,
            config=map_config,
        )
        map_summary = dict(map_result["summary"])
        map_summary["branch"] = branch
        map_summary["split"] = split
        map_metrics.append(map_summary)
        if branch == "raw":
            raw_object_lookup = {
                int(row["object_index"]): float(row["paired_rmse_m"])
                for row in result["objects"]
            }

    for row in objects:
        if row["branch"] == "raw":
            continue
        raw_value = raw_object_lookup.get(int(row["object_index"]), float("nan"))
        candidate_value = float(row["paired_rmse_m"])
        row["paired_rmse_gain_vs_raw_percent"] = _gain(raw_value, candidate_value)
        row["improved_vs_raw"] = int(
            math.isfinite(raw_value)
            and math.isfinite(candidate_value)
            and candidate_value < raw_value
        )
    for row in metrics:
        if row["branch"] == "raw":
            continue
        branch_objects = [
            value for value in objects if value["branch"] == row["branch"]
        ]
        row["improved_objects_vs_raw"] = int(
            sum(int(value["improved_vs_raw"]) for value in branch_objects)
        )
        row["improved_object_ratio_vs_raw"] = (
            float(row["improved_objects_vs_raw"] / len(branch_objects))
            if branch_objects
            else 0.0
        )
    return {
        "clip": data.name,
        "scene_id": data.scene_id,
        "split": split,
        "metrics": metrics,
        "map_metrics": map_metrics,
        "gate_rows": gate_rows,
        "objects": objects,
        "frames": frames,
    }


def _build_summary(
    *,
    protocol: Path,
    feature_dir: Path,
    model_path: Path,
    base_revision: str,
    model_spec: Mapping[str, Any],
    metric_rows: Sequence[Mapping[str, Any]],
    map_metric_rows: Sequence[Mapping[str, Any]],
    gate_rows: Sequence[Mapping[str, Any]],
    validation_clips: Sequence[Any],
    test_clips: Sequence[Any],
) -> dict[str, Any]:
    aggregate: dict[str, dict[str, Any]] = {}
    for branch in ALL_BRANCHES:
        value: dict[str, Any] = {}
        for split in ("validation", "test"):
            metrics = [
                row
                for row in metric_rows
                if row["branch"] == branch and row["split"] == split
            ]
            maps = [
                row
                for row in map_metric_rows
                if row["branch"] == branch and row["split"] == split
            ]
            prefix = f"{split}_"
            value[f"{prefix}clips"] = len(metrics)
            value[f"{prefix}object_rmse_m"] = _mean_field(
                metrics, "object_paired_rmse_m"
            )
            value[f"{prefix}object_fscore_5cm"] = _mean_field(
                metrics, "object_fscore_5cm"
            )
            value[f"{prefix}map_voxel_iou_5cm"] = _mean_field(
                maps, "voxel_iou_5cm"
            )
            value[f"{prefix}map_fscore_5cm"] = _mean_field(
                maps, "fscore_5cm"
            )
            value[f"{prefix}map_ghost_rate"] = _mean_field(
                maps, "ghost_point_ratio"
            )
        aggregate[branch] = value

    raw = aggregate["raw"]
    persistent = aggregate["persistent_dino"]
    persistent_gated = aggregate["persistent_dino_v24_gated"]
    geometry_gated = aggregate["geometry_only_v24_gated"]
    persistent_gate_not_worse = _map_not_worse(
        persistent_gated,
        persistent,
        split="test",
        ghost_tolerance=0.0,
    )
    persistent_gate_better_geometry = (
        persistent_gated["test_object_rmse_m"]
        < geometry_gated["test_object_rmse_m"]
        and persistent_gated["test_map_fscore_5cm"]
        >= geometry_gated["test_map_fscore_5cm"]
    )
    test_map_not_worse_raw = _map_not_worse(
        persistent_gated,
        raw,
        split="test",
        ghost_tolerance=0.02,
    )
    validation_map_not_worse_raw = _map_not_worse(
        aggregate["persistent_dino_v24_gated"],
        aggregate["raw"],
        split="validation",
        ghost_tolerance=0.02,
    )
    test_object_better_raw = (
        persistent_gated["test_object_rmse_m"] < raw["test_object_rmse_m"]
    )
    enough_test_scenes = len(test_clips) >= 2
    safe_signal = (
        persistent_gate_not_worse
        and persistent_gate_better_geometry
        and test_map_not_worse_raw
        and test_object_better_raw
        and validation_map_not_worse_raw
    )
    if not safe_signal:
        overall = "NO_GO"
    elif not enough_test_scenes:
        overall = "CONDITIONAL_GO"
    else:
        overall = "GO"
    decision = {
        "v24_gate_not_worse_than_ungated_persistent_test": int(
            persistent_gate_not_worse
        ),
        "v24_persistent_better_than_geometry_gated_test": int(
            persistent_gate_better_geometry
        ),
        "v24_persistent_map_not_worse_than_raw_test": int(test_map_not_worse_raw),
        "v24_persistent_map_not_worse_than_raw_validation": int(
            validation_map_not_worse_raw
        ),
        "v24_persistent_object_better_than_raw_test": int(test_object_better_raw),
        "v24_safe_signal": int(safe_signal),
        "multiple_test_scenes": int(enough_test_scenes),
        "overall": overall,
        "rule": (
            "V2.4 must be map-safe against its ungated persistent candidate, "
            "beat geometry-gated on test, not worsen raw map on test, improve "
            "test object RMSE, avoid validation map regression, and needs at "
            "least two test scenes for unconditional GO"
        ),
    }
    return {
        "schema": 1,
        "revision": REVISION,
        "base_revision": str(base_revision),
        "protocol": str(protocol),
        "feature_dir": str(feature_dir),
        "frozen_model": str(model_path),
        "branches": list(ALL_BRANCHES),
        "gated_branches": dict(GATED_TO_BASE),
        "model": dict(model_spec),
        "validation_scenes": [clip.scene_id for clip in validation_clips],
        "test_scenes": [clip.scene_id for clip in test_clips],
        "aggregate": aggregate,
        "gate_diagnostics": list(gate_rows),
        "decision": decision,
        "data_policy": {
            "streamvggt_rerun": 0,
            "sam_rerun": 0,
            "dinov3_rerun": 0,
            "residual_head_parameters_updated": 0,
            "backbone_parameters_updated": 0,
            "test_gt_read_during_gate": 0,
            "gate_uses_future_frames": 0,
            "gate_uses_gt": 0,
            "gate_memory": "causal_per_slot_world_centroid_and_robust_extent",
        },
        "metrics": list(metric_rows),
        "semantic_map_metrics": list(map_metric_rows),
    }


def _map_not_worse(
    candidate: Mapping[str, Any],
    reference: Mapping[str, Any],
    *,
    split: str,
    ghost_tolerance: float,
) -> bool:
    return (
        candidate[f"{split}_map_voxel_iou_5cm"]
        >= reference[f"{split}_map_voxel_iou_5cm"] * 0.95
        and candidate[f"{split}_map_fscore_5cm"]
        >= reference[f"{split}_map_fscore_5cm"] * 0.95
        and candidate[f"{split}_map_ghost_rate"]
        <= reference[f"{split}_map_ghost_rate"] + float(ghost_tolerance)
    )


def _print_clip_metrics(result: Mapping[str, Any], split: str) -> None:
    print(f"  {split} clip={result['clip']} scene={result['scene_id']}")
    for row in result["metrics"]:
        print(
            f"    {row['branch']} object_rmse={row['object_paired_rmse_m']:.5f} "
            f"object_fscore5={row['object_fscore_5cm']:.5f}"
        )
    for row in result["map_metrics"]:
        print(
            f"    map[{row['branch']}] voxelIoU5={row['voxel_iou_5cm']:.5f} "
            f"F5cm={row['fscore_5cm']:.5f} ghost={row['ghost_point_ratio']:.5f}"
        )
    for row in result["gate_rows"]:
        print(
            f"    gate[{row['branch']}] accepted_objects="
            f"{row['accepted_object_observations']}/"
            f"{row['total_object_observations']} "
            f"accepted_points={row['accepted_point_ratio']:.5f} "
            f"reasons={row['reject_reasons']}"
        )


def _mean_field(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    values = [float(row[field]) for row in rows if math.isfinite(float(row[field]))]
    return sum(values) / len(values) if values else float("nan")


def _gain(raw: float, candidate: float) -> float:
    if not math.isfinite(float(raw)) or not math.isfinite(float(candidate)):
        return float("nan")
    return 100.0 * (float(raw) - float(candidate)) / max(abs(float(raw)), 1e-12)


def _resolve_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {device} requested but CUDA is unavailable.")
    return device


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf8")
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_copyable(path: Path, summary: Mapping[str, Any]) -> None:
    lines = [
        "===== COPYABLE_DINOV3_OBJECT_GEOMETRY_V24_BEGIN =====",
        f"revision={summary['revision']}",
        f"base_revision={summary['base_revision']}",
        "branches=" + ",".join(summary["branches"]),
        "validation_scenes=" + ",".join(summary["validation_scenes"]),
        "test_scenes=" + ",".join(summary["test_scenes"]),
        "gate=causal_world_space_centroid_extent_memory",
        "gate_uses_gt=0",
        "gate_uses_future_frames=0",
        "",
        "branch,validation_object_rmse_m,validation_map_fscore_5cm,validation_map_ghost_rate,test_object_rmse_m,test_map_voxel_iou_5cm,test_map_fscore_5cm,test_map_ghost_rate",
    ]
    for branch in summary["branches"]:
        value = summary["aggregate"][branch]
        lines.append(
            ",".join(
                [
                    branch,
                    str(value["validation_object_rmse_m"]),
                    str(value["validation_map_fscore_5cm"]),
                    str(value["validation_map_ghost_rate"]),
                    str(value["test_object_rmse_m"]),
                    str(value["test_map_voxel_iou_5cm"]),
                    str(value["test_map_fscore_5cm"]),
                    str(value["test_map_ghost_rate"]),
                ]
            )
        )
    lines.extend(
        (
            "",
            "decision=" + json.dumps(summary["decision"], sort_keys=True),
            f"summary={path.with_name('summary.json')}",
            f"semantic_map_metrics={path.with_name('semantic_map_metrics.csv')}",
            f"gate_diagnostics={path.with_name('gate_diagnostics.csv')}",
            "===== COPYABLE_DINOV3_OBJECT_GEOMETRY_V24_END =====",
        )
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf8")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if torch.is_tensor(value):
        return value.tolist()
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--confidence-threshold", type=float, default=0.30)
    parser.add_argument("--track-score-threshold", type=float, default=0.50)
    parser.add_argument("--maximum-points-per-frame", type=int, default=8192)
    parser.add_argument("--maximum-object-points", type=int, default=1024)
    parser.add_argument("--gate-min-points", type=int, default=32)
    parser.add_argument("--gate-max-correction-m", type=float, default=0.08)
    parser.add_argument("--gate-consistency-margin-m", type=float, default=0.02)
    parser.add_argument("--gate-shape-weight", type=float, default=0.25)
    parser.add_argument("--gate-memory-momentum", type=float, default=0.80)
    parser.add_argument("--gate-shape-scale-m", type=float, default=0.05)
    return parser.parse_args()


if __name__ == "__main__":
    main()
