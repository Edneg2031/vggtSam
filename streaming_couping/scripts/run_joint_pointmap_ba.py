"""Reproduce the pointmap-blend sweep and export the adaptive final method."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path

import torch

from streaming_couping.src.config import load_config
from streaming_couping.src.learned_pose.cache import cache_path, load_feature_cache
from streaming_couping.src.learned_pose.config import load_learned_pose_config
from streaming_couping.src.learned_pose.geometry_metrics import (
    append_geometry_metrics,
    summarize_pointmap_metrics,
)
from streaming_couping.src.learned_pose.export import (
    export_final_ray_pose_outputs,
)
from streaming_couping.src.learned_pose.pipeline import (
    _append_pose_metrics,
    _evaluation_metadata,
    _evaluation_sequence_indices,
)
from streaming_couping.src.learned_pose.ray_pose import (
    FINAL_RAY_POSE_NAME,
    recover_final_ray_pose,
    reference_blend_pose_name,
)
from vggtsam.utils.imports import maybe_add_repo_to_path


CONTROL_RAW = "raw_control"
CONTROL_FIXED = "fixed_ref_050_control"
ADAPTIVE_FINAL = "adaptive_support_gate"
ADAPTIVE_SUPPORT_THRESHOLD = 0.75
POINT_BLEND_VALUES = (0.0, 0.25, 0.50, 0.75, 1.0)
SUMMARY_FIELDS = (
    "split",
    "clip",
    "variant",
    "point_blend",
    "pose_pointmap_coupled",
    "ate",
    "rotation_deg",
    "pointmap_mean",
    "ate_delta_from_raw",
    "pointmap_delta_from_raw",
    "fit_accepted",
    "support_ratio",
    "mean_center_shift",
    "raw_reference_exact",
    "learned_reference_preserved",
    "a100_pose_matches_fixed",
    "module_off_exact",
    "status",
)


def main() -> None:
    args = _parse_args()
    config = load_learned_pose_config(args.config)
    recovery = load_config(config.recovery_config)
    maybe_add_repo_to_path(recovery.streamvggt_repo)
    source_path = args.predictions or _default_predictions(config.output_dir)
    source = _torch_load(source_path)
    predictions = source.get("predictions")
    if not isinstance(predictions, dict):
        raise ValueError(f"Prediction file has no clip dictionary: {source_path}")
    output = config.output_dir / "joint_pointmap_ba"
    evaluation = output / "evaluation"
    evaluation.mkdir(parents=True, exist_ok=True)
    print(f"reusing predictions: {source_path}")
    print(f"reusing cache: {config.features.cache_dir}")
    print("pointmap blend sweep does not run SAM3, StreamVGGT, or training")

    pose_summary: list[dict] = []
    pose_frames: list[dict] = []
    pose_rpe: list[dict] = []
    pose_pairs: list[dict] = []
    pose_pair_summary: list[dict] = []
    pointmap_frames: list[dict] = []
    depth_frames: list[dict] = []
    diagnostic_rows: list[dict] = []
    saved_predictions: dict[str, dict] = {}
    adaptive_export_predictions: dict[str, dict] = {}

    for clip in config.clips:
        if not (
            clip.split.lower() in {"val", "validation", "test"}
            or clip.evaluation_frame_indices is not None
        ):
            continue
        path = cache_path(config, clip)
        if not path.is_file():
            raise FileNotFoundError(
                "Joint BA intentionally does not rebuild caches. "
                f"Missing: {path}"
            )
        if clip.name not in predictions:
            raise KeyError(
                f"Prediction file {source_path} has no clip {clip.name!r}."
            )
        print(f"pointmap blend clip={clip.name}")
        payload = load_feature_cache(path)
        predicted = predictions[clip.name]
        raw_pose = _pose_prediction(predicted, "raw_baseline_control")
        learned_pose = _learned_pose_prediction(predicted)
        fixed_pose = _pose_prediction(
            predicted,
            reference_blend_pose_name(0.50),
        )
        raw_points = payload["baseline_world_points"].float()
        refined_points = _squeeze_saved(
            predicted["refined_world_points"]
        ).float()
        refined_confidence = _squeeze_saved(
            predicted["refined_world_confidence"]
        ).float()
        raw_confidence = payload["baseline_world_confidence"].float()
        sequence_indices = _evaluation_sequence_indices(clip)
        metadata = _evaluation_metadata(clip)
        geometry_batch = _geometry_batch(payload, args.device)
        ray_batch = _ray_batch(
            payload,
            clip=clip,
            strict_identity_gate=bool(config.fusion.strict_identity_gate),
            device=args.device,
        )
        ray_config = replace(
            config.evaluation.ray_pose,
            solver_modes=(),
            reference_blend_values=(0.50,),
        )
        module_off_exact = int(
            torch.equal(
                _squeeze_saved(raw_pose).cpu(),
                payload["baseline_pose_encoding"].float(),
            )
        )

        controls = (
            (
                CONTROL_RAW,
                raw_pose,
                raw_points[None].to(args.device),
                payload["baseline_depth"][None].to(args.device),
                1,
                {"status": "raw_streamvggt_control"},
            ),
            (
                CONTROL_FIXED,
                fixed_pose,
                refined_points[None].to(args.device),
                payload["baseline_depth"][None].to(args.device),
                0,
                {"status": "previous_fixed_ref_050_control"},
            ),
        )
        clip_outputs: dict[str, dict] = {}
        blend_records: dict[float, dict[str, object]] = {}
        for name, pose, points, depth, consistent, diagnostics in controls:
            pose = pose.to(args.device)
            _append_metrics(
                pose_summary,
                pose_frames,
                pose_rpe,
                pose_pairs,
                pose_pair_summary,
                pointmap_frames,
                depth_frames,
                payload=payload,
                batch=geometry_batch,
                pose=pose,
                points=points,
                depth=depth,
                name=name,
                sequence_indices=sequence_indices,
                metadata=metadata,
            )
            diagnostic_rows.append(
                {
                    "clip": clip.name,
                    "variant": name,
                    "joint_consistent": consistent,
                    "module_off_exact": module_off_exact,
                    **diagnostics,
                }
            )
            clip_outputs[name] = {
                "pose_encoding": pose.detach().cpu(),
                "world_points": points.detach().cpu(),
                "depth": depth.detach().cpu(),
            }

        for point_blend in POINT_BLEND_VALUES:
            name = point_blend_name(point_blend)
            points = _blend_pointmap(
                raw_points,
                refined_points,
                blend=point_blend,
                reference_index=int(payload["reference_sequence_index"]),
            )
            confidence = _blend_confidence(
                raw_confidence,
                refined_confidence,
                blend=point_blend,
                reference_index=int(payload["reference_sequence_index"]),
            )
            ray_results = recover_final_ray_pose(
                batch=ray_batch,
                baseline_outputs={
                    "pose_encoding": raw_pose.to(args.device),
                },
                refined_outputs={
                    "pose_encoding": learned_pose.to(args.device),
                    "world_points": points[None].to(args.device),
                    "world_confidence": confidence[None].to(args.device),
                },
                config=ray_config,
            )
            if len(ray_results) != 1:
                raise ValueError(
                    "Pointmap blend sweep expected exactly one ray-pose result."
                )
            ray_result = ray_results[0]
            pose = ray_result.pose_encoding
            fit_rows = tuple(ray_result.diagnostics)
            nonreference_rows = [
                row for row in fit_rows if int(row.get("is_reference", 0)) == 0
            ]
            accepted = sum(
                int(row.get("fit_accepted", 0))
                for row in nonreference_rows
            )
            accepted_shifts = [
                float(row.get("applied_center_shift_native", 0.0))
                for row in nonreference_rows
                if int(row.get("fit_accepted", 0)) == 1
            ]
            pose_matches_fixed = (
                int(torch.equal(pose.cpu(), fixed_pose.cpu()))
                if point_blend == 1.0
                else ""
            )
            diagnostics = {
                "status": (
                    f"ray_fit_accepted_{accepted}_of_{len(nonreference_rows)}"
                ),
                "point_blend": point_blend,
                "pose_pointmap_coupled": 1,
                "fit_accepted": accepted,
                "fit_total": len(nonreference_rows),
                "support_ratio": (
                    accepted / len(nonreference_rows)
                    if nonreference_rows
                    else 0.0
                ),
                "mean_pose_center_shift_native": (
                    sum(accepted_shifts) / len(accepted_shifts)
                    if accepted_shifts
                    else 0.0
                ),
                "raw_reference_exact": int(
                    torch.equal(
                        pose[:, int(payload["reference_sequence_index"])].cpu(),
                        raw_pose[
                            :, int(payload["reference_sequence_index"])
                        ].cpu(),
                    )
                ),
                "learned_reference_preserved": int(
                    torch.equal(
                        pose[:, int(payload["reference_sequence_index"])].cpu(),
                        learned_pose[
                            :, int(payload["reference_sequence_index"])
                        ].cpu(),
                    )
                ),
                "a100_pose_matches_fixed": pose_matches_fixed,
            }
            _append_metrics(
                pose_summary,
                pose_frames,
                pose_rpe,
                pose_pairs,
                pose_pair_summary,
                pointmap_frames,
                depth_frames,
                payload=payload,
                batch=geometry_batch,
                pose=pose,
                points=points[None].to(args.device),
                depth=payload["baseline_depth"][None].to(args.device),
                name=name,
                sequence_indices=sequence_indices,
                metadata=metadata,
            )
            diagnostic_rows.append(
                {
                    "clip": clip.name,
                    "variant": name,
                    "module_off_exact": module_off_exact,
                    **diagnostics,
                }
            )
            clip_outputs[name] = {
                "pose_encoding": pose.detach().cpu(),
                "world_points": points[None].detach().cpu(),
                "depth": payload["baseline_depth"][None].detach().cpu(),
                "world_confidence": confidence[None].detach().cpu(),
                "diagnostics": diagnostics,
            }
            blend_records[point_blend] = {
                "pose_encoding": pose.detach().cpu(),
                "world_points": points[None].detach().cpu(),
                "world_confidence": confidence[None].detach().cpu(),
                "depth": payload["baseline_depth"][None].detach().cpu(),
                "diagnostics": diagnostics,
            }
            del ray_result
            if str(args.device).startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()

        gate_source = blend_records[1.0]
        gate_source_diagnostics = gate_source["diagnostics"]
        if not isinstance(gate_source_diagnostics, dict):
            raise TypeError("Adaptive gate diagnostics must be a dictionary.")
        support_ratio = float(
            gate_source_diagnostics.get("support_ratio", 0.0)
        )
        selected_blend = _select_adaptive_blend(
            support_ratio,
            threshold=ADAPTIVE_SUPPORT_THRESHOLD,
        )
        selected_record = blend_records[selected_blend]
        selected_diagnostics = selected_record["diagnostics"]
        if not isinstance(selected_diagnostics, dict):
            raise TypeError("Selected diagnostics must be a dictionary.")
        adaptive_diagnostics = {
            **selected_diagnostics,
            "status": (
                f"adaptive_selected_a{int(100 * selected_blend):03d}_"
                f"from_a100_support_{support_ratio:.6f}"
            ),
            "point_blend": selected_blend,
            "support_ratio": support_ratio,
            "adaptive_support_threshold": ADAPTIVE_SUPPORT_THRESHOLD,
            "a100_pose_matches_fixed": "",
        }
        adaptive_pose = selected_record["pose_encoding"]
        adaptive_points = selected_record["world_points"]
        adaptive_depth = selected_record["depth"]
        adaptive_confidence = selected_record["world_confidence"]
        if not all(
            torch.is_tensor(value)
            for value in (
                adaptive_pose,
                adaptive_points,
                adaptive_depth,
                adaptive_confidence,
            )
        ):
            raise TypeError("Adaptive prediction payload contains non-tensors.")
        _append_metrics(
            pose_summary,
            pose_frames,
            pose_rpe,
            pose_pairs,
            pose_pair_summary,
            pointmap_frames,
            depth_frames,
            payload=payload,
            batch=geometry_batch,
            pose=adaptive_pose.to(args.device),
            points=adaptive_points.to(args.device),
            depth=adaptive_depth.to(args.device),
            name=ADAPTIVE_FINAL,
            sequence_indices=sequence_indices,
            metadata=metadata,
        )
        diagnostic_rows.append(
            {
                "clip": clip.name,
                "variant": ADAPTIVE_FINAL,
                "module_off_exact": module_off_exact,
                **adaptive_diagnostics,
            }
        )
        clip_outputs[ADAPTIVE_FINAL] = {
            **selected_record,
            "diagnostics": adaptive_diagnostics,
        }
        saved_predictions[clip.name] = {
            "scene_id": clip.scene_id,
            "frame_indices": list(clip.frame_indices),
            "reference_sequence_index": clip.reference_sequence_index,
            "variants": clip_outputs,
        }
        adaptive_export_predictions[clip.name] = {
            "scene_id": clip.scene_id,
            "frame_indices": list(clip.frame_indices),
            "instance_ids": list(clip.instance_ids),
            "image_paths": list(payload["image_paths"]),
            "pose_encodings": {
                "raw_baseline_control": raw_pose.detach().cpu(),
                FINAL_RAY_POSE_NAME: adaptive_pose.detach().cpu(),
            },
            "refined_world_points": adaptive_points.detach().cpu(),
            "refined_world_confidence": adaptive_confidence.detach().cpu(),
            "tracking_masks_stream": payload[
                "trusted_tracking_masks_stream"
            ][None].detach().bool().cpu(),
            "adaptive_selected_blend": selected_blend,
            "adaptive_support_ratio": support_ratio,
            "adaptive_support_threshold": ADAPTIVE_SUPPORT_THRESHOLD,
        }

    pointmap_summary = summarize_pointmap_metrics(pointmap_frames)
    _write_csv(evaluation / "pose_summary.csv", pose_summary)
    _write_csv(evaluation / "pose_frame_metrics.csv", pose_frames)
    _write_csv(evaluation / "pose_rpe.csv", pose_rpe)
    _write_csv(evaluation / "pose_pair_metrics.csv", pose_pairs)
    _write_csv(evaluation / "pose_pair_summary.csv", pose_pair_summary)
    _write_csv(evaluation / "pointmap_frame_metrics.csv", pointmap_frames)
    _write_csv(evaluation / "pointmap_summary.csv", pointmap_summary)
    _write_csv(evaluation / "joint_ba_diagnostics.csv", diagnostic_rows)
    torch.save(
        {
            "source_predictions": str(source_path),
            "predictions": saved_predictions,
        },
        evaluation / "joint_ba_predictions.pt",
    )
    adaptive_predictions_path = (
        evaluation / "adaptive_ray_pose_predictions.pt"
    )
    torch.save(
        {
            "source_predictions": str(source_path),
            "selection_policy": {
                "name": ADAPTIVE_FINAL,
                "support_source": point_blend_name(1.0),
                "support_threshold": ADAPTIVE_SUPPORT_THRESHOLD,
                "high_support_blend": 1.0,
                "low_support_blend": 0.0,
            },
            "predictions": adaptive_export_predictions,
        },
        adaptive_predictions_path,
    )
    compact = _compact_summary(
        pose_summary,
        pointmap_summary,
        diagnostic_rows,
    )
    summary_path = config.output_dir / "joint_ba_upload_summary.csv"
    _write_csv(summary_path, compact, fieldnames=SUMMARY_FIELDS)
    print(f"joint BA upload summary: {summary_path}")
    with summary_path.open("r", encoding="utf8") as handle:
        print(handle.read().rstrip())
    if args.export:
        export_root = export_final_ray_pose_outputs(
            config,
            output_dir=config.output_dir / "final_adaptive_pointmap_pose",
            predictions_path=adaptive_predictions_path,
        )
        print(f"adaptive full export: {export_root}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="streaming_couping/configs/v5_ablation_suite.yaml",
    )
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--predictions", type=Path)
    parser.add_argument(
        "--export",
        action="store_true",
        help="Export GT/raw/ours PLY, masks, poses, and comparison tables.",
    )
    return parser.parse_args()


def _default_predictions(output_dir: Path) -> Path:
    candidate = (
        output_dir
        / "joint_solver_sweep"
        / "evaluation"
        / "ray_pose_predictions.pt"
    )
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(
        "Missing the reference-blend sweep predictions required by the "
        "fixed_ref_050 control. Run commands_joint_pointmap_ba.txt, which "
        f"creates them automatically when needed. Expected: {candidate}"
    )


def _pose_prediction(prediction: dict, name: str) -> torch.Tensor:
    poses = prediction.get("pose_encodings")
    if not isinstance(poses, dict) or name not in poses:
        raise KeyError(f"Missing pose prediction {name!r}.")
    value = poses[name]
    if not torch.is_tensor(value):
        raise TypeError(f"Pose prediction {name!r} is not a tensor.")
    return value.float()


def _learned_pose_prediction(prediction: dict) -> torch.Tensor:
    poses = prediction.get("pose_encodings")
    if not isinstance(poses, dict):
        raise KeyError("Missing pose_encodings.")
    matches = [
        value
        for name, value in poses.items()
        if "learned_pose_control" in str(name)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one learned pose control, found {len(matches)}."
        )
    return matches[0].float()


def _squeeze_saved(value: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise TypeError("Saved prediction value is not a tensor.")
    return value[0] if value.ndim >= 1 and value.shape[0] == 1 else value


def _geometry_batch(payload: dict, device: str) -> dict:
    output = {
        "clip_name": payload["clip_name"],
        "scene_id": payload["scene_id"],
        "frame_indices": payload["frame_indices"],
        "reference_sequence_index": payload["reference_sequence_index"],
        "image_size": payload["image_size"],
        "point_alignment_scale": payload["point_alignment_scale"],
    }
    for name in (
        "baseline_pose_encoding",
        "baseline_depth",
        "baseline_world_points",
        "target_world_points",
        "target_depth",
        "tracking_masks_stream",
    ):
        output[name] = payload[name][None].to(device)
    for name in (
        "point_alignment_rotation",
        "point_alignment_translation",
    ):
        output[name] = payload[name].to(device)
    return output


def _ray_batch(
    payload: dict,
    *,
    clip,
    strict_identity_gate: bool,
    device: str,
) -> dict:
    return {
        "image_size": payload["image_size"],
        "frame_indices": payload["frame_indices"],
        "reference_sequence_index": payload["reference_sequence_index"],
        "instance_ids": tuple(int(value) for value in clip.instance_ids),
        "strict_identity_gate": bool(strict_identity_gate),
        "baseline_world_points": payload["baseline_world_points"][
            None
        ].to(device),
        "baseline_world_confidence": payload[
            "baseline_world_confidence"
        ][None].to(device),
        "tracking_masks_stream": payload["tracking_masks_stream"][
            None
        ].to(device),
        "trusted_tracking_masks_stream": payload[
            "trusted_tracking_masks_stream"
        ][None].to(device),
        "trusted_instance_valid": payload["trusted_instance_valid"][
            None
        ].to(device),
    }


def point_blend_name(blend: float) -> str:
    percent = int(round(100.0 * float(blend)))
    if abs(float(blend) - percent / 100.0) > 1e-8:
        raise ValueError("Pointmap blend must be an integer percentage.")
    return f"fixed_pose_pointblend_a{percent:03d}"


def _select_adaptive_blend(
    support_ratio: float,
    *,
    threshold: float = ADAPTIVE_SUPPORT_THRESHOLD,
) -> float:
    if not 0.0 <= float(support_ratio) <= 1.0:
        raise ValueError("Adaptive support ratio must be in [0,1].")
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("Adaptive support threshold must be in [0,1].")
    return 1.0 if float(support_ratio) >= float(threshold) else 0.0


def _blend_pointmap(
    raw: torch.Tensor,
    learned: torch.Tensor,
    *,
    blend: float,
    reference_index: int,
) -> torch.Tensor:
    if raw.shape != learned.shape:
        raise ValueError("Raw and learned pointmaps must have equal shape.")
    valid = (
        torch.isfinite(raw).all(dim=-1)
        & torch.isfinite(learned).all(dim=-1)
    )
    value = torch.where(
        valid[..., None],
        raw + float(blend) * (learned - raw),
        raw,
    )
    value[int(reference_index)] = raw[int(reference_index)]
    return value


def _blend_confidence(
    raw: torch.Tensor,
    learned: torch.Tensor,
    *,
    blend: float,
    reference_index: int,
) -> torch.Tensor:
    if raw.ndim == 4 and raw.shape[-1] == 1:
        raw = raw[..., 0]
    if learned.ndim == 4 and learned.shape[-1] == 1:
        learned = learned[..., 0]
    if raw.shape != learned.shape:
        raise ValueError("Raw and learned confidence maps must have equal shape.")
    learned = torch.where(torch.isfinite(learned), learned, raw)
    value = raw + float(blend) * (learned - raw)
    value[int(reference_index)] = raw[int(reference_index)]
    return value


def _append_metrics(
    pose_summary: list[dict],
    pose_frames: list[dict],
    pose_rpe: list[dict],
    pose_pairs: list[dict],
    pose_pair_summary: list[dict],
    pointmap_frames: list[dict],
    depth_frames: list[dict],
    *,
    payload: dict,
    batch: dict,
    pose: torch.Tensor,
    points: torch.Tensor,
    depth: torch.Tensor,
    name: str,
    sequence_indices: list[int],
    metadata: dict[str, object],
) -> None:
    _append_pose_metrics(
        pose_summary,
        pose_frames,
        pose_rpe,
        pose_pairs,
        pose_pair_summary,
        payload=payload,
        pose_encoding=pose,
        mode="joint_pointmap_ba",
        perturbation=name,
        sequence_indices=sequence_indices,
        evaluation_metadata=metadata,
    )
    append_geometry_metrics(
        pointmap_frames,
        depth_frames,
        batch=batch,
        outputs={
            "pose_encoding": pose,
            "world_points": points,
            "depth": depth,
        },
        mode="joint_pointmap_ba",
        perturbation=name,
        sequence_indices=sequence_indices,
        evaluation_metadata=metadata,
    )


def _compact_summary(
    pose_rows: list[dict],
    pointmap_rows: list[dict],
    diagnostic_rows: list[dict],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    clips = sorted({str(row["clip"]) for row in pose_rows})
    order = (
        CONTROL_RAW,
        CONTROL_FIXED,
        *(point_blend_name(value) for value in POINT_BLEND_VALUES),
        ADAPTIVE_FINAL,
    )
    for clip in clips:
        raw_pose = _one(
            pose_rows,
            clip=clip,
            perturbation=CONTROL_RAW,
        )
        raw_point = _pointmap_value(
            pointmap_rows,
            clip=clip,
            variant=CONTROL_RAW,
        )
        for variant in order:
            pose = _one(
                pose_rows,
                clip=clip,
                perturbation=variant,
            )
            point = _pointmap_value(
                pointmap_rows,
                clip=clip,
                variant=variant,
            )
            diagnostics = _one(
                diagnostic_rows,
                clip=clip,
                variant=variant,
            )
            row = {
                "split": pose.get("evaluation_protocol", ""),
                "clip": clip,
                "variant": variant,
                "point_blend": diagnostics.get(
                    "point_blend",
                    "",
                ),
                "pose_pointmap_coupled": diagnostics.get(
                    "pose_pointmap_coupled",
                    "",
                ),
                "ate": float(pose["ate_rmse"]),
                "rotation_deg": float(
                    pose["rotation_error_mean_degrees"]
                ),
                "pointmap_mean": point,
                "ate_delta_from_raw": (
                    float(pose["ate_rmse"])
                    - float(raw_pose["ate_rmse"])
                ),
                "pointmap_delta_from_raw": point - raw_point,
                "fit_accepted": diagnostics.get(
                    "fit_accepted",
                    "",
                ),
                "support_ratio": diagnostics.get(
                    "support_ratio",
                    "",
                ),
                "mean_center_shift": diagnostics.get(
                    "mean_pose_center_shift_native",
                    "",
                ),
                "raw_reference_exact": diagnostics.get(
                    "raw_reference_exact",
                    "",
                ),
                "learned_reference_preserved": diagnostics.get(
                    "learned_reference_preserved",
                    "",
                ),
                "a100_pose_matches_fixed": diagnostics.get(
                    "a100_pose_matches_fixed",
                    "",
                ),
                "module_off_exact": diagnostics.get(
                    "module_off_exact",
                    "",
                ),
                "status": diagnostics.get("status", ""),
            }
            output.append(
                {
                    field: _compact_value(row.get(field, ""))
                    for field in SUMMARY_FIELDS
                }
            )
    return output


def _pointmap_value(
    rows: list[dict],
    *,
    clip: str,
    variant: str,
) -> float:
    row = _one(
        rows,
        clip=clip,
        perturbation=variant,
        spatial_region="full_scene",
        geometry_source="point_head",
        group="all_frames",
    )
    return float(row["mean_frame_paired_distance_mean"])


def _one(rows: list[dict], **criteria) -> dict:
    matches = [
        row
        for row in rows
        if all(str(row.get(key)) == str(value) for key, value in criteria.items())
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one row matching {criteria}, found {len(matches)}."
        )
    return matches[0]


def _compact_value(value: object) -> object:
    if isinstance(value, float):
        return format(value, ".8g")
    return value


def _torch_load(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, dict):
        raise TypeError(f"Expected dictionary in {path}.")
    return value


def _write_csv(
    path: Path,
    rows: list[dict],
    *,
    fieldnames: tuple[str, ...] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is not None:
        names = list(fieldnames)
    else:
        # Control rows intentionally contain fewer diagnostics than optimized
        # BA rows.  Build a stable union instead of assuming the first row
        # defines the complete schema.
        names = list(
            dict.fromkeys(
                key
                for row in rows
                for key in row
            )
        )
    with path.open("w", newline="", encoding="utf8") as handle:
        if not names:
            return
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
