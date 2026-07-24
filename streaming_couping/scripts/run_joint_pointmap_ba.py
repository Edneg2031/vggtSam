"""Run no-training shared-SE(3) pose and pointmap graph refinement."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from streaming_couping.src.config import load_config
from streaming_couping.src.learned_pose.cache import cache_path, load_feature_cache
from streaming_couping.src.learned_pose.config import load_learned_pose_config
from streaming_couping.src.learned_pose.geometry_metrics import (
    append_geometry_metrics,
    summarize_pointmap_metrics,
)
from streaming_couping.src.learned_pose.pipeline import (
    _append_pose_metrics,
    _evaluation_metadata,
    _evaluation_sequence_indices,
)
from streaming_couping.src.learned_pose.ray_pose import (
    reference_blend_pose_name,
)
from streaming_couping.src.learned_pose.shared_rigid_graph import (
    SHARED_RIGID_VARIANTS,
    SharedRigidConfig,
    run_shared_rigid_graph,
)
from vggtsam.utils.imports import maybe_add_repo_to_path


CONTROL_RAW = "raw_control"
CONTROL_FIXED = "fixed_ref_050_control"
SUMMARY_FIELDS = (
    "split",
    "clip",
    "variant",
    "joint_consistent",
    "ate",
    "rotation_deg",
    "pointmap_mean",
    "ate_delta_from_raw",
    "pointmap_delta_from_raw",
    "point_rmse_before",
    "point_rmse_after",
    "matches",
    "accepted_frames",
    "active_instance_edges",
    "rejected_instance_edges",
    "mean_center_shift",
    "reference_anchor_exact",
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
    print("shared SE3 graph does not run SAM3, StreamVGGT, or training")

    pose_summary: list[dict] = []
    pose_frames: list[dict] = []
    pose_rpe: list[dict] = []
    pose_pairs: list[dict] = []
    pose_pair_summary: list[dict] = []
    pointmap_frames: list[dict] = []
    depth_frames: list[dict] = []
    diagnostic_rows: list[dict] = []
    saved_predictions: dict[str, dict] = {}

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
        print(f"shared SE3 graph clip={clip.name}")
        payload = load_feature_cache(path)
        predicted = predictions[clip.name]
        raw_pose = _pose_prediction(predicted, "raw_baseline_control")
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

        for variant in SHARED_RIGID_VARIANTS:
            result = run_shared_rigid_graph(
                raw_pose_encoding=raw_pose.to(args.device),
                initial_pose_encoding=fixed_pose.to(args.device),
                raw_world_points=raw_points[None].to(args.device),
                learned_world_points=refined_points[None].to(args.device),
                raw_confidence=raw_confidence[None].to(args.device),
                learned_confidence=refined_confidence[None].to(args.device),
                token_levels=payload["token_levels"],
                patch_start_idx=int(payload["patch_start_idx"]),
                patch_shape=tuple(int(value) for value in payload["patch_shape"]),
                tracking_masks=payload["tracking_masks_stream"][None].to(
                    args.device
                ),
                trusted_tracking_masks=payload[
                    "trusted_tracking_masks_stream"
                ][None].to(args.device),
                trusted_instance_valid=payload["trusted_instance_valid"][
                    None
                ].to(args.device),
                image_size=tuple(int(value) for value in payload["image_size"]),
                reference_index=int(payload["reference_sequence_index"]),
                scene_scale=float(payload["scene_scale"]),
                variant=variant,
                config=SharedRigidConfig(),
            )
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
                pose=result.pose_encoding,
                points=result.world_points,
                depth=result.depth,
                name=result.name,
                sequence_indices=sequence_indices,
                metadata=metadata,
            )
            diagnostic_rows.append(
                {
                    "clip": clip.name,
                    "variant": result.name,
                    "joint_consistent": 1,
                    "module_off_exact": module_off_exact,
                    **result.diagnostics,
                }
            )
            clip_outputs[result.name] = {
                "pose_encoding": result.pose_encoding.detach().cpu(),
                "world_points": result.world_points.detach().cpu(),
                "depth": result.depth.detach().cpu(),
                "diagnostics": result.diagnostics,
            }
            del result
            if str(args.device).startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
        saved_predictions[clip.name] = {
            "scene_id": clip.scene_id,
            "frame_indices": list(clip.frame_indices),
            "reference_sequence_index": clip.reference_sequence_index,
            "variants": clip_outputs,
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="streaming_couping/configs/v5_ablation_suite.yaml",
    )
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--predictions", type=Path)
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
        *(variant.name for variant in SHARED_RIGID_VARIANTS),
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
                "joint_consistent": diagnostics.get(
                    "joint_consistent",
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
                "point_rmse_before": diagnostics.get(
                    "initial_point_rmse",
                    "",
                ),
                "point_rmse_after": diagnostics.get(
                    "final_point_rmse",
                    "",
                ),
                "matches": diagnostics.get("matches", ""),
                "accepted_frames": diagnostics.get(
                    "accepted_frames",
                    "",
                ),
                "active_instance_edges": diagnostics.get(
                    "active_instance_edges",
                    "",
                ),
                "rejected_instance_edges": diagnostics.get(
                    "rejected_instance_edges",
                    "",
                ),
                "mean_center_shift": diagnostics.get(
                    "mean_pose_center_shift_native",
                    "",
                ),
                "reference_anchor_exact": diagnostics.get(
                    "reference_anchor_exact",
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
