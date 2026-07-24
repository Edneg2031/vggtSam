"""Evaluate reference-preserving ray-pose policies with one frozen V5 model."""

from __future__ import annotations

import argparse
import csv
import math
import shutil
from dataclasses import replace
from pathlib import Path

from streaming_couping.src.learned_pose.cache import cache_path
from streaming_couping.src.learned_pose.config import (
    FINAL_MODE,
    LearnedPoseConfig,
    load_learned_pose_config,
)
from streaming_couping.src.learned_pose.pipeline import evaluate_final_method
from streaming_couping.src.learned_pose.ray_pose import (
    FINAL_RAY_POSE_NAME,
    reference_blend_pose_name,
)


SOURCE_VARIANT = "v5_residual_so3_union"
REFERENCE_BLENDS = (0.25, 0.50, 0.75, 1.00)
SUMMARY_FIELDS = (
    "split",
    "clip",
    "policy",
    "preserve_reference",
    "blend",
    "raw_ate",
    "learned_ate",
    "solver_ate",
    "raw_rotation_deg",
    "solver_rotation_deg",
    "raw_direct_pointmap_mean",
    "learned_direct_pointmap_mean",
    "solver_reposed_raw_pointmap_mean",
    "reposed_delta_from_raw",
    "fit_accepted",
    "mean_applied_center_shift_native",
    "module_off_exact",
)


def main() -> None:
    args = _parse_args()
    base = load_learned_pose_config(args.config)
    if args.training_device:
        base = replace(
            base,
            training=replace(base.training, device=args.training_device),
        )
    config = _sweep_config(base)
    _prepare_reused_assets(base, config, args.source_checkpoint)

    print(
        "evaluating one frozen checkpoint: "
        f"{SOURCE_VARIANT}; no cache build or training"
    )
    evaluate_final_method(config)

    rows = _build_joint_solver_summary(config)
    summary_path = base.output_dir / "joint_solver_sweep_summary.csv"
    _write_csv(summary_path, rows, SUMMARY_FIELDS)
    print(f"joint solver sweep summary: {summary_path}")
    with summary_path.open("r", encoding="utf8") as handle:
        print(handle.read().rstrip())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="streaming_couping/configs/v5_ablation_suite.yaml",
    )
    parser.add_argument("--training-device")
    parser.add_argument(
        "--source-checkpoint",
        type=Path,
        help=(
            "Optional trained v5_residual_so3_union checkpoint. By default "
            "it is read from the V5 suite output directory."
        ),
    )
    return parser.parse_args()


def _sweep_config(base: LearnedPoseConfig) -> LearnedPoseConfig:
    ray_pose = replace(
        base.evaluation.ray_pose,
        preserve_reference=False,
        blend=1.0,
        solver_modes=("current_refined",),
        reference_blend_values=REFERENCE_BLENDS,
    )
    return replace(
        base,
        output_dir=base.output_dir / "joint_solver_sweep",
        features=replace(base.features, rebuild=False),
        fusion=replace(
            base.fusion,
            unknown_camera_weight=0.25,
            pose_feature_mode="residual_only",
            rotation_update_mode="bounded_so3",
            spatial_attention_mode="union",
        ),
        evaluation=replace(
            base.evaluation,
            perturbations=("aligned", "module_off"),
            ray_pose=ray_pose,
        ),
    )


def _prepare_reused_assets(
    base: LearnedPoseConfig,
    config: LearnedPoseConfig,
    source_checkpoint: Path | None,
) -> None:
    missing_caches = [
        str(cache_path(config, clip))
        for clip in config.clips
        if not cache_path(config, clip).is_file()
    ]
    if missing_caches:
        joined = "\n  ".join(missing_caches)
        raise FileNotFoundError(
            "The V5 feature cache is incomplete; this command intentionally "
            f"does not rebuild it. Missing:\n  {joined}"
        )

    source = source_checkpoint or (
        base.output_dir
        / "variants"
        / SOURCE_VARIANT
        / "checkpoints"
        / FINAL_MODE
        / "checkpoint_best.pt"
    )
    if not source.is_file():
        raise FileNotFoundError(
            "Missing trained V5 residual checkpoint; run the V5 suite first "
            f"or pass --source-checkpoint. Expected: {source}"
        )
    destination = (
        config.output_dir
        / "checkpoints"
        / FINAL_MODE
        / "checkpoint_best.pt"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
    print(f"reused checkpoint: {source}")
    print(f"reused cache: {config.features.cache_dir}")


def _policy_specs() -> tuple[tuple[str, str, bool, float], ...]:
    anchored = tuple(
        (
            f"fixed_ref_{int(round(100.0 * blend)):03d}",
            reference_blend_pose_name(blend),
            True,
            blend,
        )
        for blend in REFERENCE_BLENDS
    )
    return (
        ("free_ref_100", FINAL_RAY_POSE_NAME, False, 1.0),
        *anchored,
    )


def _build_joint_solver_summary(
    config: LearnedPoseConfig,
) -> list[dict[str, object]]:
    evaluation = config.output_dir / "evaluation"
    ray_pose = _read_csv(evaluation / "ray_pose_summary.csv")
    ray_fit = _read_csv(evaluation / "ray_pose_fit_diagnostics.csv")
    pointmap = _read_csv(evaluation / "pointmap_summary.csv")
    equivalence = _read_csv(evaluation / "baseline_equivalence.csv")
    rows: list[dict[str, object]] = []

    for clip in config.clips:
        clip_ray = [row for row in ray_pose if row.get("clip") == clip.name]
        if not clip_ray:
            continue
        raw = _one(
            clip_ray,
            ray_pose_role="raw_baseline_control",
        )
        learned = _one(
            clip_ray,
            ray_pose_role="learned_pose_control",
        )
        evaluated_frames = {
            int(value)
            for value in raw.get("evaluated_frame_indices", "").split()
        }
        raw_pointmap = _pointmap_value(
            pointmap,
            clip=clip.name,
            mode="baseline",
            perturbation="module_off",
            geometry_source="point_head",
        )
        learned_pointmap = _pointmap_value(
            pointmap,
            clip=clip.name,
            mode=FINAL_MODE,
            perturbation="aligned",
            geometry_source="point_head",
        )
        exact_rows = [
            row
            for row in equivalence
            if row.get("clip") == clip.name
            and row.get("mode") == FINAL_MODE
        ]
        module_off_exact = (
            min(int(row["strict_equal"]) for row in exact_rows)
            if exact_rows
            else 0
        )

        for policy, perturbation, preserve_reference, blend in _policy_specs():
            solver = _one(clip_ray, perturbation=perturbation)
            solver_pointmap = _pointmap_value(
                pointmap,
                clip=clip.name,
                mode="ray_pose_raw_geometry",
                perturbation=perturbation,
                geometry_source="baseline_point_head_refined_pose",
            )
            diagnostics = [
                row
                for row in ray_fit
                if row.get("clip") == clip.name
                and row.get("perturbation") == perturbation
                and int(row["frame_index"]) in evaluated_frames
            ]
            if len(diagnostics) != len(evaluated_frames):
                raise ValueError(
                    "Expected one solver diagnostic per evaluated frame for "
                    f"{clip.name}/{perturbation}; found {len(diagnostics)} "
                    f"for {len(evaluated_frames)} frames."
                )
            rows.append(
                {
                    "split": raw.get("evaluation_protocol", ""),
                    "clip": clip.name,
                    "policy": policy,
                    "preserve_reference": int(preserve_reference),
                    "blend": blend,
                    "raw_ate": _metric(raw, "ate_rmse"),
                    "learned_ate": _metric(learned, "ate_rmse"),
                    "solver_ate": _metric(solver, "ate_rmse"),
                    "raw_rotation_deg": _metric(
                        raw,
                        "rotation_error_mean_degrees",
                    ),
                    "solver_rotation_deg": _metric(
                        solver,
                        "rotation_error_mean_degrees",
                    ),
                    "raw_direct_pointmap_mean": raw_pointmap,
                    "learned_direct_pointmap_mean": learned_pointmap,
                    "solver_reposed_raw_pointmap_mean": solver_pointmap,
                    "reposed_delta_from_raw": solver_pointmap - raw_pointmap,
                    "fit_accepted": sum(
                        int(row["fit_accepted"]) for row in diagnostics
                    ),
                    "mean_applied_center_shift_native": _finite_mean(
                        float(row["applied_center_shift_native"])
                        for row in diagnostics
                    ),
                    "module_off_exact": module_off_exact,
                }
            )
    return [
        {
            field: _compact_value(row.get(field, ""))
            for field in SUMMARY_FIELDS
        }
        for row in rows
    ]


def _one(rows: list[dict[str, str]], **criteria: str) -> dict[str, str]:
    matches = [
        row
        for row in rows
        if all(row.get(key) == value for key, value in criteria.items())
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one CSV row matching {criteria}, found {len(matches)}."
        )
    return matches[0]


def _pointmap_value(
    rows: list[dict[str, str]],
    *,
    clip: str,
    mode: str,
    perturbation: str,
    geometry_source: str,
) -> float:
    row = _one(
        rows,
        clip=clip,
        mode=mode,
        perturbation=perturbation,
        spatial_region="full_scene",
        geometry_source=geometry_source,
        group="all_frames",
    )
    return float(row["mean_frame_paired_distance_mean"])


def _metric(row: dict[str, str], name: str) -> float:
    return float(row[name])


def _finite_mean(values) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return sum(finite) / len(finite) if finite else float("nan")


def _compact_value(value: object) -> object:
    if isinstance(value, float):
        return format(value, ".8g")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing solver sweep CSV: {path}")
    with path.open("r", newline="", encoding="utf8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(
    path: Path,
    rows: list[dict[str, object]],
    fieldnames: tuple[str, ...],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
