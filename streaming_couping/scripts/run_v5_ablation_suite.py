"""Train and evaluate the compact V5 instance-guided ablation suite."""

from __future__ import annotations

import argparse
import csv
import gc
from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path

import torch

from streaming_couping.src.learned_pose.cache import build_feature_caches
from streaming_couping.src.learned_pose.config import (
    FINAL_MODE,
    LearnedPoseConfig,
    load_learned_pose_config,
)
from streaming_couping.src.learned_pose.pipeline import (
    evaluate_final_method,
    train_final_adapter,
)
from streaming_couping.src.learned_pose.ray_pose import (
    CURRENT_RAW_POSE_NAME,
    FINAL_RAY_POSE_NAME,
    HISTORICAL_ANCHOR_POSE_NAME,
)


@dataclass(frozen=True)
class Variant:
    name: str
    unknown_camera_weight: float
    pose_feature_mode: str
    rotation_update_mode: str
    spatial_attention_mode: str
    final_candidate: bool = False


VARIANTS = (
    Variant(
        "v4_match_additive_union",
        0.0,
        "appearance_only",
        "additive_encoding",
        "union",
    ),
    Variant(
        "v5_unknown_additive_union",
        0.25,
        "appearance_only",
        "additive_encoding",
        "union",
    ),
    Variant(
        "v5_unknown_so3_union",
        0.25,
        "appearance_only",
        "bounded_so3",
        "union",
    ),
    Variant(
        "v5_residual_so3_union",
        0.25,
        "residual_only",
        "bounded_so3",
        "union",
    ),
    Variant(
        "v5_combined_so3_union",
        0.25,
        "appearance_and_residual",
        "bounded_so3",
        "union",
    ),
    Variant(
        "v5_combined_so3_per_instance",
        0.25,
        "appearance_and_residual",
        "bounded_so3",
        "per_instance",
        final_candidate=True,
    ),
)

SUMMARY_FIELDS = (
    "split",
    "clip",
    "variant",
    "raw_ate",
    "learned_ate",
    "current_raw_ate",
    "current_refined_ate",
    "historical_ate",
    "raw_rotation_deg",
    "learned_rotation_deg",
    "raw_pointmap_mean",
    "learned_pointmap_mean",
    "pointmap_delta",
    "learned_pose_reposed_raw_pointmap_mean",
    "current_refined_reposed_raw_pointmap_mean",
    "historical_reposed_raw_pointmap_mean",
    "background_delta",
    "memory_off_ate",
    "wrong_id_memory_ate",
    "spatial_shuffle_pointmap_mean",
    "module_off_exact",
    "match_count",
    "unknown_count",
    "mismatch_count",
    "current_raw_fit_accepted",
    "current_refined_fit_accepted",
    "historical_fit_accepted",
)


def main() -> None:
    args = _parse_args()
    base = load_learned_pose_config(args.config)
    base = _override_devices(base, args)
    selected = _select_variants(args.variants)
    root = base.output_dir
    root.mkdir(parents=True, exist_ok=True)
    _write_variant_manifest(root / "v5_variant_manifest.csv", selected)
    _write_split_audit(root / "v5_split_audit.csv", base)

    if args.dry_run:
        print(f"V5 dry run: {len(base.clips)} clips, {len(selected)} variants")
        for variant in selected:
            print(variant.name)
        return

    cache_config = replace(
        base,
        features=replace(
            base.features,
            rebuild=bool(args.rebuild_cache),
        ),
    )
    build_feature_caches(cache_config)
    base = replace(base, features=replace(base.features, rebuild=False))

    variant_configs: list[tuple[Variant, LearnedPoseConfig]] = []
    for variant in selected:
        current = _variant_config(base, variant)
        variant_configs.append((variant, current))
        evaluation = current.output_dir / "evaluation"
        complete = all(
            (evaluation / name).is_file()
            for name in (
                "ray_pose_summary.csv",
                "ray_pose_fit_diagnostics.csv",
                "pointmap_summary.csv",
                "baseline_equivalence.csv",
                "identity_gate_diagnostics.csv",
            )
        )
        if args.resume and complete:
            print(f"reusing completed V5 variant={variant.name}")
            continue
        checkpoint = (
            current.output_dir
            / "checkpoints"
            / FINAL_MODE
            / "checkpoint_best.pt"
        )
        if args.resume and checkpoint.is_file():
            print(f"reusing trained V5 checkpoint variant={variant.name}")
        else:
            print(f"V5 training variant={variant.name}")
            train_final_adapter(current)
        print(f"V5 evaluating variant={variant.name}")
        evaluate_final_method(current)
        _release_memory()

    rows = _build_upload_summary(variant_configs)
    summary_path = root / "v5_upload_summary.csv"
    _write_csv(summary_path, rows, fieldnames=SUMMARY_FIELDS)
    print(f"V5 upload summary: {summary_path}")
    with summary_path.open("r", encoding="utf8") as handle:
        print(handle.read().rstrip())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="streaming_couping/configs/v5_ablation_suite.yaml",
    )
    parser.add_argument("--sam3-device")
    parser.add_argument("--geometry-device")
    parser.add_argument("--training-device")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse variants whose complete evaluation CSV set already exists.",
    )
    parser.add_argument(
        "--variants",
        nargs="*",
        help="Optional subset of the six registered variant names.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _override_devices(
    config: LearnedPoseConfig,
    args: argparse.Namespace,
) -> LearnedPoseConfig:
    config = replace(
        config,
        sam3_device=args.sam3_device or config.sam3_device,
        geometry_device=args.geometry_device or config.geometry_device,
    )
    if args.training_device:
        config = replace(
            config,
            training=replace(
                config.training,
                device=args.training_device,
            ),
        )
    return config


def _select_variants(names: Iterable[str] | None) -> tuple[Variant, ...]:
    if not names:
        return VARIANTS
    lookup = {variant.name: variant for variant in VARIANTS}
    unknown = sorted(set(names) - set(lookup))
    if unknown:
        raise ValueError(f"Unknown V5 variants: {unknown}")
    return tuple(lookup[name] for name in names)


def _variant_config(
    base: LearnedPoseConfig,
    variant: Variant,
) -> LearnedPoseConfig:
    perturbations = (
        (
            "aligned",
            "module_off",
            "memory_off",
            "wrong_id_memory",
            "spatial_token_shuffle",
        )
        if variant.final_candidate
        else ("aligned", "module_off")
    )
    return replace(
        base,
        output_dir=base.output_dir / "variants" / variant.name,
        fusion=replace(
            base.fusion,
            unknown_camera_weight=variant.unknown_camera_weight,
            pose_feature_mode=variant.pose_feature_mode,
            rotation_update_mode=variant.rotation_update_mode,
            spatial_attention_mode=variant.spatial_attention_mode,
        ),
        evaluation=replace(
            base.evaluation,
            perturbations=perturbations,
        ),
    )


def _build_upload_summary(
    variants: Iterable[tuple[Variant, LearnedPoseConfig]],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for variant, config in variants:
        evaluation = config.output_dir / "evaluation"
        ray_pose = _read_csv(evaluation / "ray_pose_summary.csv")
        ray_fit = _read_csv(evaluation / "ray_pose_fit_diagnostics.csv")
        pointmap = _read_csv(evaluation / "pointmap_summary.csv")
        equivalence = _read_csv(evaluation / "baseline_equivalence.csv")
        identity = _read_csv(evaluation / "identity_gate_diagnostics.csv")
        pose = _read_csv(evaluation / "pose_summary.csv")
        for clip in config.clips:
            if not (
                clip.split.lower() in {"val", "validation", "test"}
                or clip.evaluation_frame_indices is not None
            ):
                continue
            current_ray = [
                row for row in ray_pose if row.get("clip") == clip.name
            ]
            raw = _one_by_role(current_ray, "raw_baseline_control")
            learned = _one_by_role(current_ray, "learned_pose_control")
            current_raw = _one_by_perturbation(
                current_ray,
                CURRENT_RAW_POSE_NAME,
            )
            current_refined = _one_by_perturbation(
                current_ray,
                FINAL_RAY_POSE_NAME,
            )
            historical = _one_by_perturbation(
                current_ray,
                HISTORICAL_ANCHOR_POSE_NAME,
            )
            raw_pointmap = _pointmap_value(
                pointmap,
                clip=clip.name,
                mode="baseline",
                perturbation="module_off",
                spatial_region="full_scene",
            )
            learned_pointmap = _pointmap_value(
                pointmap,
                clip=clip.name,
                mode=FINAL_MODE,
                perturbation="aligned",
                spatial_region="full_scene",
            )
            learned_pose_reposed = _pointmap_value(
                pointmap,
                clip=clip.name,
                mode=FINAL_MODE,
                perturbation="aligned",
                spatial_region="full_scene",
                geometry_source="baseline_point_head_refined_pose",
            )
            current_refined_reposed = _pointmap_value(
                pointmap,
                clip=clip.name,
                mode="ray_pose_raw_geometry",
                perturbation=FINAL_RAY_POSE_NAME,
                spatial_region="full_scene",
                geometry_source="baseline_point_head_refined_pose",
            )
            historical_reposed = _pointmap_value(
                pointmap,
                clip=clip.name,
                mode="ray_pose_raw_geometry",
                perturbation=HISTORICAL_ANCHOR_POSE_NAME,
                spatial_region="full_scene",
                geometry_source="baseline_point_head_refined_pose",
            )
            raw_background = _pointmap_value(
                pointmap,
                clip=clip.name,
                mode="baseline",
                perturbation="module_off",
                spatial_region="background",
            )
            learned_background = _pointmap_value(
                pointmap,
                clip=clip.name,
                mode=FINAL_MODE,
                perturbation="aligned",
                spatial_region="background",
            )
            evaluated = {
                int(value)
                for value in str(
                    raw.get("evaluated_frame_indices", "")
                ).split()
            }
            states = _identity_counts(
                identity,
                clip=clip.name,
                evaluated_frames=evaluated,
            )
            exact_rows = [
                row
                for row in equivalence
                if row.get("clip") == clip.name
                and row.get("mode") == FINAL_MODE
            ]
            exact = (
                min(int(row["strict_equal"]) for row in exact_rows)
                if exact_rows
                else 0
            )
            row: dict[str, object] = {
                "split": raw.get("evaluation_protocol", ""),
                "clip": clip.name,
                "variant": variant.name,
                "raw_ate": _metric(raw, "ate_rmse"),
                "learned_ate": _metric(learned, "ate_rmse"),
                "current_raw_ate": _metric(current_raw, "ate_rmse"),
                "current_refined_ate": _metric(
                    current_refined,
                    "ate_rmse",
                ),
                "historical_ate": _metric(historical, "ate_rmse"),
                "raw_rotation_deg": _metric(
                    raw,
                    "rotation_error_mean_degrees",
                ),
                "learned_rotation_deg": _metric(
                    learned,
                    "rotation_error_mean_degrees",
                ),
                "raw_pointmap_mean": raw_pointmap,
                "learned_pointmap_mean": learned_pointmap,
                "pointmap_delta": _difference(
                    learned_pointmap,
                    raw_pointmap,
                ),
                "learned_pose_reposed_raw_pointmap_mean": (
                    learned_pose_reposed
                ),
                "current_refined_reposed_raw_pointmap_mean": (
                    current_refined_reposed
                ),
                "historical_reposed_raw_pointmap_mean": (
                    historical_reposed
                ),
                "background_delta": _difference(
                    learned_background,
                    raw_background,
                ),
                "memory_off_ate": "",
                "wrong_id_memory_ate": "",
                "spatial_shuffle_pointmap_mean": "",
                "module_off_exact": exact,
                **states,
                "current_raw_fit_accepted": _accepted_fits(
                    ray_fit,
                    clip=clip.name,
                    perturbation=CURRENT_RAW_POSE_NAME,
                    evaluated_frames=evaluated,
                ),
                "current_refined_fit_accepted": _accepted_fits(
                    ray_fit,
                    clip=clip.name,
                    perturbation=FINAL_RAY_POSE_NAME,
                    evaluated_frames=evaluated,
                ),
                "historical_fit_accepted": _accepted_fits(
                    ray_fit,
                    clip=clip.name,
                    perturbation=HISTORICAL_ANCHOR_POSE_NAME,
                    evaluated_frames=evaluated,
                ),
            }
            if variant.final_candidate:
                row["memory_off_ate"] = _pose_value(
                    pose,
                    clip=clip.name,
                    perturbation="memory_off",
                )
                row["wrong_id_memory_ate"] = _pose_value(
                    pose,
                    clip=clip.name,
                    perturbation="wrong_id_memory",
                )
                row["spatial_shuffle_pointmap_mean"] = _pointmap_value(
                    pointmap,
                    clip=clip.name,
                    mode=FINAL_MODE,
                    perturbation="spatial_token_shuffle",
                    spatial_region="full_scene",
                )
            output.append(
                {
                    name: _compact_value(row.get(name, ""))
                    for name in SUMMARY_FIELDS
                }
            )
    return output


def _one_by_role(rows: list[dict[str, str]], role: str) -> dict[str, str]:
    matches = [row for row in rows if row.get("ray_pose_role") == role]
    if len(matches) != 1:
        raise ValueError(f"Expected one ray-pose role={role}, found {len(matches)}.")
    return matches[0]


def _one_by_perturbation(
    rows: list[dict[str, str]],
    perturbation: str,
) -> dict[str, str]:
    matches = [
        row for row in rows if row.get("perturbation") == perturbation
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one ray-pose perturbation={perturbation}, "
            f"found {len(matches)}."
        )
    return matches[0]


def _pointmap_value(
    rows: list[dict[str, str]],
    *,
    clip: str,
    mode: str,
    perturbation: str,
    spatial_region: str,
    geometry_source: str = "point_head",
) -> float:
    matches = [
        row
        for row in rows
        if row.get("clip") == clip
        and row.get("mode") == mode
        and row.get("perturbation") == perturbation
        and row.get("spatial_region") == spatial_region
        and row.get("geometry_source") == geometry_source
        and row.get("group") == "all_frames"
    ]
    if len(matches) != 1:
        raise ValueError(
            "Expected one pointmap summary for "
            f"{clip}/{mode}/{perturbation}/{spatial_region}, "
            f"found {len(matches)}."
        )
    return float(matches[0]["mean_frame_paired_distance_mean"])


def _pose_value(
    rows: list[dict[str, str]],
    *,
    clip: str,
    perturbation: str,
) -> float:
    matches = [
        row
        for row in rows
        if row.get("clip") == clip
        and row.get("mode") == FINAL_MODE
        and row.get("perturbation") == perturbation
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one pose row for {clip}/{perturbation}, "
            f"found {len(matches)}."
        )
    return float(matches[0]["ate_rmse"])


def _identity_counts(
    rows: list[dict[str, str]],
    *,
    clip: str,
    evaluated_frames: set[int],
) -> dict[str, int]:
    selected = [
        row
        for row in rows
        if row.get("clip") == clip
        and (
            not evaluated_frames
            or int(row["frame_index"]) in evaluated_frames
        )
    ]
    counts = {"MATCH": 0, "UNKNOWN": 0, "MISMATCH": 0}
    for row in selected:
        state = row.get("identity_state", "")
        if state in counts:
            counts[state] += 1
    return {
        "match_count": counts["MATCH"],
        "unknown_count": counts["UNKNOWN"],
        "mismatch_count": counts["MISMATCH"],
    }


def _accepted_fits(
    rows: list[dict[str, str]],
    *,
    clip: str,
    perturbation: str,
    evaluated_frames: set[int],
) -> int:
    return sum(
        int(row["fit_accepted"])
        for row in rows
        if row.get("clip") == clip
        and row.get("perturbation") == perturbation
        and (
            not evaluated_frames
            or int(row["frame_index"]) in evaluated_frames
        )
    )


def _metric(row: dict[str, str], name: str) -> float:
    return float(row[name])


def _difference(first: float, second: float) -> float:
    return float(first) - float(second)


def _compact_value(value: object) -> object:
    if isinstance(value, float):
        return format(value, ".8g")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing V5 evaluation CSV: {path}")
    with path.open("r", newline="", encoding="utf8") as handle:
        return list(csv.DictReader(handle))


def _write_variant_manifest(
    path: Path,
    variants: Iterable[Variant],
) -> None:
    rows = [
        {
            "variant": variant.name,
            "unknown_camera_weight": variant.unknown_camera_weight,
            "pose_feature_mode": variant.pose_feature_mode,
            "rotation_update_mode": variant.rotation_update_mode,
            "spatial_attention_mode": variant.spatial_attention_mode,
            "independent_training": 1,
            "final_candidate": int(variant.final_candidate),
        }
        for variant in variants
    ]
    _write_csv(path, rows)


def _write_split_audit(
    path: Path,
    config: LearnedPoseConfig,
) -> None:
    train_clips = [
        clip for clip in config.clips if clip.split.lower() == "train"
    ]
    evaluation_clips = [
        clip
        for clip in config.clips
        if clip.split.lower() in {"val", "validation", "test"}
        or clip.evaluation_frame_indices is not None
    ]
    rows = []
    for train in train_clips:
        training_frames = set(
            train.training_frame_indices or train.frame_indices
        )
        for evaluation in evaluation_clips:
            evaluated_frames = set(
                evaluation.evaluation_frame_indices
                or evaluation.frame_indices
            )
            overlap = sorted(training_frames & evaluated_frames)
            if overlap:
                raise ValueError(
                    "V5 split leakage between "
                    f"{train.name} and {evaluation.name}: {overlap}"
                )
            distances = [
                abs(int(left) - int(right))
                for left in training_frames
                for right in evaluated_frames
            ]
            rows.append(
                {
                    "train_clip": train.name,
                    "evaluation_clip": evaluation.name,
                    "training_frames": " ".join(
                        str(value) for value in sorted(training_frames)
                    ),
                    "evaluated_frames": " ".join(
                        str(value) for value in sorted(evaluated_frames)
                    ),
                    "intersection": "",
                    "minimum_numeric_frame_distance": min(distances),
                }
            )
    _write_csv(path, rows)


def _write_csv(
    path: Path,
    rows: Iterable[dict[str, object]],
    *,
    fieldnames: Iterable[str] | None = None,
) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(fieldnames or (rows[0].keys() if rows else ()))
    with path.open("w", newline="", encoding="utf8") as handle:
        if not names:
            return
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def _release_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
