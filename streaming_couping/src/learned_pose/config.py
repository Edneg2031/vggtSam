"""Configuration for persistent-instance StreamVGGT pose refinement."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ClipConfig:
    name: str
    scene_id: str
    frame_indices: tuple[int, ...]
    instance_ids: tuple[int, ...]
    split: str = "train"
    reference_sequence_index: int = 0
    tracking_cache: Path | None = None
    training_frame_indices: tuple[int, ...] | None = None
    evaluation_frame_indices: tuple[int, ...] | None = None


@dataclass(frozen=True)
class FeatureConfig:
    cache_dir: Path
    rebuild: bool = False
    sam_source: str = "detector_fpn2"
    sam_resolution: int = 1008
    sam_grid: tuple[int, int] = (72, 72)
    point_confidence_threshold: float = 0.30
    min_instance_points: int = 128
    sampled_instance_points: int = 128


@dataclass(frozen=True)
class FusionConfig:
    instance_dim: int = 512
    attention_dim: int = 512
    num_heads: int = 8
    dropout: float = 0.0
    memory_momentum: float = 0.90
    min_track_confidence: float = 0.50
    min_geometry_confidence: float = 0.20
    min_static_score: float = 0.20
    dpt_layer_indices: tuple[int, ...] = (4, 11, 17, 23)


@dataclass(frozen=True)
class LossConfig:
    camera: float = 20.0
    relative_rotation: float = 2.0
    translation_direction: float = 2.0
    rigid: float = 1.0
    centroid: float = 0.25
    residual: float = 1e-4
    depth: float = 10.0
    depth_fixed: float = 0.0
    pointmap: float = 5.0
    rigid_trim_quantile: float = 0.70


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 20
    repeats_per_epoch: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    seed: int = 0
    device: str = "cuda:1"
    amp: bool = False


@dataclass(frozen=True)
class RayPoseConfig:
    """Deployable pointmap-to-camera translation recovery at evaluation time."""

    confidence_threshold: float = 0.30
    min_points: int = 1024
    max_points: int = 65536
    max_iterations: int = 6
    max_condition_number: float = 1e8
    max_center_shift: float = 0.75
    max_residual_rmse: float = 0.20
    blend: float = 1.0
    angular_huber_delta: float = 0.02
    angular_min_range: float = 0.05
    preserve_reference: bool = True
    export_confidence_threshold: float = 0.30
    export_max_full_scene_points: int = 1_000_000
    export_max_instance_points: int = 250_000


@dataclass(frozen=True)
class EvaluationConfig:
    perturbations: tuple[str, ...] = (
        "aligned",
        "module_off",
    )
    strict_equivalence: bool = True
    ray_pose: RayPoseConfig = RayPoseConfig()


@dataclass(frozen=True)
class LearnedPoseConfig:
    source_path: Path
    recovery_config: Path
    manifest: Path
    output_dir: Path
    sam3_device: str
    geometry_device: str
    clips: tuple[ClipConfig, ...]
    features: FeatureConfig
    fusion: FusionConfig
    loss: LossConfig
    training: TrainingConfig
    evaluation: EvaluationConfig


FINAL_MODE = "decoupled_dual_branch"

VALID_PERTURBATIONS = {
    "aligned",
    "module_off",
}


def load_learned_pose_config(path: str | Path) -> LearnedPoseConfig:
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf8") as handle:
        raw = yaml.safe_load(handle) or {}

    dataset = raw.get("dataset", {})
    features = raw.get("features", {})
    fusion = raw.get("fusion", {})
    loss = raw.get("loss", {})
    training = raw.get("training", {})
    evaluation = raw.get("evaluation", {})
    ray_pose = evaluation.get("ray_pose", {})
    runtime = raw.get("runtime", {})

    clips = tuple(_parse_clip(value, source.parent) for value in dataset.get("clips", []))
    if not clips:
        raise ValueError("dataset.clips must contain at least one clip.")

    config = LearnedPoseConfig(
        source_path=source,
        recovery_config=_path(raw.get("recovery_config"), source.parent),
        manifest=_path(dataset.get("manifest"), source.parent),
        output_dir=_path(raw.get("output_dir"), source.parent),
        sam3_device=str(runtime.get("sam3_device", "cuda:3")),
        geometry_device=str(runtime.get("geometry_device", "cuda:1")),
        clips=clips,
        features=FeatureConfig(
            cache_dir=_path(features.get("cache_dir"), source.parent),
            rebuild=bool(features.get("rebuild", False)),
            sam_source=str(features.get("sam_source", "detector_fpn2")),
            sam_resolution=int(features.get("sam_resolution", 1008)),
            sam_grid=_pair(features.get("sam_grid", [72, 72]), "features.sam_grid"),
            point_confidence_threshold=float(features.get("point_confidence_threshold", 0.30)),
            min_instance_points=int(features.get("min_instance_points", 128)),
            sampled_instance_points=int(features.get("sampled_instance_points", 128)),
        ),
        fusion=FusionConfig(
            instance_dim=int(fusion.get("instance_dim", 512)),
            attention_dim=int(fusion.get("attention_dim", 512)),
            num_heads=int(fusion.get("num_heads", 8)),
            dropout=float(fusion.get("dropout", 0.0)),
            memory_momentum=float(fusion.get("memory_momentum", 0.90)),
            min_track_confidence=float(fusion.get("min_track_confidence", 0.50)),
            min_geometry_confidence=float(fusion.get("min_geometry_confidence", 0.20)),
            min_static_score=float(fusion.get("min_static_score", 0.20)),
            dpt_layer_indices=tuple(int(v) for v in fusion.get("dpt_layer_indices", [4, 11, 17, 23])),
        ),
        loss=LossConfig(
            camera=float(loss.get("camera", 20.0)),
            relative_rotation=float(loss.get("relative_rotation", 2.0)),
            translation_direction=float(loss.get("translation_direction", 2.0)),
            rigid=float(loss.get("rigid", 1.0)),
            centroid=float(loss.get("centroid", 0.25)),
            residual=float(loss.get("residual", 1e-4)),
            depth=float(loss.get("depth", 10.0)),
            depth_fixed=float(loss.get("depth_fixed", 0.0)),
            pointmap=float(loss.get("pointmap", 5.0)),
            rigid_trim_quantile=float(loss.get("rigid_trim_quantile", 0.70)),
        ),
        training=TrainingConfig(
            epochs=int(training.get("epochs", 20)),
            repeats_per_epoch=int(training.get("repeats_per_epoch", 8)),
            learning_rate=float(training.get("learning_rate", 1e-4)),
            weight_decay=float(training.get("weight_decay", 1e-4)),
            grad_clip_norm=float(training.get("grad_clip_norm", 1.0)),
            seed=int(training.get("seed", 0)),
            device=str(training.get("device", runtime.get("geometry_device", "cuda:1"))),
            amp=bool(training.get("amp", False)),
        ),
        evaluation=EvaluationConfig(
            perturbations=tuple(
                str(v)
                for v in evaluation.get(
                    "perturbations",
                    list(EvaluationConfig().perturbations),
                )
            ),
            strict_equivalence=bool(evaluation.get("strict_equivalence", True)),
            ray_pose=RayPoseConfig(
                confidence_threshold=float(
                    ray_pose.get("confidence_threshold", 0.30)
                ),
                min_points=int(ray_pose.get("min_points", 1024)),
                max_points=int(ray_pose.get("max_points", 65536)),
                max_iterations=int(ray_pose.get("max_iterations", 6)),
                max_condition_number=float(
                    ray_pose.get("max_condition_number", 1e8)
                ),
                max_center_shift=float(
                    ray_pose.get("max_center_shift", 0.75)
                ),
                max_residual_rmse=float(
                    ray_pose.get("max_residual_rmse", 0.20)
                ),
                blend=float(ray_pose.get("blend", 1.0)),
                angular_huber_delta=float(
                    ray_pose.get("angular_huber_delta", 0.02)
                ),
                angular_min_range=float(
                    ray_pose.get("angular_min_range", 0.05)
                ),
                preserve_reference=bool(
                    ray_pose.get("preserve_reference", True)
                ),
                export_confidence_threshold=float(
                    ray_pose.get("export_confidence_threshold", 0.30)
                ),
                export_max_full_scene_points=int(
                    ray_pose.get("export_max_full_scene_points", 1_000_000)
                ),
                export_max_instance_points=int(
                    ray_pose.get("export_max_instance_points", 250_000)
                ),
            ),
        ),
    )
    _validate(config)
    return config


def _parse_clip(value: dict[str, Any], base: Path) -> ClipConfig:
    if not isinstance(value, dict):
        raise TypeError("Each dataset.clips item must be a mapping.")
    frames = tuple(int(v) for v in value.get("frame_indices", []))
    instances = tuple(int(v) for v in value.get("instance_ids", []))
    if not frames or not instances:
        raise ValueError("Each clip requires frame_indices and instance_ids.")
    cache_value = value.get("tracking_cache")
    return ClipConfig(
        name=str(value.get("name") or f"{value.get('scene_id')}_{frames[0]}_{frames[-1]}"),
        scene_id=str(value.get("scene_id")),
        frame_indices=frames,
        instance_ids=instances,
        split=str(value.get("split", "train")),
        reference_sequence_index=int(value.get("reference_sequence_index", 0)),
        tracking_cache=(_path(cache_value, base) if cache_value else None),
        training_frame_indices=_optional_int_tuple(
            value.get("training_frame_indices")
        ),
        evaluation_frame_indices=_optional_int_tuple(
            value.get("evaluation_frame_indices")
        ),
    )


def _validate(config: LearnedPoseConfig) -> None:
    bad_perturbations = sorted(set(config.evaluation.perturbations) - VALID_PERTURBATIONS)
    if bad_perturbations:
        raise ValueError(f"Unknown evaluation perturbations: {bad_perturbations}")
    ray_pose = config.evaluation.ray_pose
    if ray_pose.export_confidence_threshold < 0.0:
        raise ValueError("Ray-pose export confidence threshold must be nonnegative.")
    if ray_pose.export_max_full_scene_points < 1:
        raise ValueError("Ray-pose full-scene export limit must be positive.")
    if ray_pose.export_max_instance_points < 1:
        raise ValueError("Ray-pose instance export limit must be positive.")
    if not 0.0 <= ray_pose.confidence_threshold:
        raise ValueError("ray_pose.confidence_threshold must be non-negative.")
    if ray_pose.min_points < 3 or ray_pose.max_points < ray_pose.min_points:
        raise ValueError("ray_pose point limits are invalid.")
    if ray_pose.max_iterations < 1:
        raise ValueError("ray_pose.max_iterations must be positive.")
    if ray_pose.max_condition_number <= 1.0:
        raise ValueError("ray_pose.max_condition_number must be greater than 1.")
    if ray_pose.max_center_shift <= 0.0 or ray_pose.max_residual_rmse <= 0.0:
        raise ValueError("ray_pose shift/residual limits must be positive.")
    if not 0.0 <= ray_pose.blend <= 1.0:
        raise ValueError("ray_pose.blend must be in [0,1].")
    if ray_pose.angular_huber_delta <= 0.0 or ray_pose.angular_min_range <= 0.0:
        raise ValueError("ray_pose angular robust-fit values must be positive.")
    if config.fusion.attention_dim % config.fusion.num_heads:
        raise ValueError("fusion.attention_dim must be divisible by fusion.num_heads.")
    if config.fusion.dpt_layer_indices != (4, 11, 17, 23):
        raise ValueError(
            "Current StreamVGGT DPT heads require fusion.dpt_layer_indices "
            "to be exactly [4, 11, 17, 23]."
        )
    for name, value in (
        ("features.point_confidence_threshold", config.features.point_confidence_threshold),
        ("fusion.memory_momentum", config.fusion.memory_momentum),
        ("fusion.min_track_confidence", config.fusion.min_track_confidence),
        ("fusion.min_geometry_confidence", config.fusion.min_geometry_confidence),
        ("fusion.min_static_score", config.fusion.min_static_score),
        ("loss.rigid_trim_quantile", config.loss.rigid_trim_quantile),
    ):
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must be in [0, 1].")
    if config.features.sampled_instance_points < 8:
        raise ValueError("features.sampled_instance_points must be at least 8.")
    if config.training.epochs < 1 or config.training.repeats_per_epoch < 1:
        raise ValueError("training epochs and repeats_per_epoch must be positive.")
    for clip in config.clips:
        if clip.reference_sequence_index != 0:
            raise ValueError("Learned causal pose refinement requires reference_sequence_index=0.")
        # ``frame_indices`` defines the model's observation order.  A
        # deliberately hard view sequence may jump forwards and backwards in
        # the source video, so numeric frame order is not a validity
        # requirement.  Repeated frames remain invalid because they make
        # sequence-level metrics and tracking-cache identity ambiguous.
        if len(set(clip.frame_indices)) != len(clip.frame_indices):
            raise ValueError(f"Clip {clip.name!r} frame_indices contains duplicates.")
        frame_set = set(clip.frame_indices)
        for field, values in (
            ("training_frame_indices", clip.training_frame_indices),
            ("evaluation_frame_indices", clip.evaluation_frame_indices),
        ):
            if values is None:
                continue
            if not values:
                raise ValueError(f"Clip {clip.name!r} {field} must not be empty.")
            if len(set(values)) != len(values):
                raise ValueError(f"Clip {clip.name!r} {field} contains duplicates.")
            if any(value not in frame_set for value in values):
                raise ValueError(
                    f"Clip {clip.name!r} {field} must be a subset of frame_indices."
                )
            ordered = tuple(value for value in clip.frame_indices if value in set(values))
            if values != ordered:
                raise ValueError(
                    f"Clip {clip.name!r} {field} must follow frame_indices order."
                )
        if clip.training_frame_indices is not None:
            reference_frame = clip.frame_indices[clip.reference_sequence_index]
            if reference_frame not in clip.training_frame_indices:
                raise ValueError(
                    f"Clip {clip.name!r} training_frame_indices must include the "
                    f"reference frame {reference_frame}."
                )
        if (
            clip.split.lower() == "train"
            and clip.evaluation_frame_indices is not None
            and clip.training_frame_indices is None
        ):
            raise ValueError(
                f"Clip {clip.name!r} with an evaluation_frame_indices holdout "
                "must explicitly set training_frame_indices."
            )
        if (
            clip.training_frame_indices is not None
            and clip.evaluation_frame_indices is not None
        ):
            overlap = set(clip.training_frame_indices) & set(
                clip.evaluation_frame_indices
            )
            if overlap:
                raise ValueError(
                    f"Clip {clip.name!r} training/evaluation frames overlap: "
                    f"{sorted(overlap)}."
                )
            position = {
                frame_index: sequence_index
                for sequence_index, frame_index in enumerate(clip.frame_indices)
            }
            if max(position[value] for value in clip.training_frame_indices) >= min(
                position[value] for value in clip.evaluation_frame_indices
            ):
                raise ValueError(
                    f"Clip {clip.name!r} temporal holdout requires every training "
                    "observation to precede every evaluation observation in "
                    "frame_indices order."
                )


def _path(value: Any, base: Path) -> Path:
    if value is None or str(value).strip() == "":
        raise ValueError("A required path is missing from the learned-pose config.")
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    # Repository commands are run from the repository root. Prefer that
    # interpretation, then fall back to YAML-relative paths.
    cwd_path = (Path.cwd() / path).resolve()
    if cwd_path.exists() or not (base / path).exists():
        return cwd_path
    return (base / path).resolve()


def _pair(value: Any, field: str) -> tuple[int, int]:
    values = tuple(int(v) for v in value)
    if len(values) != 2 or min(values) <= 0:
        raise ValueError(f"{field} must contain two positive integers.")
    return values


def _optional_int_tuple(value: Any) -> tuple[int, ...] | None:
    if value is None:
        return None
    return tuple(int(item) for item in value)
