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
    cache_all_token_levels: bool = True
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
    modes: tuple[str, ...] = (
        "camera_geometry_only",
        "camera_sam_only",
        "camera_token_fusion",
        "all_token_fusion",
    )
    epochs: int = 20
    repeats_per_epoch: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    seed: int = 0
    device: str = "cuda:1"
    amp: bool = False


@dataclass(frozen=True)
class EvaluationConfig:
    perturbations: tuple[str, ...] = (
        "aligned",
        "module_off",
        "zero_appearance",
        "zero_geometry",
        "shuffle_instance_ids",
        "shuffle_time",
    )
    strict_equivalence: bool = True


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


VALID_MODES = {
    "baseline",
    "camera_geometry_only",
    "camera_sam_only",
    "camera_token_fusion",
    "all_token_fusion",
    "patch_sam_only",
    "patch_sam_geometry_strict",
    "patch_sam_geometry_tracker_gate",
    "decoupled_dual_branch",
}

VALID_PERTURBATIONS = {
    "aligned",
    "module_off",
    "zero_appearance",
    "zero_geometry",
    "shuffle_instance_ids",
    "shuffle_time",
    "pose_branch_off",
    "geometry_branch_off",
}

POSE_MODES = frozenset(
    {
        "camera_geometry_only",
        "camera_sam_only",
        "camera_token_fusion",
        "all_token_fusion",
        "decoupled_dual_branch",
    }
)

PATCH_MODES = frozenset(
    {
        "patch_sam_only",
        "patch_sam_geometry_strict",
        "patch_sam_geometry_tracker_gate",
    }
)

GEOMETRY_MODES = frozenset(
    {
        "all_token_fusion",
        "decoupled_dual_branch",
        *PATCH_MODES,
    }
)

V2_MODES = frozenset({"decoupled_dual_branch", *PATCH_MODES})


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
            cache_all_token_levels=bool(features.get("cache_all_token_levels", True)),
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
            modes=tuple(str(v) for v in training.get("modes", list(TrainingConfig().modes))),
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
    bad_modes = sorted(set(config.training.modes) - VALID_MODES)
    if bad_modes:
        raise ValueError(f"Unknown training modes: {bad_modes}")
    if "baseline" in config.training.modes:
        raise ValueError("baseline has no trainable parameters and must not be in training.modes.")
    bad_perturbations = sorted(set(config.evaluation.perturbations) - VALID_PERTURBATIONS)
    if bad_perturbations:
        raise ValueError(f"Unknown evaluation perturbations: {bad_perturbations}")
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
        if any(b <= a for a, b in zip(clip.frame_indices, clip.frame_indices[1:])):
            raise ValueError(f"Clip {clip.name!r} frame_indices must be strictly increasing.")
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
            if max(clip.training_frame_indices) >= min(
                clip.evaluation_frame_indices
            ):
                raise ValueError(
                    f"Clip {clip.name!r} temporal holdout requires every training "
                    "frame to precede every evaluation frame."
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
