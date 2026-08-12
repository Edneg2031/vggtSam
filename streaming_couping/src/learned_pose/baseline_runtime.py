"""Runtime helpers for the retained raw-pose dynamic-tracking baseline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import yaml


@dataclass(frozen=True)
class BaselineRunConfig:
    source_path: Path
    version: str
    output_dir: Path
    clip_name: str
    audit_device: str
    evaluation_frames: tuple[int, ...]


@dataclass(frozen=True)
class TrackBACandidateConfig:
    """Locked deployable inputs for the V0 fixed-structure pose candidate."""

    enabled: bool
    output_dir: Path
    device: str
    track_token_source: str
    query_source: str
    primary_method: str
    methods: tuple[str, ...]
    window_frames: int
    query_count: int
    query_grid: tuple[int, int]
    track_iterations: int
    visibility_threshold: float
    track_confidence_threshold: float
    point_confidence_threshold: float
    min_correspondences: int
    optimizer_steps: int
    learning_rate: float
    robust_delta_pixels: float
    rotation_prior_weight: float
    translation_prior_weight: float
    temporal_prior_weight: float
    max_rotation_degrees: float
    max_translation_scene_fraction: float


@dataclass(frozen=True)
class FeaturePnPCandidateConfig:
    """Frozen local-feature 2D-3D pose candidate without learned fitting."""

    enabled: bool
    output_dir: Path
    primary_method: str
    methods: tuple[str, ...]
    primary_landmark_source: str
    landmark_sources: tuple[str, ...]
    primary_history_scope: str
    history_scopes: tuple[str, ...]
    anchor_lookback: int
    nfeatures: int
    contrast_threshold: float
    ratio_threshold: float
    max_correspondences: int
    min_correspondences: int
    min_inliers: int
    point_confidence_threshold: float
    ransac_iterations: int
    ransac_reprojection_pixels: float
    ransac_confidence: float
    max_rotation_degrees: float
    max_translation_scene_fraction: float


def load_baseline_run_config(path: str | Path) -> BaselineRunConfig:
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf8") as handle:
        raw = yaml.safe_load(handle) or {}
    section = raw.get("baseline", {})
    frames = section.get("frames", {})
    config = BaselineRunConfig(
        source_path=source,
        version=str(section.get("version", "")).strip().lower(),
        output_dir=_path(section.get("output_dir", "outputs/streaming_couping_v0")),
        clip_name=str(section.get("clip_name", "")),
        audit_device=str(
            section.get(
                "audit_device",
                section.get("training_device", "cuda:0"),
            )
        ),
        evaluation_frames=_int_tuple(frames.get("evaluation", ())),
    )
    _validate_config(config)
    return config


def load_track_ba_candidate_config(
    path: str | Path,
) -> TrackBACandidateConfig:
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf8") as handle:
        raw = yaml.safe_load(handle) or {}
    baseline = raw.get("baseline", {})
    section = baseline.get("track_ba_candidate", {})
    methods = tuple(
        str(value).strip().lower()
        for value in section.get(
            "methods",
            (
                "full_image",
                "sam_dynamic_excluded",
                "sam_instance_background_stratified",
                "bbox_instance_background_stratified",
                "random_instance_background_stratified",
            ),
        )
    )
    grid = _int_tuple(section.get("query_grid", (64, 64)))
    config = TrackBACandidateConfig(
        enabled=bool(section.get("enabled", True)),
        output_dir=_path(
            section.get(
                "output_dir",
                Path(baseline.get("output_dir", "outputs/streaming_couping_v0"))
                / "track_ba_candidate",
            )
        ),
        device=str(section.get("device", baseline.get("audit_device", "cuda:0"))),
        track_token_source=str(
            section.get("track_token_source", "window_reaggregated")
        ).strip().lower(),
        query_source=str(
            section.get("query_source", "shi_tomasi")
        ).strip().lower(),
        primary_method=str(
            section.get("primary_method", "sam_dynamic_excluded")
        ).strip().lower(),
        methods=methods,
        window_frames=int(section.get("window_frames", 5)),
        query_count=int(section.get("query_count", 256)),
        query_grid=(int(grid[0]), int(grid[1])) if len(grid) == 2 else grid,
        track_iterations=int(section.get("track_iterations", 4)),
        visibility_threshold=float(section.get("visibility_threshold", 0.05)),
        track_confidence_threshold=float(
            section.get("track_confidence_threshold", 0.05)
        ),
        point_confidence_threshold=float(
            section.get("point_confidence_threshold", 0.30)
        ),
        min_correspondences=int(section.get("min_correspondences", 32)),
        optimizer_steps=int(section.get("optimizer_steps", 120)),
        learning_rate=float(section.get("learning_rate", 0.03)),
        robust_delta_pixels=float(section.get("robust_delta_pixels", 4.0)),
        rotation_prior_weight=float(section.get("rotation_prior_weight", 0.50)),
        translation_prior_weight=float(
            section.get("translation_prior_weight", 0.50)
        ),
        temporal_prior_weight=float(section.get("temporal_prior_weight", 0.10)),
        max_rotation_degrees=float(section.get("max_rotation_degrees", 10.0)),
        max_translation_scene_fraction=float(
            section.get("max_translation_scene_fraction", 0.25)
        ),
    )
    _validate_track_ba_config(config)
    return config


def load_feature_pnp_candidate_config(
    path: str | Path,
) -> FeaturePnPCandidateConfig:
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf8") as handle:
        raw = yaml.safe_load(handle) or {}
    baseline = raw.get("baseline", {})
    section = baseline.get("feature_pnp_candidate", {})
    methods = tuple(
        str(value).strip().lower()
        for value in section.get(
            "methods",
            (
                "full_image",
                "sam_dynamic_excluded",
                "sam_instance_background_stratified",
                "bbox_instance_background_stratified",
                "random_instance_background_stratified",
            ),
        )
    )
    landmark_sources = tuple(
        str(value).strip().lower()
        for value in section.get(
            "landmark_sources",
            ("raw_depth_unprojected", "native_world_pointmap"),
        )
    )
    history_scopes = tuple(
        str(value).strip().lower()
        for value in section.get(
            "history_scopes",
            ("recent", "all_causal"),
        )
    )
    config = FeaturePnPCandidateConfig(
        enabled=bool(section.get("enabled", True)),
        output_dir=_path(
            section.get(
                "output_dir",
                Path(baseline.get("output_dir", "outputs/streaming_couping_v0"))
                / "feature_pnp_candidate",
            )
        ),
        primary_method=str(
            section.get("primary_method", "sam_dynamic_excluded")
        ).strip().lower(),
        methods=methods,
        primary_landmark_source=str(
            section.get("primary_landmark_source", "native_world_pointmap")
        ).strip().lower(),
        landmark_sources=landmark_sources,
        primary_history_scope=str(
            section.get("primary_history_scope", "all_causal")
        ).strip().lower(),
        history_scopes=history_scopes,
        anchor_lookback=int(section.get("anchor_lookback", 4)),
        nfeatures=int(section.get("nfeatures", 4096)),
        contrast_threshold=float(section.get("contrast_threshold", 0.01)),
        ratio_threshold=float(section.get("ratio_threshold", 0.75)),
        max_correspondences=int(section.get("max_correspondences", 256)),
        min_correspondences=int(section.get("min_correspondences", 32)),
        min_inliers=int(section.get("min_inliers", 16)),
        point_confidence_threshold=float(
            section.get("point_confidence_threshold", 0.30)
        ),
        ransac_iterations=int(section.get("ransac_iterations", 1000)),
        ransac_reprojection_pixels=float(
            section.get("ransac_reprojection_pixels", 4.0)
        ),
        ransac_confidence=float(section.get("ransac_confidence", 0.999)),
        max_rotation_degrees=float(section.get("max_rotation_degrees", 10.0)),
        max_translation_scene_fraction=float(
            section.get("max_translation_scene_fraction", 0.25)
        ),
    )
    _validate_feature_pnp_config(config)
    return config


def decode_cached_poses(
    payload: dict,
    *,
    pose_decoder,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    baseline_encoding = payload.get("baseline_pose_encoding")
    target_encoding = payload.get("target_pose_encoding")
    if not torch.is_tensor(baseline_encoding):
        raise ValueError("Baseline cache lacks tensor field 'baseline_pose_encoding'.")
    if not torch.is_tensor(target_encoding):
        raise ValueError("Baseline cache lacks tensor field 'target_pose_encoding'.")
    image_size = tuple(int(value) for value in payload["image_size"])
    baseline, _ = pose_decoder(
        baseline_encoding.unsqueeze(0).to(device),
        image_size_hw=image_size,
    )
    target, _ = pose_decoder(
        target_encoding.unsqueeze(0).to(device),
        image_size_hw=image_size,
    )
    return baseline, target


def pose_metrics(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    reference_index: int,
    evaluation_indices: list[int] | None,
) -> dict[str, float]:
    indices = evaluation_indices_or_default(
        predicted.shape[1],
        reference_index=reference_index,
        evaluation_indices=evaluation_indices,
    )
    index = torch.tensor(indices, dtype=torch.long, device=predicted.device)
    predicted_eval = predicted.index_select(1, index)
    target_eval = target.index_select(1, index)
    relative = (
        predicted_eval[..., :3, :3]
        @ target_eval[..., :3, :3].transpose(-1, -2)
    )
    cosine = (
        torch.diagonal(relative, dim1=-2, dim2=-1).sum(dim=-1) - 1.0
    ) * 0.5
    rotation = torch.rad2deg(torch.acos(cosine.clamp(-1, 1))).mean()
    center = torch.linalg.vector_norm(
        camera_centers(predicted_eval) - camera_centers(target_eval),
        dim=-1,
    ).mean()
    return {
        "rotation_degrees": float(rotation.cpu()),
        "center_error_native": float(center.cpu()),
    }


def evaluation_indices_or_default(
    sequence_length: int,
    *,
    reference_index: int,
    evaluation_indices: list[int] | None,
) -> list[int]:
    indices = (
        [index for index in range(sequence_length) if index != int(reference_index)]
        if evaluation_indices is None
        else [int(value) for value in evaluation_indices]
    )
    if not indices or int(reference_index) in indices:
        raise ValueError("Evaluation indices are empty or include the reference.")
    if len(indices) != len(set(indices)):
        raise ValueError("Evaluation indices contain duplicates.")
    if any(index < 0 or index >= sequence_length for index in indices):
        raise ValueError("Evaluation index is outside the sequence.")
    return indices


def camera_centers(world_to_camera: torch.Tensor) -> torch.Tensor:
    rotation = world_to_camera[..., :3, :3]
    translation = world_to_camera[..., :3, 3]
    return -(rotation.transpose(-1, -2) @ translation[..., None]).squeeze(-1)


def tracking_audit(
    payload: dict,
    *,
    reference_index: int,
) -> dict[str, object]:
    """Validate causal persistent-track registry invariants from a V0 cache."""

    frames = tuple(int(value) for value in payload.get("frame_indices", ()))
    births = tuple(int(value) for value in payload.get("sam_birth_indices", ()))
    geometry_births = tuple(
        int(value) for value in payload.get("instance_birth_indices", ())
    )
    instance_ids = tuple(int(value) for value in payload.get("instance_ids", ()))
    sam_track_ids = tuple(
        int(value) for value in payload.get("sam_track_ids", ())
    )
    sam_track_prompts = tuple(
        str(value) for value in payload.get("sam_track_prompts", ())
    )
    registry_shapes_match = (
        len(instance_ids) == len(births)
        and len(geometry_births) == len(births)
        and len(sam_track_ids) == len(births)
        and len(sam_track_prompts) == len(births)
    )
    rows = sorted(
        (dict(row) for row in payload.get("dynamic_instance_diagnostics", ())),
        key=lambda row: int(row["sequence_index"]),
    )
    rows_aligned = len(rows) == len(frames) and all(
        int(row["sequence_index"]) == index
        and int(row["frame_index"]) == frames[index]
        for index, row in enumerate(rows)
    )
    expected_discovered = tuple(
        sum(birth >= 0 and birth <= index for birth in births)
        for index in range(len(frames))
    )
    reported_discovered = tuple(
        int(row.get("discovered_tracks", -1)) for row in rows
    )
    discovery_exact = rows_aligned and reported_discovered == expected_discovered
    discovery_monotonic = all(
        left <= right
        for left, right in zip(
            reported_discovered,
            reported_discovered[1:],
        )
    )
    mature_is_causal = rows_aligned and all(
        int(row.get("mature_tracks", -1))
        <= sum(birth >= 0 and birth < index for birth in geometry_births)
        for index, row in enumerate(rows)
    )
    valid_track_keys = tuple(
        (sam_track_prompts[index], track_id)
        for index, track_id in enumerate(sam_track_ids)
        if track_id >= 0
    )
    track_keys_unique = len(valid_track_keys) == len(set(valid_track_keys))
    discovered_track_count = sum(birth >= 0 for birth in births)
    track_count_matches_births = len(valid_track_keys) == discovered_track_count
    late_births = tuple(
        birth for birth in births if birth > int(reference_index)
    )
    future_birth_supported = bool(late_births)
    passed = all(
        (
            rows_aligned,
            registry_shapes_match,
            discovery_exact,
            discovery_monotonic,
            mature_is_causal,
            track_keys_unique,
            track_count_matches_births,
            future_birth_supported,
        )
    )
    return {
        "rows_aligned": int(rows_aligned),
        "registry_shapes_match": int(registry_shapes_match),
        "discovery_exact_from_birth_registry": int(discovery_exact),
        "discovery_monotonic": int(discovery_monotonic),
        "mature_tracks_require_prior_geometry_birth": int(mature_is_causal),
        "persistent_prompt_track_keys_unique": int(track_keys_unique),
        "track_count_matches_birth_registry": int(track_count_matches_births),
        "future_birth_supported": int(future_birth_supported),
        "late_birth_count": len(late_births),
        "late_birth_sequence_indices": late_births,
        "discovered_track_count": discovered_track_count,
        "permanent_slot_capacity": len(births),
        "tracking_audit_pass": int(passed),
    }


def _int_tuple(value) -> tuple[int, ...]:
    return tuple(int(item) for item in (value or ()))


def _path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def _validate_config(config: BaselineRunConfig) -> None:
    if config.version != "v0":
        raise ValueError("The retained baseline config must declare baseline.version=v0.")
    if not config.clip_name:
        raise ValueError("baseline.clip_name is required.")
    if len(config.evaluation_frames) != 12:
        raise ValueError("V0 requires twelve future evaluation frames.")
    if len(set(config.evaluation_frames)) != len(config.evaluation_frames):
        raise ValueError("V0 evaluation frames must be unique.")
    if tuple(sorted(config.evaluation_frames)) != config.evaluation_frames:
        raise ValueError("V0 evaluation frames must be strictly time ordered.")


def _validate_track_ba_config(config: TrackBACandidateConfig) -> None:
    if config.track_token_source not in {
        "cached_streaming",
        "window_reaggregated",
    }:
        raise ValueError(
            "Track-BA track_token_source must be cached_streaming or "
            "window_reaggregated."
        )
    if config.query_source not in {"uniform_grid", "shi_tomasi"}:
        raise ValueError(
            "Track-BA query_source must be uniform_grid or shi_tomasi."
        )
    allowed = {
        "full_image",
        "sam_dynamic_excluded",
        "sam_instance_background_stratified",
        "bbox_instance_background_stratified",
        "random_instance_background_stratified",
    }
    if not config.methods or len(config.methods) != len(set(config.methods)):
        raise ValueError("V0 Track-BA methods must be non-empty and unique.")
    unknown = set(config.methods) - allowed
    if unknown:
        raise ValueError(f"Unknown V0 Track-BA methods={sorted(unknown)}.")
    if config.primary_method not in config.methods:
        raise ValueError("Track-BA primary_method must appear in methods.")
    if config.window_frames < 2:
        raise ValueError("Track-BA window_frames must be at least two.")
    if config.query_count < 2 or config.query_count % 2:
        raise ValueError("Track-BA query_count must be a positive even number.")
    if len(config.query_grid) != 2 or min(config.query_grid) < 2:
        raise ValueError("Track-BA query_grid must contain two values >=2.")
    if config.query_grid[0] * config.query_grid[1] < config.query_count:
        raise ValueError("Track-BA query_grid has fewer cells than query_count.")
    if config.track_iterations < 1 or config.optimizer_steps < 1:
        raise ValueError("Track/optimizer iterations must be positive.")
    for name, value in (
        ("visibility_threshold", config.visibility_threshold),
        ("track_confidence_threshold", config.track_confidence_threshold),
        ("point_confidence_threshold", config.point_confidence_threshold),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Track-BA {name} must be in [0,1].")
    if config.min_correspondences < 8:
        raise ValueError("Track-BA needs at least eight correspondences.")
    if config.learning_rate <= 0 or config.robust_delta_pixels <= 0:
        raise ValueError("Track-BA learning rate/robust delta must be positive.")
    if min(
        config.rotation_prior_weight,
        config.translation_prior_weight,
        config.temporal_prior_weight,
    ) < 0:
        raise ValueError("Track-BA prior weights cannot be negative.")
    if config.max_rotation_degrees <= 0:
        raise ValueError("Track-BA max_rotation_degrees must be positive.")
    if config.max_translation_scene_fraction <= 0:
        raise ValueError(
            "Track-BA max_translation_scene_fraction must be positive."
        )


def _validate_feature_pnp_config(config: FeaturePnPCandidateConfig) -> None:
    allowed = {
        "full_image",
        "sam_dynamic_excluded",
        "sam_instance_background_stratified",
        "bbox_instance_background_stratified",
        "random_instance_background_stratified",
    }
    if not config.methods or len(config.methods) != len(set(config.methods)):
        raise ValueError("Feature-PnP methods must be non-empty and unique.")
    unknown = set(config.methods) - allowed
    if unknown:
        raise ValueError(f"Unknown Feature-PnP methods={sorted(unknown)}.")
    if config.primary_method not in config.methods:
        raise ValueError("Feature-PnP primary method must appear in methods.")
    landmark_sources = {"raw_depth_unprojected", "native_world_pointmap"}
    if (
        not config.landmark_sources
        or len(config.landmark_sources) != len(set(config.landmark_sources))
        or set(config.landmark_sources) - landmark_sources
    ):
        raise ValueError("Feature-PnP landmark sources are invalid or duplicated.")
    if config.primary_landmark_source not in config.landmark_sources:
        raise ValueError(
            "Feature-PnP primary landmark source must appear in sources."
        )
    history_scopes = {"recent", "all_causal"}
    if (
        not config.history_scopes
        or len(config.history_scopes) != len(set(config.history_scopes))
        or set(config.history_scopes) - history_scopes
    ):
        raise ValueError("Feature-PnP history scopes are invalid or duplicated.")
    if config.primary_history_scope not in config.history_scopes:
        raise ValueError(
            "Feature-PnP primary history scope must appear in scopes."
        )
    if config.anchor_lookback < 1 or config.nfeatures < 64:
        raise ValueError("Feature-PnP lookback/nfeatures are too small.")
    if not 0 < config.contrast_threshold < 1:
        raise ValueError("Feature-PnP contrast threshold must be in (0,1).")
    if not 0 < config.ratio_threshold < 1:
        raise ValueError("Feature-PnP ratio threshold must be in (0,1).")
    if config.max_correspondences < config.min_correspondences:
        raise ValueError("Feature-PnP max correspondences is below minimum.")
    if config.min_correspondences < 8 or config.min_inliers < 6:
        raise ValueError("Feature-PnP correspondence/inlier gates are too small.")
    if config.min_inliers > config.min_correspondences:
        raise ValueError(
            "Feature-PnP min inliers cannot exceed min correspondences."
        )
    if not 0 <= config.point_confidence_threshold <= 1:
        raise ValueError("Feature-PnP point confidence must be in [0,1].")
    if config.ransac_iterations < 1 or config.ransac_reprojection_pixels <= 0:
        raise ValueError("Feature-PnP RANSAC settings must be positive.")
    if not 0 < config.ransac_confidence < 1:
        raise ValueError("Feature-PnP RANSAC confidence must be in (0,1).")
    if config.max_rotation_degrees <= 0:
        raise ValueError("Feature-PnP rotation bound must be positive.")
    if config.max_translation_scene_fraction <= 0:
        raise ValueError("Feature-PnP translation bound must be positive.")
