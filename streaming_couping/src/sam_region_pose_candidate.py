"""MESA-style SAM-region filtering for a frozen StreamVGGT pose candidate.

SAM contributes masks and persistent region identity only.  SIFT supplies
candidate point correspondences, the frozen StreamVGGT world pointmap supplies
historical 3D, and one fixed PnP-RANSAC solver estimates the current pose.
Ground-truth fields are intentionally absent from every function in this file.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F
import yaml


METHODS = (
    "full_image_match",
    "sam_region_identity",
    "shuffled_instance_identity",
)


@dataclass(frozen=True)
class SamRegionPoseConfig:
    source_path: Path
    enabled: bool
    output_dir: Path
    methods: tuple[str, ...]
    primary_method: str
    anchor_offsets: tuple[int, ...]
    nfeatures: int
    contrast_threshold: float
    ratio_threshold: float
    point_confidence_threshold: float
    raw_reprojection_gate_pixels: float
    mask_erosion_pixels: int
    min_track_confidence: float
    max_pose_instances: int
    max_correspondences: int
    min_correspondences: int
    min_instance_correspondences: int
    validation_stride: int
    min_validation_correspondences: int
    ransac_iterations: int
    ransac_reprojection_pixels: float
    ransac_confidence: float
    min_inliers: int
    min_inlier_ratio: float
    min_validation_gain_fraction: float
    max_validation_rmse_pixels: float
    max_rotation_degrees: float
    max_translation_scene_fraction: float
    spatial_grid: tuple[int, int]


@dataclass(frozen=True)
class FeatureSet:
    points: torch.Tensor
    descriptors: object
    responses: torch.Tensor


@dataclass(frozen=True)
class MatchPool:
    current_points: torch.Tensor
    world_points: torch.Tensor
    descriptor_distances: torch.Tensor
    raw_reprojection_errors: torch.Tensor
    anchor_indices: torch.Tensor
    anchor_labels: torch.Tensor
    current_labels: torch.Tensor
    shuffled_current_labels: torch.Tensor


def load_sam_region_pose_config(path: str | Path) -> SamRegionPoseConfig:
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf8") as handle:
        raw = yaml.safe_load(handle) or {}
    section = raw.get("sam_region_pose_candidate", {})
    config = SamRegionPoseConfig(
        source_path=source,
        enabled=bool(section.get("enabled", False)),
        output_dir=_path(
            section.get(
                "output_dir",
                "outputs/streaming_couping_v0/sam_region_pose_candidate",
            )
        ),
        methods=tuple(str(value) for value in section.get("methods", METHODS)),
        primary_method=str(
            section.get("primary_method", "sam_region_identity")
        ),
        anchor_offsets=tuple(
            int(value) for value in section.get("anchor_offsets", (1, 2, 4, 8, 16))
        ),
        nfeatures=int(section.get("nfeatures", 4096)),
        contrast_threshold=float(section.get("contrast_threshold", 0.01)),
        ratio_threshold=float(section.get("ratio_threshold", 0.80)),
        point_confidence_threshold=float(
            section.get("point_confidence_threshold", 0.30)
        ),
        raw_reprojection_gate_pixels=float(
            section.get("raw_reprojection_gate_pixels", 48.0)
        ),
        mask_erosion_pixels=int(section.get("mask_erosion_pixels", 3)),
        min_track_confidence=float(section.get("min_track_confidence", 0.50)),
        max_pose_instances=int(section.get("max_pose_instances", 5)),
        max_correspondences=int(section.get("max_correspondences", 512)),
        min_correspondences=int(section.get("min_correspondences", 32)),
        min_instance_correspondences=int(
            section.get("min_instance_correspondences", 8)
        ),
        validation_stride=int(section.get("validation_stride", 5)),
        min_validation_correspondences=int(
            section.get("min_validation_correspondences", 8)
        ),
        ransac_iterations=int(section.get("ransac_iterations", 2000)),
        ransac_reprojection_pixels=float(
            section.get("ransac_reprojection_pixels", 3.0)
        ),
        ransac_confidence=float(section.get("ransac_confidence", 0.999)),
        min_inliers=int(section.get("min_inliers", 20)),
        min_inlier_ratio=float(section.get("min_inlier_ratio", 0.25)),
        min_validation_gain_fraction=float(
            section.get("min_validation_gain_fraction", 0.05)
        ),
        max_validation_rmse_pixels=float(
            section.get("max_validation_rmse_pixels", 8.0)
        ),
        max_rotation_degrees=float(section.get("max_rotation_degrees", 5.0)),
        max_translation_scene_fraction=float(
            section.get("max_translation_scene_fraction", 0.10)
        ),
        spatial_grid=tuple(
            int(value) for value in section.get("spatial_grid", (8, 12))
        ),
    )
    _validate_config(config)
    return config


def deployable_cache_view(payload: Mapping[str, object]) -> dict[str, object]:
    """Copy only inference-time fields, excluding every target/GT field."""

    names = (
        "frame_indices",
        "instance_ids",
        "instance_prompts",
        "sam_track_ids",
        "sam_birth_indices",
        "stream_images",
        "baseline_world_points",
        "baseline_world_confidence",
        # These masks have already passed through the exact StreamVGGT image
        # crop/resize transform.  Resizing tracking_masks_output here would
        # misregister masks whenever preprocessing crops the source image.
        "tracking_masks_stream",
        "tracking_scores",
        "scene_scale",
    )
    missing = [name for name in names if name not in payload]
    if missing:
        raise ValueError(f"SAM-region pose cache lacks fields={missing}.")
    view = {name: payload[name] for name in names}
    forbidden = [name for name in view if name.startswith("target_") or "ground_truth" in name]
    if forbidden:
        raise RuntimeError(f"Candidate view leaked GT fields={forbidden}.")
    validate_deployable_cache(view)
    return view


def validate_deployable_cache(payload: Mapping[str, object]) -> None:
    images = payload["stream_images"]
    points = payload["baseline_world_points"]
    confidence = payload["baseline_world_confidence"]
    masks = payload["tracking_masks_stream"]
    scores = payload["tracking_scores"]
    if not torch.is_tensor(images) or images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("stream_images must be [S,3,H,W].")
    if not torch.is_tensor(points) or points.ndim != 4 or points.shape[-1] != 3:
        raise ValueError("baseline_world_points must be [S,H,W,3].")
    if not torch.is_tensor(confidence) or confidence.ndim != 3:
        raise ValueError("baseline_world_confidence must be [S,H,W].")
    if not torch.is_tensor(masks) or masks.ndim != 4:
        raise ValueError("tracking_masks_stream must be [S,I,H,W].")
    if not torch.is_tensor(scores) or scores.ndim != 2:
        raise ValueError("tracking_scores must be [S,I].")
    sequence = int(images.shape[0])
    instances = int(masks.shape[1])
    if any(int(value.shape[0]) != sequence for value in (points, confidence, masks, scores)):
        raise ValueError("SAM-region pose cache sequence dimensions disagree.")
    if int(scores.shape[1]) != instances:
        raise ValueError("SAM mask/score instance dimensions disagree.")
    if tuple(masks.shape[-2:]) != tuple(images.shape[-2:]):
        raise ValueError(
            "tracking_masks_stream must be aligned to the StreamVGGT image grid."
        )
    if tuple(points.shape[1:3]) != tuple(images.shape[-2:]):
        raise ValueError(
            "baseline_world_points must be aligned to the StreamVGGT image grid."
        )
    if tuple(confidence.shape[1:3]) != tuple(images.shape[-2:]):
        raise ValueError(
            "baseline_world_confidence must be aligned to the StreamVGGT image grid."
        )
    if len(payload["instance_ids"]) != instances:
        raise ValueError("SAM registry size differs from mask tensor.")
    if len(payload["sam_track_ids"]) != instances or len(payload["sam_birth_indices"]) != instances:
        raise ValueError("SAM track registry shapes disagree.")


def extract_sift_features(
    image: torch.Tensor,
    config: SamRegionPoseConfig,
) -> FeatureSet:
    cv2, np = _opencv()
    cv2.setRNGSeed(0)
    cv2.setNumThreads(1)
    gray = _gray_uint8(image, cv2, np)
    detector = cv2.SIFT_create(
        nfeatures=int(config.nfeatures),
        contrastThreshold=float(config.contrast_threshold),
    )
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    if not keypoints or descriptors is None:
        return FeatureSet(
            points=torch.empty(0, 2, dtype=torch.float32),
            descriptors=np.empty((0, 128), dtype=np.float32),
            responses=torch.empty(0, dtype=torch.float32),
        )
    order = sorted(
        range(len(keypoints)),
        key=lambda index: (
            -float(keypoints[index].response),
            float(keypoints[index].pt[1]),
            float(keypoints[index].pt[0]),
        ),
    )
    return FeatureSet(
        points=torch.tensor(
            [keypoints[index].pt for index in order],
            dtype=torch.float32,
        ),
        descriptors=np.ascontiguousarray(descriptors[order], dtype=np.float32),
        responses=torch.tensor(
            [keypoints[index].response for index in order],
            dtype=torch.float32,
        ),
    )


def mutual_ratio_matches(left, right, *, ratio: float) -> list[tuple[int, int, float]]:
    cv2, _ = _opencv()
    if (
        left is None
        or right is None
        or len(left) < 2
        or len(right) < 2
        or getattr(left, "ndim", 0) != 2
        or getattr(right, "ndim", 0) != 2
        or left.shape[1] != right.shape[1]
    ):
        return []
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    forward = matcher.knnMatch(left, right, k=2)
    reverse = matcher.knnMatch(right, left, k=2)
    forward_pass = {
        int(pair[0].queryIdx): (int(pair[0].trainIdx), float(pair[0].distance))
        for pair in forward
        if len(pair) == 2 and pair[0].distance < float(ratio) * pair[1].distance
    }
    reverse_pass = {
        int(pair[0].queryIdx): int(pair[0].trainIdx)
        for pair in reverse
        if len(pair) == 2 and pair[0].distance < float(ratio) * pair[1].distance
    }
    return [
        (left_index, right_index, distance)
        for left_index, (right_index, distance) in sorted(forward_pass.items())
        if reverse_pass.get(right_index) == left_index
    ]


def build_instance_label_map(
    payload: Mapping[str, object],
    *,
    frame_index: int,
    output_size: tuple[int, int],
    config: SamRegionPoseConfig,
) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Return -1=uncertain boundary/overlap, 0=background, slot+1=instance."""

    masks = payload["tracking_masks_stream"][int(frame_index)].detach().bool().cpu()
    if tuple(masks.shape[-2:]) != tuple(output_size):
        raise ValueError("SAM stream masks and matching image use different grids.")
    scores = payload["tracking_scores"][int(frame_index)].detach().float().cpu()
    births = tuple(int(value) for value in payload["sam_birth_indices"])
    track_ids = tuple(int(value) for value in payload["sam_track_ids"])
    registered = torch.tensor(
        [
            track_ids[slot] >= 0
            and births[slot] >= 0
            and births[slot] <= int(frame_index)
            for slot in range(len(track_ids))
        ],
        dtype=torch.bool,
    )
    present = masks.flatten(1).any(dim=1)
    eligible = (
        registered
        & present
        & (scores >= float(config.min_track_confidence))
    )
    active_slots = torch.nonzero(eligible, as_tuple=False).flatten().tolist()
    active_slots.sort(
        key=lambda slot: (
            -float(scores[slot]),
            -int(masks[slot].sum()),
            int(slot),
        )
    )
    active = torch.zeros_like(registered)
    for slot in active_slots[: int(config.max_pose_instances)]:
        active[int(slot)] = True
    ignored_masks = masks & (registered & ~active)[:, None, None]
    masks = masks & active[:, None, None]
    radius = int(config.mask_erosion_pixels)
    if radius > 0:
        kernel = 2 * radius + 1
        dilated = F.max_pool2d(
            masks[:, None].float(), kernel, stride=1, padding=radius
        )[:, 0].bool()
        eroded = ~F.max_pool2d(
            (~masks)[:, None].float(), kernel, stride=1, padding=radius
        )[:, 0].bool()
    else:
        dilated = masks
        eroded = masks
    if radius > 0:
        ignored_dilated = F.max_pool2d(
            ignored_masks[:, None].float(), kernel, stride=1, padding=radius
        )[:, 0].bool()
    else:
        ignored_dilated = ignored_masks
    count = eroded.sum(dim=0)
    dilated_count = dilated.sum(dim=0)
    uncertain = dilated.any(dim=0) & (count != 1)
    # The dilated-minus-eroded ring is deliberately unknown on both sides of
    # each mask boundary, where segmentation and depth mixing are least safe.
    uncertain |= dilated.any(dim=0) & ~eroded.any(dim=0)
    uncertain |= dilated_count > 1
    # A registered region that was not selected into the per-frame active-5
    # budget is unknown, never background.  Otherwise features on the sixth
    # object could silently enter the background-to-background pool.
    uncertain |= ignored_dilated.any(dim=0)
    labels = torch.zeros(output_size, dtype=torch.long)
    for slot in range(int(eroded.shape[0])):
        labels[eroded[slot] & (count == 1)] = slot + 1
    labels[uncertain] = -1
    active_labels = tuple(
        slot + 1
        for slot in range(int(eroded.shape[0]))
        if bool(eroded[slot].any())
    )
    return labels, active_labels


def shuffled_labels(
    labels: torch.Tensor,
    *,
    active_labels: Sequence[int],
    registry_capacity: int,
) -> torch.Tensor:
    """Deterministically break instance identity without moving mask shapes."""

    output = labels.clone()
    active = tuple(sorted(int(value) for value in active_labels))
    if len(active) >= 2:
        mapping = {
            value: active[(index + 1) % len(active)]
            for index, value in enumerate(active)
        }
    elif len(active) == 1:
        value = active[0]
        mapping = {value: value % max(int(registry_capacity), 2) + 1}
    else:
        mapping = {}
    for source, target in mapping.items():
        output[labels == source] = int(target)
    return output


def build_match_pool(
    *,
    payload: Mapping[str, object],
    current_index: int,
    features: Sequence[FeatureSet | None],
    raw_world_to_camera: torch.Tensor,
    intrinsics: torch.Tensor,
    config: SamRegionPoseConfig,
) -> MatchPool:
    current = features[int(current_index)]
    if current is None:
        raise ValueError(f"Current frame {current_index} has no extracted features.")
    height, width = tuple(int(value) for value in payload["stream_images"].shape[-2:])
    current_map, current_active = build_instance_label_map(
        payload,
        frame_index=int(current_index),
        output_size=(height, width),
        config=config,
    )
    current_shuffled = shuffled_labels(
        current_map,
        active_labels=current_active,
        registry_capacity=len(payload["instance_ids"]),
    )
    records: list[tuple] = []
    label_maps: dict[int, torch.Tensor] = {}
    for offset in config.anchor_offsets:
        anchor_index = int(current_index) - int(offset)
        if anchor_index < 0:
            continue
        anchor = features[anchor_index]
        if anchor is None:
            raise ValueError(f"Anchor frame {anchor_index} has no extracted features.")
        matches = mutual_ratio_matches(
            anchor.descriptors,
            current.descriptors,
            ratio=config.ratio_threshold,
        )
        if not matches:
            continue
        anchor_ids = torch.tensor([row[0] for row in matches], dtype=torch.long)
        current_ids = torch.tensor([row[1] for row in matches], dtype=torch.long)
        distance = torch.tensor([row[2] for row in matches], dtype=torch.float32)
        anchor_uv = anchor.points.index_select(0, anchor_ids)
        current_uv = current.points.index_select(0, current_ids)
        world = bilinear_sample_map(
            payload["baseline_world_points"][anchor_index], anchor_uv
        )
        confidence = bilinear_sample_map(
            payload["baseline_world_confidence"][anchor_index][..., None],
            anchor_uv,
        )[:, 0]
        raw_projection, positive = project_world_points(
            world,
            raw_world_to_camera[0, current_index].detach().float().cpu(),
            intrinsics[0, current_index].detach().float().cpu(),
        )
        reprojection = torch.linalg.vector_norm(raw_projection - current_uv, dim=-1)
        valid = torch.isfinite(world).all(dim=-1)
        valid &= torch.isfinite(confidence)
        valid &= confidence >= float(config.point_confidence_threshold)
        valid &= positive
        valid &= torch.isfinite(reprojection)
        valid &= reprojection <= float(config.raw_reprojection_gate_pixels)
        if not bool(valid.any()):
            continue
        if anchor_index not in label_maps:
            label_maps[anchor_index] = build_instance_label_map(
                payload,
                frame_index=anchor_index,
                output_size=(height, width),
                config=config,
            )[0]
        anchor_map = label_maps[anchor_index]
        valid_distance = distance[valid]
        valid_reprojection = reprojection[valid]
        valid_anchor_uv = anchor_uv[valid]
        valid_current_uv = current_uv[valid]
        valid_world = world[valid]
        anchor_labels = labels_at_points(anchor_map, valid_anchor_uv)
        current_labels = labels_at_points(current_map, valid_current_uv)
        shifted_labels = labels_at_points(current_shuffled, valid_current_uv)
        valid_current_ids = current_ids[valid]
        for row in range(int(valid_current_ids.shape[0])):
            records.append(
                (
                    float(valid_distance[row]),
                    float(valid_reprojection[row]),
                    -anchor_index,
                    int(valid_current_ids[row]),
                    anchor_index,
                    valid_current_uv[row],
                    valid_world[row],
                    int(anchor_labels[row]),
                    int(current_labels[row]),
                    int(shifted_labels[row]),
                )
            )
    if not records:
        return empty_match_pool()
    # One current keypoint contributes at most one historical 3D observation.
    records.sort(key=lambda item: item[:4])
    retained = []
    used_current: set[int] = set()
    for row in records:
        if row[3] in used_current:
            continue
        used_current.add(row[3])
        retained.append(row)
    return MatchPool(
        current_points=torch.stack([row[5] for row in retained]),
        world_points=torch.stack([row[6] for row in retained]),
        descriptor_distances=torch.tensor([row[0] for row in retained]),
        raw_reprojection_errors=torch.tensor([row[1] for row in retained]),
        anchor_indices=torch.tensor([row[4] for row in retained], dtype=torch.long),
        anchor_labels=torch.tensor([row[7] for row in retained], dtype=torch.long),
        current_labels=torch.tensor([row[8] for row in retained], dtype=torch.long),
        shuffled_current_labels=torch.tensor(
            [row[9] for row in retained], dtype=torch.long
        ),
    )


def method_mask(pool: MatchPool, method: str) -> torch.Tensor:
    if method == "full_image_match":
        return torch.ones(pool.current_points.shape[0], dtype=torch.bool)
    if method == "sam_region_identity":
        return (
            (pool.anchor_labels >= 0)
            & (pool.current_labels >= 0)
            & (pool.anchor_labels == pool.current_labels)
        )
    if method == "shuffled_instance_identity":
        return (
            (pool.anchor_labels >= 0)
            & (pool.shuffled_current_labels >= 0)
            & (pool.anchor_labels == pool.shuffled_current_labels)
        )
    raise ValueError(f"Unknown SAM-region pose method={method!r}.")


def select_equal_count_correspondences(
    pool: MatchPool,
    *,
    config: SamRegionPoseConfig,
    image_size: tuple[int, int],
) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, int]]]:
    available = {method: torch.nonzero(method_mask(pool, method)).flatten() for method in config.methods}
    locked_count = min(
        [int(config.max_correspondences)]
        + [int(indices.shape[0]) for indices in available.values()]
    )
    selected = {
        method: spatially_diverse_selection(
            indices,
            points=pool.current_points,
            distances=pool.descriptor_distances,
            count=locked_count,
            image_size=image_size,
            grid=config.spatial_grid,
        )
        for method, indices in available.items()
    }
    diagnostics = {}
    for method, indices in selected.items():
        labels = pool.current_labels.index_select(0, indices) if indices.numel() else torch.empty(0, dtype=torch.long)
        diagnostics[method] = {
            "available_correspondences": int(available[method].shape[0]),
            "locked_equal_count": int(locked_count),
            "selected_instance_correspondences": int((labels > 0).sum()),
            "selected_instance_regions": int(torch.unique(labels[labels > 0]).numel()),
            "selected_background_correspondences": int((labels == 0).sum()),
            "selected_uncertain_correspondences": int((labels < 0).sum()),
        }
    return selected, diagnostics


def spatially_diverse_selection(
    indices: torch.Tensor,
    *,
    points: torch.Tensor,
    distances: torch.Tensor,
    count: int,
    image_size: tuple[int, int],
    grid: tuple[int, int],
) -> torch.Tensor:
    if count <= 0 or indices.numel() == 0:
        return torch.empty(0, dtype=torch.long)
    ordered = indices.index_select(
        0, torch.argsort(distances.index_select(0, indices), stable=True)
    )
    height, width = image_size
    rows, columns = grid
    seen: set[tuple[int, int]] = set()
    first: list[int] = []
    remaining: list[int] = []
    for raw_index in ordered.tolist():
        point = points[raw_index]
        cell = (
            min(int(float(point[1]) * rows / max(height, 1)), rows - 1),
            min(int(float(point[0]) * columns / max(width, 1)), columns - 1),
        )
        if cell not in seen:
            seen.add(cell)
            first.append(raw_index)
        else:
            remaining.append(raw_index)
    values = (first + remaining)[: int(count)]
    return torch.tensor(values, dtype=torch.long)


def solve_pose_candidate(
    *,
    pool: MatchPool,
    selected: torch.Tensor,
    raw_pose: torch.Tensor,
    intrinsics: torch.Tensor,
    scene_scale: float,
    config: SamRegionPoseConfig,
    sam_evidence_active: bool,
) -> tuple[torch.Tensor, dict[str, float | int | str]]:
    raw = raw_pose.detach().float().cpu()
    common: dict[str, float | int | str] = {
        "selected_correspondences": int(selected.shape[0]),
        "fit_correspondences": 0,
        "validation_correspondences": 0,
        "inliers": 0,
        "inlier_ratio": 0.0,
        "raw_validation_rmse_pixels": float("nan"),
        "candidate_validation_rmse_pixels": float("nan"),
        "validation_gain_fraction": float("nan"),
        "rotation_update_degrees": 0.0,
        "center_update_native": 0.0,
        "sam_evidence_active": int(bool(sam_evidence_active)),
    }
    if selected.shape[0] < int(config.min_correspondences):
        return raw, {"optimized": 0, "reason": "fewer_than_min_correspondences", **common}
    position = torch.arange(selected.shape[0])
    validation_mask = position.remainder(int(config.validation_stride)) == 0
    validation = selected[validation_mask]
    fit = selected[~validation_mask]
    common["fit_correspondences"] = int(fit.shape[0])
    common["validation_correspondences"] = int(validation.shape[0])
    if validation.shape[0] < int(config.min_validation_correspondences) or fit.shape[0] < 6:
        return raw, {"optimized": 0, "reason": "fit_validation_split_too_small", **common}

    cv2, np = _opencv()
    object_points = np.ascontiguousarray(
        pool.world_points.index_select(0, fit).double().numpy(), dtype=np.float64
    )
    image_points = np.ascontiguousarray(
        pool.current_points.index_select(0, fit).double().numpy(), dtype=np.float64
    )
    calibration = np.ascontiguousarray(intrinsics.detach().double().numpy(), dtype=np.float64)
    raw_rotation = np.ascontiguousarray(raw[:3, :3].double().numpy(), dtype=np.float64)
    raw_translation = np.ascontiguousarray(raw[:3, 3].double().numpy(), dtype=np.float64)
    rvec, _ = cv2.Rodrigues(raw_rotation)
    tvec = raw_translation.reshape(3, 1).copy()
    cv2.setRNGSeed(0)
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        object_points,
        image_points,
        calibration,
        None,
        rvec=rvec,
        tvec=tvec,
        useExtrinsicGuess=True,
        iterationsCount=int(config.ransac_iterations),
        reprojectionError=float(config.ransac_reprojection_pixels),
        confidence=float(config.ransac_confidence),
        flags=cv2.SOLVEPNP_EPNP,
    )
    inlier_count = 0 if inliers is None else int(len(inliers))
    inlier_ratio = inlier_count / max(int(fit.shape[0]), 1)
    common["inliers"] = inlier_count
    common["inlier_ratio"] = float(inlier_ratio)
    if (
        not ok
        or inlier_count < int(config.min_inliers)
        or inlier_ratio < float(config.min_inlier_ratio)
    ):
        return raw, {"optimized": 0, "reason": "pnp_inlier_gate_failed", **common}
    inlier_ids = inliers.reshape(-1)
    rvec, tvec = cv2.solvePnPRefineLM(
        object_points[inlier_ids],
        image_points[inlier_ids],
        calibration,
        None,
        rvec,
        tvec,
    )
    rotation, _ = cv2.Rodrigues(rvec)
    translation = tvec.reshape(3)
    if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
        return raw, {"optimized": 0, "reason": "nonfinite_pnp_pose", **common}

    validation_world = pool.world_points.index_select(0, validation)
    validation_uv = pool.current_points.index_select(0, validation)
    candidate = torch.from_numpy(
        np.concatenate((rotation, translation[:, None]), axis=1)
    ).float()
    raw_rmse = reprojection_rmse(validation_world, validation_uv, raw, intrinsics)
    candidate_rmse = reprojection_rmse(
        validation_world, validation_uv, candidate, intrinsics
    )
    gain = (raw_rmse - candidate_rmse) / max(raw_rmse, 1e-6)
    common["raw_validation_rmse_pixels"] = raw_rmse
    common["candidate_validation_rmse_pixels"] = candidate_rmse
    common["validation_gain_fraction"] = gain

    delta = rotation @ raw_rotation.T
    cosine = float(np.clip((np.trace(delta) - 1.0) * 0.5, -1.0, 1.0))
    rotation_update = math.degrees(math.acos(cosine))
    raw_center = -(raw_rotation.T @ raw_translation)
    center = -(rotation.T @ translation)
    center_update = float(np.linalg.norm(center - raw_center))
    common["rotation_update_degrees"] = rotation_update
    common["center_update_native"] = center_update
    positive = project_world_points(validation_world, candidate, intrinsics)[1]
    if float(positive.float().mean()) < 0.95:
        return raw, {"optimized": 0, "reason": "validation_positive_depth_failed", **common}
    if (
        not math.isfinite(candidate_rmse)
        or candidate_rmse > float(config.max_validation_rmse_pixels)
        or gain < float(config.min_validation_gain_fraction)
    ):
        return raw, {"optimized": 0, "reason": "heldout_reprojection_gate_failed", **common}
    if (
        rotation_update > float(config.max_rotation_degrees)
        or center_update
        > max(float(scene_scale) * float(config.max_translation_scene_fraction), 1e-6)
    ):
        return raw, {"optimized": 0, "reason": "pose_update_outside_locked_bounds", **common}
    return candidate, {"optimized": 1, "reason": "ok", **common}


def labels_at_points(labels: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    height, width = labels.shape
    x = points[:, 0].round().long().clamp(0, width - 1)
    y = points[:, 1].round().long().clamp(0, height - 1)
    return labels[y, x]


def bilinear_sample_map(value: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    if value.ndim != 3:
        raise ValueError("Bilinear map must be [H,W,C].")
    height, width, _ = value.shape
    normalized = points.detach().float().clone()
    normalized[:, 0] = 2.0 * normalized[:, 0] / max(width - 1, 1) - 1.0
    normalized[:, 1] = 2.0 * normalized[:, 1] / max(height - 1, 1) - 1.0
    sampled = F.grid_sample(
        value.detach().float().permute(2, 0, 1)[None],
        normalized[None, None],
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return sampled[0, :, 0].T


def project_world_points(
    world: torch.Tensor,
    world_to_camera: torch.Tensor,
    intrinsics: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    camera = world.float() @ world_to_camera[:3, :3].float().T
    camera = camera + world_to_camera[:3, 3].float()
    positive = torch.isfinite(camera).all(dim=-1) & (camera[:, 2] > 1e-6)
    z = camera[:, 2].clamp_min(1e-6)
    u = intrinsics[0, 0] * camera[:, 0] / z + intrinsics[0, 2]
    v = intrinsics[1, 1] * camera[:, 1] / z + intrinsics[1, 2]
    return torch.stack((u, v), dim=-1), positive


def reprojection_rmse(
    world: torch.Tensor,
    image: torch.Tensor,
    pose: torch.Tensor,
    intrinsics: torch.Tensor,
) -> float:
    projected, positive = project_world_points(world, pose, intrinsics)
    valid = positive & torch.isfinite(projected).all(dim=-1)
    if not bool(valid.any()):
        return float("inf")
    error2 = (projected[valid] - image[valid]).square().sum(dim=-1)
    return float(torch.sqrt(error2.mean()))


def scene_scale_from_cache(payload: Mapping[str, object], threshold: float) -> float:
    cached = float(payload.get("scene_scale", float("nan")))
    if math.isfinite(cached) and cached > 0.0:
        return cached
    points = payload["baseline_world_points"][0].detach().float().cpu()
    confidence = payload["baseline_world_confidence"][0].detach().float().cpu()
    finite = torch.isfinite(points).all(dim=-1) & torch.isfinite(confidence)
    finite &= confidence >= float(threshold)
    selected = points[finite]
    if selected.shape[0] < 128:
        selected = points[torch.isfinite(points).all(dim=-1)]
    if not selected.numel():
        return 1.0
    center = selected.median(dim=0).values
    radius = torch.linalg.vector_norm(selected - center, dim=-1)
    return float(torch.quantile(radius, 0.50).clamp_min(1e-3))


def empty_match_pool() -> MatchPool:
    return MatchPool(
        current_points=torch.empty(0, 2),
        world_points=torch.empty(0, 3),
        descriptor_distances=torch.empty(0),
        raw_reprojection_errors=torch.empty(0),
        anchor_indices=torch.empty(0, dtype=torch.long),
        anchor_labels=torch.empty(0, dtype=torch.long),
        current_labels=torch.empty(0, dtype=torch.long),
        shuffled_current_labels=torch.empty(0, dtype=torch.long),
    )


def _gray_uint8(image: torch.Tensor, cv2, np):
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("SIFT input must be RGB [3,H,W].")
    rgb = (
        image.detach()
        .float()
        .clamp(0, 1)
        .mul(255)
        .byte()
        .permute(1, 2, 0)
        .contiguous()
        .numpy()
    )
    return np.ascontiguousarray(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))


def _opencv():
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "V0 SAM-region pose candidate requires OpenCV with SIFT support."
        ) from exc
    if not hasattr(cv2, "SIFT_create"):
        raise RuntimeError("Installed OpenCV does not expose cv2.SIFT_create().")
    return cv2, np


def _path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def _validate_config(config: SamRegionPoseConfig) -> None:
    if config.methods != METHODS:
        raise ValueError(f"SAM-region methods must be exactly {METHODS}.")
    if config.primary_method != "sam_region_identity":
        raise ValueError("SAM-region primary method must be sam_region_identity.")
    if not config.anchor_offsets or any(value <= 0 for value in config.anchor_offsets):
        raise ValueError("anchor_offsets must be positive causal lags.")
    if len(set(config.anchor_offsets)) != len(config.anchor_offsets):
        raise ValueError("anchor_offsets must be unique.")
    if config.max_correspondences < config.min_correspondences:
        raise ValueError("max_correspondences must cover min_correspondences.")
    if config.max_pose_instances < 1:
        raise ValueError("max_pose_instances must be positive.")
    if config.min_correspondences < 8 or config.min_inliers < 6:
        raise ValueError("PnP support thresholds are too small.")
    if config.validation_stride < 2 or config.min_validation_correspondences < 4:
        raise ValueError("A nontrivial held-out correspondence split is required.")
    if len(config.spatial_grid) != 2 or any(value < 1 for value in config.spatial_grid):
        raise ValueError("spatial_grid must contain two positive values.")
    for name, value in (
        ("ratio_threshold", config.ratio_threshold),
        ("point_confidence_threshold", config.point_confidence_threshold),
        ("min_track_confidence", config.min_track_confidence),
        ("ransac_confidence", config.ransac_confidence),
        ("min_inlier_ratio", config.min_inlier_ratio),
        ("min_validation_gain_fraction", config.min_validation_gain_fraction),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0,1].")
