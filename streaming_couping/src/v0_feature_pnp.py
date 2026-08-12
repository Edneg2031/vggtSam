"""Causal local-feature 2D-3D PnP factors for the V0 pose candidate.

No model is trained. SIFT matches provide current 2D to historical 2D
correspondence; frozen raw StreamVGGT depth/K/pose lifts only the historical
observations into the baseline world gauge. SAM contributes deployable masks
for dynamic exclusion or equal-count instance/background stratification.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .learned_pose.baseline_runtime import FeaturePnPCandidateConfig
from .v0_track_ba import (
    bilinear_sample_map,
    support_region,
    unproject_depth_samples,
)


@dataclass(frozen=True)
class FeatureSet:
    points: torch.Tensor
    descriptors: object
    responses: torch.Tensor


@dataclass(frozen=True)
class MatchPool:
    current_points: torch.Tensor
    world_points: torch.Tensor
    distances: torch.Tensor
    anchor_indices: torch.Tensor
    anchor_region: dict[str, torch.Tensor]
    current_region: dict[str, torch.Tensor]


def extract_sift_features(
    image: torch.Tensor,
    config: FeaturePnPCandidateConfig,
) -> FeatureSet:
    cv2, np = _opencv()
    cv2.setRNGSeed(0)
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
    points = torch.tensor(
        [keypoints[index].pt for index in order],
        dtype=torch.float32,
    )
    responses = torch.tensor(
        [keypoints[index].response for index in order],
        dtype=torch.float32,
    )
    return FeatureSet(
        points=points,
        descriptors=np.ascontiguousarray(descriptors[order], dtype=np.float32),
        responses=responses,
    )


def build_match_pool(
    *,
    payload: dict,
    current_index: int,
    features: list[FeatureSet | None],
    raw_world_to_camera: torch.Tensor,
    intrinsics: torch.Tensor,
    config: FeaturePnPCandidateConfig,
) -> MatchPool:
    current = features[int(current_index)]
    if current is None:
        raise ValueError(
            f"Feature-PnP current frame {current_index} was not extracted."
        )
    height, width = tuple(int(v) for v in payload["stream_images"].shape[-2:])
    current_regions = {
        method: support_region(
            payload,
            frame_index=int(current_index),
            method=method,
            output_size=(height, width),
        )
        for method in config.methods
    }
    records = []
    start = max(0, int(current_index) - int(config.anchor_lookback))
    for anchor_index in range(start, int(current_index)):
        anchor = features[anchor_index]
        if anchor is None:
            raise ValueError(
                f"Feature-PnP anchor frame {anchor_index} was not extracted."
            )
        matches = mutual_ratio_matches(
            anchor.descriptors,
            current.descriptors,
            ratio=float(config.ratio_threshold),
        )
        if not matches:
            continue
        anchor_ids = torch.tensor([item[0] for item in matches], dtype=torch.long)
        current_ids = torch.tensor([item[1] for item in matches], dtype=torch.long)
        distances = torch.tensor([item[2] for item in matches], dtype=torch.float32)
        anchor_points = anchor.points.index_select(0, anchor_ids)
        current_points = current.points.index_select(0, current_ids)
        depth = bilinear_sample_map(
            payload["baseline_depth"][anchor_index],
            anchor_points,
        )[..., 0]
        point_confidence = bilinear_sample_map(
            payload["baseline_world_confidence"][anchor_index][..., None],
            anchor_points,
        )[..., 0]
        world = unproject_depth_samples(
            anchor_points,
            depth,
            raw_world_to_camera[0, anchor_index].detach().float().cpu(),
            intrinsics[0, anchor_index].detach().float().cpu(),
        )
        valid = torch.isfinite(world).all(dim=-1)
        valid &= torch.isfinite(depth) & (depth > 1e-6)
        valid &= torch.isfinite(point_confidence)
        valid &= point_confidence >= float(config.point_confidence_threshold)
        if not valid.any():
            continue
        records.append(
            (
                anchor_index,
                anchor_points[valid],
                current_points[valid],
                world[valid],
                distances[valid],
                current_ids[valid],
            )
        )
    if not records:
        return _empty_pool(current_regions)

    # One current feature must vote for at most one historical 3D point.
    # Keep its strongest descriptor match, with recent anchors as tie-breaker.
    candidates = []
    for anchor_index, anchor_uv, current_uv, world, distances, current_ids in records:
        for row in range(int(current_uv.shape[0])):
            candidates.append(
                (
                    float(distances[row]),
                    -int(anchor_index),
                    int(current_ids[row]),
                    int(anchor_index),
                    anchor_uv[row],
                    current_uv[row],
                    world[row],
                )
            )
    candidates.sort(key=lambda item: item[:3])
    used_current: set[int] = set()
    retained = []
    for item in candidates:
        current_id = item[2]
        if current_id in used_current:
            continue
        used_current.add(current_id)
        retained.append(item)
    anchor_indices = torch.tensor([item[3] for item in retained], dtype=torch.long)
    anchor_points = torch.stack([item[4] for item in retained])
    current_points = torch.stack([item[5] for item in retained])
    world_points = torch.stack([item[6] for item in retained])
    distances = torch.tensor([item[0] for item in retained], dtype=torch.float32)

    unique_anchors = sorted(set(anchor_indices.tolist()))
    region_cache = {
        (method, anchor_index): support_region(
            payload,
            frame_index=anchor_index,
            method=method,
            output_size=(height, width),
        )
        for method in config.methods
        for anchor_index in unique_anchors
    }
    anchor_regions = {}
    for method in config.methods:
        flags = []
        for row, anchor_index in enumerate(anchor_indices.tolist()):
            region = region_cache[(method, anchor_index)]
            flags.append(_points_in_mask(anchor_points[row : row + 1], region)[0])
        anchor_regions[method] = torch.stack(flags).bool()
    current_region_flags = {
        method: _points_in_mask(current_points, region)
        for method, region in current_regions.items()
    }
    return MatchPool(
        current_points=current_points,
        world_points=world_points,
        distances=distances,
        anchor_indices=anchor_indices,
        anchor_region=anchor_regions,
        current_region=current_region_flags,
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
    left_knn = matcher.knnMatch(left, right, k=2)
    right_knn = matcher.knnMatch(right, left, k=2)
    left_pass = {
        int(pair[0].queryIdx): (int(pair[0].trainIdx), float(pair[0].distance))
        for pair in left_knn
        if len(pair) == 2 and pair[0].distance < float(ratio) * pair[1].distance
    }
    right_pass = {
        int(pair[0].queryIdx): int(pair[0].trainIdx)
        for pair in right_knn
        if len(pair) == 2 and pair[0].distance < float(ratio) * pair[1].distance
    }
    return [
        (left_index, right_index, distance)
        for left_index, (right_index, distance) in sorted(left_pass.items())
        if right_pass.get(right_index) == left_index
    ]


def select_method_correspondences(
    pool: MatchPool,
    *,
    method: str,
    count: int,
) -> tuple[torch.Tensor, bool, dict[str, int]]:
    if int(count) <= 0 or not pool.current_points.shape[0]:
        empty = torch.empty(0, dtype=torch.long)
        return empty, False, {"available": 0, "region": 0, "background": 0}
    allowed_method = (
        "sam_dynamic_excluded"
        if method.endswith("instance_background_stratified")
        else method
    )
    allowed = pool.anchor_region[allowed_method] & pool.current_region[allowed_method]
    order = torch.argsort(pool.distances, stable=True)
    if method.endswith("instance_background_stratified"):
        anchor_region = pool.anchor_region[method]
        current_region = pool.current_region[method]
        # Do not classify cross-boundary matches as background. The two
        # strata require the same deployable region label in both views.
        region = anchor_region & current_region & allowed
        background = ~anchor_region & ~current_region & allowed
        half = int(count) // 2
        left = order[region.index_select(0, order)][:half]
        right = order[background.index_select(0, order)][: int(count) - half]
        selected = torch.cat((left, right))
        feasible = selected.shape[0] == int(count)
        diagnostics = {
            "available": int(allowed.sum()),
            "region": int(region.sum()),
            "background": int(background.sum()),
        }
    else:
        selected = order[allowed.index_select(0, order)][: int(count)]
        feasible = selected.shape[0] == int(count)
        diagnostics = {
            "available": int(allowed.sum()),
            "region": 0,
            "background": int(allowed.sum()),
        }
    return selected, bool(feasible), diagnostics


def solve_feature_pnp(
    *,
    pool: MatchPool,
    selected: torch.Tensor,
    raw_world_to_camera: torch.Tensor,
    intrinsics: torch.Tensor,
    current_index: int,
    scene_scale: float,
    config: FeaturePnPCandidateConfig,
    force_fallback_reason: str | None = None,
) -> tuple[torch.Tensor, dict[str, float | int | str]]:
    raw = raw_world_to_camera[0, current_index].detach().float().cpu()
    common = {
        "selected_correspondences": int(selected.shape[0]),
        "inliers": 0,
        "inlier_ratio": 0.0,
        "reprojection_rmse_pixels": float("nan"),
        "inlier_positive_depth_fraction": float("nan"),
        "rotation_update_degrees": 0.0,
        "center_update_native": 0.0,
    }
    if force_fallback_reason is not None:
        return raw, {"optimized": 0, "reason": force_fallback_reason, **common}
    if selected.shape[0] < int(config.min_correspondences):
        return raw, {"optimized": 0, "reason": "fewer_than_min_correspondences", **common}
    cv2, np = _opencv()
    object_points = np.ascontiguousarray(
        pool.world_points.index_select(0, selected).double().numpy(),
        dtype=np.float64,
    )
    image_points = np.ascontiguousarray(
        pool.current_points.index_select(0, selected).double().numpy(),
        dtype=np.float64,
    )
    calibration = np.ascontiguousarray(
        intrinsics[0, current_index].detach().double().cpu().numpy(),
        dtype=np.float64,
    )
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
    if not ok or inlier_count < int(config.min_inliers):
        return raw, {
            "optimized": 0,
            "reason": "pnp_failed_or_fewer_than_min_inliers",
            **common,
            "inliers": inlier_count,
            "inlier_ratio": inlier_count / max(int(selected.shape[0]), 1),
        }
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
    projected, _ = cv2.projectPoints(
        object_points[inlier_ids],
        rvec,
        tvec,
        calibration,
        None,
    )
    residual = projected[:, 0] - image_points[inlier_ids]
    rmse = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
    inlier_camera = object_points[inlier_ids] @ rotation.T + translation
    positive_depth_fraction = float(np.mean(inlier_camera[:, 2] > 1e-6))
    delta = rotation @ raw_rotation.T
    rotation_cosine = float(np.clip((np.trace(delta) - 1.0) * 0.5, -1.0, 1.0))
    rotation_update = math.degrees(math.acos(rotation_cosine))
    raw_center = -(raw_rotation.T @ raw_translation)
    center = -(rotation.T @ translation)
    center_update = float(np.linalg.norm(center - raw_center))
    bounded = (
        rotation_update <= float(config.max_rotation_degrees)
        and center_update
        <= max(float(scene_scale) * float(config.max_translation_scene_fraction), 1e-6)
    )
    if not bounded or not np.isfinite(rotation).all() or not np.isfinite(translation).all():
        return raw, {
            "optimized": 0,
            "reason": "pnp_update_outside_locked_bounds",
            **common,
            "inliers": inlier_count,
            "inlier_ratio": inlier_count / int(selected.shape[0]),
            "reprojection_rmse_pixels": rmse,
            "inlier_positive_depth_fraction": positive_depth_fraction,
            "rotation_update_degrees": rotation_update,
            "center_update_native": center_update,
        }
    candidate = torch.from_numpy(
        np.concatenate((rotation, translation[:, None]), axis=1)
    ).float()
    return candidate, {
        "optimized": 1,
        "reason": "ok",
        **common,
        "inliers": inlier_count,
        "inlier_ratio": inlier_count / int(selected.shape[0]),
        "reprojection_rmse_pixels": rmse,
        "inlier_positive_depth_fraction": positive_depth_fraction,
        "rotation_update_degrees": rotation_update,
        "center_update_native": center_update,
    }


def _points_in_mask(points: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    height, width = mask.shape
    x = points[:, 0].round().long().clamp(0, width - 1)
    y = points[:, 1].round().long().clamp(0, height - 1)
    return mask[y, x].bool()


def _empty_pool(current_regions: dict[str, torch.Tensor]) -> MatchPool:
    return MatchPool(
        current_points=torch.empty(0, 2),
        world_points=torch.empty(0, 3),
        distances=torch.empty(0),
        anchor_indices=torch.empty(0, dtype=torch.long),
        anchor_region={name: torch.empty(0, dtype=torch.bool) for name in current_regions},
        current_region={name: torch.empty(0, dtype=torch.bool) for name in current_regions},
    )


def _gray_uint8(image: torch.Tensor, cv2, np):
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("Feature-PnP image must be RGB [3,H,W].")
    rgb = (
        image.detach().float().clamp(0, 1).mul(255).byte()
        .permute(1, 2, 0).contiguous().numpy()
    )
    cv2.setNumThreads(1)
    return np.ascontiguousarray(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))


def _opencv():
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("V0 Feature-PnP requires opencv-python.") from exc
    return cv2, np
