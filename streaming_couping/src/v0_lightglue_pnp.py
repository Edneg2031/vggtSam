"""Frozen ALIKED-LightGlue correspondences for the V0 PnP audit.

This module has no GT inputs and trains no parameters.  It changes only the
correspondence source from the completed SIFT factor experiment: landmarks are
the frozen StreamVGGT native world pointmap, history is all causal, and pose is
the same calibrated PnP-RANSAC + LM implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .external_repos import maybe_add_repo_to_path
from .learned_pose.baseline_runtime import LightGluePnPCandidateConfig
from .v0_feature_pnp import MatchPool, _empty_pool, _points_in_mask
from .v0_track_ba import bilinear_sample_map, support_region


@dataclass(frozen=True)
class LightGlueFeatureSet:
    points: torch.Tensor
    descriptors: torch.Tensor
    scores: torch.Tensor
    image_size: torch.Tensor


def load_frozen_aliked_lightglue(
    *,
    repo_path: str | Path,
    device: str,
    nfeatures: int,
    detection_threshold: float,
):
    """Load the public frozen ALIKED extractor and its matched LightGlue."""

    maybe_add_repo_to_path(repo_path)
    try:
        from lightglue import ALIKED, LightGlue
    except ImportError as exc:
        raise RuntimeError(
            "V0 r7 requires the pinned LightGlue repository and kornia. "
            "Run commands_v0_track_ba_candidate.txt so its preflight can "
            "prepare them."
        ) from exc
    extractor = ALIKED(
        max_num_keypoints=int(nfeatures),
        detection_threshold=float(detection_threshold),
    ).to(device).eval()
    matcher = LightGlue(features="aliked").to(device).eval()
    for module in (extractor, matcher):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    return extractor, matcher


@torch.no_grad()
def extract_aliked_features(
    image: torch.Tensor,
    *,
    extractor,
    device: str,
) -> LightGlueFeatureSet:
    data = extractor.extract(image.detach().float().clamp(0, 1).to(device))
    required = ("keypoints", "descriptors", "keypoint_scores", "image_size")
    missing = [name for name in required if name not in data]
    if missing:
        raise ValueError(f"ALIKED output lacks fields={missing}.")
    return LightGlueFeatureSet(
        points=data["keypoints"][0].detach().float().cpu(),
        descriptors=data["descriptors"][0].detach().float().cpu(),
        scores=data["keypoint_scores"][0].detach().float().cpu(),
        image_size=data["image_size"][0].detach().float().cpu(),
    )


@torch.no_grad()
def match_aliked_features(
    left: LightGlueFeatureSet,
    right: LightGlueFeatureSet,
    *,
    matcher,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return compact [left,right] indices and pretrained match scores."""

    if not left.points.shape[0] or not right.points.shape[0]:
        return torch.empty(0, 2, dtype=torch.long), torch.empty(0)

    def packed(features: LightGlueFeatureSet) -> dict[str, torch.Tensor]:
        return {
            "keypoints": features.points[None].to(device),
            "descriptors": features.descriptors[None].to(device),
            "keypoint_scores": features.scores[None].to(device),
            "image_size": features.image_size[None].to(device),
        }

    prediction = matcher({"image0": packed(left), "image1": packed(right)})
    matches = prediction["matches"][0].detach().long().cpu()
    scores = prediction["scores"][0].detach().float().cpu()
    if matches.ndim != 2 or matches.shape[-1] != 2:
        raise ValueError("LightGlue compact matches must have shape [K,2].")
    if scores.shape != matches.shape[:1]:
        raise ValueError("LightGlue match score count is inconsistent.")
    return matches, scores


def build_lightglue_match_pool(
    *,
    payload: dict,
    current_index: int,
    features: list[LightGlueFeatureSet | None],
    matcher,
    device: str,
    config: LightGluePnPCandidateConfig,
) -> tuple[MatchPool, list[dict[str, int | float]]]:
    """Pool current-to-every-history matches with one current key at most."""

    current = features[int(current_index)]
    if current is None:
        raise ValueError(f"Missing ALIKED features for current={current_index}.")
    height, width = tuple(int(value) for value in payload["stream_images"].shape[-2:])
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
    pair_rows: list[dict[str, int | float]] = []
    for anchor_index in range(int(current_index)):
        anchor = features[anchor_index]
        if anchor is None:
            raise ValueError(f"Missing ALIKED features for anchor={anchor_index}.")
        matches, scores = match_aliked_features(
            anchor,
            current,
            matcher=matcher,
            device=device,
        )
        if matches.shape[0]:
            anchor_points = anchor.points.index_select(0, matches[:, 0])
            current_points = current.points.index_select(0, matches[:, 1])
            native_world = bilinear_sample_map(
                payload["baseline_world_points"][anchor_index],
                anchor_points,
            )
            confidence = bilinear_sample_map(
                payload["baseline_world_confidence"][anchor_index][..., None],
                anchor_points,
            )[..., 0]
            valid = torch.isfinite(native_world).all(dim=-1)
            valid &= torch.isfinite(confidence)
            valid &= confidence >= float(config.point_confidence_threshold)
        else:
            anchor_points = torch.empty(0, 2)
            current_points = torch.empty(0, 2)
            native_world = torch.empty(0, 3)
            valid = torch.empty(0, dtype=torch.bool)
        pair_rows.append(
            {
                "anchor_sequence_index": int(anchor_index),
                "current_sequence_index": int(current_index),
                "raw_pair_matches": int(matches.shape[0]),
                "geometry_valid_pair_matches": int(valid.sum()),
            }
        )
        if valid.any():
            records.append(
                (
                    anchor_index,
                    matches[valid, 1],
                    anchor_points[valid],
                    current_points[valid],
                    native_world[valid],
                    scores[valid],
                )
            )
    if not records:
        return _empty_pool(current_regions), pair_rows

    # Every current ALIKED keypoint may vote for only one historical landmark.
    # Prefer pretrained match confidence, then the most recent history frame.
    candidates = []
    for (
        anchor_index,
        current_ids,
        anchor_points,
        current_points,
        native_world,
        scores,
    ) in records:
        for row in range(int(current_points.shape[0])):
            candidates.append(
                (
                    -float(scores[row]),
                    -int(anchor_index),
                    int(current_ids[row]),
                    int(anchor_index),
                    anchor_points[row],
                    current_points[row],
                    native_world[row],
                    float(scores[row]),
                )
            )
    candidates.sort(key=lambda item: item[:3])
    retained = []
    used_current: set[int] = set()
    for item in candidates:
        if item[2] in used_current:
            continue
        used_current.add(item[2])
        retained.append(item)

    anchor_indices = torch.tensor([item[3] for item in retained], dtype=torch.long)
    anchor_points = torch.stack([item[4] for item in retained])
    current_points = torch.stack([item[5] for item in retained])
    world_points = torch.stack([item[6] for item in retained])
    match_scores = torch.tensor([item[7] for item in retained], dtype=torch.float32)
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
        anchor_regions[method] = torch.stack(
            [
                _points_in_mask(
                    anchor_points[row : row + 1],
                    region_cache[(method, anchor_index)],
                )[0]
                for row, anchor_index in enumerate(anchor_indices.tolist())
            ]
        ).bool()
    current_region_flags = {
        method: _points_in_mask(current_points, region)
        for method, region in current_regions.items()
    }
    return (
        MatchPool(
            current_points=current_points,
            world_points=world_points,
            # Existing equal-count selection sorts ascending distances.
            distances=1.0 - match_scores,
            anchor_indices=anchor_indices,
            anchor_region=anchor_regions,
            current_region=current_region_flags,
        ),
        pair_rows,
    )


def spatial_hull_coverage(
    points: torch.Tensor,
    *,
    image_size: tuple[int, int],
) -> float:
    """Convex-hull area divided by image area, for correspondence audit."""

    if points.shape[0] < 3:
        return 0.0
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("V0 r7 spatial audit requires opencv-python.") from exc
    array = np.ascontiguousarray(points.detach().float().cpu().numpy(), dtype=np.float32)
    hull = cv2.convexHull(array)
    area = float(cv2.contourArea(hull))
    height, width = image_size
    return area / max(float(height * width), 1.0)
