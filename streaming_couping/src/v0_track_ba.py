"""Causal StreamVGGT TrackHead factors for the V0 pose candidate.

This module deliberately has no GT inputs.  SAM contributes only masks,
identity state and spatial stratification; metric residuals come from frozen
StreamVGGT tracks and anchor depth.  Camera updates are explicit,
bounded SE(3) variables regularized to the raw StreamVGGT trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from .external_repos import maybe_add_repo_to_path
from .learned_pose.baseline_runtime import TrackBACandidateConfig


@dataclass(frozen=True)
class TrackWindow:
    anchor_index: int
    current_index: int
    sequence_indices: tuple[int, ...]
    query_points: torch.Tensor
    world_points: torch.Tensor
    tracks: torch.Tensor
    valid: torch.Tensor
    visibility: torch.Tensor
    confidence: torch.Tensor
    region_mask: torch.Tensor
    equal_count_region_feasible: bool
    validity_diagnostics: dict[str, float | int] = field(default_factory=dict)


def load_frozen_track_head(
    *,
    repo_path: str | Path,
    checkpoint_path: str | Path,
    device: str,
):
    """Load exactly the checkpoint TrackHead, then release unused modules."""

    maybe_add_repo_to_path(repo_path)
    from streamvggt.heads.track_head import TrackHead

    head = TrackHead(dim_in=2048, patch_size=14)
    try:
        state = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise ValueError("StreamVGGT checkpoint is not a state dictionary.")
    prefix = "track_head."
    selected = {}
    for name, value in state.items():
        normalized = str(name)
        while normalized.startswith("module."):
            normalized = normalized[len("module.") :]
        if normalized.startswith(prefix):
            selected[normalized[len(prefix) :]] = value
    if not selected:
        raise ValueError("StreamVGGT checkpoint contains no track_head weights.")
    head.load_state_dict(selected, strict=True)
    head = head.to(device).eval()
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    return head


def validate_track_cache(payload: dict) -> None:
    required = (
        "token_levels",
        "dpt_layer_indices",
        "stream_images",
        "patch_start_idx",
        "baseline_depth",
        "baseline_world_confidence",
        "tracking_masks_stream",
        "trusted_tracking_masks_stream",
        "quality",
        "identity_valid",
        "identity_unknown",
        "sam_birth_indices",
    )
    missing = [name for name in required if name not in payload]
    if missing:
        raise ValueError(f"V0 Track-BA cache lacks fields={missing}.")
    levels = payload["token_levels"]
    images = payload["stream_images"]
    if not torch.is_tensor(levels) or levels.ndim != 4:
        raise ValueError("token_levels must be [L,S,T,C].")
    if not torch.is_tensor(images) or images.ndim != 4:
        raise ValueError("stream_images must be [S,3,H,W].")
    if levels.shape[0] != len(payload["dpt_layer_indices"]):
        raise ValueError("DPT cache level names/shapes disagree.")
    if levels.shape[1] != images.shape[0]:
        raise ValueError("Cached token/image sequence lengths disagree.")
    if tuple(int(value) for value in payload.get("patch_shape", ())) != (
        int(images.shape[-2]) // 14,
        int(images.shape[-1]) // 14,
    ):
        raise ValueError("Cached patch shape does not match TrackHead images.")
    if not torch.is_tensor(payload["baseline_depth"]):
        raise ValueError("baseline_depth must be a tensor.")
    if not torch.is_tensor(payload["baseline_world_confidence"]):
        raise ValueError("baseline_world_confidence must be a tensor.")


def cache_tokens_for_track_head(
    payload: dict,
    *,
    indices: list[int],
    device: str,
    depth: int = 24,
) -> list[torch.Tensor | None]:
    levels = payload["token_levels"]
    layer_indices = tuple(int(value) for value in payload["dpt_layer_indices"])
    if tuple(layer_indices) != (4, 11, 17, 23):
        raise ValueError(
            "V0 Track-BA requires exact TrackHead DPT levels (4,11,17,23)."
        )
    selected = torch.tensor(indices, dtype=torch.long)
    output: list[torch.Tensor | None] = [None] * int(depth)
    for slot, layer_index in enumerate(layer_indices):
        output[layer_index] = (
            levels[slot]
            .index_select(0, selected)
            .unsqueeze(0)
            .to(device=device, dtype=torch.float32)
        )
    return output


@torch.no_grad()
def build_track_window(
    *,
    payload: dict,
    head,
    current_index: int,
    method: str,
    config: TrackBACandidateConfig,
    raw_world_to_camera: torch.Tensor,
    intrinsics: torch.Tensor,
) -> TrackWindow:
    """Produce one causal fixed-count factor window.

    The first frame in the prefix window is always the TrackHead query frame;
    neither tokens nor masks after ``current_index`` are materialized.
    """

    # Every cached token is causal at its original time. Re-anchoring the
    # TrackHead to the oldest frame of a recent consecutive window therefore
    # reads history but never future state. It also lets an object born after
    # frame 0 enter pose support once it is mature at a later window anchor.
    anchor = max(0, int(current_index) - int(config.window_frames) + 1)
    indices = list(range(anchor, int(current_index) + 1))
    device = config.device
    images = payload["stream_images"].index_select(
        0, torch.tensor(indices, dtype=torch.long)
    )
    height, width = (int(images.shape[-2]), int(images.shape[-1]))
    region = support_region(
        payload,
        frame_index=anchor,
        method=method,
        output_size=(height, width),
    )
    allowed = (
        support_region(
            payload,
            frame_index=anchor,
            method="sam_dynamic_excluded",
            output_size=(height, width),
        )
        if method.endswith("instance_background_stratified")
        else region
    )
    queries, region_feasible = sample_query_points(
        region,
        allowed=allowed,
        count=config.query_count,
        grid_shape=config.query_grid,
        method=method,
        frame_index=anchor,
    )
    token_list = cache_tokens_for_track_head(
        payload,
        indices=indices,
        device=device,
    )
    batch_images = images.unsqueeze(0).to(device=device, dtype=torch.float32)
    batch_queries = queries.unsqueeze(0).to(device=device, dtype=torch.float32)
    with _autocast_off(device):
        coordinate_iterations, visibility, confidence = head(
            token_list,
            batch_images,
            int(payload["patch_start_idx"]),
            query_points=batch_queries,
            iters=int(config.track_iterations),
        )
    tracks = coordinate_iterations[-1][0].detach().float().cpu()
    visibility = visibility[0].detach().float().cpu()
    confidence = confidence[0].detach().float().cpu()
    queries = queries.detach().float().cpu()
    depth = bilinear_sample_map(
        payload["baseline_depth"][anchor],
        queries,
    )[..., 0]
    point_confidence = bilinear_sample_map(
        payload["baseline_world_confidence"][anchor][..., None],
        queries,
    )[..., 0]
    world = unproject_depth_samples(
        queries,
        depth,
        raw_world_to_camera[0, anchor].detach().float().cpu(),
        intrinsics[0, anchor].detach().float().cpu(),
    )
    geometry_valid = torch.isfinite(world).all(dim=-1)
    geometry_valid &= torch.isfinite(depth) & (depth > 1e-6)
    geometry_valid &= torch.isfinite(point_confidence)
    geometry_valid &= point_confidence >= float(
        config.point_confidence_threshold
    )
    valid, validity_diagnostics = track_validity_and_diagnostics(
        tracks=tracks,
        visibility=visibility,
        confidence=confidence,
        geometry_valid=geometry_valid,
        width=width,
        height=height,
        visibility_threshold=float(config.visibility_threshold),
        confidence_threshold=float(config.track_confidence_threshold),
    )
    # The query observation itself is exact by definition and must survive
    # confidence filtering so every 3D point keeps its anchor observation.
    valid[0] = geometry_valid
    tracks[0] = queries
    return TrackWindow(
        anchor_index=anchor,
        current_index=int(current_index),
        sequence_indices=tuple(indices),
        query_points=queries,
        world_points=world,
        tracks=tracks,
        valid=valid,
        visibility=visibility,
        confidence=confidence,
        region_mask=region,
        equal_count_region_feasible=region_feasible,
        validity_diagnostics=validity_diagnostics,
    )


def track_validity_and_diagnostics(
    *,
    tracks: torch.Tensor,
    visibility: torch.Tensor,
    confidence: torch.Tensor,
    geometry_valid: torch.Tensor,
    width: int,
    height: int,
    visibility_threshold: float,
    confidence_threshold: float,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Apply the TrackHead gates and expose every current-frame factor.

    StreamVGGT's tracker already applies sigmoid to visibility/confidence.
    Keeping independent and cumulative counts here distinguishes a coordinate
    scale failure from low scores or invalid anchor geometry without changing
    any acceptance threshold.
    """

    if tracks.ndim != 3 or tracks.shape[-1] != 2:
        raise ValueError("TrackHead tracks must be [S,N,2].")
    if visibility.shape != tracks.shape[:2]:
        raise ValueError("TrackHead visibility shape disagrees with tracks.")
    if confidence.shape != tracks.shape[:2]:
        raise ValueError("TrackHead confidence shape disagrees with tracks.")
    if geometry_valid.shape != tracks.shape[1:2]:
        raise ValueError("TrackHead anchor geometry must be [N].")

    track_finite = torch.isfinite(tracks).all(dim=-1)
    in_bounds = (
        track_finite
        & (tracks[..., 0] >= 0)
        & (tracks[..., 0] <= int(width) - 1)
        & (tracks[..., 1] >= 0)
        & (tracks[..., 1] <= int(height) - 1)
    )
    visibility_finite = torch.isfinite(visibility)
    confidence_finite = torch.isfinite(confidence)
    visibility_pass = visibility_finite & (
        visibility >= float(visibility_threshold)
    )
    confidence_pass = confidence_finite & (
        confidence >= float(confidence_threshold)
    )
    finite_bounds_visibility = in_bounds & visibility_pass
    track_gate_pass = finite_bounds_visibility & confidence_pass
    valid = track_gate_pass & geometry_valid[None]

    current = -1
    query_count = int(tracks.shape[1])
    current_tracks = tracks[current]
    current_track_finite = track_finite[current]
    current_visibility = visibility[current]
    current_confidence = confidence[current]
    diagnostics: dict[str, float | int] = {
        "validity_image_width": int(width),
        "validity_image_height": int(height),
        "current_query_count": query_count,
        "current_geometry_valid_count": int(geometry_valid.sum()),
        "current_track_finite_count": int(current_track_finite.sum()),
        "current_track_in_bounds_count": int(in_bounds[current].sum()),
        "current_visibility_finite_count": int(
            visibility_finite[current].sum()
        ),
        "current_visibility_pass_count": int(
            visibility_pass[current].sum()
        ),
        "current_confidence_finite_count": int(
            confidence_finite[current].sum()
        ),
        "current_confidence_pass_count": int(confidence_pass[current].sum()),
        "current_finite_bounds_visibility_count": int(
            finite_bounds_visibility[current].sum()
        ),
        "current_track_gate_pass_count": int(track_gate_pass[current].sum()),
        "current_valid_after_geometry_count": int(valid[current].sum()),
        "current_track_in_bounds_fraction": _fraction(
            in_bounds[current], query_count
        ),
        "current_visibility_pass_fraction": _fraction(
            visibility_pass[current], query_count
        ),
        "current_confidence_pass_fraction": _fraction(
            confidence_pass[current], query_count
        ),
        "current_track_gate_pass_fraction": _fraction(
            track_gate_pass[current], query_count
        ),
        **_finite_range("current_visibility", current_visibility),
        **_finite_range("current_confidence", current_confidence),
        **_finite_range(
            "current_track_x",
            current_tracks[:, 0],
            mask=current_track_finite,
        ),
        **_finite_range(
            "current_track_y",
            current_tracks[:, 1],
            mask=current_track_finite,
        ),
    }
    return valid, diagnostics


def _fraction(mask: torch.Tensor, denominator: int) -> float:
    return float(mask.sum()) / max(int(denominator), 1)


def _finite_range(
    prefix: str,
    values: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
) -> dict[str, float]:
    finite = torch.isfinite(values)
    if mask is not None:
        finite &= mask
    selected = values[finite].float()
    if not selected.numel():
        return {
            f"{prefix}_min": float("nan"),
            f"{prefix}_mean": float("nan"),
            f"{prefix}_max": float("nan"),
        }
    return {
        f"{prefix}_min": float(selected.min()),
        f"{prefix}_mean": float(selected.mean()),
        f"{prefix}_max": float(selected.max()),
    }


def support_region(
    payload: dict,
    *,
    frame_index: int,
    method: str,
    output_size: tuple[int, int],
) -> torch.Tensor:
    """Build deployable equal-count selection regions from current SAM state."""

    height, width = output_size
    masks = _resize_masks(
        payload["tracking_masks_stream"][int(frame_index)].bool(),
        output_size,
    )
    trusted = _resize_masks(
        payload["trusted_tracking_masks_stream"][int(frame_index)].bool(),
        output_size,
    )
    observed = masks.any(dim=0)
    rigid_weight = payload.get("instance_rigid_weight")
    if torch.is_tensor(rigid_weight):
        static_slots = rigid_weight[int(frame_index)].float() > 0
        trusted = trusted & static_slots[:, None, None]
    trusted_union = trusted.any(dim=0)
    # Observed regions that cannot be associated to a persistent ID are
    # treated as possibly dynamic. Trusted static regions remain eligible.
    uncertain_or_dynamic = observed & ~trusted_union
    if method == "full_image":
        return torch.ones(height, width, dtype=torch.bool)
    if method == "sam_dynamic_excluded":
        return ~uncertain_or_dynamic
    if method == "sam_instance_background_stratified":
        return trusted_union & ~uncertain_or_dynamic
    if method == "bbox_instance_background_stratified":
        return bounding_box_mask(trusted_union) & ~uncertain_or_dynamic
    if method == "random_instance_background_stratified":
        shift_y = 1 + (int(frame_index) * 17) % max(height - 1, 1)
        shift_x = 1 + (int(frame_index) * 29) % max(width - 1, 1)
        return torch.roll(
            trusted_union,
            shifts=(shift_y, shift_x),
            dims=(0, 1),
        ) & ~uncertain_or_dynamic
    raise ValueError(f"Unknown Track-BA support method={method!r}.")


def sample_query_points(
    region: torch.Tensor,
    *,
    allowed: torch.Tensor | None = None,
    count: int,
    grid_shape: tuple[int, int],
    method: str,
    frame_index: int,
) -> tuple[torch.Tensor, bool]:
    """Deterministic UV-stratified equal-count sampling.

    The stratified methods allocate exactly half their support to the supplied
    region and half to its complement.  If either half is infeasible they use
    the whole image, and the diagnostic can report that SAM was unavailable;
    the point count never changes across controls.
    """

    if region.ndim != 2:
        raise ValueError("Track-BA sampling region must be [H,W].")
    if allowed is None:
        allowed = torch.ones_like(region)
    if allowed.shape != region.shape:
        raise ValueError("Track-BA allowed/region masks must share shape.")
    height, width = region.shape
    grid_y = torch.linspace(0, height - 1, int(grid_shape[0]))
    grid_x = torch.linspace(0, width - 1, int(grid_shape[1]))
    y, x = torch.meshgrid(grid_y, grid_x, indexing="ij")
    candidates = torch.stack((x.reshape(-1), y.reshape(-1)), dim=-1)
    pixel_x = candidates[:, 0].round().long().clamp(0, width - 1)
    pixel_y = candidates[:, 1].round().long().clamp(0, height - 1)
    candidate_allowed = allowed[pixel_y, pixel_x]
    inside = region[pixel_y, pixel_x] & candidate_allowed
    stratified = method.endswith("instance_background_stratified")
    if stratified:
        half = int(count) // 2
        left = candidates[inside]
        right = candidates[candidate_allowed & ~inside]
        if left.shape[0] >= half and right.shape[0] >= half:
            feasible = True
            selected = torch.cat(
                (
                    farthest_uv(left, half),
                    farthest_uv(right, int(count) - half),
                ),
                dim=0,
            )
        else:
            feasible = False
            selected = farthest_uv(candidates, int(count))
    else:
        allowed_points = candidates[inside]
        feasible = allowed_points.shape[0] >= int(count)
        selected = farthest_uv(
            allowed_points if feasible else candidates,
            int(count),
        )
    if selected.shape[0] != int(count):
        raise RuntimeError(
            f"Track-BA equal-count sampling failed method={method} "
            f"frame={frame_index}: {selected.shape[0]}/{count}."
        )
    return selected.float(), bool(feasible)


def farthest_uv(points: torch.Tensor, count: int) -> torch.Tensor:
    count = min(int(count), int(points.shape[0]))
    if count <= 0:
        return points[:0]
    center = points.mean(dim=0, keepdim=True)
    selected = torch.empty(count, dtype=torch.long)
    selected[0] = torch.linalg.vector_norm(points - center, dim=-1).argmin()
    distance = (points - points[selected[0]]).square().sum(dim=-1)
    for index in range(1, count):
        current = distance.argmax()
        selected[index] = current
        distance = torch.minimum(
            distance,
            (points - points[current]).square().sum(dim=-1),
        )
    return points.index_select(0, selected)


def bilinear_sample_map(value: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    if value.ndim != 3:
        raise ValueError("Bilinear map must be [H,W,C].")
    height, width, _ = value.shape
    normalized = points.clone().float()
    normalized[:, 0] = 2.0 * normalized[:, 0] / max(width - 1, 1) - 1.0
    normalized[:, 1] = 2.0 * normalized[:, 1] / max(height - 1, 1) - 1.0
    sampled = F.grid_sample(
        value.permute(2, 0, 1)[None].float(),
        normalized[None, None],
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return sampled[0, :, 0].T


def unproject_depth_samples(
    points: torch.Tensor,
    depth: torch.Tensor,
    world_to_camera: torch.Tensor,
    intrinsics: torch.Tensor,
) -> torch.Tensor:
    """Unproject raw depth through raw K/pose into the baseline world gauge."""

    x = (points[:, 0] - intrinsics[0, 2]) / intrinsics[0, 0].clamp_min(1e-6)
    y = (points[:, 1] - intrinsics[1, 2]) / intrinsics[1, 1].clamp_min(1e-6)
    camera = torch.stack((x * depth, y * depth, depth), dim=-1)
    rotation = world_to_camera[:3, :3]
    translation = world_to_camera[:3, 3]
    return (camera - translation) @ rotation


def optimize_track_window(
    *,
    raw_world_to_camera: torch.Tensor,
    intrinsics: torch.Tensor,
    window: TrackWindow,
    scene_scale: float,
    config: TrackBACandidateConfig,
) -> tuple[torch.Tensor, dict[str, float | int | str]]:
    """Optimize bounded current-camera delta; 3D, K and history stay frozen."""

    device = config.device
    selected_raw = torch.tensor(
        window.sequence_indices,
        dtype=torch.long,
        device=raw_world_to_camera.device,
    )
    raw = (
        raw_world_to_camera[0]
        .index_select(0, selected_raw)
        .detach()
        .double()
        .to(device)
    )
    raw_full = _homogeneous_pose(raw)
    raw_anchor_inverse = torch.linalg.inv(raw_full[0])
    raw_relative = (raw_full @ raw_anchor_inverse)[:, :3]
    selected_k = selected_raw.to(intrinsics.device)
    calibration = (
        intrinsics[0]
        .index_select(0, selected_k)
        .detach()
        .double()
        .to(device)
    )
    points_global = window.world_points.detach().double().to(device)
    points = points_global @ raw[0, :3, :3].T + raw[0, :3, 3]
    tracks = window.tracks.detach().double().to(device)
    valid = window.valid.detach().bool().to(device)
    counts = valid.sum(dim=1)
    if min(int(counts[0]), int(counts[-1])) < int(config.min_correspondences):
        return raw[-1].float().cpu(), {
            "optimized": 0,
            "reason": "fewer_than_min_correspondences",
            "anchor_correspondences": int(counts[0]),
            "current_correspondences": int(counts[-1]),
            "mean_correspondences": float(counts.float().mean().cpu()),
            "initial_reprojection_rmse_pixels": float("nan"),
            "final_reprojection_rmse_pixels": float("nan"),
            "optimizer_loss": float("nan"),
            "current_rotation_update_degrees": 0.0,
            "current_translation_update_native": 0.0,
        }
    # Fix the anchor camera as the gauge. All deployable corrections affect
    # later cameras only and cannot retroactively alter the query 3D frame.
    rotation_raw = raw_relative[:, :3, :3]
    translation_raw = raw_relative[:, :3, 3]
    delta_rotation = torch.zeros(
        1,
        3,
        dtype=torch.double,
        device=device,
        requires_grad=True,
    )
    delta_translation = torch.zeros_like(
        delta_rotation,
        requires_grad=True,
    )
    optimizer = torch.optim.Adam(
        (delta_rotation, delta_translation),
        lr=float(config.learning_rate),
    )
    maximum_rotation = math.radians(float(config.max_rotation_degrees))
    maximum_translation = max(
        float(scene_scale) * float(config.max_translation_scene_fraction),
        1e-4,
    )

    def build_pose() -> tuple[torch.Tensor, torch.Tensor]:
        bounded_rotation = maximum_rotation * torch.tanh(delta_rotation)
        bounded_translation = maximum_translation * torch.tanh(
            delta_translation
        )
        update = so3_exp(bounded_rotation)[0]
        rotation = torch.cat(
            (rotation_raw[:-1], (update @ rotation_raw[-1])[None]),
            dim=0,
        )
        translation = torch.cat(
            (
                translation_raw[:-1],
                (translation_raw[-1] + bounded_translation[0])[None],
            ),
            dim=0,
        )
        return rotation, translation

    with torch.no_grad():
        initial_reprojection = _reprojection_rmse(
            points,
            rotation_raw[-1:],
            translation_raw[-1:],
            calibration[-1:],
            tracks[-1:],
            valid[-1:],
        )
    best_loss = float("inf")
    best_rotation = rotation_raw.clone()
    best_translation = translation_raw.clone()
    for _ in range(int(config.optimizer_steps)):
        optimizer.zero_grad(set_to_none=True)
        rotation, translation = build_pose()
        projected, positive_depth = project_points(
            points,
            rotation,
            translation,
            calibration,
        )
        active = valid[-1:] & positive_depth[-1:]
        residual = torch.linalg.vector_norm(
            projected[-1:] - tracks[-1:],
            dim=-1,
        )
        data_loss = pseudo_huber(
            residual[active],
            float(config.robust_delta_pixels),
        ).mean()
        rotation_prior = delta_rotation.square().mean()
        translation_prior = delta_translation.square().mean()
        # The current correction is explicitly shrunk toward the frozen raw
        # history; there are no free historical camera variables.
        temporal_prior = rotation_prior + translation_prior
        loss = (
            data_loss
            + float(config.rotation_prior_weight) * rotation_prior
            + float(config.translation_prior_weight) * translation_prior
            + float(config.temporal_prior_weight) * temporal_prior
        )
        if not torch.isfinite(loss):
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            (delta_rotation, delta_translation),
            max_norm=5.0,
        )
        optimizer.step()
        value = float(loss.detach().cpu())
        if value < best_loss:
            best_loss = value
            with torch.no_grad():
                best_rotation, best_translation = build_pose()
                best_rotation = best_rotation.detach().clone()
                best_translation = best_translation.detach().clone()
    candidate = torch.cat(
        (best_rotation[-1], best_translation[-1, :, None]),
        dim=-1,
    )
    bounded_rotation = maximum_rotation * torch.tanh(delta_rotation.detach())
    bounded_translation = maximum_translation * torch.tanh(
        delta_translation.detach()
    )
    final_reprojection = _reprojection_rmse(
        points,
        best_rotation[-1:],
        best_translation[-1:],
        calibration[-1:],
        tracks[-1:],
        valid[-1:],
    )
    improved_objective = (
        math.isfinite(final_reprojection)
        and math.isfinite(initial_reprojection)
        and final_reprojection < initial_reprojection
    )
    if not improved_objective:
        candidate = raw_relative[-1]
    return candidate.float().cpu(), {
        "optimized": int(math.isfinite(best_loss) and improved_objective),
        "reason": (
            "ok"
            if math.isfinite(best_loss) and improved_objective
            else (
                "nonfinite_loss"
                if not math.isfinite(best_loss)
                else "reprojection_not_improved"
            )
        ),
        "anchor_correspondences": int(counts[0]),
        "current_correspondences": int(counts[-1]),
        "mean_correspondences": float(counts.float().mean().cpu()),
        "initial_reprojection_rmse_pixels": initial_reprojection,
        "final_reprojection_rmse_pixels": final_reprojection,
        "optimizer_loss": best_loss,
        "current_rotation_update_degrees": float(
            torch.rad2deg(torch.linalg.vector_norm(bounded_rotation[0])).cpu()
        ),
        "current_translation_update_native": float(
            torch.linalg.vector_norm(bounded_translation[0]).cpu()
        ),
    }


def compose_relative_candidate(
    *,
    raw_world_to_camera: torch.Tensor,
    anchor_index: int,
    candidate_relative: torch.Tensor,
) -> torch.Tensor:
    """Map an anchor-relative W2C result back into the global raw gauge."""

    anchor = raw_world_to_camera[0, int(anchor_index)].detach().float().cpu()
    anchor_h = _homogeneous_pose(anchor[None])[0]
    relative_h = _homogeneous_pose(candidate_relative[None])[0]
    return (relative_h @ anchor_h)[:3]


def project_points(
    points: torch.Tensor,
    rotation: torch.Tensor,
    translation: torch.Tensor,
    intrinsics: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    camera = torch.einsum("sij,pj->spi", rotation, points)
    camera = camera + translation[:, None, :]
    depth = camera[..., 2]
    safe_depth = depth.clamp_min(1e-6)
    u = intrinsics[:, None, 0, 0] * camera[..., 0] / safe_depth
    u = u + intrinsics[:, None, 0, 2]
    v = intrinsics[:, None, 1, 1] * camera[..., 1] / safe_depth
    v = v + intrinsics[:, None, 1, 2]
    return torch.stack((u, v), dim=-1), depth > 1e-6


def _homogeneous_pose(world_to_camera: torch.Tensor) -> torch.Tensor:
    if world_to_camera.ndim != 3 or world_to_camera.shape[-2:] != (3, 4):
        raise ValueError("Expected world-to-camera poses [S,3,4].")
    bottom = torch.zeros(
        world_to_camera.shape[0],
        1,
        4,
        dtype=world_to_camera.dtype,
        device=world_to_camera.device,
    )
    bottom[:, 0, 3] = 1.0
    return torch.cat((world_to_camera, bottom), dim=-2)


def so3_exp(vector: torch.Tensor) -> torch.Tensor:
    """Stable Rodrigues map for batched axis-angle vectors."""

    theta2 = vector.square().sum(dim=-1, keepdim=True)
    theta = torch.sqrt(theta2.clamp_min(1e-16))
    safe_theta2 = theta2.clamp_min(1e-16)
    a = torch.where(
        theta2 > 1e-8,
        torch.sin(theta) / theta,
        1.0 - theta2 / 6.0 + theta2.square() / 120.0,
    )
    b = torch.where(
        theta2 > 1e-8,
        (1.0 - torch.cos(theta)) / safe_theta2,
        0.5 - theta2 / 24.0 + theta2.square() / 720.0,
    )
    x, y, z = vector.unbind(dim=-1)
    zero = torch.zeros_like(x)
    skew = torch.stack(
        (zero, -z, y, z, zero, -x, -y, x, zero),
        dim=-1,
    ).reshape(vector.shape[:-1] + (3, 3))
    identity = torch.eye(3, dtype=vector.dtype, device=vector.device)
    identity = identity.expand(vector.shape[:-1] + (3, 3))
    return identity + a[..., None] * skew + b[..., None] * (skew @ skew)


def pseudo_huber(residual: torch.Tensor, delta: float) -> torch.Tensor:
    scaled = residual / float(delta)
    return float(delta) ** 2 * (torch.sqrt(1.0 + scaled.square()) - 1.0)


def scene_scale_from_cache(payload: dict, threshold: float) -> float:
    depth = payload["baseline_depth"][0].float()
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    confidence = payload["baseline_world_confidence"][0].float()
    # Median depth is a stable native-gauge proxy and does not read GT.
    valid = (
        torch.isfinite(depth)
        & (depth > 1e-6)
        & torch.isfinite(confidence)
        & (confidence >= float(threshold))
    )
    selected = depth[valid]
    if selected.numel() < 128:
        selected = depth[torch.isfinite(depth) & (depth > 1e-6)]
    if selected.numel() == 0:
        return 1.0
    return float(torch.quantile(selected, 0.50).clamp_min(1e-3))


def region_coverage(mask: torch.Tensor) -> float:
    return float(mask.float().mean())


def bounding_box_mask(mask: torch.Tensor) -> torch.Tensor:
    result = torch.zeros_like(mask)
    indices = torch.nonzero(mask, as_tuple=False)
    if indices.numel():
        y0, x0 = indices.amin(dim=0)
        y1, x1 = indices.amax(dim=0)
        result[y0 : y1 + 1, x0 : x1 + 1] = True
    return result


def _resize_masks(
    masks: torch.Tensor,
    size: tuple[int, int],
) -> torch.Tensor:
    if tuple(masks.shape[-2:]) == tuple(size):
        return masks
    return F.interpolate(
        masks[:, None].float(),
        size=size,
        mode="nearest",
    )[:, 0].bool()


def _reprojection_rmse(
    points: torch.Tensor,
    rotation: torch.Tensor,
    translation: torch.Tensor,
    intrinsics: torch.Tensor,
    tracks: torch.Tensor,
    valid: torch.Tensor,
) -> float:
    with torch.no_grad():
        projected, positive = project_points(
            points,
            rotation,
            translation,
            intrinsics,
        )
        active = valid & positive
        if not bool(active.any()):
            return float("nan")
        error2 = (projected - tracks).square().sum(dim=-1)
        return float(torch.sqrt(error2[active].mean()).cpu())


def _autocast_off(device: str):
    device_type = "cuda" if str(device).startswith("cuda") else "cpu"
    return torch.autocast(device_type=device_type, enabled=False)
