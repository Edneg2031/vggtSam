#!/usr/bin/env python3
"""Dependency-light tensor smoke for V8 O1/O2 theory validation."""

from __future__ import annotations

import torch

from streaming_couping.src.solvers.weighted_kabsch import (
    KabschConfig,
    weighted_kabsch,
)
from streaming_couping.src.v80_pose_geometry import (
    backproject_depth_at_local_tokens,
    causal_gt_nearest_pairs,
    causal_history_indices,
    gather_pair_points,
    invert_rigid,
    rotation_error_degrees,
    transform_points,
)


def main() -> None:
    torch.manual_seed(80)
    _smoke_kabsch()
    _smoke_coordinates_and_causality()
    print("V8 theory-validation smoke passed")


def _smoke_kabsch() -> None:
    source = torch.randn(2, 20, 3, dtype=torch.float64)
    angle = torch.tensor([0.20, -0.15], dtype=torch.float64)
    cosine, sine = torch.cos(angle), torch.sin(angle)
    rotation = torch.stack(
        [
            cosine,
            -sine,
            torch.zeros_like(angle),
            sine,
            cosine,
            torch.zeros_like(angle),
            torch.zeros_like(angle),
            torch.zeros_like(angle),
            torch.ones_like(angle),
        ],
        dim=-1,
    ).reshape(2, 3, 3)
    translation = torch.tensor([[0.1, -0.2, 0.05], [-0.3, 0.1, 0.2]]).double()
    target = torch.einsum("bij,bnj->bni", rotation, source) + translation[:, None]
    result = weighted_kabsch(
        source,
        target,
        config=KabschConfig(min_points=6, inlier_distance=1e-4),
    )
    _require(bool(result.accepted.all()), "batched Kabsch rejected valid geometry")
    _require(
        float((result.rotation - rotation).abs().max()) < 1e-8,
        "batched Kabsch rotation mismatch",
    )
    _require(
        float((result.translation - translation).abs().max()) < 1e-8,
        "batched Kabsch translation mismatch",
    )

    contaminated = target[0].clone()
    contaminated[-4:] += 20.0
    trimmed = weighted_kabsch(
        source[0],
        contaminated,
        config=KabschConfig(
            min_points=6,
            trim_quantile=0.70,
            trim_iterations=4,
            inlier_distance=1e-3,
        ),
    )
    _require(bool(trimmed.accepted), "trimmed Kabsch rejected")
    _require(float(trimmed.rmse) < 1e-7, "trimmed Kabsch did not reject outliers")

    empty = weighted_kabsch(
        torch.empty(0, 3),
        torch.empty(0, 3),
        config=KabschConfig(min_points=6),
    )
    _require(not bool(empty.accepted), "empty Kabsch was accepted")
    _require(
        torch.equal(empty.transform, torch.eye(4)),
        "empty Kabsch fallback is not exact identity",
    )

    partly_invalid = source[0].clone()
    partly_invalid[:2] = float("nan")
    robust = weighted_kabsch(
        partly_invalid,
        target[0],
        config=KabschConfig(min_points=6),
    )
    _require(bool(robust.accepted), "Kabsch did not ignore non-finite pairs")
    _require(int(robust.point_count) == 18, "Kabsch finite-pair count is wrong")


def _smoke_coordinates_and_causality() -> None:
    sequence, instances, points = 3, 1, 4
    uv = torch.tensor(
        [[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]]
    )
    local = torch.zeros(sequence, instances, points, 7)
    local[..., 3:5] = uv
    valid = torch.ones(sequence, instances, points, dtype=torch.bool)
    depth = torch.full((sequence, 2, 2, 1), 2.0)
    intrinsics = torch.tensor(
        [[[2.0, 0.0, 0.5], [0.0, 2.0, 0.5], [0.0, 0.0, 1.0]]]
    ).repeat(sequence, 1, 1)
    camera, camera_valid = backproject_depth_at_local_tokens(
        depth,
        intrinsics,
        local_features=local,
        local_valid=valid,
    )
    _require(bool(camera_valid.all()), "depth backprojection lost valid points")
    _require(
        torch.allclose(camera[..., 2], torch.full_like(camera[..., 2], 2.0)),
        "depth backprojection changed Z",
    )
    local_with_nan = local.clone()
    local_with_nan[0, 0, 0, 3] = float("nan")
    _, nan_valid = backproject_depth_at_local_tokens(
        depth,
        intrinsics,
        local_features=local_with_nan,
        local_valid=valid,
    )
    _require(not bool(nan_valid[0, 0, 0]), "non-finite UV was accepted")

    write = torch.tensor([[True], [False], [True]])
    history = causal_history_indices(write, valid)
    _require(history[:, 0].tolist() == [-1, 0, 0], "causal history is not previous-only")
    world = camera.clone()
    pairs = causal_gt_nearest_pairs(
        current_frame=2,
        history_indices=history[2],
        gt_world_metric=world,
        gt_valid=valid,
        max_distance_metric=0.01,
        require_mutual_nearest=True,
    )
    _require(pairs.count == points, "mutual GT pseudo-match count mismatch")
    current, previous, pair_valid = gather_pair_points(
        camera, valid, current_frame=2, pairs=pairs
    )
    _require(bool(pair_valid.all()), "gathered causal pair became invalid")
    _require(torch.equal(current, previous), "causal point gathering is misaligned")

    no_pairs = causal_gt_nearest_pairs(
        current_frame=0,
        history_indices=history[0],
        gt_world_metric=world,
        gt_valid=valid,
        max_distance_metric=0.01,
        require_mutual_nearest=True,
    )
    _require(no_pairs.count == 0, "missing history produced correspondences")
    no_current, no_previous, no_valid = gather_pair_points(
        camera, valid, current_frame=0, pairs=no_pairs
    )
    _require(
        no_current.shape == no_previous.shape == (0, 3)
        and no_valid.shape == (0,),
        "empty pair gathering changed tensor shapes",
    )

    c2w = torch.eye(4)
    c2w[:3, 3] = torch.tensor([0.1, -0.2, 0.3])
    transformed = transform_points(c2w, current)
    recovered = transform_points(invert_rigid(c2w), transformed)
    _require(torch.allclose(recovered, current, atol=1e-6), "pose round-trip failed")
    mixed_precision = transform_points(c2w.double(), current.float())
    _require(
        mixed_precision.dtype == torch.float64,
        "mixed-precision transform did not promote to a common dtype",
    )
    _require(
        torch.allclose(mixed_precision, transformed.double(), atol=1e-8),
        "mixed-precision transform changed coordinates",
    )
    batched_c2w = c2w[None].repeat(sequence, 1, 1)
    batched_c2w[:, :3, 3] += torch.arange(sequence).float()[:, None] * 0.01
    world = transform_points(batched_c2w[:, None], camera)
    camera_roundtrip = transform_points(
        invert_rigid(batched_c2w)[:, None], world
    )
    _require(
        torch.allclose(camera_roundtrip, camera, atol=1e-6),
        "batched pose/token round-trip failed",
    )
    _require(
        float(rotation_error_degrees(c2w, c2w)) < 1e-6,
        "identity rotation error is nonzero",
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"V8 theory smoke failed: {message}")


if __name__ == "__main__":
    main()
