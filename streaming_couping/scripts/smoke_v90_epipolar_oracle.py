#!/usr/bin/env python3
"""Dependency-light V9 Stage-O smoke: visibility, solver and fallback."""

from __future__ import annotations

import math
from pathlib import Path

import torch

from streaming_couping.scripts.run_v90_epipolar_oracle import (
    FRAME_COLUMNS,
    V90Config,
    _evaluate_frame,
)
from streaming_couping.src.v80_pose_geometry import homogeneous, invert_rigid
from streaming_couping.src.v90_epipolar_geometry import (
    EpipolarConfig,
    VisibilityConfig,
    causal_mask_history_indices,
    estimate_relative_epipolar_pose,
    recover_absolute_pose,
    relative_translation_direction_error_degrees,
    surface_reprojection_correspondences,
)


def main() -> None:
    torch.manual_seed(90)
    _smoke_surface_reprojection_and_causality()
    _smoke_epipolar_pose()
    _smoke_exact_fallback()
    _smoke_runner_frame()
    print("V9 epipolar-oracle smoke passed")


def _smoke_surface_reprojection_and_causality() -> None:
    sequence, slots, height, width = 3, 2, 24, 32
    masks = torch.zeros(sequence, slots, height, width, dtype=torch.bool)
    masks[0, 0, 3:21, 4:28] = True
    masks[1, 0, 3:21, 4:28] = True
    masks[1, 1, 6:18, 8:24] = True
    masks[2, 1, 6:18, 8:24] = True
    history = causal_mask_history_indices(masks, max_history=2)
    _require(history[0].eq(-1).all(), "birth frame cannot read itself")
    _require(int(history[1, 0, 0]) == 0, "mature slot reads frame zero")
    _require(int(history[1, 1, 0]) == -1, "late birth only writes memory")
    _require(int(history[2, 1, 0]) == 1, "late-born slot matures next observation")

    focal = 30.0
    k = torch.tensor(
        [[focal, 0.0, 15.5], [0.0, focal, 11.5], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    ).repeat(sequence, 1, 1)
    y, x = torch.meshgrid(
        torch.arange(height, dtype=torch.float64),
        torch.arange(width, dtype=torch.float64),
        indexing="ij",
    )
    z = torch.full_like(x, 2.0)
    world = torch.stack(
        [(x - k[0, 0, 2]) * z / focal, (y - k[0, 1, 2]) * z / focal, z],
        dim=-1,
    ).repeat(sequence, 1, 1, 1)
    depth = z.repeat(sequence, 1, 1)
    pose = torch.eye(4, dtype=torch.float64).repeat(sequence, 1, 1)
    pairs = surface_reprojection_correspondences(
        current_frame=1,
        history_frame=0,
        slot=0,
        masks=masks,
        world_points_metric=world,
        depth_metric=depth,
        global_world_to_camera=pose,
        intrinsics=k,
        config=VisibilityConfig(max_queries_per_instance=64),
    )
    _require(pairs.count == 64, "visible surface reprojection count")
    _require(
        float((pairs.current_uv - pairs.history_uv).abs().max()) < 1e-10,
        "identity reprojection preserves subpixel target",
    )


def _smoke_epipolar_pose() -> None:
    count = 96
    world = torch.randn(count, 3, dtype=torch.float64)
    world[:, 0] *= 0.8
    world[:, 1] *= 0.5
    world[:, 2] = torch.linspace(3.0, 8.0, count)
    history_w2c = torch.eye(4, dtype=torch.float64)
    target_center = torch.tensor([0.45, 0.08, 0.03], dtype=torch.float64)
    target_rotation = _rotation_y(0.09) @ _rotation_x(-0.04)
    current_w2c = _w2c(target_rotation, target_center)
    relative = history_w2c @ invert_rigid(current_w2c)
    k = torch.tensor(
        [[420.0, 0.0, 320.0], [0.0, 415.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    current_uv = _project(current_w2c, world, k)
    history_uv = _project(history_w2c, world, k)

    l0_center = torch.tensor([0.42, 0.15, 0.02], dtype=torch.float64)
    l0_rotation = _rotation_y(0.13) @ _rotation_x(-0.02)
    l0_current = _w2c(l0_rotation, l0_center)
    l0_relative = history_w2c @ invert_rigid(l0_current)
    estimate = estimate_relative_epipolar_pose(
        current_uv,
        history_uv,
        torch.ones(count, dtype=torch.float64),
        k,
        k,
        l0_relative,
        config=EpipolarConfig(),
    )
    _require(estimate.success, f"essential solve: {estimate.reason}")
    rotation_error = _rotation_error(
        estimate.rotation_current_to_history, relative[:3, :3]
    )
    direction_error = relative_translation_direction_error_degrees(
        estimate.rotation_current_to_history,
        estimate.translation_current_origin_in_history,
        relative,
    )
    _require(rotation_error < 0.1, "relative rotation recovery")
    _require(direction_error < 0.1, "relative translation direction recovery")

    baseline = torch.stack([history_w2c, l0_current])
    absolute = recover_absolute_pose(
        current_index=1,
        baseline_world_to_camera=baseline,
        edge_history_indices=[0],
        edge_estimates=[estimate],
        config=EpipolarConfig(),
    )
    _require(absolute.success, "absolute pose composition")
    _require(
        _rotation_error(absolute.world_to_camera[:3, :3], target_rotation) < 0.1,
        "absolute rotation recovery",
    )


def _smoke_exact_fallback() -> None:
    uv = torch.arange(14, dtype=torch.float64).reshape(7, 2)
    k = torch.eye(3, dtype=torch.float64)
    estimate = estimate_relative_epipolar_pose(
        uv,
        uv,
        torch.ones(7, dtype=torch.float64),
        k,
        k,
        torch.eye(4, dtype=torch.float64),
        config=EpipolarConfig(),
    )
    _require(not estimate.success, "seven points must be inactive")
    baseline = torch.eye(4, dtype=torch.float64).repeat(2, 1, 1)
    baseline[1, 0, 3] = 0.25
    absolute = recover_absolute_pose(
        current_index=1,
        baseline_world_to_camera=baseline,
        edge_history_indices=[0],
        edge_estimates=[estimate],
        config=EpipolarConfig(),
    )
    _require(not absolute.success, "failed edge must not update pose")
    _require(torch.equal(absolute.world_to_camera, baseline[1]), "fallback is bit exact")


def _smoke_runner_frame() -> None:
    sequence, height, width = 3, 48, 64
    k = torch.tensor(
        [[90.0, 0.0, 31.5], [0.0, 92.0, 23.5], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    ).repeat(sequence, 1, 1)
    centers = (
        torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64),
        torch.tensor([0.22, 0.03, 0.00], dtype=torch.float64),
        torch.tensor([0.47, -0.02, 0.02], dtype=torch.float64),
    )
    rotations = (
        _rotation_y(0.00),
        _rotation_y(0.025),
        _rotation_y(0.050) @ _rotation_x(-0.015),
    )
    target = torch.stack(
        [_w2c(rotation, center) for rotation, center in zip(rotations, centers)]
    )
    world, depth, masks = _render_sphere_sequence(
        target,
        k,
        height=height,
        width=width,
    )
    baseline = target.clone()
    baseline[2] = _w2c(
        _rotation_y(0.085) @ _rotation_x(-0.005),
        torch.tensor([0.44, 0.06, 0.01], dtype=torch.float64),
    )
    history = causal_mask_history_indices(masks, max_history=2)
    config = V90Config(
        source_path=Path("synthetic-v90.yaml"),
        data_config=Path("synthetic-v74.yaml"),
        output_dir=Path("synthetic-output"),
        clip_name="synthetic",
        max_history=2,
        visibility=VisibilityConfig(
            max_queries_per_instance=192,
            depth_tolerance_metric=0.05,
            relative_depth_tolerance=0.01,
        ),
        epipolar=EpipolarConfig(),
    )
    row, diagnostics = _evaluate_frame(
        fold="smoke",
        test_position=0,
        current=2,
        frames=(90, 105, 120),
        masks=masks,
        history_bank=history,
        world_points=world,
        depth=depth,
        global_w2c=target,
        intrinsics=k,
        baseline=baseline,
        target=target,
        sam_track_ids=(7,),
        sam_prompts=("sphere",),
        config=config,
    )
    _require(set(row) == set(FRAME_COLUMNS), "runner frame CSV schema")
    _require(int(row["history_edges_attempted"]) == 2, "runner reads two histories")
    _require(int(row["history_edges_solved"]) >= 1, "runner solves an oracle edge")
    _require(int(row["active"]) == 1, "runner activates fixed solver")
    _require(
        any(item["row_type"] == "slot_surface_reprojection" for item in diagnostics),
        "runner emits visibility diagnostics",
    )
    _require(
        any(item["row_type"] == "history_edge_aggregate" for item in diagnostics),
        "runner emits edge diagnostics",
    )


def _render_sphere_sequence(
    world_to_camera: torch.Tensor,
    intrinsics: torch.Tensor,
    *,
    height: int,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    y, x = torch.meshgrid(
        torch.arange(height, dtype=torch.float64),
        torch.arange(width, dtype=torch.float64),
        indexing="ij",
    )
    sphere_center = torch.tensor([0.0, 0.0, 6.0], dtype=torch.float64)
    radius = 4.0
    world_rows = []
    depth_rows = []
    mask_rows = []
    for pose, k in zip(world_to_camera, intrinsics):
        camera_ray = torch.stack(
            [
                (x - k[0, 2]) / k[0, 0],
                (y - k[1, 2]) / k[1, 1],
                torch.ones_like(x),
            ],
            dim=-1,
        )
        camera_ray = camera_ray / torch.linalg.vector_norm(
            camera_ray, dim=-1, keepdim=True
        )
        center = invert_rigid(pose)[:3, 3]
        world_ray = torch.einsum(
            "ij,hwj->hwi", pose[:3, :3].transpose(0, 1), camera_ray
        )
        offset = center - sphere_center
        projection = torch.einsum("hwi,i->hw", world_ray, offset)
        discriminant = projection.square() - (
            torch.dot(offset, offset) - radius * radius
        )
        valid = discriminant.gt(0.0)
        distance = -projection - torch.sqrt(discriminant.clamp_min(0.0))
        valid = valid & distance.gt(0.0)
        points = center + distance[..., None] * world_ray
        camera = torch.einsum("ij,hwj->hwi", pose[:3, :3], points)
        camera = camera + pose[:3, 3]
        frame_depth = torch.where(valid, camera[..., 2], torch.full_like(x, torch.nan))
        points = torch.where(valid[..., None], points, torch.full_like(points, torch.nan))
        world_rows.append(points)
        depth_rows.append(frame_depth)
        mask_rows.append(valid[None])
    return (
        torch.stack(world_rows),
        torch.stack(depth_rows),
        torch.stack(mask_rows),
    )


def _project(pose: torch.Tensor, points: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    pose = homogeneous(pose)
    camera = torch.einsum("ij,nj->ni", pose[:3, :3], points) + pose[:3, 3]
    projected = torch.einsum("ij,nj->ni", k, camera)
    return projected[:, :2] / projected[:, 2:3]


def _w2c(rotation: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
    output = torch.eye(4, dtype=torch.float64)
    output[:3, :3] = rotation
    output[:3, 3] = -(rotation @ center)
    return output


def _rotation_x(angle: float) -> torch.Tensor:
    cosine, sine = math.cos(angle), math.sin(angle)
    return torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]],
        dtype=torch.float64,
    )


def _rotation_y(angle: float) -> torch.Tensor:
    cosine, sine = math.cos(angle), math.sin(angle)
    return torch.tensor(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
        dtype=torch.float64,
    )


def _rotation_error(first: torch.Tensor, second: torch.Tensor) -> float:
    relative = first @ second.transpose(0, 1)
    cosine = ((torch.trace(relative) - 1.0) * 0.5).clamp(-1.0, 1.0)
    return float(torch.rad2deg(torch.acos(cosine)))


def _require(condition: bool, label: str) -> None:
    if not condition:
        raise RuntimeError(f"V9 smoke failed: {label}")


if __name__ == "__main__":
    main()
