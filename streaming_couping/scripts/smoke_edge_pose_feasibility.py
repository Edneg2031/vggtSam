#!/usr/bin/env python3
"""CPU smoke checks for the E0-r2 edge pose-basin experiment."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from PIL import Image

from streaming_couping.src.edge_pose_feasibility import (
    EdgeConfig,
    EdgeFeasibilityConfig,
    ProjectionConfig,
    RecoveryConfig,
    bilinear_sample_image,
    equal_count_branch_edges,
    project_depth_points,
    run_edge_feasibility,
    shifted_masks_no_wrap,
    sobel_magnitude,
    threshold_edges,
    truncated_distance_transform,
)


def main() -> None:
    size = 32
    yy, xx = torch.meshgrid(
        torch.arange(size), torch.arange(size), indexing="ij"
    )
    pattern = (
        (((xx - 15) ** 2 + (yy - 14) ** 2) < 75)
        | ((xx - 2 * yy - 3).abs() < 2)
        | ((2 * xx + yy - 49).abs() < 2)
    ).float()
    image = pattern[None]
    strength = sobel_magnitude(image)
    edges = threshold_edges(strength, quantile=0.70, max_edges_per_frame=384)
    assert edges.any()
    distance = truncated_distance_transform(edges[0], max_distance=6)
    assert float(distance[edges[0]].max()) == 0.0
    assert float(distance.max()) <= 6.0

    # A fractional distance-field sample must propagate a useful UV gradient.
    probe_u = torch.tensor([10.35], requires_grad=True)
    probe_v = torch.tensor([11.20], requires_grad=True)
    probe_field = xx.float() + 2.0 * yy.float()
    bilinear_sample_image(probe_field, probe_u, probe_v).sum().backward()
    assert probe_u.grad is not None and abs(float(probe_u.grad)) > 0.5
    assert probe_v.grad is not None and abs(float(probe_v.grad)) > 1.0

    mask = torch.zeros(1, size, size, dtype=torch.bool)
    mask[:, :3, :4] = True
    shifted = shifted_masks_no_wrap(mask)
    assert not bool(shifted[:, :3, :4].any())
    assert int(shifted.sum()) == int(mask.sum())
    branches = equal_count_branch_edges(
        all_edges=edges.repeat(2, 1, 1),
        edge_strength=strength.repeat(2, 1, 1),
        exclusion_masks=mask.repeat(2, 1, 1),
        branches=(
            "all_edge",
            "sam_object_excluded_edge",
            "shifted_object_mask_control",
        ),
    )
    counts = [value.sum(dim=(-2, -1)) for value in branches.values()]
    assert all(torch.equal(counts[0], value) for value in counts[1:])

    focal = 28.0
    k = torch.tensor(
        [[focal, 0.0, 15.5], [0.0, focal, 15.5], [0.0, 0.0, 1.0]]
    )
    w2c = torch.eye(4)[:3]
    projected = project_depth_points(
        x=torch.tensor([3.0]),
        y=torch.tensor([4.0]),
        depth=torch.tensor([2.0]),
        source_w2c=w2c,
        target_w2c=w2c,
        source_k=k,
        target_k=k,
    )
    assert torch.allclose(projected[0, :2], torch.tensor([3.0, 4.0]), atol=1e-5)
    assert torch.allclose(projected[0, 2], torch.tensor(2.0))

    with TemporaryDirectory() as tmp:
        paths = [Path(tmp) / f"f{index}.png" for index in range(2)]
        pil = Image.fromarray((image[0].numpy() * 255).astype("uint8"))
        for path in paths:
            pil.save(path)
        payload = {
            "clip_name": "smoke",
            "frame_indices": [0, 1],
            "image_paths": [str(value) for value in paths],
            "image_size": [size, size],
            "scene_scale": 2.0,
            "baseline_depth": torch.full((2, size, size), 2.0),
            "baseline_depth_confidence": torch.ones(2, size, size),
            "associated_tracking_masks_output": torch.zeros(
                2, 1, size, size
            ).bool(),
        }
        config = EdgeFeasibilityConfig(
            source_path=Path("smoke.yaml"),
            base_config=Path("v0.yaml"),
            output_dir=Path(tmp),
            clip_name="smoke",
            evaluation_frames=(1,),
            branches=(
                "raw",
                "all_edge",
                "sam_object_excluded_edge",
                "shifted_object_mask_control",
            ),
            edge=EdgeConfig(
                sobel_quantile=0.70,
                max_edges_per_frame=384,
                distance_truncate_px=6,
                huber_delta_px=1.0,
                mask_dilation_px=0,
            ),
            projection=ProjectionConfig(
                source_offsets=(1,),
                min_depth=0.01,
                max_depth=10.0,
                min_depth_confidence=0.1,
                depth_abs_tolerance=0.01,
                depth_rel_tolerance=0.01,
                min_locked_points_per_direction=16,
                max_locked_points_per_direction=128,
            ),
            recovery=RecoveryConfig(
                device="cpu",
                axes=("x",),
                signs=(1,),
                rotation_degrees=(5.0,),
                translation_scene_fractions=(0.10,),
                optimizer_steps=30,
                learning_rate=0.05,
                gradient_probe_step=0.001,
                max_rotation_degrees=10.0,
                max_translation_scene_fraction=0.20,
                min_loss_order_pass_rate=0.5,
                min_mean_recovery_fraction=0.0,
            ),
        )
        result = run_edge_feasibility(
            payload=payload,
            world_to_camera=torch.stack((w2c, w2c), dim=0),
            intrinsics=torch.stack((k, k), dim=0),
            config=config,
        )
        assert len(result["rows"]) == 4
        assert len(result["summary"]) == 3
        for row in result["summary"]:
            assert row["mean_in_bounds_rate"] > 0.0
            assert row["mean_depth_consistency_pass_rate"] > 0.0
        recovery_rows = result["recovery_rows"]
        assert len(recovery_rows) == 2
        assert all(int(row["active"]) == 1 for row in recovery_rows)
        assert all(float(row["gradient_norm"]) > 0.0 for row in recovery_rows)
        assert all(int(row["gradient_probe_pass"]) == 1 for row in recovery_rows)
        assert all(int(row["optimizer_active"]) == 1 for row in recovery_rows)
    print("E0-r2 differentiable edge pose-basin smoke passed")


if __name__ == "__main__":
    main()
