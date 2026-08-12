#!/usr/bin/env python3
"""CPU smoke checks for E0 masked-edge feasibility helpers."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from PIL import Image

from streaming_couping.src.edge_pose_feasibility import (
    EdgeConfig,
    EdgeFeasibilityConfig,
    ProjectionConfig,
    project_depth_points,
    run_edge_feasibility,
    sobel_magnitude,
    threshold_edges,
    truncated_distance_transform,
)


def main() -> None:
    image = torch.zeros(1, 16, 16)
    image[:, :, 8:] = 1.0
    edges = threshold_edges(
        sobel_magnitude(image),
        quantile=0.80,
        max_edges_per_frame=64,
    )
    assert edges.any()
    distance = truncated_distance_transform(edges[0], max_distance=4)
    assert float(distance[edges[0]].max()) == 0.0
    assert float(distance.max()) <= 4.0

    x = torch.tensor([3.0])
    y = torch.tensor([4.0])
    depth = torch.tensor([2.0])
    k = torch.eye(3)
    w2c = torch.eye(4)[:3]
    projected = project_depth_points(
        x=x,
        y=y,
        depth=depth,
        source_w2c=w2c,
        target_w2c=w2c,
        source_k=k,
        target_k=k,
    )
    assert torch.allclose(projected[0, :2], torch.tensor([3.0, 4.0]))
    assert torch.allclose(projected[0, 2], torch.tensor(2.0))

    with TemporaryDirectory() as tmp:
        path0 = Path(tmp) / "f0.png"
        path1 = Path(tmp) / "f1.png"
        pil = Image.fromarray((image[0].numpy() * 255).astype("uint8"))
        pil.save(path0)
        pil.save(path1)
        payload = {
            "clip_name": "smoke",
            "frame_indices": [0, 1],
            "image_paths": [str(path0), str(path1)],
            "image_size": [16, 16],
            "baseline_depth": torch.full((2, 16, 16), 2.0),
            "baseline_depth_confidence": torch.ones(2, 16, 16),
            "associated_tracking_masks_output": torch.zeros(2, 1, 16, 16).bool(),
        }
        config = EdgeFeasibilityConfig(
            source_path=Path("smoke.yaml"),
            base_config=Path("v0.yaml"),
            output_dir=Path(tmp),
            clip_name="smoke",
            evaluation_frames=(1,),
            branches=("raw", "all_edge", "sam_static_edge", "shuffled_static_mask"),
            edge=EdgeConfig(sobel_quantile=0.80, max_edges_per_frame=64, distance_truncate_px=4),
            projection=ProjectionConfig(source_offsets=(1,), min_depth=0.01, max_depth=10.0),
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
            assert row["mean_depth_cycle_pass_rate"] > 0.0
    print("masked-edge feasibility smoke passed")


if __name__ == "__main__":
    main()
