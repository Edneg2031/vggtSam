"""CPU tests for bridge geometry and confidence-gate semantics."""

from __future__ import annotations

import unittest

import torch

from .gates import GateConfig, decide_gates
from .geometry import project_world_points


class GeometryBridgeTest(unittest.TestCase):
    def test_world_points_project_to_expected_region(self) -> None:
        y, x = torch.meshgrid(
            torch.linspace(-0.2, 0.2, 21),
            torch.linspace(-0.2, 0.2, 21),
            indexing="ij",
        )
        points = torch.stack([x.flatten(), y.flatten(), torch.ones(441) * 2.0], dim=-1)
        extrinsic = torch.eye(4)[:3]
        intrinsic = torch.tensor(
            [[300.0, 0.0, 259.0], [0.0, 300.0, 259.0], [0.0, 0.0, 1.0]]
        )
        mask = project_world_points(
            points,
            world_to_camera=extrinsic,
            intrinsics=intrinsic,
            source_size=(518, 518),
            processed_size=(518, 518),
            output_size=(128, 128),
            image_mode="crop",
            splat_radius=1,
        )
        self.assertTrue(mask.any())
        ys, xs = mask.nonzero(as_tuple=True)
        self.assertLess(abs(float(xs.float().mean()) - 64.0), 2.0)
        self.assertLess(abs(float(ys.float().mean()) - 64.0), 2.0)

    def test_update_and_fallback_are_distinct_gates(self) -> None:
        config = GateConfig(
            track_update_threshold=0.7,
            track_fallback_threshold=0.5,
            geometry_threshold=0.4,
            min_persistence=1,
        )
        reliable = decide_gates(
            track_confidence=0.9,
            geometry_confidence=0.8,
            persistence=2,
            has_object_map=True,
            config=config,
        )
        lost = decide_gates(
            track_confidence=0.1,
            geometry_confidence=0.8,
            persistence=2,
            has_object_map=True,
            config=config,
        )
        self.assertTrue(reliable.update_map)
        self.assertFalse(reliable.use_fallback)
        self.assertFalse(lost.update_map)
        self.assertTrue(lost.use_fallback)


if __name__ == "__main__":
    unittest.main()

