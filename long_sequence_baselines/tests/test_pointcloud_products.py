from pathlib import Path

import numpy as np
from PIL import Image

from long_sequence_baselines.common import write_intrinsics_txt, write_w2c_txt
from long_sequence_baselines.pointcloud_products import (
    PointCloudProtocol,
    WeightedVoxelFusion,
    camera_to_world,
    rebuild_depth_pose_products,
    unproject_depth,
)


def test_unproject_and_inverse_w2c_are_consistent() -> None:
    depth = np.asarray([[2.0]], dtype=np.float32)
    intrinsics = np.eye(3, dtype=np.float64)
    camera = unproject_depth(depth, intrinsics).reshape(-1, 3)
    assert np.allclose(camera, [[0.0, 0.0, 2.0]])

    world_to_camera = np.asarray(
        [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
    )
    world = camera_to_world(camera, world_to_camera)
    assert np.allclose(world, [[-1.0, 0.0, 2.0]])


def test_weighted_voxel_requires_two_frame_observations() -> None:
    fusion = WeightedVoxelFusion(voxel_size=0.1)
    fusion.add_frame(
        np.asarray([[0.01, 0.0, 1.0]], dtype=np.float32),
        np.asarray([[255, 0, 0]], dtype=np.uint8),
        np.asarray([1.0], dtype=np.float32),
    )
    empty, _, _ = fusion.arrays(min_observations=2, max_points=10)
    assert len(empty) == 0

    fusion.add_frame(
        np.asarray([[0.03, 0.0, 1.0]], dtype=np.float32),
        np.asarray([[0, 0, 255]], dtype=np.uint8),
        np.asarray([1.0], dtype=np.float32),
    )
    points, colors, observations = fusion.arrays(min_observations=2, max_points=10)
    assert np.allclose(points, [[0.02, 0.0, 1.0]])
    assert np.array_equal(colors, [[128, 0, 128]])
    assert np.array_equal(observations, [2])


def test_rebuild_depth_pose_writes_all_three_products(tmp_path: Path) -> None:
    scene = tmp_path / "scene"
    for frame in range(2):
        (scene / "depth" / "dpt").mkdir(parents=True, exist_ok=True)
        (scene / "depth" / "conf").mkdir(parents=True, exist_ok=True)
        (scene / "images" / "rgb").mkdir(parents=True, exist_ok=True)
        np.save(
            scene / "depth" / "dpt" / f"frame_{frame:06d}.npy",
            np.full((2, 2), 2.0, dtype=np.float32),
        )
        np.save(
            scene / "depth" / "conf" / f"frame_{frame:06d}.npy",
            np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        )
        Image.fromarray(np.full((2, 2, 3), 64 + frame, dtype=np.uint8)).save(
            scene / "images" / "rgb" / f"frame_{frame:06d}.png"
        )

    extrinsics = np.repeat(np.eye(4, dtype=np.float64)[None, :3], 2, axis=0)
    intrinsics = np.repeat(np.eye(3, dtype=np.float64)[None], 2, axis=0)
    write_w2c_txt(scene / "poses" / "abs_pose.txt", extrinsics, (0, 1))
    write_intrinsics_txt(scene / "poses" / "intri.txt", intrinsics, (0, 1))
    result = rebuild_depth_pose_products(
        scene,
        protocol=PointCloudProtocol(
            confidence_percentile=0.0,
            depth_percentile_low=0.0,
            depth_percentile_high=100.0,
            voxel_size_ratio=0.1,
            min_voxel_observations=2,
            max_points=100,
        ),
    )

    assert result["frames"] == 2
    assert result["fused_ply_points"] == 4
    for name in (
        "depthpose_raw.ply",
        "depthpose_conf.ply",
        "depthpose_conf_voxel.ply",
        "depthpose_summary.json",
    ):
        assert (scene / "points" / name).is_file()
