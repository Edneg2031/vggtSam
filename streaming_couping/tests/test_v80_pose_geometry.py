import torch

from streaming_couping.src.v80_pose_geometry import (
    causal_gt_nearest_pairs,
    causal_history_indices,
    gather_pair_points,
    invert_rigid,
    transform_points,
)


def test_causal_history_reads_before_writing_current_frame():
    write = torch.tensor([[True, False], [True, True], [False, True]])
    valid = torch.ones(3, 2, 4, dtype=torch.bool)
    history = causal_history_indices(write, valid)
    assert history.tolist() == [[-1, -1], [0, -1], [1, 1]]


def test_mutual_gt_pairs_respect_instance_and_history():
    points = torch.tensor(
        [
            [[[[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]]]],
            [[[[0.01, 0.0, 1.0], [1.01, 0.0, 1.0]]]],
        ]
    ).reshape(2, 1, 2, 3)
    valid = torch.ones(2, 1, 2, dtype=torch.bool)
    pairs = causal_gt_nearest_pairs(
        current_frame=1,
        history_indices=torch.tensor([0]),
        gt_world_metric=points,
        gt_valid=valid,
        max_distance_metric=0.05,
        require_mutual_nearest=True,
    )
    assert pairs.count == 2
    assert pairs.current_points.tolist() == [0, 1]
    assert pairs.history_points.tolist() == [0, 1]


def test_rigid_inverse_round_trip():
    pose = torch.eye(4)
    pose[:3, 3] = torch.tensor([0.2, -0.1, 0.3])
    points = torch.randn(7, 3)
    assert torch.allclose(
        transform_points(invert_rigid(pose), transform_points(pose, points)),
        points,
        atol=1e-6,
    )


def test_no_history_returns_shape_safe_empty_pairs():
    points = torch.zeros(2, 1, 4, 3)
    valid = torch.ones(2, 1, 4, dtype=torch.bool)
    pairs = causal_gt_nearest_pairs(
        current_frame=0,
        history_indices=torch.tensor([-1]),
        gt_world_metric=points,
        gt_valid=valid,
        max_distance_metric=0.1,
        require_mutual_nearest=True,
    )
    current, history, pair_valid = gather_pair_points(
        points, valid, current_frame=0, pairs=pairs
    )
    assert pairs.count == 0
    assert current.shape == history.shape == (0, 3)
    assert pair_valid.shape == (0,)


def test_gt_pair_builder_ignores_nonfinite_points():
    points = torch.zeros(2, 1, 4, 3)
    points[0, 0, 0] = float("nan")
    valid = torch.ones(2, 1, 4, dtype=torch.bool)
    pairs = causal_gt_nearest_pairs(
        current_frame=1,
        history_indices=torch.tensor([0]),
        gt_world_metric=points,
        gt_valid=valid,
        max_distance_metric=0.1,
        require_mutual_nearest=True,
    )
    assert pairs.count == 1
    assert pairs.history_points.tolist() != [0]
