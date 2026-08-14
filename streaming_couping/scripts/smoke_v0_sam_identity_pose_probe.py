#!/usr/bin/env python3
"""CPU smoke for the causal equal-count SAM identity pose probe."""

from __future__ import annotations

import torch

from streaming_couping.scripts.run_v0_sam_identity_pose_probe import (
    BRANCHES,
    _build_equal_count_groups,
)
from streaming_couping.src.sam_identity_pose_probe import (
    PointBank,
    ProjectionGroup,
    evaluate_projection_groups,
    fixed_pose_candidates,
    shift_mask,
)


def main() -> None:
    _test_fixed_candidates_and_projection_loss()
    _test_causal_bank_and_equal_count_controls()
    _test_shift_has_no_wraparound()
    print("V0 causal SAM identity fixed-direction pose probe smoke passed")


def _test_fixed_candidates_and_projection_loss() -> None:
    base = torch.eye(4)[:3]
    candidates = fixed_pose_candidates(
        base,
        rotation_step_degrees=0.25,
        translation_step=0.01,
    )
    assert len(candidates) == 13
    assert candidates[0][0] == "identity"
    assert len({name for name, _ in candidates}) == 13
    for _, pose in candidates:
        rotation = pose[:3, :3]
        torch.testing.assert_close(
            rotation @ rotation.T,
            torch.eye(3),
            atol=1e-6,
            rtol=1e-6,
        )

    y, x = torch.meshgrid(torch.arange(4), torch.arange(5), indexing="ij")
    points = torch.stack(
        (x.float(), y.float(), torch.ones_like(x).float()), dim=-1
    ).reshape(-1, 3)
    mask = torch.ones(4, 5, dtype=torch.bool)
    group = [ProjectionGroup(points, mask)]
    kwargs = {
        "intrinsics": torch.eye(3),
        "depth": torch.ones(4, 5),
        "normalized_depth_confidence": torch.ones(4, 5),
        "confidence_threshold": 0.3,
        "relative_depth_cap": 0.25,
        "mask_miss_weight": 0.05,
    }
    identity = evaluate_projection_groups(
        group,
        world_to_camera=base,
        **kwargs,
    )
    z_shift = next(
        pose for name, pose in candidates if name == "translation_z_pos"
    )
    shifted = evaluate_projection_groups(
        group,
        world_to_camera=z_shift,
        **kwargs,
    )
    assert identity["source_points"] == shifted["source_points"] == 20
    assert identity["loss"] < 1e-7
    assert shifted["loss"] > identity["loss"]


def _test_causal_bank_and_equal_count_controls() -> None:
    first = PointBank.empty()
    second = PointBank.empty()
    global_bank = PointBank.empty()
    first_points = torch.stack(
        (torch.arange(200).float(), torch.zeros(200), torch.ones(200)), dim=-1
    )
    second_points = first_points + torch.tensor([0.0, 10.0, 0.0])
    weights = torch.linspace(0.1, 1.0, 200)
    first.update(first_points, weights, max_points=200)
    second.update(second_points, weights, max_points=200)
    global_bank.update(
        torch.cat((first_points, second_points), dim=0),
        torch.cat((weights, weights), dim=0),
        max_points=400,
    )
    assert first.observations == second.observations == 1
    masks = [torch.zeros(20, 20, dtype=torch.bool) for _ in range(2)]
    masks[0][2:8, 2:8] = True
    masks[1][10:16, 10:16] = True
    groups, reason = _build_equal_count_groups(
        eligible=(0, 1),
        current_masks=masks,
        instance_banks=(first, second),
        global_bank=global_bank,
        points_per_instance=160,
        min_history_points=128,
        shift_y=2,
        shift_x=3,
    )
    assert reason == "active" and groups is not None
    assert set(groups) == set(BRANCHES)
    counts = {
        branch: sum(group.points.shape[0] for group in branch_groups)
        for branch, branch_groups in groups.items()
    }
    assert set(counts.values()) == {320}
    assert not torch.equal(
        groups["correct_persistent_id"][0].points,
        groups["shuffled_persistent_id"][0].points,
    )


def _test_shift_has_no_wraparound() -> None:
    mask = torch.zeros(6, 7, dtype=torch.bool)
    mask[1, 1] = True
    shifted = shift_mask(mask, shift_y=-2, shift_x=-2)
    assert not shifted.any()
    mask.zero_()
    mask[2, 2] = True
    shifted = shift_mask(mask, shift_y=1, shift_x=2)
    assert shifted[3, 4]
    assert int(shifted.sum()) == 1


if __name__ == "__main__":
    main()
