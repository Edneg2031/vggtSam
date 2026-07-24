from pathlib import Path

import torch

from streaming_couping.scripts.run_joint_pointmap_ba import (
    CONTROL_FIXED,
    CONTROL_RAW,
    SUMMARY_FIELDS,
    _compact_summary,
)
from streaming_couping.src.learned_pose.joint_ba import (
    JOINT_BA_VARIANTS,
    JointBAConfig,
    dcs_switch,
    regauge_camera_to_world,
    run_joint_ba,
    world_points_from_depth,
)
from vggtsam.utils.imports import maybe_add_repo_to_path


def test_regauge_anchors_raw_reference_and_preserves_relative_poses() -> None:
    raw = torch.eye(4).repeat(3, 1, 1)
    raw[:, 0, 3] = torch.tensor([4.0, 5.0, 6.0])
    learned = torch.eye(4).repeat(3, 1, 1)
    learned[:, 0, 3] = torch.tensor([1.0, 1.5, 2.0])
    learned[:, 1, 3] = torch.tensor([0.0, 0.2, 0.4])

    anchored = regauge_camera_to_world(
        learned,
        raw,
        reference_index=1,
    )

    assert torch.equal(anchored[1], raw[1])
    learned_relative = torch.linalg.inv(learned[1]) @ learned
    anchored_relative = torch.linalg.inv(anchored[1]) @ anchored
    assert torch.allclose(anchored_relative, learned_relative, atol=1e-6)


def test_world_points_from_depth_recovers_camera_rays_and_depth() -> None:
    depth = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[2.0, 3.0], [4.0, 5.0]],
        ]
    )
    camera_to_world = torch.eye(4).repeat(2, 1, 1)
    camera_to_world[1, :3, 3] = torch.tensor([0.5, -0.2, 0.1])
    intrinsics = torch.tensor(
        [
            [[2.0, 0.0, 1.0], [0.0, 2.0, 1.0], [0.0, 0.0, 1.0]],
            [[3.0, 0.0, 1.0], [0.0, 3.0, 1.0], [0.0, 0.0, 1.0]],
        ]
    )

    world = world_points_from_depth(
        depth,
        camera_to_world,
        intrinsics,
        image_size=(2, 2),
    )
    local = torch.einsum(
        "sji,shwj->shwi",
        camera_to_world[:, :3, :3],
        world - camera_to_world[:, None, None, :3, 3],
    )
    y, x = torch.meshgrid(torch.arange(2), torch.arange(2), indexing="ij")
    expected_x_over_z = (
        x[None] - intrinsics[:, 0, 2, None, None]
    ) / intrinsics[:, 0, 0, None, None]
    expected_y_over_z = (
        y[None] - intrinsics[:, 1, 2, None, None]
    ) / intrinsics[:, 1, 1, None, None]

    assert torch.allclose(local[..., 2], depth, atol=1e-6)
    assert torch.allclose(
        local[..., 0] / local[..., 2],
        expected_x_over_z,
        atol=1e-6,
    )
    assert torch.allclose(
        local[..., 1] / local[..., 2],
        expected_y_over_z,
        atol=1e-6,
    )


def test_dcs_switch_keeps_small_errors_and_suppresses_large_errors() -> None:
    weights = dcs_switch(torch.tensor([0.0, 1.0, 99.0]))

    assert torch.equal(weights[:2], torch.ones(2))
    assert torch.allclose(weights[2], torch.tensor(0.02))
    assert torch.equal(
        dcs_switch(torch.tensor(99.0), minimum=0.15),
        torch.tensor(0.15),
    )


def test_insufficient_matches_fall_back_to_exact_raw_outputs() -> None:
    inputs = _synthetic_inputs()
    result = run_joint_ba(
        **inputs,
        variant=JOINT_BA_VARIANTS[0],
        config=JointBAConfig(
            outer_iterations=1,
            inner_steps=1,
            min_total_matches=10_000,
        ),
    )

    assert result.diagnostics["status"] == (
        "fallback_insufficient_cross_view_matches"
    )
    assert torch.equal(result.pose_encoding, inputs["raw_pose_encoding"])
    assert torch.equal(
        result.world_points,
        inputs["raw_world_points"],
    )


def test_small_joint_ba_case_is_finite_anchored_and_reduces_ray_error() -> None:
    inputs = _synthetic_inputs()
    result = run_joint_ba(
        **inputs,
        variant=JOINT_BA_VARIANTS[0],
        config=JointBAConfig(
            outer_iterations=2,
            inner_steps=40,
            learning_rate=0.05,
            match_radius_patches=1,
            max_matches_per_edge_region=32,
            min_matches_per_edge_region=2,
            min_total_matches=4,
            feature_dim_limit=16,
            min_feature_cosine=0.5,
            max_log_depth_residual=1.0,
            max_forward_backward_patches=4.0,
            rotation_prior_weight=0.05,
            translation_prior_weight=0.01,
            max_translation_scene_ratio=0.5,
        ),
    )

    assert result.diagnostics["status"] == "accepted_joint_ba"
    assert torch.isfinite(result.pose_encoding).all()
    assert torch.isfinite(result.world_points).all()
    assert torch.isfinite(result.depth).all()
    assert torch.equal(
        result.pose_encoding[:, 0],
        inputs["raw_pose_encoding"][:, 0],
    )
    assert result.diagnostics["reference_pose_max_abs_diff"] == 0.0
    assert (
        result.diagnostics["final_ray_rmse"]
        < result.diagnostics["initial_ray_rmse"]
    )


def test_compact_summary_has_five_ordered_rows_and_raw_deltas() -> None:
    variants = (
        CONTROL_RAW,
        CONTROL_FIXED,
        *(variant.name for variant in JOINT_BA_VARIANTS),
    )
    pose_rows = []
    pointmap_rows = []
    diagnostics = []
    for index, variant in enumerate(variants):
        pose_rows.append(
            {
                "clip": "clip",
                "perturbation": variant,
                "evaluation_protocol": "held_out_clip",
                "ate_rmse": 0.4 - 0.01 * index,
                "rotation_error_mean_degrees": 2.0 - 0.1 * index,
            }
        )
        pointmap_rows.append(
            {
                "clip": "clip",
                "perturbation": variant,
                "spatial_region": "full_scene",
                "geometry_source": "point_head",
                "group": "all_frames",
                "mean_frame_paired_distance_mean": 0.2 - 0.01 * index,
            }
        )
        diagnostics.append(
            {
                "clip": "clip",
                "variant": variant,
                "joint_consistent": int(variant != CONTROL_FIXED),
                "module_off_exact": 1,
                "status": "control" if index < 2 else "accepted_joint_ba",
                "initial_ray_rmse": 0.1,
                "final_ray_rmse": 0.05,
                "matches": 32,
                "active_instance_edges": 2,
                "rejected_instance_edges": 1,
                "beta_mean": 0.1,
                "reference_pose_max_abs_diff": 0.0,
            }
        )

    rows = _compact_summary(pose_rows, pointmap_rows, diagnostics)

    assert len(rows) == 5
    assert tuple(rows[0]) == SUMMARY_FIELDS
    assert [row["variant"] for row in rows] == list(variants)
    assert rows[0]["ate_delta_from_raw"] == "0"
    assert rows[0]["pointmap_delta_from_raw"] == "0"
    assert rows[-1]["ate_delta_from_raw"] == "-0.04"
    assert rows[-1]["pointmap_delta_from_raw"] == "-0.04"
    assert rows[-1]["reference_anchor_exact"] == 1


def _synthetic_inputs() -> dict[str, object]:
    maybe_add_repo_to_path(
        Path(__file__).resolve().parents[2] / "externals/streamvggt"
    )
    from streamvggt.utils.pose_enc import extri_intri_to_pose_encoding

    image_size = (16, 16)
    sequence = 2
    intrinsics = torch.tensor(
        [[16.0, 0.0, 8.0], [0.0, 16.0, 8.0], [0.0, 0.0, 1.0]]
    ).repeat(sequence, 1, 1)
    raw_c2w = torch.eye(4).repeat(sequence, 1, 1)
    raw_c2w[1, 0, 3] = 0.05
    learned_c2w = raw_c2w.clone()
    learned_c2w[1, 0, 3] = 0.25
    raw_pose = extri_intri_to_pose_encoding(
        _c2w_to_w2c(raw_c2w)[None],
        intrinsics[None],
        image_size_hw=image_size,
    )
    learned_pose = extri_intri_to_pose_encoding(
        _c2w_to_w2c(learned_c2w)[None],
        intrinsics[None],
        image_size_hw=image_size,
    )
    depth = torch.full((sequence, *image_size), 2.0)
    raw_points = world_points_from_depth(
        depth,
        raw_c2w,
        intrinsics,
        image_size=image_size,
    )[None]
    patch_shape = (4, 4)
    patch_count = patch_shape[0] * patch_shape[1]
    token_levels = torch.eye(patch_count)[None, None].expand(
        1,
        sequence,
        patch_count,
        patch_count,
    ).clone()
    tracking_masks = torch.zeros(
        1,
        sequence,
        1,
        *image_size,
        dtype=torch.bool,
    )
    trusted_instance_valid = torch.zeros(
        1,
        sequence,
        1,
        dtype=torch.bool,
    )
    return {
        "raw_pose_encoding": raw_pose,
        "learned_pose_encoding": learned_pose,
        "raw_world_points": raw_points,
        "learned_world_points": raw_points.clone(),
        "raw_confidence": torch.ones(1, sequence, *image_size),
        "learned_confidence": torch.ones(1, sequence, *image_size),
        "token_levels": token_levels,
        "patch_start_idx": 0,
        "patch_shape": patch_shape,
        "tracking_masks": tracking_masks,
        "trusted_tracking_masks": tracking_masks.clone(),
        "trusted_instance_valid": trusted_instance_valid,
        "image_size": image_size,
        "reference_index": 0,
        "scene_scale": 1.0,
    }


def _c2w_to_w2c(camera_to_world: torch.Tensor) -> torch.Tensor:
    rotation = camera_to_world[:, :3, :3].transpose(-1, -2)
    translation = -torch.einsum(
        "sij,sj->si",
        rotation,
        camera_to_world[:, :3, 3],
    )
    return torch.cat([rotation, translation[..., None]], dim=-1)
