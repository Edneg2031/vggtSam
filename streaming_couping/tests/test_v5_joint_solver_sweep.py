from dataclasses import replace

from streaming_couping.scripts.run_v5_ablation_suite import _write_csv
from streaming_couping.scripts.run_v5_joint_solver_sweep import (
    REFERENCE_BLENDS,
    SUMMARY_FIELDS,
    _build_joint_solver_summary,
    _policy_specs,
    _sweep_config,
)
from streaming_couping.src.learned_pose.config import (
    FINAL_MODE,
    load_learned_pose_config,
)


def test_joint_solver_summary_is_five_policies_per_clip(tmp_path) -> None:
    base = load_learned_pose_config(
        "streaming_couping/configs/v5_ablation_suite.yaml"
    )
    clip = base.clips[1]
    config = _sweep_config(
        replace(
            base,
            clips=(clip,),
            output_dir=tmp_path,
        )
    )
    evaluation = config.output_dir / "evaluation"
    common = {
        "clip": clip.name,
        "evaluation_protocol": "held_out_clip",
        "evaluated_frame_indices": "492 512",
    }
    ray_rows = [
        {
            **common,
            "perturbation": "raw",
            "ray_pose_role": "raw_baseline_control",
            "ate_rmse": 0.4,
            "rotation_error_mean_degrees": 2.0,
        },
        {
            **common,
            "perturbation": "learned",
            "ray_pose_role": "learned_pose_control",
            "ate_rmse": 0.3,
            "rotation_error_mean_degrees": 1.5,
        },
    ]
    for index, (_, perturbation, _, _) in enumerate(_policy_specs()):
        ray_rows.append(
            {
                **common,
                "perturbation": perturbation,
                "ray_pose_role": "solver",
                "ate_rmse": 0.25 - 0.01 * index,
                "rotation_error_mean_degrees": 1.5,
            }
        )
    _write_csv(evaluation / "ray_pose_summary.csv", ray_rows)

    diagnostic_rows = []
    for _, perturbation, preserve_reference, blend in _policy_specs():
        for frame_index in (492, 512):
            is_preserved = preserve_reference and frame_index == 492
            diagnostic_rows.append(
                {
                    "clip": clip.name,
                    "perturbation": perturbation,
                    "frame_index": frame_index,
                    "fit_accepted": int(not is_preserved),
                    "fit_status": (
                        "preserved_reference"
                        if is_preserved
                        else "accepted"
                    ),
                    "applied_center_shift_native": (
                        0.0 if is_preserved else 0.1 * blend
                    ),
                }
            )
    _write_csv(
        evaluation / "ray_pose_fit_diagnostics.csv",
        diagnostic_rows,
    )

    pointmap_rows = []
    for mode, perturbation, source, value in (
        ("baseline", "module_off", "point_head", 0.20),
        (FINAL_MODE, "aligned", "point_head", 0.18),
    ):
        pointmap_rows.append(
            {
                "clip": clip.name,
                "mode": mode,
                "perturbation": perturbation,
                "spatial_region": "full_scene",
                "geometry_source": source,
                "group": "all_frames",
                "mean_frame_paired_distance_mean": value,
            }
        )
    for index, (_, perturbation, _, _) in enumerate(_policy_specs()):
        pointmap_rows.append(
            {
                "clip": clip.name,
                "mode": "ray_pose_raw_geometry",
                "perturbation": perturbation,
                "spatial_region": "full_scene",
                "geometry_source": "baseline_point_head_refined_pose",
                "group": "all_frames",
                "mean_frame_paired_distance_mean": 0.24 - 0.01 * index,
            }
        )
    _write_csv(evaluation / "pointmap_summary.csv", pointmap_rows)
    _write_csv(
        evaluation / "baseline_equivalence.csv",
        [{"clip": clip.name, "mode": FINAL_MODE, "strict_equal": 1}],
    )

    rows = _build_joint_solver_summary(config)

    assert len(rows) == 5
    assert tuple(rows[0]) == SUMMARY_FIELDS
    assert rows[0]["policy"] == "free_ref_100"
    assert rows[0]["solver_ate"] == "0.25"
    assert rows[0]["reposed_delta_from_raw"] == "0.04"
    assert rows[1]["policy"] == "fixed_ref_025"
    assert rows[1]["fit_accepted"] == 1
    assert rows[1]["mean_applied_center_shift_native"] == "0.0125"
    assert rows[-1]["module_off_exact"] == 1


def test_sweep_reuses_residual_so3_union_structure() -> None:
    base = load_learned_pose_config(
        "streaming_couping/configs/v5_ablation_suite.yaml"
    )

    config = _sweep_config(base)

    assert config.features.cache_dir == base.features.cache_dir
    assert config.features.rebuild is False
    assert config.fusion.pose_feature_mode == "residual_only"
    assert config.fusion.rotation_update_mode == "bounded_so3"
    assert config.fusion.spatial_attention_mode == "union"
    assert config.evaluation.ray_pose.reference_blend_values == REFERENCE_BLENDS
    assert config.evaluation.ray_pose.preserve_reference is False
