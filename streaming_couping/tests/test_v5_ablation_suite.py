from dataclasses import replace

from streaming_couping.scripts.run_v5_ablation_suite import (
    CURRENT_RAW_POSE_NAME,
    FINAL_MODE,
    FINAL_RAY_POSE_NAME,
    HISTORICAL_ANCHOR_POSE_NAME,
    VARIANTS,
    _build_upload_summary,
    _write_csv,
)
from streaming_couping.src.learned_pose.config import load_learned_pose_config


def test_v5_upload_summary_joins_only_compact_required_metrics(tmp_path):
    config = load_learned_pose_config(
        "streaming_couping/configs/v5_ablation_suite.yaml"
    )
    clip = config.clips[1]
    variant = VARIANTS[-1]
    config = replace(
        config,
        clips=(clip,),
        output_dir=tmp_path / "variant",
    )
    evaluation = config.output_dir / "evaluation"
    common = {
        "clip": clip.name,
        "evaluation_protocol": "held_out_clip",
        "evaluated_frame_indices": "492 512",
    }
    _write_csv(
        evaluation / "ray_pose_summary.csv",
        [
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
            {
                **common,
                "perturbation": CURRENT_RAW_POSE_NAME,
                "ray_pose_role": "solver_source_ablation",
                "ate_rmse": 0.25,
                "rotation_error_mean_degrees": 1.5,
            },
            {
                **common,
                "perturbation": FINAL_RAY_POSE_NAME,
                "ray_pose_role": "deployable",
                "ate_rmse": 0.2,
                "rotation_error_mean_degrees": 1.5,
            },
            {
                **common,
                "perturbation": HISTORICAL_ANCHOR_POSE_NAME,
                "ray_pose_role": "historical_anchor_ablation",
                "ate_rmse": 0.18,
                "rotation_error_mean_degrees": 1.5,
            },
        ],
    )
    _write_csv(
        evaluation / "ray_pose_fit_diagnostics.csv",
        [
            {
                "clip": clip.name,
                "perturbation": perturbation,
                "frame_index": 492,
                "fit_accepted": accepted,
            }
            for perturbation, accepted in (
                (CURRENT_RAW_POSE_NAME, 1),
                (FINAL_RAY_POSE_NAME, 1),
                (HISTORICAL_ANCHOR_POSE_NAME, 0),
            )
        ],
    )
    pointmap_rows = []
    for mode, perturbation, region, value in (
        ("baseline", "module_off", "full_scene", 0.20),
        (FINAL_MODE, "aligned", "full_scene", 0.15),
        ("baseline", "module_off", "background", 0.10),
        (FINAL_MODE, "aligned", "background", 0.10),
        (
            FINAL_MODE,
            "spatial_token_shuffle",
            "full_scene",
            0.19,
        ),
    ):
        pointmap_rows.append(
            {
                "clip": clip.name,
                "mode": mode,
                "perturbation": perturbation,
                "spatial_region": region,
                "geometry_source": "point_head",
                "group": "all_frames",
                "mean_frame_paired_distance_mean": value,
            }
        )
    for mode, perturbation, value in (
        (FINAL_MODE, "aligned", 0.17),
        ("ray_pose_raw_geometry", FINAL_RAY_POSE_NAME, 0.16),
        (
            "ray_pose_raw_geometry",
            HISTORICAL_ANCHOR_POSE_NAME,
            0.14,
        ),
    ):
        pointmap_rows.append(
            {
                "clip": clip.name,
                "mode": mode,
                "perturbation": perturbation,
                "spatial_region": "full_scene",
                "geometry_source": "baseline_point_head_refined_pose",
                "group": "all_frames",
                "mean_frame_paired_distance_mean": value,
            }
        )
    _write_csv(evaluation / "pointmap_summary.csv", pointmap_rows)
    _write_csv(
        evaluation / "baseline_equivalence.csv",
        [
            {
                "clip": clip.name,
                "mode": FINAL_MODE,
                "strict_equal": 1,
            }
        ],
    )
    _write_csv(
        evaluation / "identity_gate_diagnostics.csv",
        [
            {
                "clip": clip.name,
                "frame_index": 492,
                "identity_state": state,
            }
            for state in ("MATCH", "UNKNOWN", "MISMATCH")
        ],
    )
    _write_csv(
        evaluation / "pose_summary.csv",
        [
            {
                "clip": clip.name,
                "mode": FINAL_MODE,
                "perturbation": "memory_off",
                "ate_rmse": 0.31,
            },
            {
                "clip": clip.name,
                "mode": FINAL_MODE,
                "perturbation": "wrong_id_memory",
                "ate_rmse": 0.35,
            },
        ],
    )

    rows = _build_upload_summary([(variant, config)])

    assert len(rows) == 1
    row = rows[0]
    assert row["raw_ate"] == "0.4"
    assert row["historical_ate"] == "0.18"
    assert row["pointmap_delta"] == "-0.05"
    assert row["learned_pose_reposed_raw_pointmap_mean"] == "0.17"
    assert row["current_refined_reposed_raw_pointmap_mean"] == "0.16"
    assert row["historical_reposed_raw_pointmap_mean"] == "0.14"
    assert row["background_delta"] == "0"
    assert row["memory_off_ate"] == "0.31"
    assert row["match_count"] == 1
    assert row["unknown_count"] == 1
    assert row["mismatch_count"] == 1
    assert row["historical_fit_accepted"] == 0
