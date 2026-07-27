from streaming_couping.src.learned_pose.config import load_learned_pose_config


def test_v4_coverage_first_structure_is_frozen() -> None:
    config = load_learned_pose_config(
        "streaming_couping/configs/v4_coverage_first.yaml"
    )

    assert config.output_dir.name == "streaming_couping_v4_coverage_first"
    assert config.fusion.strict_identity_gate
    assert config.fusion.unknown_camera_weight == 0.0
    assert config.fusion.pose_feature_mode == "appearance_only"
    assert config.fusion.rotation_update_mode == "additive_encoding"
    assert config.evaluation.ray_pose.solver_modes == ("current_refined",)


def test_v5_adaptive_best_structure_is_frozen() -> None:
    config = load_learned_pose_config(
        "streaming_couping/configs/v5_adaptive_best.yaml"
    )

    assert config.output_dir.name == "streaming_couping_v5_adaptive_best"
    assert config.fusion.strict_identity_gate
    assert config.fusion.unknown_camera_weight == 0.25
    assert config.fusion.pose_feature_mode == "residual_only"
    assert config.fusion.rotation_update_mode == "bounded_so3"
    assert config.evaluation.ray_pose.solver_modes == ("current_refined",)


def test_retained_configs_use_identical_clips_and_cache() -> None:
    v4 = load_learned_pose_config(
        "streaming_couping/configs/v4_coverage_first.yaml"
    )
    v5 = load_learned_pose_config(
        "streaming_couping/configs/v5_adaptive_best.yaml"
    )

    assert v4.features.cache_dir == v5.features.cache_dir
    assert tuple(clip.frame_indices for clip in v4.clips) == tuple(
        clip.frame_indices for clip in v5.clips
    )
