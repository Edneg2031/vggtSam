"""Train V6 pose branches and evaluate fixed checkpoints on held-out clips."""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from streaming_couping.src.config import load_config
from streaming_couping.src.external_repos import maybe_add_repo_to_path
from streaming_couping.src.learned_pose.cache import cache_path, load_feature_cache
from streaming_couping.src.learned_pose.config import load_learned_pose_config
from streaming_couping.src.learned_pose.pipeline import _slice_training_payload
from streaming_couping.src.learned_pose.v6_camera_fusion import (
    V6_TRAINING_MODES,
    V6_VARIANTS,
    V6CameraFusion,
    V6FusionConfig,
    perturb_instance_inputs,
)


@dataclass(frozen=True)
class V6TrainingConfig:
    steps: int = 1200
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    grad_clip_norm: float = 5.0
    translation_weight: float = 10.0
    seed: int = 0
    log_every: int = 200


@dataclass(frozen=True)
class V6SuccessConfig:
    minimum_loss_drop_percent: float = 95.0
    maximum_rotation_degrees: float = 0.10
    maximum_translation_native: float = 0.005
    instance_loss_ratio: float = 0.50
    shuffle_loss_ratio: float = 0.80
    camera_loss_ratio: float = 0.80


@dataclass(frozen=True)
class V6ExperimentConfig:
    source_path: Path
    base_config: Path
    output_dir: Path
    clip_name: str
    test_clip_name: str
    device: str
    fusion: V6FusionConfig
    training: V6TrainingConfig
    success: V6SuccessConfig


def main() -> None:
    args = _parse_args()
    config = load_v6_config(args.config)
    if args.device:
        config = replace(config, device=args.device)
    run_v6_camera_overfit(config)


def run_v6_camera_overfit(config: V6ExperimentConfig) -> Path:
    _seed_everything(config.training.seed)
    base = load_learned_pose_config(config.base_config)
    clips = [clip for clip in base.clips if clip.name == config.clip_name]
    if len(clips) != 1:
        raise ValueError(
            f"V6 clip {config.clip_name!r} was not found exactly once in "
            f"{config.base_config}."
        )
    clip = clips[0]
    test_clips = [
        item for item in base.clips if item.name == config.test_clip_name
    ]
    if len(test_clips) != 1:
        raise ValueError(
            f"V6 test clip {config.test_clip_name!r} was not found exactly "
            f"once in {config.base_config}."
        )
    test_clip = test_clips[0]
    path = cache_path(base, clip)
    if not path.is_file():
        raise FileNotFoundError(
            f"Frozen feature cache is missing: {path}. Run the V6 command file "
            "once; its first stage builds/reuses the cache."
        )
    print(f"V6 reusing frozen cache: {path}")
    print("V6 trains Feature Merger + SE(3) head only; pointmap and solver are off")
    full_payload = load_feature_cache(path)
    payload = _slice_training_payload(full_payload, clip)
    batch = _camera_batch(payload, device=config.device)
    full_batch = _camera_batch(full_payload, device=config.device)
    test_path = cache_path(base, test_clip)
    if not test_path.is_file():
        raise FileNotFoundError(
            f"Frozen cross-clip cache is missing: {test_path}. Run the V6 "
            "command file to build/reuse both caches."
        )
    print(f"V6 reusing frozen cross-clip cache: {test_path}")
    test_payload = load_feature_cache(test_path)
    test_batch = _camera_batch(test_payload, device=config.device)

    recovery = load_config(base.recovery_config)
    maybe_add_repo_to_path(recovery.streamvggt_repo)
    from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri

    image_size = tuple(int(value) for value in payload["image_size"])
    baseline_w2c, _ = pose_encoding_to_extri_intri(
        batch["baseline_pose_encoding"],
        image_size_hw=image_size,
    )
    target_w2c, _ = pose_encoding_to_extri_intri(
        batch["target_pose_encoding"],
        image_size_hw=image_size,
    )
    full_baseline_w2c, _ = pose_encoding_to_extri_intri(
        full_batch["baseline_pose_encoding"],
        image_size_hw=image_size,
    )
    full_target_w2c, _ = pose_encoding_to_extri_intri(
        full_batch["target_pose_encoding"],
        image_size_hw=image_size,
    )
    test_image_size = tuple(int(value) for value in test_payload["image_size"])
    test_baseline_w2c, _ = pose_encoding_to_extri_intri(
        test_batch["baseline_pose_encoding"],
        image_size_hw=test_image_size,
    )
    test_target_w2c, _ = pose_encoding_to_extri_intri(
        test_batch["target_pose_encoding"],
        image_size_hw=test_image_size,
    )
    reference_index = int(payload["reference_sequence_index"])
    heldout_frames = tuple(clip.evaluation_frame_indices or ())
    if not heldout_frames:
        raise ValueError(
            f"V6 clip {clip.name!r} requires evaluation_frame_indices."
        )
    frame_position = {
        int(frame): index
        for index, frame in enumerate(full_payload["frame_indices"])
    }
    heldout_indices = [frame_position[int(frame)] for frame in heldout_frames]
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_metrics = _pose_metrics(
        baseline_w2c,
        target_w2c,
        reference_index=reference_index,
        translation_weight=config.training.translation_weight,
    )
    initial_loss = float(raw_metrics["loss"])
    trained = {}
    for mode in V6_TRAINING_MODES:
        _seed_everything(config.training.seed)
        model = _new_model(batch, config)
        training_result = _train_model(
            model,
            mode=mode,
            batch=batch,
            baseline_w2c=baseline_w2c,
            target_w2c=target_w2c,
            reference_index=reference_index,
            config=config,
        )
        metrics = _evaluate_model(
            model,
            mode=mode,
            variant="normal",
            batch=batch,
            baseline_w2c=baseline_w2c,
            target_w2c=target_w2c,
            reference_index=reference_index,
            config=config,
        )
        capacity_pass = _capacity_pass(metrics, initial_loss, config)
        trained[mode] = {
            "model": model,
            "metrics": metrics,
            "capacity_pass": capacity_pass,
            **training_result,
        }
        _save_checkpoint(
            output_dir / f"v6_checkpoint_{mode}.pt",
            model=model,
            mode=mode,
            batch=batch,
            payload=payload,
            reference_index=reference_index,
            config=config,
            training_result=training_result,
        )

    specialized = {}
    for name, mode, head_component in (
        ("camera_rotation", "camera_only", "rotation"),
        ("instance_center", "instance_only", "center"),
    ):
        _seed_everything(config.training.seed)
        model = _new_model(
            batch,
            config,
            head_component=head_component,
        )
        training_result = _train_model(
            model,
            mode=mode,
            objective=head_component,
            batch=batch,
            baseline_w2c=baseline_w2c,
            target_w2c=target_w2c,
            reference_index=reference_index,
            config=config,
        )
        specialized[name] = {
            "model": model,
            "mode": mode,
            **training_result,
        }
        _save_checkpoint(
            output_dir / f"v6_checkpoint_specialized_{name}.pt",
            model=model,
            mode=mode,
            batch=batch,
            payload=payload,
            reference_index=reference_index,
            config=config,
            training_result=training_result,
        )

    specialized_train_output = _compose_decoupled_output(
        _predict_model(
            specialized["camera_rotation"]["model"],
            mode="camera_only",
            variant="normal",
            batch=batch,
            baseline_w2c=baseline_w2c,
            reference_index=reference_index,
        ),
        _predict_model(
            specialized["instance_center"]["model"],
            mode="instance_only",
            variant="normal",
            batch=batch,
            baseline_w2c=baseline_w2c,
            reference_index=reference_index,
        ),
        baseline_w2c=baseline_w2c,
        reference_index=reference_index,
    )
    specialized_train_metrics = _prediction_metrics(
        specialized_train_output,
        baseline_w2c=baseline_w2c,
        target_w2c=target_w2c,
        reference_index=reference_index,
        translation_weight=config.training.translation_weight,
    )

    fusion_model = trained["fusion"]["model"]
    evaluations = {
        variant: _evaluate_model(
            fusion_model,
            mode="fusion",
            variant=variant,
            batch=batch,
            baseline_w2c=baseline_w2c,
            target_w2c=target_w2c,
            reference_index=reference_index,
            config=config,
        )
        for variant in V6_VARIANTS
    }
    heldout_raw = _pose_metrics(
        full_baseline_w2c,
        full_target_w2c,
        reference_index=int(full_payload["reference_sequence_index"]),
        translation_weight=config.training.translation_weight,
        evaluation_indices=heldout_indices,
    )
    heldout_outputs = {
        mode: _predict_model(
            trained[mode]["model"],
            mode=mode,
            variant="normal",
            batch=full_batch,
            baseline_w2c=full_baseline_w2c,
            reference_index=int(full_payload["reference_sequence_index"]),
        )
        for mode in V6_TRAINING_MODES
    }
    heldout_decoupled = {
        "instanceR_cameraC": _compose_decoupled_output(
            heldout_outputs["instance_only"],
            heldout_outputs["camera_only"],
            baseline_w2c=full_baseline_w2c,
            reference_index=int(full_payload["reference_sequence_index"]),
        ),
        "cameraR_instanceC": _compose_decoupled_output(
            heldout_outputs["camera_only"],
            heldout_outputs["instance_only"],
            baseline_w2c=full_baseline_w2c,
            reference_index=int(full_payload["reference_sequence_index"]),
        ),
    }
    heldout_specialized_output = _compose_decoupled_output(
        _predict_model(
            specialized["camera_rotation"]["model"],
            mode="camera_only",
            variant="normal",
            batch=full_batch,
            baseline_w2c=full_baseline_w2c,
            reference_index=int(full_payload["reference_sequence_index"]),
        ),
        _predict_model(
            specialized["instance_center"]["model"],
            mode="instance_only",
            variant="normal",
            batch=full_batch,
            baseline_w2c=full_baseline_w2c,
            reference_index=int(full_payload["reference_sequence_index"]),
        ),
        baseline_w2c=full_baseline_w2c,
        reference_index=int(full_payload["reference_sequence_index"]),
    )
    heldout = {
        mode: _prediction_metrics(
            output,
            baseline_w2c=full_baseline_w2c,
            target_w2c=full_target_w2c,
            reference_index=int(full_payload["reference_sequence_index"]),
            translation_weight=config.training.translation_weight,
            evaluation_indices=heldout_indices,
        )
        for mode, output in heldout_outputs.items()
    }
    heldout_decoupled_metrics = {
        name: _prediction_metrics(
            output,
            baseline_w2c=full_baseline_w2c,
            target_w2c=full_target_w2c,
            reference_index=int(full_payload["reference_sequence_index"]),
            translation_weight=config.training.translation_weight,
            evaluation_indices=heldout_indices,
        )
        for name, output in heldout_decoupled.items()
    }
    heldout_specialized_metrics = _prediction_metrics(
        heldout_specialized_output,
        baseline_w2c=full_baseline_w2c,
        target_w2c=full_target_w2c,
        reference_index=int(full_payload["reference_sequence_index"]),
        translation_weight=config.training.translation_weight,
        evaluation_indices=heldout_indices,
    )
    heldout_fusion_best = int(
        float(heldout["fusion"]["loss"])
        < min(
            float(heldout_raw["loss"]),
            float(heldout["camera_only"]["loss"]),
            float(heldout["instance_only"]["loss"]),
        )
    )
    test_reference_index = int(test_payload["reference_sequence_index"])
    test_indices = _evaluation_indices(
        len(test_payload["frame_indices"]),
        reference_index=test_reference_index,
        evaluation_indices=None,
    )
    cross_raw = _pose_metrics(
        test_baseline_w2c,
        test_target_w2c,
        reference_index=test_reference_index,
        translation_weight=config.training.translation_weight,
        evaluation_indices=test_indices,
    )
    cross_outputs = {
        mode: _predict_model(
            trained[mode]["model"],
            mode=mode,
            variant="normal",
            batch=test_batch,
            baseline_w2c=test_baseline_w2c,
            reference_index=test_reference_index,
        )
        for mode in V6_TRAINING_MODES
    }
    cross_decoupled = {
        "instanceR_cameraC": _compose_decoupled_output(
            cross_outputs["instance_only"],
            cross_outputs["camera_only"],
            baseline_w2c=test_baseline_w2c,
            reference_index=test_reference_index,
        ),
        "cameraR_instanceC": _compose_decoupled_output(
            cross_outputs["camera_only"],
            cross_outputs["instance_only"],
            baseline_w2c=test_baseline_w2c,
            reference_index=test_reference_index,
        ),
    }
    cross_specialized_output = _compose_decoupled_output(
        _predict_model(
            specialized["camera_rotation"]["model"],
            mode="camera_only",
            variant="normal",
            batch=test_batch,
            baseline_w2c=test_baseline_w2c,
            reference_index=test_reference_index,
        ),
        _predict_model(
            specialized["instance_center"]["model"],
            mode="instance_only",
            variant="normal",
            batch=test_batch,
            baseline_w2c=test_baseline_w2c,
            reference_index=test_reference_index,
        ),
        baseline_w2c=test_baseline_w2c,
        reference_index=test_reference_index,
    )
    cross_metrics = {
        mode: _prediction_metrics(
            output,
            baseline_w2c=test_baseline_w2c,
            target_w2c=test_target_w2c,
            reference_index=test_reference_index,
            translation_weight=config.training.translation_weight,
            evaluation_indices=test_indices,
        )
        for mode, output in cross_outputs.items()
    }
    cross_decoupled_metrics = {
        name: _prediction_metrics(
            output,
            baseline_w2c=test_baseline_w2c,
            target_w2c=test_target_w2c,
            reference_index=test_reference_index,
            translation_weight=config.training.translation_weight,
            evaluation_indices=test_indices,
        )
        for name, output in cross_decoupled.items()
    }
    cross_specialized_metrics = _prediction_metrics(
        cross_specialized_output,
        baseline_w2c=test_baseline_w2c,
        target_w2c=test_target_w2c,
        reference_index=test_reference_index,
        translation_weight=config.training.translation_weight,
        evaluation_indices=test_indices,
    )
    cross_clip_decoupled_best = int(
        float(cross_decoupled_metrics["instanceR_cameraC"]["loss"])
        < min(
            float(cross_raw["loss"]),
            *(float(metrics["loss"]) for metrics in cross_metrics.values()),
            float(cross_decoupled_metrics["cameraR_instanceC"]["loss"]),
        )
    )
    cross_clip_specialized_best = int(
        float(cross_specialized_metrics["loss"])
        < min(
            float(cross_raw["loss"]),
            *(float(metrics["loss"]) for metrics in cross_metrics.values()),
            *(
                float(metrics["loss"])
                for metrics in cross_decoupled_metrics.values()
            ),
        )
    )
    frame_diagnostic_rows = _frame_diagnostic_rows(
        split="temporal_holdout",
        frame_indices=full_payload["frame_indices"],
        evaluation_indices=heldout_indices,
        batch=full_batch,
        baseline_w2c=full_baseline_w2c,
        target_w2c=full_target_w2c,
        camera_output=heldout_outputs["camera_only"],
        instance_output=heldout_outputs["instance_only"],
        specialized_output=heldout_specialized_output,
        instance_model=trained["instance_only"]["model"],
    )
    frame_diagnostic_rows.extend(
        _frame_diagnostic_rows(
            split="cross_clip",
            frame_indices=test_payload["frame_indices"],
            evaluation_indices=test_indices,
            batch=test_batch,
            baseline_w2c=test_baseline_w2c,
            target_w2c=test_target_w2c,
            camera_output=cross_outputs["camera_only"],
            instance_output=cross_outputs["instance_only"],
            specialized_output=cross_specialized_output,
            instance_model=trained["instance_only"]["model"],
        )
    )
    normal = evaluations["normal"]
    instance_used = int(
        float(normal["loss"])
        <= config.success.instance_loss_ratio
        * float(evaluations["instance_off"]["loss"])
        and float(normal["loss"])
        <= config.success.shuffle_loss_ratio
        * float(evaluations["shuffle_time"]["loss"])
    )
    camera_used = int(
        float(normal["loss"])
        <= config.success.camera_loss_ratio
        * float(evaluations["camera_off"]["loss"])
    )
    summary_path = output_dir / "v6_summary.csv"
    frames = " ".join(str(value) for value in payload["frame_indices"])
    heldout_frame_text = " ".join(str(value) for value in heldout_frames)
    test_frame_text = " ".join(
        str(value) for value in test_payload["frame_indices"]
    )
    rows = [
        _summary_row(
            experiment="control",
            variant="raw",
            trained_mode="none",
            test_input="raw",
            frames=frames,
            metrics=raw_metrics,
            initial_loss=initial_loss,
            capacity_pass=0,
            instance_used=instance_used,
            camera_used=camera_used,
            heldout_fusion_best=heldout_fusion_best,
        )
    ]
    for mode in V6_TRAINING_MODES:
        rows.append(
            _summary_row(
                experiment="capacity",
                variant=f"trained_{mode}",
                trained_mode=mode,
                test_input="normal",
                frames=frames,
                metrics=trained[mode]["metrics"],
                initial_loss=initial_loss,
                capacity_pass=trained[mode]["capacity_pass"],
                instance_used=instance_used,
                camera_used=camera_used,
                heldout_fusion_best=heldout_fusion_best,
            )
        )
    specialized_capacity_pass = int(
        all(
            _loss_drop(
                float(specialized[name]["initial_loss"]),
                float(specialized[name]["best_logged_loss"]),
            )
            >= config.success.minimum_loss_drop_percent
            for name in ("camera_rotation", "instance_center")
        )
        and int(specialized_train_metrics["reference_exact"]) == 1
    )
    rows.append(
        _summary_row(
            experiment="specialized_capacity",
            variant="trained_specialized_cameraR_instanceC",
            trained_mode="camera_rotation+instance_center",
            test_input="normal",
            frames=frames,
            metrics=specialized_train_metrics,
            initial_loss=initial_loss,
            capacity_pass=specialized_capacity_pass,
            instance_used=instance_used,
            camera_used=camera_used,
            heldout_fusion_best=heldout_fusion_best,
        )
    )
    for name, metrics in heldout_decoupled_metrics.items():
        rows.append(
            _summary_row(
                experiment="heldout_decoupled",
                variant=f"heldout_{name}",
                trained_mode="camera_only+instance_only",
                test_input=name,
                frames=heldout_frame_text,
                metrics=metrics,
                initial_loss=float(heldout_raw["loss"]),
                capacity_pass=int(
                    trained["camera_only"]["capacity_pass"]
                    and trained["instance_only"]["capacity_pass"]
                ),
                instance_used=instance_used,
                camera_used=camera_used,
                heldout_fusion_best=heldout_fusion_best,
            )
        )
    rows.append(
        _summary_row(
            experiment="heldout_specialized",
            variant="heldout_specialized_cameraR_instanceC",
            trained_mode="camera_rotation+instance_center",
            test_input="normal",
            frames=heldout_frame_text,
            metrics=heldout_specialized_metrics,
            initial_loss=float(heldout_raw["loss"]),
            capacity_pass=specialized_capacity_pass,
            instance_used=instance_used,
            camera_used=camera_used,
            heldout_fusion_best=heldout_fusion_best,
        )
    )
    rows.append(
        _summary_row(
            experiment="cross_clip",
            variant="cross_clip_raw",
            trained_mode="none",
            test_input="raw",
            frames=test_frame_text,
            metrics=cross_raw,
            initial_loss=float(cross_raw["loss"]),
            capacity_pass=0,
            instance_used=instance_used,
            camera_used=camera_used,
            heldout_fusion_best=heldout_fusion_best,
        )
    )
    for mode in V6_TRAINING_MODES:
        rows.append(
            _summary_row(
                experiment="cross_clip",
                variant=f"cross_clip_{mode}",
                trained_mode=mode,
                test_input="normal",
                frames=test_frame_text,
                metrics=cross_metrics[mode],
                initial_loss=float(cross_raw["loss"]),
                capacity_pass=trained[mode]["capacity_pass"],
                instance_used=instance_used,
                camera_used=camera_used,
                heldout_fusion_best=heldout_fusion_best,
            )
        )
    for name, metrics in cross_decoupled_metrics.items():
        rows.append(
            _summary_row(
                experiment="cross_clip_decoupled",
                variant=f"cross_clip_{name}",
                trained_mode="camera_only+instance_only",
                test_input=name,
                frames=test_frame_text,
                metrics=metrics,
                initial_loss=float(cross_raw["loss"]),
                capacity_pass=int(
                    trained["camera_only"]["capacity_pass"]
                    and trained["instance_only"]["capacity_pass"]
                ),
                instance_used=instance_used,
                camera_used=camera_used,
                heldout_fusion_best=heldout_fusion_best,
            )
        )
    rows.append(
        _summary_row(
            experiment="cross_clip_specialized",
            variant="cross_clip_specialized_cameraR_instanceC",
            trained_mode="camera_rotation+instance_center",
            test_input="normal",
            frames=test_frame_text,
            metrics=cross_specialized_metrics,
            initial_loss=float(cross_raw["loss"]),
            capacity_pass=specialized_capacity_pass,
            instance_used=instance_used,
            camera_used=camera_used,
            heldout_fusion_best=heldout_fusion_best,
        )
    )
    rows.append(
        _summary_row(
            experiment="heldout",
            variant="heldout_raw",
            trained_mode="none",
            test_input="raw",
            frames=heldout_frame_text,
            metrics=heldout_raw,
            initial_loss=float(heldout_raw["loss"]),
            capacity_pass=0,
            instance_used=instance_used,
            camera_used=camera_used,
            heldout_fusion_best=heldout_fusion_best,
        )
    )
    for mode in V6_TRAINING_MODES:
        rows.append(
            _summary_row(
                experiment="heldout",
                variant=f"heldout_{mode}",
                trained_mode=mode,
                test_input="normal",
                frames=heldout_frame_text,
                metrics=heldout[mode],
                initial_loss=float(heldout_raw["loss"]),
                capacity_pass=trained[mode]["capacity_pass"],
                instance_used=instance_used,
                camera_used=camera_used,
                heldout_fusion_best=heldout_fusion_best,
            )
        )
    for variant in V6_VARIANTS:
        if variant == "normal":
            continue
        rows.append(
            _summary_row(
                experiment="fusion_dependency",
                variant=f"fusion_{variant}",
                trained_mode="fusion",
                test_input=variant,
                frames=frames,
                metrics=evaluations[variant],
                initial_loss=initial_loss,
                capacity_pass=trained["fusion"]["capacity_pass"],
                instance_used=instance_used,
                camera_used=camera_used,
                heldout_fusion_best=heldout_fusion_best,
            )
        )
    experiment_order = {
        "control": 0,
        "capacity": 1,
        "specialized_capacity": 2,
        "heldout": 3,
        "heldout_decoupled": 4,
        "heldout_specialized": 5,
        "cross_clip": 6,
        "cross_clip_decoupled": 7,
        "cross_clip_specialized": 8,
        "fusion_dependency": 9,
    }
    rows.sort(key=lambda row: experiment_order[str(row["experiment"])])
    for row in rows:
        row["cross_clip_decoupled_best"] = cross_clip_decoupled_best
        row["cross_clip_specialized_best"] = cross_clip_specialized_best
    _write_csv(summary_path, rows)
    cross_summary_path = output_dir / "v6_cross_clip_summary.csv"
    cross_rows = [
        row for row in rows if str(row["experiment"]).startswith("cross_clip")
    ]
    _write_csv(
        cross_summary_path,
        [
            {
                "variant": row["variant"],
                "frames": row["frames"],
                "rotation_deg": row["rotation_deg"],
                "translation_native": row["translation_native"],
                "loss": row["loss"],
                "better_than_raw": row["better_than_split_raw"],
                "decoupled_best": cross_clip_decoupled_best,
                "specialized_best": cross_clip_specialized_best,
            }
            for row in cross_rows
        ],
    )
    frame_summary_path = output_dir / "v6_frame_diagnostics.csv"
    _write_csv(frame_summary_path, frame_diagnostic_rows)
    with (output_dir / "v6_run.json").open("w", encoding="utf8") as handle:
        json.dump(
            {
                "purpose": (
                    "three full-SE3 controls, two specialized 3DoF heads, "
                    "fusion dependency, and fixed-checkpoint development "
                    "evaluation"
                ),
                "uses_gt_during_training": True,
                "runs_pointmap_branch": False,
                "runs_analytic_pose_solver": False,
                "specialized_components": {
                    "rotation": "camera_only",
                    "center": "instance_only",
                },
                "capacity_pass": {
                    mode: bool(trained[mode]["capacity_pass"])
                    for mode in V6_TRAINING_MODES
                },
                "instance_used": bool(instance_used),
                "camera_used": bool(camera_used),
                "heldout_frames": list(heldout_frames),
                "heldout_fusion_best": bool(heldout_fusion_best),
                "cross_clip": test_clip.name,
                "cross_clip_role": "development_after_component_analysis",
                "cross_clip_decoupled_best": bool(cross_clip_decoupled_best),
                "cross_clip_specialized_best": bool(
                    cross_clip_specialized_best
                ),
                "frame_diagnostics": str(frame_summary_path),
                "config": str(config.source_path),
                "cache": str(path),
                "test_cache": str(test_path),
            },
            handle,
            indent=2,
        )

    print("\nV6 FAIR CAMERA FUSION ABLATION (GT-supervised)")
    print(
        "raw:     "
        f"rot={float(raw_metrics['rotation_degrees']):.4f} deg  "
        f"trans={float(raw_metrics['translation_native']):.6f}"
    )
    for mode in V6_TRAINING_MODES:
        metrics = trained[mode]["metrics"]
        print(
            f"{mode:13s} "
            f"rot={float(metrics['rotation_degrees']):.4f} deg  "
            f"trans={float(metrics['translation_native']):.6f}  "
            f"pass={int(trained[mode]['capacity_pass'])}"
        )
    print(
        f"fusion_instance_used={instance_used}  camera_used={camera_used}  "
        f"reference_exact={int(normal['reference_exact'])}"
    )
    print(f"HELD-OUT {heldout_frame_text}")
    print(
        "raw:          "
        f"rot={float(heldout_raw['rotation_degrees']):.4f} deg  "
        f"trans={float(heldout_raw['translation_native']):.6f}"
    )
    for mode in V6_TRAINING_MODES:
        metrics = heldout[mode]
        print(
            f"{mode:13s} "
            f"rot={float(metrics['rotation_degrees']):.4f} deg  "
            f"trans={float(metrics['translation_native']):.6f}"
        )
    print(
        "specialized   "
        f"rot={float(heldout_specialized_metrics['rotation_degrees']):.4f} deg  "
        f"trans={float(heldout_specialized_metrics['translation_native']):.6f}"
    )
    print(f"heldout_fusion_best={heldout_fusion_best}")
    print(f"CROSS-CLIP {test_frame_text}")
    print(
        "raw:          "
        f"rot={float(cross_raw['rotation_degrees']):.4f} deg  "
        f"trans={float(cross_raw['translation_native']):.6f}"
    )
    for mode in V6_TRAINING_MODES:
        metrics = cross_metrics[mode]
        print(
            f"{mode:13s} "
            f"rot={float(metrics['rotation_degrees']):.4f} deg  "
            f"trans={float(metrics['translation_native']):.6f}"
        )
    for name, metrics in cross_decoupled_metrics.items():
        print(
            f"{name:18s} "
            f"rot={float(metrics['rotation_degrees']):.4f} deg  "
            f"trans={float(metrics['translation_native']):.6f}"
        )
    print(
        "specialized       "
        f"rot={float(cross_specialized_metrics['rotation_degrees']):.4f} deg  "
        f"trans={float(cross_specialized_metrics['translation_native']):.6f}"
    )
    print(f"cross_clip_decoupled_best={cross_clip_decoupled_best}")
    print(f"cross_clip_specialized_best={cross_clip_specialized_best}")
    print(f"cross_summary={cross_summary_path}")
    print(f"frame_diagnostics={frame_summary_path}")
    print(f"full_summary={summary_path}")
    return summary_path


def load_v6_config(path: str | Path) -> V6ExperimentConfig:
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf8") as handle:
        raw = yaml.safe_load(handle) or {}
    model = raw.get("model", {})
    training = raw.get("training", {})
    success = raw.get("success", {})
    config = V6ExperimentConfig(
        source_path=source,
        base_config=_path(raw.get("base_config"), source.parent),
        output_dir=_path(raw.get("output_dir"), source.parent),
        clip_name=str(raw.get("clip_name", "")),
        test_clip_name=str(raw.get("test_clip_name", "")),
        device=str(raw.get("device", "cuda:1")),
        fusion=V6FusionConfig(
            hidden_dim=int(model.get("hidden_dim", 256)),
            num_heads=int(model.get("num_heads", 8)),
            dropout=float(model.get("dropout", 0.0)),
            memory_momentum=float(model.get("memory_momentum", 0.90)),
            min_track_confidence=float(model.get("min_track_confidence", 0.25)),
            unknown_reliability=float(model.get("unknown_reliability", 0.50)),
            max_rotation_degrees=float(model.get("max_rotation_degrees", 15.0)),
            max_translation_native=float(model.get("max_translation_native", 0.50)),
        ),
        training=V6TrainingConfig(
            steps=int(training.get("steps", 1200)),
            learning_rate=float(training.get("learning_rate", 1e-3)),
            weight_decay=float(training.get("weight_decay", 0.0)),
            grad_clip_norm=float(training.get("grad_clip_norm", 5.0)),
            translation_weight=float(training.get("translation_weight", 10.0)),
            seed=int(training.get("seed", 0)),
            log_every=int(training.get("log_every", 200)),
        ),
        success=V6SuccessConfig(
            minimum_loss_drop_percent=float(
                success.get("minimum_loss_drop_percent", 95.0)
            ),
            maximum_rotation_degrees=float(
                success.get("maximum_rotation_degrees", 0.10)
            ),
            maximum_translation_native=float(
                success.get("maximum_translation_native", 0.005)
            ),
            instance_loss_ratio=float(success.get("instance_loss_ratio", 0.50)),
            shuffle_loss_ratio=float(success.get("shuffle_loss_ratio", 0.80)),
            camera_loss_ratio=float(success.get("camera_loss_ratio", 0.80)),
        ),
    )
    _validate_config(config)
    return config


def _camera_batch(payload: dict, *, device: str) -> dict[str, torch.Tensor]:
    output = {}
    for field in (
        "camera_hidden",
        "baseline_pose_encoding",
        "target_pose_encoding",
        "appearance",
        "pose_geometry",
        "quality",
        "observed",
        "identity_valid",
        "identity_unknown",
    ):
        value = payload.get(field)
        if not torch.is_tensor(value):
            raise ValueError(f"V6 cache is missing tensor field {field!r}.")
        output[field] = value.unsqueeze(0).to(device)
    return output


def _new_model(
    batch: dict[str, torch.Tensor],
    config: V6ExperimentConfig,
    *,
    head_component: str = "se3",
) -> V6CameraFusion:
    return V6CameraFusion(
        camera_dim=int(batch["camera_hidden"].shape[-1]),
        appearance_dim=int(batch["appearance"].shape[-1]),
        geometry_dim=int(batch["pose_geometry"].shape[-1]),
        config=config.fusion,
        head_component=head_component,
    ).to(config.device)


def _train_model(
    model: V6CameraFusion,
    *,
    mode: str,
    batch: dict[str, torch.Tensor],
    baseline_w2c: torch.Tensor,
    target_w2c: torch.Tensor,
    reference_index: int,
    config: V6ExperimentConfig,
    objective: str = "se3",
) -> dict[str, float | int]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    initial_output = _predict_model(
        model,
        mode=mode,
        variant="normal",
        batch=batch,
        baseline_w2c=baseline_w2c,
        reference_index=reference_index,
    )
    initial_loss = float(
        _pose_loss(
            initial_output["world_to_camera"],
            target_w2c,
            reference_index=reference_index,
            translation_weight=config.training.translation_weight,
            component=objective,
        ).cpu()
    )
    if not bool(initial_output["active_frames"].any()):
        raise RuntimeError(
            f"V6 {mode} found no usable non-reference frame. Check the "
            "persistent-instance cache."
        )

    best_loss = float("inf")
    best_step = 0
    best_state = None
    model.train()
    for step in range(1, config.training.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        output = _forward_variant(
            model,
            batch,
            baseline_w2c,
            reference_index=reference_index,
            variant="normal",
            mode=mode,
        )
        loss = _pose_loss(
            output["world_to_camera"],
            target_w2c,
            reference_index=reference_index,
            translation_weight=config.training.translation_weight,
            component=objective,
        )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"Non-finite V6 {mode} loss at step {step}.")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config.training.grad_clip_norm,
        )
        if not bool(torch.isfinite(grad_norm)):
            raise RuntimeError(f"Non-finite V6 {mode} gradient at step {step}.")
        optimizer.step()
        if step % config.training.log_every == 0 or step == config.training.steps:
            checked = _predict_model(
                model,
                mode=mode,
                variant="normal",
                batch=batch,
                baseline_w2c=baseline_w2c,
                reference_index=reference_index,
            )
            current = float(
                _pose_loss(
                    checked["world_to_camera"],
                    target_w2c,
                    reference_index=reference_index,
                    translation_weight=config.training.translation_weight,
                    component=objective,
                ).cpu()
            )
            model.train()
            print(
                f"V6 {mode:13s}/{objective:8s} "
                f"{step:4d}/{config.training.steps}: loss={current:.6g}"
            )
            if current < best_loss:
                best_loss = current
                best_step = step
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
    if best_state is None:
        raise RuntimeError(f"V6 {mode} did not produce a checkpoint.")
    model.load_state_dict(best_state)
    model.eval()
    return {
        "best_step": best_step,
        "initial_loss": initial_loss,
        "best_logged_loss": best_loss,
    }


def _evaluate_model(
    model: V6CameraFusion,
    *,
    mode: str,
    variant: str,
    batch: dict[str, torch.Tensor],
    baseline_w2c: torch.Tensor,
    target_w2c: torch.Tensor,
    reference_index: int,
    config: V6ExperimentConfig,
    evaluation_indices: list[int] | None = None,
) -> dict[str, float | int]:
    output = _predict_model(
        model,
        mode=mode,
        variant=variant,
        batch=batch,
        baseline_w2c=baseline_w2c,
        reference_index=reference_index,
    )
    return _prediction_metrics(
        output,
        baseline_w2c=baseline_w2c,
        target_w2c=target_w2c,
        reference_index=reference_index,
        translation_weight=config.training.translation_weight,
        evaluation_indices=evaluation_indices,
    )


def _predict_model(
    model: V6CameraFusion,
    *,
    mode: str,
    variant: str,
    batch: dict[str, torch.Tensor],
    baseline_w2c: torch.Tensor,
    reference_index: int,
) -> dict[str, torch.Tensor]:
    model.eval()
    with torch.no_grad():
        return _forward_variant(
            model,
            batch,
            baseline_w2c,
            reference_index=reference_index,
            variant=variant,
            mode=mode,
        )


def _prediction_metrics(
    output: dict[str, torch.Tensor],
    *,
    baseline_w2c: torch.Tensor,
    target_w2c: torch.Tensor,
    reference_index: int,
    translation_weight: float,
    evaluation_indices: list[int] | None = None,
) -> dict[str, float | int]:
    metrics = _pose_metrics(
        output["world_to_camera"],
        target_w2c,
        reference_index=reference_index,
        translation_weight=translation_weight,
        evaluation_indices=evaluation_indices,
    )
    metrics["reference_exact"] = int(
        torch.equal(
            output["world_to_camera"][:, reference_index].cpu(),
            baseline_w2c[:, reference_index].cpu(),
        )
    )
    active_indices = _evaluation_indices(
        output["active_frames"].shape[1],
        reference_index=reference_index,
        evaluation_indices=evaluation_indices,
    )
    active_index = torch.tensor(
        active_indices,
        dtype=torch.long,
        device=output["active_frames"].device,
    )
    metrics["active_frames"] = int(
        output["active_frames"].index_select(1, active_index).sum().cpu()
    )
    return metrics


def _compose_decoupled_output(
    rotation_output: dict[str, torch.Tensor],
    center_output: dict[str, torch.Tensor],
    *,
    baseline_w2c: torch.Tensor,
    reference_index: int,
) -> dict[str, torch.Tensor]:
    rotation_pose = rotation_output["world_to_camera"]
    center_pose = center_output["world_to_camera"]
    if (
        rotation_pose.shape != center_pose.shape
        or rotation_pose.shape != baseline_w2c.shape
    ):
        raise ValueError("Decoupled V6 pose shapes disagree.")
    rotation = rotation_pose[..., :3, :3]
    center = _camera_centers(center_pose)
    translation = -(rotation @ center[..., None])
    composed = torch.cat([rotation, translation], dim=-1)
    composed = composed.clone()
    composed[:, int(reference_index)] = baseline_w2c[:, int(reference_index)]
    return {
        "world_to_camera": composed,
        "active_frames": (
            rotation_output["active_frames"]
            & center_output["active_frames"]
        ),
    }


def _frame_diagnostic_rows(
    *,
    split: str,
    frame_indices: list[int] | tuple[int, ...],
    evaluation_indices: list[int],
    batch: dict[str, torch.Tensor],
    baseline_w2c: torch.Tensor,
    target_w2c: torch.Tensor,
    camera_output: dict[str, torch.Tensor],
    instance_output: dict[str, torch.Tensor],
    specialized_output: dict[str, torch.Tensor],
    instance_model: V6CameraFusion,
) -> list[dict[str, object]]:
    """Relate per-frame error changes to inference-time instance support."""

    if baseline_w2c.shape[0] != 1:
        raise ValueError("V6 frame diagnostics require a batch size of one.")
    sequence = baseline_w2c.shape[1]
    if len(frame_indices) != sequence:
        raise ValueError("V6 frame diagnostics frame/pose lengths disagree.")
    with torch.no_grad():
        _, usable, _ = instance_model.instance_encoder(
            appearance=batch["appearance"],
            geometry=batch["pose_geometry"],
            quality=batch["quality"],
            observed=batch["observed"],
            identity_valid=batch["identity_valid"],
            identity_unknown=batch["identity_unknown"],
        )
    raw_rotation, raw_center = _pose_errors_per_frame(
        baseline_w2c,
        target_w2c,
    )
    camera_rotation, camera_center = _pose_errors_per_frame(
        camera_output["world_to_camera"],
        target_w2c,
    )
    instance_rotation, instance_center = _pose_errors_per_frame(
        instance_output["world_to_camera"],
        target_w2c,
    )
    specialized_rotation, specialized_center = _pose_errors_per_frame(
        specialized_output["world_to_camera"],
        target_w2c,
    )
    geometry = batch["quality"][..., 1]
    rows = []
    for index in evaluation_indices:
        index = int(index)
        if index < 0 or index >= sequence:
            raise ValueError("V6 frame diagnostic index is outside the sequence.")
        current_usable = usable[0, index]
        usable_count = int(current_usable.sum().cpu())
        geometry_confidence = (
            float(geometry[0, index][current_usable].mean().cpu())
            if usable_count
            else 0.0
        )
        rows.append(
            {
                "split": split,
                "frame": int(frame_indices[index]),
                "usable_instances": usable_count,
                "geometry_confidence": _short(geometry_confidence),
                "camera_rotation_delta": _short(
                    float((camera_rotation[0, index] - raw_rotation[0, index]).cpu())
                ),
                "instance_rotation_delta": _short(
                    float(
                        (instance_rotation[0, index] - raw_rotation[0, index]).cpu()
                    )
                ),
                "camera_center_delta": _short(
                    float((camera_center[0, index] - raw_center[0, index]).cpu())
                ),
                "instance_center_delta": _short(
                    float((instance_center[0, index] - raw_center[0, index]).cpu())
                ),
                "specialized_rotation_delta": _short(
                    float(
                        (
                            specialized_rotation[0, index]
                            - raw_rotation[0, index]
                        ).cpu()
                    )
                ),
                "specialized_center_delta": _short(
                    float(
                        (
                            specialized_center[0, index]
                            - raw_center[0, index]
                        ).cpu()
                    )
                ),
            }
        )
    return rows


def _pose_errors_per_frame(
    predicted: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if predicted.shape != target.shape or predicted.shape[-2:] != (3, 4):
        raise ValueError("V6 per-frame pose tensors must share shape [B,S,3,4].")
    relative = (
        predicted[..., :3, :3]
        @ target[..., :3, :3].transpose(-1, -2)
    )
    cosine = (
        torch.diagonal(relative, dim1=-2, dim2=-1).sum(dim=-1) - 1.0
    ) * 0.5
    rotation = torch.rad2deg(torch.acos(cosine.clamp(-1.0, 1.0)))
    center = torch.linalg.vector_norm(
        _camera_centers(predicted) - _camera_centers(target),
        dim=-1,
    )
    return rotation, center


def _capacity_pass(
    metrics: dict[str, float | int],
    initial_loss: float,
    config: V6ExperimentConfig,
) -> int:
    return int(
        _loss_drop(initial_loss, float(metrics["loss"]))
        >= config.success.minimum_loss_drop_percent
        and float(metrics["rotation_degrees"])
        <= config.success.maximum_rotation_degrees
        and float(metrics["translation_native"])
        <= config.success.maximum_translation_native
        and int(metrics["reference_exact"]) == 1
    )


def _save_checkpoint(
    path: Path,
    *,
    model: V6CameraFusion,
    mode: str,
    batch: dict[str, torch.Tensor],
    payload: dict,
    reference_index: int,
    config: V6ExperimentConfig,
    training_result: dict[str, float | int],
) -> None:
    torch.save(
        {
            "model": {
                name: value.detach().cpu()
                for name, value in model.state_dict().items()
            },
            "mode": mode,
            "head_component": model.head_component,
            "fusion": asdict(config.fusion),
            "training": asdict(config.training),
            "camera_dim": int(batch["camera_hidden"].shape[-1]),
            "appearance_dim": int(batch["appearance"].shape[-1]),
            "geometry_dim": int(batch["pose_geometry"].shape[-1]),
            "clip": payload["clip_name"],
            "frame_indices": list(payload["frame_indices"]),
            "reference_index": reference_index,
            **training_result,
        },
        path,
    )


def _summary_row(
    *,
    experiment: str,
    variant: str,
    trained_mode: str,
    test_input: str,
    frames: str,
    metrics: dict[str, float | int],
    initial_loss: float,
    capacity_pass: int,
    instance_used: int,
    camera_used: int,
    heldout_fusion_best: int,
) -> dict[str, object]:
    row_loss = float(metrics["loss"])
    return {
        "experiment": experiment,
        "variant": variant,
        "trained_mode": trained_mode,
        "test_input": test_input,
        "frames": frames,
        "rotation_deg": _short(float(metrics["rotation_degrees"])),
        "translation_native": _short(float(metrics["translation_native"])),
        "loss": _short(row_loss),
        "loss_drop_percent": _short(_loss_drop(initial_loss, row_loss)),
        "active_frames": metrics.get("active_frames", 0),
        "reference_exact": metrics.get("reference_exact", 1),
        "model_overfit_pass": capacity_pass,
        "better_than_split_raw": int(row_loss < initial_loss),
        "fusion_instance_used": instance_used,
        "fusion_camera_used": camera_used,
        "heldout_fusion_best": heldout_fusion_best,
    }


def _forward_variant(
    model: V6CameraFusion,
    batch: dict[str, torch.Tensor],
    baseline_w2c: torch.Tensor,
    *,
    reference_index: int,
    variant: str,
    mode: str,
) -> dict[str, torch.Tensor]:
    inputs = perturb_instance_inputs(batch, variant)
    return model(
        camera_hidden=inputs["camera_hidden"],
        baseline_world_to_camera=baseline_w2c,
        appearance=inputs["appearance"],
        geometry=inputs["pose_geometry"],
        quality=inputs["quality"],
        observed=inputs["observed"],
        identity_valid=inputs["identity_valid"],
        identity_unknown=inputs["identity_unknown"],
        reference_index=reference_index,
        mode=mode,
    )


def _pose_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    reference_index: int,
    translation_weight: float,
    evaluation_indices: list[int] | None = None,
    component: str = "se3",
) -> torch.Tensor:
    if component not in ("se3", "rotation", "center"):
        raise ValueError(f"Unknown V6 pose-loss component: {component!r}.")
    indices = _evaluation_indices(
        predicted.shape[1],
        reference_index=reference_index,
        evaluation_indices=evaluation_indices,
    )
    if not indices:
        raise ValueError("V6 needs at least one non-reference frame.")
    index = torch.tensor(indices, dtype=torch.long, device=predicted.device)
    predicted = predicted.index_select(1, index)
    target = target.index_select(1, index)
    rotation = F.mse_loss(predicted[..., :3, :3], target[..., :3, :3])
    translation = F.smooth_l1_loss(
        _camera_centers(predicted),
        _camera_centers(target),
        beta=0.01,
    )
    if component == "rotation":
        return rotation
    if component == "center":
        return float(translation_weight) * translation
    return rotation + float(translation_weight) * translation


def _pose_metrics(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    reference_index: int,
    translation_weight: float,
    evaluation_indices: list[int] | None = None,
) -> dict[str, float]:
    indices = _evaluation_indices(
        predicted.shape[1],
        reference_index=reference_index,
        evaluation_indices=evaluation_indices,
    )
    index = torch.tensor(indices, dtype=torch.long, device=predicted.device)
    predicted_eval = predicted.index_select(1, index)
    target_eval = target.index_select(1, index)
    relative = (
        predicted_eval[..., :3, :3]
        @ target_eval[..., :3, :3].transpose(-1, -2)
    )
    cosine = (
        torch.diagonal(relative, dim1=-2, dim2=-1).sum(dim=-1) - 1.0
    ) * 0.5
    rotation = torch.rad2deg(torch.acos(cosine.clamp(-1.0, 1.0))).mean()
    translation = torch.linalg.vector_norm(
        _camera_centers(predicted_eval) - _camera_centers(target_eval),
        dim=-1,
    ).mean()
    loss = _pose_loss(
        predicted,
        target,
        reference_index=reference_index,
        translation_weight=translation_weight,
        evaluation_indices=evaluation_indices,
    )
    return {
        "rotation_degrees": float(rotation.cpu()),
        "translation_native": float(translation.cpu()),
        "loss": float(loss.cpu()),
    }


def _evaluation_indices(
    sequence_length: int,
    *,
    reference_index: int,
    evaluation_indices: list[int] | None,
) -> list[int]:
    indices = (
        [
            index
            for index in range(sequence_length)
            if index != int(reference_index)
        ]
        if evaluation_indices is None
        else [int(index) for index in evaluation_indices]
    )
    if not indices:
        raise ValueError("V6 evaluation requires at least one frame.")
    if len(set(indices)) != len(indices):
        raise ValueError("V6 evaluation indices contain duplicates.")
    if any(index < 0 or index >= sequence_length for index in indices):
        raise ValueError("V6 evaluation index is outside the sequence.")
    if int(reference_index) in indices:
        raise ValueError("V6 held-out evaluation must not include the reference.")
    return indices


def _camera_centers(world_to_camera: torch.Tensor) -> torch.Tensor:
    rotation = world_to_camera[..., :3, :3]
    translation = world_to_camera[..., :3, 3]
    return -(rotation.transpose(-1, -2) @ translation[..., None]).squeeze(-1)


def _loss_drop(initial: float, final: float) -> float:
    if initial <= 1e-12:
        return 0.0
    return 100.0 * (initial - final) / initial


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _short(value: float) -> str:
    return f"{float(value):.8g}"


def _path(value: object, base: Path) -> Path:
    if value is None or not str(value).strip():
        raise ValueError("V6 configuration contains an empty path.")
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    cwd = (Path.cwd() / path).resolve()
    if cwd.exists() or not (base / path).exists():
        return cwd
    return (base / path).resolve()


def _validate_config(config: V6ExperimentConfig) -> None:
    if not config.clip_name:
        raise ValueError("clip_name is required.")
    if not config.test_clip_name:
        raise ValueError("test_clip_name is required.")
    if config.test_clip_name == config.clip_name:
        raise ValueError("test_clip_name must differ from the training clip.")
    if config.training.steps < 1 or config.training.log_every < 1:
        raise ValueError("V6 training steps and log_every must be positive.")
    if config.training.learning_rate <= 0.0:
        raise ValueError("V6 learning_rate must be positive.")
    if config.fusion.hidden_dim < 8:
        raise ValueError("V6 hidden_dim is too small.")
    if config.fusion.hidden_dim % config.fusion.num_heads:
        raise ValueError("V6 hidden_dim must be divisible by num_heads.")
    for name, value in (
        ("memory_momentum", config.fusion.memory_momentum),
        ("min_track_confidence", config.fusion.min_track_confidence),
        ("unknown_reliability", config.fusion.unknown_reliability),
        ("instance_loss_ratio", config.success.instance_loss_ratio),
        ("shuffle_loss_ratio", config.success.shuffle_loss_ratio),
        ("camera_loss_ratio", config.success.camera_loss_ratio),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"V6 {name} must be in [0,1].")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="streaming_couping/configs/v6_camera_overfit.yaml",
    )
    parser.add_argument("--device", help="Override the V6 training device.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
