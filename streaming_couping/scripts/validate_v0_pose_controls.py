#!/usr/bin/env python3
"""Validate V0 temporal folds and parameter-matched camera input controls."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
import json
from pathlib import Path

import torch

from streaming_couping.src.config import load_config
from streaming_couping.src.external_repos import maybe_add_repo_to_path
from streaming_couping.src.learned_pose.baseline_runtime import (
    camera_centers,
    load_baseline_run_config,
    pose_metrics,
    prepare_cached_batch,
    seed_everything,
    slice_batch_prefix,
    train_pose_model,
)
from streaming_couping.src.learned_pose.cache import cache_path, load_feature_cache
from streaming_couping.src.learned_pose.config import load_learned_pose_config
from streaming_couping.src.learned_pose.dynamic_instance_baseline import (
    CameraPoseBaseline,
)


CONTROL_MODES = (
    ("normal", "camera_pose"),
    ("pose_only", "pose_only"),
    ("camera_token_only", "camera_only"),
    ("time_only", "time_only"),
)
FOLD_NAMES = ("short", "medium", "long")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="streaming_couping/configs/v0_baseline.yaml"
    )
    parser.add_argument("--device")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    data = load_learned_pose_config(args.config)
    run = load_baseline_run_config(args.config)
    if args.device:
        run = replace(run, training_device=str(args.device))
    if args.steps is not None:
        run = replace(
            run,
            optimizer=replace(run.optimizer, base_steps=int(args.steps)),
        )
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else run.output_dir / "validation"
    )
    result = validate_controls(data, run, output_dir=output_dir)
    print(f"V0 control validation={result}")


def validate_controls(data, run, *, output_dir: Path) -> Path:
    matches = [clip for clip in data.clips if clip.name == run.clip_name]
    if len(matches) != 1:
        raise ValueError(f"V0 validation clip {run.clip_name!r} is not unique.")
    clip = matches[0]
    path = cache_path(data, clip)
    payload = load_feature_cache(path)
    recovery = load_config(data.recovery_config)
    maybe_add_repo_to_path(recovery.streamvggt_repo)
    from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri

    batch, raw_pose, target_pose = prepare_cached_batch(
        payload,
        pose_decoder=pose_encoding_to_extri_intri,
        device=run.training_device,
        local_point_count=run.local_point_count,
    )
    frames = tuple(int(value) for value in payload["frame_indices"])
    positions = {frame: index for index, frame in enumerate(frames)}
    reference = int(payload["reference_sequence_index"])
    training_indices = [positions[frame] for frame in run.base_train_frames]
    evaluation_indices = [positions[frame] for frame in run.evaluation_frames]
    if len(evaluation_indices) != 12:
        raise ValueError("V0 controls require twelve future frames.")
    training_length = positions[max(run.base_train_frames)] + 1
    training_batch = slice_batch_prefix(batch, length=training_length)
    training_raw = raw_pose[:, :training_length]
    training_target = target_pose[:, :training_length]

    rows: list[dict[str, object]] = []
    training_results: dict[str, dict[str, object]] = {}
    parameter_counts: dict[str, int] = {}
    for label, input_mode in CONTROL_MODES:
        seed_everything(run.optimizer.seed)
        model = CameraPoseBaseline(
            camera_dim=int(batch["camera_hidden"].shape[-1]),
            config=run.model,
            input_mode=input_mode,
        ).to(run.training_device)
        parameter_counts[label] = sum(
            parameter.numel() for parameter in model.parameters()
        )
        try:
            training = train_pose_model(
                model,
                batch=training_batch,
                baseline=training_raw,
                target=training_target,
                reference_index=reference,
                training_indices=training_indices,
                steps=run.optimizer.base_steps,
                config=run.optimizer,
            )
            training_record: dict[str, object] = {
                "training_pass": 1,
                **training,
            }
        except RuntimeError as error:
            if not _is_expected_training_rejection(error):
                raise
            training_record = {
                "training_pass": 0,
                "training_error": str(error),
            }
        training_results[label] = training_record
        model.eval()
        with torch.no_grad():
            prediction = model(
                camera_hidden=batch["camera_hidden"],
                baseline_world_to_camera=raw_pose,
                reference_index=reference,
            )["world_to_camera"]
        for fold_index, fold in enumerate(FOLD_NAMES):
            indices = evaluation_indices[
                4 * fold_index : 4 * (fold_index + 1)
            ]
            rows.append(
                _fold_row(
                    method=label,
                    input_mode=input_mode,
                    fold=fold,
                    indices=indices,
                    frames=frames,
                    prediction=prediction,
                    raw_pose=raw_pose,
                    target_pose=target_pose,
                    reference_index=reference,
                    translation_weight=run.optimizer.translation_weight,
                    training_pass=int(training_record["training_pass"]),
                    minimum_gain=run.optimizer.min_evaluation_gain_percent,
                )
            )

    normal_rows = [row for row in rows if row["method"] == "normal"]
    all_fold_pass = int(all(int(row["fold_pass"]) for row in normal_rows))
    parameter_counts_match = int(len(set(parameter_counts.values())) == 1)
    beats_pose_only = int(
        _normal_beats_control_all_folds(rows, control="pose_only")
    )
    beats_camera_only = int(
        _normal_beats_control_all_folds(rows, control="camera_token_only")
    )
    beats_time_only = int(
        _normal_beats_control_all_folds(rows, control="time_only")
    )
    decision = {
        "schema": 1,
        "baseline_version": "v0",
        "protocol": "early_train_then_three_locked_future_folds",
        "clip": run.clip_name,
        "training_frames": run.base_train_frames,
        "evaluation_frames": run.evaluation_frames,
        "primary_metric": "mean_camera_center_error_native",
        "pose_loss_role": "secondary_only",
        "parameter_matched_controls": [label for label, _ in CONTROL_MODES],
        "parameter_counts": parameter_counts,
        "parameter_counts_match": parameter_counts_match,
        "training": training_results,
        "rows": rows,
        "normal_all_fold_pass": all_fold_pass,
        "normal_beats_pose_only_all_folds": beats_pose_only,
        "normal_beats_camera_token_only_all_folds": beats_camera_only,
        "normal_beats_time_only_all_folds": beats_time_only,
        "v0_pose_validation_pass": int(
            bool(all_fold_pass)
            and bool(parameter_counts_match)
            and bool(beats_pose_only)
            and bool(beats_time_only)
        ),
        "interpretation": (
            "pass=0: retain V0 only as an aggregate same-scene diagnostic; "
            "do not tune geometry or claim robust pose improvement."
        ),
        "config": asdict(run),
        "cache": str(path),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "v0_pose_control_validation.csv"
    _write_csv(csv_path, rows)
    result = output_dir / "v0_pose_control_decision.json"
    result.write_text(
        json.dumps(decision, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf8",
    )
    print("V0 POSE CONTROL DECISION")
    for row in normal_rows:
        print(
            f"  {row['fold']}: center_gain="
            f"{float(row['center_gain_vs_raw_percent']):.6g}% "
            f"worse={row['center_worse_frames']} "
            f"pass={row['fold_pass']}"
        )
    print(
        "  controls: "
        f"parameter_match={parameter_counts_match} "
        f"beats_pose_only={beats_pose_only} "
        f"beats_camera_only={beats_camera_only} "
        f"beats_time_only={beats_time_only}"
    )
    print(f"  final={decision['v0_pose_validation_pass']} output={result}")
    return result


def _fold_row(
    *,
    method: str,
    input_mode: str,
    fold: str,
    indices: list[int],
    frames: tuple[int, ...],
    prediction: torch.Tensor,
    raw_pose: torch.Tensor,
    target_pose: torch.Tensor,
    reference_index: int,
    translation_weight: float,
    training_pass: int,
    minimum_gain: float,
) -> dict[str, object]:
    raw = pose_metrics(
        raw_pose,
        target_pose,
        reference_index=reference_index,
        translation_weight=translation_weight,
        evaluation_indices=indices,
    )
    current = pose_metrics(
        prediction,
        target_pose,
        reference_index=reference_index,
        translation_weight=translation_weight,
        evaluation_indices=indices,
    )
    raw_error = _center_errors(raw_pose, target_pose, indices)
    current_error = _center_errors(prediction, target_pose, indices)
    center_gain = _gain(raw["center_error_native"], current["center_error_native"])
    worse_frames = int(current_error.gt(raw_error).sum().cpu())
    return {
        "fold": fold,
        "method": method,
        "input_mode": input_mode,
        "test_frames": " ".join(str(frames[index]) for index in indices),
        "training_pass": training_pass,
        "raw_center_error_native": raw["center_error_native"],
        "predicted_center_error_native": current["center_error_native"],
        "center_gain_vs_raw_percent": center_gain,
        "center_worse_frames": worse_frames,
        "raw_rotation_degrees": raw["rotation_degrees"],
        "predicted_rotation_degrees": current["rotation_degrees"],
        "rotation_gain_vs_raw_percent": _gain(
            raw["rotation_degrees"], current["rotation_degrees"]
        ),
        "pose_loss_gain_vs_raw_percent_secondary": _gain(
            raw["loss"], current["loss"]
        ),
        "fold_pass": int(
            training_pass
            and center_gain > float(minimum_gain)
            and worse_frames == 0
        ),
    }


def _center_errors(
    predicted: torch.Tensor,
    target: torch.Tensor,
    indices: list[int],
) -> torch.Tensor:
    index = torch.tensor(indices, dtype=torch.long, device=predicted.device)
    left = camera_centers(predicted.index_select(1, index))
    right = camera_centers(target.index_select(1, index))
    return torch.linalg.vector_norm(left - right, dim=-1)


def _gain(reference: float, candidate: float) -> float:
    return 100.0 * (float(reference) - float(candidate)) / max(
        abs(float(reference)), 1e-12
    )


def _normal_beats_control(
    rows: list[dict[str, object]], *, fold: str, control: str
) -> bool:
    normal = next(
        row for row in rows if row["fold"] == fold and row["method"] == "normal"
    )
    compared = next(
        row for row in rows if row["fold"] == fold and row["method"] == control
    )
    return (
        bool(normal["training_pass"])
        and bool(compared["training_pass"])
        and float(normal["predicted_center_error_native"])
        < float(compared["predicted_center_error_native"])
    )


def _normal_beats_control_all_folds(
    rows: list[dict[str, object]], *, control: str
) -> bool:
    return all(
        _normal_beats_control(rows, fold=fold, control=control)
        for fold in FOLD_NAMES
    )


def _is_expected_training_rejection(error: RuntimeError) -> bool:
    message = str(error)
    return (
        "V0 training is a no-op" in message
        or "V0 training loss did not pass the no-op gate" in message
    )


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("V0 validation produced no rows.")
    with path.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
