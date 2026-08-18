#!/usr/bin/env python3
"""Run the frozen training-free StreamVGGT QK pose retrieval policy."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any

import torch
import yaml

from streaming_couping.src.backbones.streamvggt_latent import (
    load_streamvggt_latent_model,
)
from streaming_couping.src.backbones.streamvggt_parallel import (
    LayerShardedStreamVGGT,
    assert_frame_repository_cache_equivalence,
    assert_processed_key_cache_equivalence,
)
from streaming_couping.src.config import load_config
from streaming_couping.src.external_repos import maybe_add_repo_to_path
from streaming_couping.src.learned_pose.baseline_runtime import (
    camera_centers,
    load_baseline_run_config,
    pose_metrics,
)
from streaming_couping.src.learned_pose.cache import cache_path, load_feature_cache
from streaming_couping.src.learned_pose.config import (
    ClipConfig,
    LearnedPoseConfig,
    load_learned_pose_config,
)
from streaming_couping.src.qk_pose_retrieval import (
    QKRetrievalPolicy,
    rank_qk_history,
    select_qk_history,
)
from streaming_couping.src.storage import expand_storage_path


REVISION = "v0_streamvggt_qk_pose_retrieval_r1"


@dataclass(frozen=True)
class QKRun:
    source_path: Path
    output_dir: Path
    policy: QKRetrievalPolicy


def main() -> None:
    args = _parse_args()
    data = load_learned_pose_config(args.config)
    baseline = load_baseline_run_config(args.config)
    run = _load_run(args.config)
    if args.output_dir:
        run = replace(
            run,
            output_dir=Path(args.output_dir).expanduser().resolve(),
        )
    if args.streamvggt_devices:
        data = replace(
            data,
            streamvggt_devices=tuple(
                value.strip()
                for value in args.streamvggt_devices.split(",")
                if value.strip()
            ),
        )
    clip = _find_clip(data, baseline.clip_name)
    path = cache_path(data, clip)
    payload = load_feature_cache(path)
    _validate_payload(payload, clip=clip)
    recovery = load_config(data.recovery_config)
    if not data.streamvggt_devices or not recovery.streaming_cache:
        raise ValueError("Clean QK retrieval requires layer-sharded streaming.")

    maybe_add_repo_to_path(recovery.streamvggt_repo)
    assert_processed_key_cache_equivalence()
    assert_frame_repository_cache_equivalence()
    print("V0 QK pose processed-key/repository equivalence passed")
    model = load_streamvggt_latent_model(
        repo_path=recovery.streamvggt_repo,
        checkpoint_path=recovery.streamvggt_checkpoint,
        device="cpu",
        strict=True,
    )
    runner = LayerShardedStreamVGGT(
        model,
        data.streamvggt_devices,
        selected_layer_indices=data.fusion.dpt_layer_indices,
        amp_dtype=data.streamvggt_amp_dtype,
    )
    for line in runner.layout_summary():
        print(f"  {line}")

    candidate_payload = _candidate_generation_payload(payload)
    candidate = _generate_qk_candidate(
        runner,
        candidate_payload,
        policy=run.policy,
    )
    result = _score_and_write(
        payload=payload,
        cache_path_value=path,
        run=run,
        candidate=candidate,
    )
    print(f"V0 QK pose result={result}")


@torch.inference_mode()
def _generate_qk_candidate(
    runner: LayerShardedStreamVGGT,
    payload: dict[str, Any],
    *,
    policy: QKRetrievalPolicy,
) -> dict[str, Any]:
    """Generate camera pose from one RGB/QK replay."""

    images = payload["stream_images"].detach().float().cpu()
    frame_numbers = tuple(int(value) for value in payload["frame_indices"])
    pose_encodings = []
    retrieval_rows = []
    runner.reset()

    def selector(
        frame_index: int,
        qk_scores: torch.Tensor,
    ) -> tuple[int, ...]:
        selected = select_qk_history(
            frame_index,
            qk_scores,
            policy=policy,
        )
        ranked = rank_qk_history(qk_scores)
        retrieval_rows.append(
            {
                "sequence_index": int(frame_index),
                "frame_index": frame_numbers[int(frame_index)],
                "history_frames": int(frame_index),
                "selected_history_budget": min(
                    int(policy.total_frame_budget),
                    int(frame_index),
                ),
                "selected_sequence_indices": _space(selected),
                "selected_frame_indices": _space(
                    frame_numbers[index] for index in selected
                ),
                "qk_ranked_sequence_indices": _space(ranked),
                "qk_ranked_frame_indices": _space(
                    frame_numbers[index] for index in ranked
                ),
                "qk_scores": " ".join(
                    f"{float(value):.8g}" for value in qk_scores
                ),
            }
        )
        return selected

    for frame_index in range(images.shape[0]):
        batch_frame = images[frame_index : frame_index + 1].unsqueeze(0)
        selected_tokens = runner.aggregate_frame(
            batch_frame,
            frame_index,
            history_selector=selector,
        )
        pose_encoding = runner.camera(selected_tokens)
        pose_encodings.append(pose_encoding[0, 0].detach().float().cpu())
        del selected_tokens, pose_encoding
    runner.reset()
    return {
        "pose_encoding": torch.stack(pose_encodings),
        "retrieval_rows": retrieval_rows,
    }


def _candidate_generation_payload(payload: dict) -> dict[str, Any]:
    allowed = ("stream_images", "frame_indices")
    output = {name: payload[name] for name in allowed}
    forbidden = (
        "tracking_masks_output",
        "tracking_masks_stream",
        "tracking_scores",
        "sam_track_ids",
        "sam_track_prompts",
        "target_pose_encoding",
        "target_world_points",
    )
    if any(name in output for name in forbidden):
        raise RuntimeError("Clean QK candidate payload contains a forbidden field.")
    return output


def _score_and_write(
    *,
    payload: dict,
    cache_path_value: Path,
    run: QKRun,
    candidate: dict[str, Any],
) -> Path:
    """Introduce raw/GT pose only after the RGB-only candidate is complete."""

    from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri

    image_size = tuple(int(value) for value in payload["image_size"])
    raw_pose, _ = pose_encoding_to_extri_intri(
        payload["baseline_pose_encoding"].unsqueeze(0).float(),
        image_size_hw=image_size,
    )
    candidate_pose, _ = pose_encoding_to_extri_intri(
        candidate["pose_encoding"].unsqueeze(0).float(),
        image_size_hw=image_size,
    )
    target_pose, _ = pose_encoding_to_extri_intri(
        payload["target_pose_encoding"].unsqueeze(0).float(),
        image_size_hw=image_size,
    )
    frames = tuple(int(value) for value in payload["frame_indices"])
    reference = int(payload["reference_sequence_index"])
    evaluation_indices = [
        index for index in range(len(frames)) if index != reference
    ]
    raw_metrics = pose_metrics(
        raw_pose,
        target_pose,
        reference_index=reference,
        evaluation_indices=evaluation_indices,
    )
    candidate_metrics = pose_metrics(
        candidate_pose,
        target_pose,
        reference_index=reference,
        evaluation_indices=evaluation_indices,
    )
    center_gain = _gain(
        raw_metrics["center_error_native"],
        candidate_metrics["center_error_native"],
    )
    rotation_gain = _gain(
        raw_metrics["rotation_degrees"],
        candidate_metrics["rotation_degrees"],
    )
    raw_rotation, raw_center = _pose_frame_errors(raw_pose, target_pose)
    candidate_rotation, candidate_center = _pose_frame_errors(
        candidate_pose,
        target_pose,
    )
    center_worse = _worse_count(
        raw_center,
        candidate_center,
        evaluation_indices,
    )
    rotation_worse = _worse_count(
        raw_rotation,
        candidate_rotation,
        evaluation_indices,
    )
    pose_pass = int(center_gain > 0.0 and rotation_gain > 0.0)

    frame_rows = []
    for index, frame in enumerate(frames):
        frame_rows.append(
            {
                "sequence_index": index,
                "frame_index": frame,
                "is_reference": int(index == reference),
                "raw_center_error_native": float(raw_center[index]),
                "candidate_center_error_native": float(candidate_center[index]),
                "raw_rotation_degrees": float(raw_rotation[index]),
                "candidate_rotation_degrees": float(candidate_rotation[index]),
            }
        )
    summary = {
        "schema": 1,
        "revision": REVISION,
        "baseline_version": "v0",
        "clip": payload["clip_name"],
        "cache": str(cache_path_value),
        "config": str(run.source_path),
        "stream_frames": len(frames),
        "frames": frames,
        "reference_sequence_index": reference,
        "reference_frame": frames[reference],
        "evaluated_pose_frames": len(evaluation_indices),
        "method": "retrieve_qk",
        "total_frame_budget": run.policy.total_frame_budget,
        "anchor_frames": run.policy.anchor_frames,
        "candidate_generation_fields": tuple(
            _candidate_generation_payload(payload)
        ),
        "candidate_generation_gt_fields": 0,
        "sam_pose_inputs": 0,
        "sam_mask_used_for_pose": 0,
        "sam_identity_used_for_pose": 0,
        "sam_hidden_features_used": 0,
        "model_trained": 0,
        "pose_loss_used": 0,
        "depth_head_run": 0,
        "point_head_run": 0,
        "pose_only_artifact": 1,
        "raw_metrics": raw_metrics,
        "candidate_metrics": candidate_metrics,
        "center_gain_percent": center_gain,
        "rotation_gain_percent": rotation_gain,
        "center_worse_frames": center_worse,
        "rotation_worse_frames": rotation_worse,
        "full_sequence_pose_pass": pose_pass,
        "selected_v0_pose_modified": 0,
        "claim": (
            "training_free_qk_retrieval_improves_pose_on_single_sequence"
            if pose_pass
            else "qk_pose_candidate_not_improved"
        ),
    }
    run.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(run.output_dir / "frame_metrics.csv", frame_rows)
    _write_csv(
        run.output_dir / "retrieval_diagnostics.csv",
        candidate["retrieval_rows"],
    )
    result = run.output_dir / "qk_pose_summary.json"
    result.write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf8",
    )
    torch.save(
        {
            "schema": 1,
            "revision": REVISION,
            "frame_indices": frames,
            "pose_encoding": candidate["pose_encoding"],
            "selected_world_to_camera": candidate_pose.detach().float().cpu(),
            "raw_world_to_camera": raw_pose.detach().float().cpu(),
            "selected_pose_branch": "retrieve_qk",
            "artifact_role": "pose_only",
        },
        run.output_dir / "qk_pose_output.pt",
    )
    _write_copyable(
        run.output_dir / "copyable_result.txt",
        summary=summary,
        output_dir=run.output_dir,
    )
    print("V0 TRAINING-FREE STREAMVGGT QK POSE RETRIEVAL")
    print(
        f"  frames={len(frames)} evaluated={len(evaluation_indices)} "
        f"center_gain={center_gain:.4f}% R_gain={rotation_gain:.4f}% "
        f"pass={pose_pass}"
    )
    print(f"  copyable_report={run.output_dir / 'copyable_result.txt'}")
    return result


def _pose_frame_errors(
    predicted: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    relative = predicted[..., :3, :3] @ target[..., :3, :3].transpose(-1, -2)
    cosine = (
        torch.diagonal(relative, dim1=-2, dim2=-1).sum(dim=-1) - 1.0
    ) * 0.5
    rotation = torch.rad2deg(torch.acos(cosine.clamp(-1, 1)))[0]
    center = torch.linalg.vector_norm(
        camera_centers(predicted) - camera_centers(target),
        dim=-1,
    )[0]
    return rotation.detach().float().cpu(), center.detach().float().cpu()


def _worse_count(
    raw: torch.Tensor,
    candidate: torch.Tensor,
    indices: list[int],
) -> int:
    index = torch.tensor(indices, dtype=torch.long)
    return int(
        (
            candidate.index_select(0, index)
            > raw.index_select(0, index)
        ).sum()
    )


def _gain(raw: float, candidate: float) -> float:
    if raw <= 0:
        raise ValueError("Raw pose metric must be positive.")
    return 100.0 * (float(raw) - float(candidate)) / float(raw)


def _write_copyable(path: Path, *, summary: dict, output_dir: Path) -> None:
    lines = [
        "===== COPYABLE_V0_QK_POSE_BEGIN =====",
        f"revision={REVISION}",
        f"clip={summary['clip']}",
        f"stream_frames={summary['stream_frames']}",
        f"reference_frame={summary['reference_frame']}",
        f"evaluated_pose_frames={summary['evaluated_pose_frames']}",
        "method=retrieve_qk",
        f"total_frame_budget={summary['total_frame_budget']}",
        f"anchor_frames={summary['anchor_frames']}",
        "candidate_generation_fields=stream_images,frame_indices",
        "sam_pose_inputs=0",
        "model_trained=0",
        "depth_head_run=0",
        "point_head_run=0",
        "pose_only_artifact=1",
        "",
        "raw_center_error_native,candidate_center_error_native,center_gain_percent,center_worse_frames,raw_rotation_degrees,candidate_rotation_degrees,rotation_gain_percent,rotation_worse_frames,full_sequence_pose_pass",
        ",".join(
            str(value)
            for value in (
                summary["raw_metrics"]["center_error_native"],
                summary["candidate_metrics"]["center_error_native"],
                summary["center_gain_percent"],
                summary["center_worse_frames"],
                summary["raw_metrics"]["rotation_degrees"],
                summary["candidate_metrics"]["rotation_degrees"],
                summary["rotation_gain_percent"],
                summary["rotation_worse_frames"],
                summary["full_sequence_pose_pass"],
            )
        ),
        "",
        f"claim={summary['claim']}",
        f"selected_v0_pose_modified={summary['selected_v0_pose_modified']}",
        "",
        "outputs:",
        f"summary={output_dir / 'qk_pose_summary.json'}",
        f"pose={output_dir / 'qk_pose_output.pt'}",
        f"frame_csv={output_dir / 'frame_metrics.csv'}",
        f"retrieval_csv={output_dir / 'retrieval_diagnostics.csv'}",
        f"copyable_report={path}",
        "===== COPYABLE_V0_QK_POSE_END =====",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}.")
    with path.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _load_run(path: str | Path) -> QKRun:
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf8") as handle:
        raw = yaml.safe_load(handle) or {}
    section = raw.get("qk_pose_retrieval", {})
    policy = QKRetrievalPolicy(
        total_frame_budget=int(section.get("total_frame_budget", 5)),
        anchor_frames=int(section.get("anchor_frames", 1)),
    )
    policy.validate()
    return QKRun(
        source_path=source,
        output_dir=expand_storage_path(
            section.get(
                "output_dir",
                "outputs/streaming_couping_v0/qk_pose_retrieval",
            )
        ),
        policy=policy,
    )


def _validate_payload(payload: dict, *, clip: ClipConfig) -> None:
    required = (
        "stream_images",
        "frame_indices",
        "image_size",
        "baseline_pose_encoding",
        "target_pose_encoding",
        "reference_sequence_index",
        "clip_name",
    )
    missing = [name for name in required if name not in payload]
    if missing:
        raise ValueError(f"V0 cache lacks QK pose-retrieval fields: {missing}.")
    if tuple(int(value) for value in payload["frame_indices"]) != clip.frame_indices:
        raise ValueError("V0 QK pose-retrieval cache frames differ from config.")


def _find_clip(config: LearnedPoseConfig, name: str) -> ClipConfig:
    selected = [clip for clip in config.clips if clip.name == name]
    if len(selected) != 1:
        raise ValueError(f"Clip {name!r} was not found exactly once.")
    return selected[0]


def _space(values) -> str:
    return " ".join(str(int(value)) for value in values)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="streaming_couping/configs/v0_baseline.yaml",
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--streamvggt-devices")
    return parser.parse_args()


if __name__ == "__main__":
    main()
