#!/usr/bin/env python3
"""Evaluate the frozen r2 residual head on one independent r3 scene.

The StreamVGGT backbone and original heads are inference-only.  The only
learned weights loaded here are the r2 ``residual_no_gate`` best-validation
checkpoint.  The r3 manifest is annotation-free during candidate generation;
its mesh-rasterized pointmaps are opened only after predictions are frozen for
the final raw-vs-residual evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

import torch
import yaml

from streaming_couping.src.backbones.streamvggt_latent import (
    StreamVGGTLatentAdapter,
    load_streamvggt_latent_model,
)
from streaming_couping.src.backbones.streamvggt_parallel import (
    LayerShardedStreamVGGT,
    assert_processed_key_cache_equivalence,
)
from streaming_couping.src.external_repos import maybe_add_repo_to_path
from streaming_couping.src.pointmap_alignment import (
    _load_gt_pointmaps,
    _paired_limit,
    _robust_similarity,
)
from streaming_couping.src.semantic_map import normalize_confidence
from streaming_couping.src.storage import expand_storage_path
from streaming_couping.src.trust_aware_residual import (
    TrustAwareResidualHead,
    apply_similarity,
    point_head_patch_features,
)


REVISION = "r3_cross_scene_residual_inference_r1"
BRANCH = "residual_no_gate"
EXPECTED_DPT_LAYERS = (4, 11, 17, 23)


def main() -> None:
    args = _parse_args()
    r3_manifest = Path(args.r3_manifest).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    v0_config = Path(args.v0_config).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    scene_id, frames = _load_r3_manifest(r3_manifest, args.scene_id)
    model_payload = _load_residual_checkpoint(checkpoint)
    run_cfg = _load_v0_runtime(v0_config)
    image_paths = [Path(row["image_path"]).expanduser().resolve() for row in frames]
    if not all(path.is_file() for path in image_paths):
        missing = [str(path) for path in image_paths if not path.is_file()]
        raise FileNotFoundError(f"r3 manifest has missing images: {missing[:3]}")

    devices = tuple(value.strip() for value in args.devices.split(",") if value.strip())
    if devices:
        # The equivalence audit imports the external StreamVGGT package
        # directly, so register its repo before running the audit.  The model
        # loader performs the same registration later, but that is too late
        # for this preflight check.
        maybe_add_repo_to_path(run_cfg["streamvggt_repo"])
        assert_processed_key_cache_equivalence()
    print("R3 CROSS-SCENE FROZEN RESIDUAL EVALUATION")
    print(
        f"  scene={scene_id} frames={len(frames)} branch={BRANCH} "
        f"devices={','.join(devices) if devices else args.device}"
    )
    print("  candidate generation=RGB + frozen StreamVGGT only; SAM=0; training=0")

    model = load_streamvggt_latent_model(
        repo_path=run_cfg["streamvggt_repo"],
        checkpoint_path=run_cfg["streamvggt_checkpoint"],
        device="cpu" if devices else args.device,
        strict=True,
    )
    runner = None
    if devices:
        runner = LayerShardedStreamVGGT(
            model,
            devices,
            selected_layer_indices=EXPECTED_DPT_LAYERS,
            amp_dtype=run_cfg["amp_dtype"],
        )
        for line in runner.layout_summary():
            print(f"  {line}")
    adapter = StreamVGGTLatentAdapter(
        model,
        device=devices[0] if devices else args.device,
        image_mode=run_cfg["image_mode"],
        dpt_layer_indices=EXPECTED_DPT_LAYERS,
        parallel_runner=runner,
    )
    with torch.inference_mode():
        output = adapter.extract_from_paths(
            image_paths,
            return_pointmap=True,
            streaming_cache=True,
        )
    candidate = _run_candidate_head(output, model_payload)
    del adapter, runner, model, output
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("  candidate predictions frozen; opening r3 GT pointmaps for scoring")

    target = _load_gt_pointmaps(
        r3_manifest,
        scene_id=scene_id,
        frame_indices=tuple(range(len(frames))),
        processed_size=candidate["image_size"],
        image_mode=run_cfg["image_mode"],
    )
    target = _normalize_dense_layout(
        target,
        candidate["image_size"],
        channels=3,
    )
    summary, branch_rows, frame_rows = _score(
        candidate,
        target,
        confidence_threshold=float(args.confidence_threshold),
        maximum_points=int(args.maximum_points),
    )
    summary.update(
        {
            "schema": 1,
            "revision": REVISION,
            "scene_id": scene_id,
            "frame_count": len(frames),
            "source_frame_indices": [int(row["source_sequence_index"]) for row in frames],
            "r3_manifest": str(r3_manifest),
            "residual_checkpoint": str(checkpoint),
            "residual_checkpoint_best_epoch_one_based": int(
                model_payload["branches"][BRANCH]["checkpoint_selection"]["best_epoch"]
            )
            + 1,
            "dpt_layer_indices": list(EXPECTED_DPT_LAYERS),
            "sam_inputs": 0,
            "model_trained": 0,
            "backbone_parameters_updated": 0,
            "point_head_parameters_updated": 0,
            "camera_head_parameters_updated": 0,
            "formal_v0_modified": 0,
            "gt_values_used_for_candidate_generation": 0,
            "evaluation_gauge": "first_frame_fixed_raw_reference_sim3",
        }
    )
    decision = summary["decision"]
    summary["claim"] = (
        "cross_scene_residual_beats_raw_under_frozen_protocol"
        if decision["cross_scene_decision"] == "GO"
        else "cross_scene_residual_generalization_not_established"
    )
    _write_json(output_dir / "summary.json", summary)
    _write_csv(output_dir / "branch_summary.csv", branch_rows)
    _write_csv(output_dir / "frame_metrics.csv", frame_rows)
    _write_copyable(output_dir / "copyable_result.txt", summary)
    print(
        f"  raw_rmse={summary['branch_lookup']['raw_full_history']['rmse']:.6f} "
        f"residual_rmse={summary['branch_lookup'][BRANCH]['rmse']:.6f} "
        f"decision={decision['cross_scene_decision']}"
    )
    print(f"  result={output_dir / 'summary.json'}")


def _load_r3_manifest(path: Path, scene_id: str | None) -> tuple[str, list[dict[str, Any]]]:
    manifest = json.loads(path.read_text(encoding="utf8"))
    scenes = manifest.get("scenes", [])
    if len(scenes) != 1:
        raise ValueError(f"r3 manifest must contain one scene, got {len(scenes)}")
    scene = scenes[0]
    actual = str(scene.get("scene_id", ""))
    if scene_id and actual != str(scene_id):
        raise ValueError(f"r3 scene mismatch: manifest={actual!r}, requested={scene_id!r}")
    frames = list(scene.get("frames", []))
    if len(frames) != 30:
        raise ValueError(f"r3 protocol requires 30 frames, got {len(frames)}")
    if any(int(row.get("sequence_index", -1)) != i for i, row in enumerate(frames)):
        raise ValueError("r3 manifest sequence_index values are not contiguous")
    if any("semantic_mask" in row or "instance_mask" in row for row in frames):
        raise ValueError("r3 residual evaluation must not consume semantic or instance masks")
    return actual, frames


def _load_residual_checkpoint(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("revision") != "phase1_temporal_trust_aware_pointmap_residual_r2":
        raise ValueError(f"Unsupported r2 residual checkpoint: {path}")
    branch = payload.get("branches", {}).get(BRANCH)
    if not isinstance(branch, dict) or "state_dict" not in branch:
        raise ValueError(f"Checkpoint lacks frozen {BRANCH} state: {path}")
    selection = branch.get("checkpoint_selection", {})
    if int(selection.get("test_metrics_read_during_selection", 1)) != 0:
        raise ValueError("r2 checkpoint was not selected with test-blind validation")
    return payload


def _load_v0_runtime(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf8")) or {}
    recovery_path = expand_storage_path(raw["recovery_config"], base=path.parent)
    recovery = yaml.safe_load(recovery_path.read_text(encoding="utf8")) or {}
    stream = recovery.get("streamvggt", {})
    return {
        "streamvggt_repo": expand_storage_path(stream.get("repo", "externals/streamvggt"), base=path.parent),
        "streamvggt_checkpoint": expand_storage_path(stream["checkpoint"], base=path.parent),
        "image_mode": str(stream.get("image_mode", "crop")),
        "amp_dtype": "bfloat16",
    }


def _run_candidate_head(output: Any, checkpoint: dict[str, Any]) -> dict[str, Any]:
    aux = output.geometry.aux
    # ``image_shape`` is returned in the adapter-level auxiliary payload;
    # geometry.aux contains the feature/head tensors but not that metadata.
    image_shape = output.aux.get("image_shape")
    if image_shape is None:
        image_shape = aux.get("image_shape")
    if image_shape is None:
        raise ValueError("StreamVGGT output lacks processed image_shape metadata")
    expected_image_size = tuple(int(v) for v in image_shape)
    dpt = aux.get("stream_dpt_tokens")
    if not isinstance(dpt, list) or len(dpt) != 4:
        raise ValueError("StreamVGGT output lacks four DPT feature levels")
    levels = torch.stack([value.detach().float().cpu()[0] for value in dpt], dim=0)
    patch_shape = tuple(int(v) for v in aux["patch_shape"])
    features = point_head_patch_features(
        levels,
        patch_start_idx=int(aux["patch_start_idx"]),
        patch_shape=patch_shape,
    )
    pointmap = aux.get("pointmap_dense")
    confidence = aux.get("confidence_dense")
    if pointmap is None or confidence is None:
        raise ValueError("StreamVGGT output lacks raw pointmap/confidence")
    raw = _normalize_dense_layout(
        pointmap.detach().float().cpu(),
        expected_image_size,
        channels=3,
    )
    confidence = normalize_confidence(confidence.detach().float().cpu())
    confidence = _normalize_dense_layout(
        confidence,
        expected_image_size,
        channels=None,
    )
    model_cfg = checkpoint["model"]
    training_patch_shape = tuple(int(v) for v in model_cfg["patch_shape"])
    model = TrustAwareResidualHead(
        feature_channels=int(model_cfg["feature_channels"]),
        level_count=int(model_cfg["level_count"]),
        # All learned layers are 1x1/3x3 convolutions, so the frozen head is
        # spatially transferable across valid DPT grids.
        patch_shape=patch_shape,
        projection_channels=int(model_cfg["projection_channels"]),
        hidden_channels=int(model_cfg["hidden_channels"]),
        use_gate=False,
        use_uncertainty=False,
        gate_bias=float(model_cfg["gate_bias"]),
    )
    model.load_state_dict(checkpoint["branches"][BRANCH]["state_dict"], strict=True)
    model.eval()
    with torch.inference_mode():
        prediction = raw + model(features.float(), output_size=tuple(raw.shape[1:3])).correction.cpu()
    return {
        "raw": raw,
        "prediction": prediction,
        "confidence": confidence,
        "image_size": expected_image_size,
        "feature_shape": tuple(int(v) for v in features.shape),
        "source_patch_shape": patch_shape,
        "training_patch_shape": training_patch_shape,
        "fully_convolutional_patch_shape_transfer": 1,
    }


def _normalize_dense_layout(
    values: torch.Tensor,
    expected_hw: tuple[int, int],
    *,
    channels: int | None,
) -> torch.Tensor:
    """Return dense predictions in the canonical ``[T,H,W,C]`` layout.

    Some StreamVGGT point-head builds expose the spatial axes in the reverse
    order for non-square crops.  The adapter's ``image_shape`` is the source
    of truth for the processed image, so transpose only when the returned
    tensor is exactly ``[T,W,H,...]``.  This keeps the original scene behavior
    unchanged while making the cross-scene aspect-ratio case explicit.
    """

    tensor = values
    if tensor.ndim == 5 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.ndim == 4 and channels is not None and tensor.shape[-1] != channels:
        raise ValueError(f"Expected dense channels={channels}, got {tuple(tensor.shape)}")
    if tensor.ndim not in (3, 4):
        raise ValueError(f"Expected dense [T,H,W,(C)], got {tuple(tensor.shape)}")
    height, width = expected_hw
    if tuple(int(v) for v in tensor.shape[1:3]) == (height, width):
        return tensor.contiguous()
    if tuple(int(v) for v in tensor.shape[1:3]) == (width, height):
        return tensor.transpose(1, 2).contiguous()
    raise ValueError(
        "StreamVGGT dense output does not match processed image shape: "
        f"output={tuple(tensor.shape)} expected_hw={expected_hw}"
    )


def _score(
    candidate: dict[str, Any],
    target: torch.Tensor,
    *,
    confidence_threshold: float,
    maximum_points: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    raw = candidate["raw"].float()
    prediction = candidate["prediction"].float()
    confidence = candidate["confidence"].float()
    target = target.float()
    if raw.shape != target.shape or prediction.shape != target.shape:
        raise ValueError(f"Candidate/GT shape mismatch: raw={raw.shape}, gt={target.shape}")
    valid = (
        torch.isfinite(raw).all(dim=-1)
        & torch.isfinite(prediction).all(dim=-1)
        & torch.isfinite(target).all(dim=-1)
        & torch.isfinite(confidence)
        & (confidence >= float(confidence_threshold))
    )
    if any(int(valid[i].sum()) < 128 for i in range(raw.shape[0])):
        raise ValueError("A cross-scene frame has fewer than 128 common valid points")
    source, truth = _paired_limit(
        raw[0].reshape(-1, 3)[valid[0].reshape(-1)],
        target[0].reshape(-1, 3)[valid[0].reshape(-1)],
        max_points=30000,
    )
    scale, rotation, translation, inliers, fit_rmse = _robust_similarity(
        source, truth, min_points=128
    )
    aligned_raw = apply_similarity(raw, scale=scale, rotation=rotation, translation=translation)
    aligned_prediction = apply_similarity(prediction, scale=scale, rotation=rotation, translation=translation)
    branches = {"raw_full_history": aligned_raw, BRANCH: aligned_prediction}
    branch_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    for name, points in branches.items():
        errors = []
        for index in range(points.shape[0]):
            selected = _limited_indices(valid[index], maximum_points)
            error = torch.linalg.vector_norm(
                points[index].reshape(-1, 3).index_select(0, selected)
                - target[index].reshape(-1, 3).index_select(0, selected),
                dim=-1,
            )
            errors.append(error)
            frame_rows.append({
                "branch": name,
                "sequence_index": index,
                "supported_points": int(valid[index].sum()),
                "evaluated_points": int(error.numel()),
                "rmse": _rmse(error),
                "median": float(error.median()),
                "p90": float(torch.quantile(error, 0.90)),
            })
        joined = torch.cat(errors)
        branch_rows.append({
            "branch": name,
            "supported_points": int(valid.sum()),
            "evaluated_points": int(joined.numel()),
            "rmse": _rmse(joined),
            "median": float(joined.median()),
            "p90": float(torch.quantile(joined, 0.90)),
            "alignment_scale": float(scale),
            "alignment_inliers": int(inliers),
            "alignment_fit_rmse": float(fit_rmse),
        })
    lookup = {row["branch"]: row for row in branch_rows}
    raw_row = lookup["raw_full_history"]
    residual_row = lookup[BRANCH]
    residual_frames = {row["sequence_index"]: row for row in frame_rows if row["branch"] == BRANCH}
    raw_frames = {row["sequence_index"]: row for row in frame_rows if row["branch"] == "raw_full_history"}
    improved = sum(residual_frames[i]["rmse"] < raw_frames[i]["rmse"] for i in raw_frames)
    decision = {
        "cross_scene_decision": "GO" if residual_row["rmse"] < raw_row["rmse"] and residual_row["p90"] <= raw_row["p90"] else "NO_GO",
        "go_rule": "residual_rmse_below_raw_and_p90_not_above_raw",
        "improved_frames": int(improved),
        "improved_frame_ratio": float(improved / len(raw_frames)),
        "cross_scene_generalization_claim": 0,
    }
    return {"branches": branch_rows, "branch_lookup": lookup, "decision": decision}, branch_rows, frame_rows


def _limited_indices(mask: torch.Tensor, maximum: int) -> torch.Tensor:
    selected = torch.nonzero(mask.reshape(-1), as_tuple=False)[:, 0]
    if selected.numel() <= int(maximum):
        return selected
    return selected.index_select(0, torch.linspace(0, selected.numel() - 1, int(maximum)).long())


def _rmse(values: torch.Tensor) -> float:
    return float(torch.sqrt(values.square().mean()))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf8")


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_copyable(path: Path, summary: dict[str, Any]) -> None:
    raw = summary["branch_lookup"]["raw_full_history"]
    residual = summary["branch_lookup"][BRANCH]
    decision = summary["decision"]
    lines = [
        "===== COPYABLE_R3_CROSS_SCENE_RESIDUAL_BEGIN =====",
        f"revision={summary['revision']}",
        f"scene_id={summary['scene_id']}",
        f"frames={summary['frame_count']}",
        f"branch={BRANCH}",
        "pose=not_modified_inference_only",
        "sam_inputs=0",
        "model_trained=0",
        "gt_values_used_for_candidate_generation=0",
        "formal_v0_modified=0",
        "branch,rmse,median,p90,rmse_gain_vs_raw_percent,p90_gain_vs_raw_percent",
    ]
    for row in (raw, residual):
        lines.append(",".join([
            str(row["branch"]),
            str(row["rmse"]),
            str(row["median"]),
            str(row["p90"]),
            "0.0" if row["branch"] == "raw_full_history" else str(_gain(raw["rmse"], row["rmse"])),
            "0.0" if row["branch"] == "raw_full_history" else str(_gain(raw["p90"], row["p90"])),
        ]))
    lines.extend([
        "",
        "decision=" + json.dumps(decision, sort_keys=True),
        "claim=" + summary["claim"],
        "summary=" + str(path.with_name("summary.json")),
        "branch_csv=" + str(path.with_name("branch_summary.csv")),
        "frame_csv=" + str(path.with_name("frame_metrics.csv")),
        "===== COPYABLE_R3_CROSS_SCENE_RESIDUAL_END =====",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf8")


def _gain(raw: float, candidate: float) -> float:
    return 100.0 * (float(raw) - float(candidate)) / max(abs(float(raw)), 1e-12)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if torch.is_tensor(value):
        return value.tolist()
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r3-manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--v0-config", default="streaming_couping/configs/v0_baseline.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scene-id", default=None)
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--confidence-threshold", type=float, default=0.30)
    parser.add_argument("--maximum-points", type=int, default=8192)
    return parser.parse_args()


if __name__ == "__main__":
    main()
