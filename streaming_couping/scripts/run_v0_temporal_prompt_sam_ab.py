#!/usr/bin/env python3
"""Run a small, real-SAM A/B ablation on the frozen V0 scene.

This experiment is the first stage after the frozen A--E temporal-prompt
matrix.  It does *not* rerun StreamVGGT, change a pose/pointmap, or propagate
a new SAM video track.  Instead, for a deterministic subset of causal
frame/slot queries it performs a single-frame SAM3.1 re-segmentation with the
historical positive points and writes the returned mask back into the same
frozen V0 slot.  All other frames remain byte-for-byte equivalent to the raw
V0 branch.

The five evaluated branches are:

``raw_v0``
    The cached V0 mask, with no SAM call.
``A_center``
    One historical center point.
``C_surface_5``
    Five historical, spatially spread surface points.
``E_depth_gate_rel_0.15`` / ``E_depth_gate_rel_0.20``
    The corresponding depth-gated point sets from the frozen matrix.

Candidate CSVs are read and written before SAM is loaded/called.  Ground truth
is opened only after every SAM output and fallback decision has been frozen.
The resulting tracking/map metrics are therefore an evaluation of a fixed
causal experiment, not a GT-guided prompt selection.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from streaming_couping.scripts.run_semantic_map_evaluation import (
    _load_run as _load_evaluation_run,
)
from streaming_couping.scripts.run_v0_bidirectional_feedback import (
    V0Arrays,
    _find_clip,
    _load_arrays,
    _validate_v0_inputs,
)
from streaming_couping.src.backbones.sam3_wrapper import SAM3Wrapper
from streaming_couping.src.config import load_config
from streaming_couping.src.learned_pose.baseline_runtime import (
    load_baseline_run_config,
)
from streaming_couping.src.learned_pose.cache import cache_path, load_feature_cache
from streaming_couping.src.learned_pose.config import load_learned_pose_config
from streaming_couping.src.semantic_map_metrics import (
    apply_similarity,
    evaluate_semantic_object_map,
    load_ground_truth_stream_masks,
)
from streaming_couping.src.semantic_tracking_metrics import (
    GroundTruthInstances,
    evaluate_tracking_variants,
    load_ground_truth_instances,
)
from streaming_couping.src.storage import expand_storage_path
from streaming_couping.src.temporal_prompt_sam_ab import (
    REAL_SAM_BRANCHES,
    PromptBranchSpec,
    build_prompt_mask,
    decide_score_fallback,
    deterministic_query_subset,
    query_key,
    rows_for_branch,
    summarize_prompt_events,
    summarize_worsened_frames,
)


REVISION = "v0_temporal_prompt_real_sam_ab_r1"
DEFAULT_CONFIG = "streaming_couping/configs/v0_baseline.yaml"
DEFAULT_CANDIDATE_DIR = (
    "${VGGT_SAM_STORAGE_ROOT}/outputs/"
    "streaming_couping_v0_temporal_prompt_matrix"
)
DEFAULT_OUTPUT = (
    "${VGGT_SAM_STORAGE_ROOT}/outputs/"
    "streaming_couping_v0_temporal_prompt_sam_ab"
)
EXPECTED_CLIP = "00a231a370_90_525_step15_37_68_54"
CAUSAL_GT_FIELDS = (
    "gt_assignment_available",
    "gt_visible",
    "gt_mask_hit",
    "gt_instance_id",
    "gt_correct_count",
    "gt_area_pixels",
)


def main() -> None:
    args = _parse_args()
    _validate_args(args)
    config_path = Path(args.config).expanduser().resolve()
    data = load_learned_pose_config(config_path)
    baseline = load_baseline_run_config(config_path)
    if baseline.version != "v0":
        raise ValueError("This experiment requires baseline.version=v0.")
    if len(data.clips) != 1:
        raise ValueError(
            "The real-SAM A/B experiment is intentionally sealed to one V0 clip."
        )
    clip = _find_clip(data, baseline.clip_name)
    if clip.name != EXPECTED_CLIP:
        raise ValueError(
            "The retained real-SAM A/B experiment is sealed to clip "
            f"{EXPECTED_CLIP!r}; got {clip.name!r}."
        )

    candidate_dir = expand_storage_path(
        args.candidate_dir or DEFAULT_CANDIDATE_DIR,
        base=config_path.parent,
    )
    output_dir = expand_storage_path(
        args.output_dir or DEFAULT_OUTPUT,
        base=config_path.parent,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_value = cache_path(data, clip)
    if not cache_value.is_file():
        raise FileNotFoundError(
            f"Missing frozen V0 cache {cache_value}; run commands_v0_baseline.txt first."
        )
    payload = load_feature_cache(cache_value)
    baseline_summary, poses = _validate_v0_inputs(
        payload=payload,
        summary_path=baseline.output_dir / "baseline_summary.json",
        poses_path=baseline.output_dir / "poses.pt",
        clip=clip,
    )
    arrays = _load_arrays(payload, poses)
    frames = tuple(int(value) for value in payload["frame_indices"])
    track_ids = tuple(int(value) for value in payload["sam_track_ids"])
    track_prompts = tuple(str(value) for value in payload["sam_track_prompts"])
    image_paths = tuple(Path(str(value)).expanduser() for value in payload["image_paths"])
    _validate_image_paths(image_paths, expected_count=len(frames))

    print("V0 REAL SAM TEMPORAL PROMPT A/B ABLATION")
    print(f"config={config_path}")
    print(f"cache={cache_value}")
    print(f"candidate_dir={candidate_dir}")
    print(f"output={output_dir}")
    print(
        "scope=one sealed V0 scene; StreamVGGT pose/pointmap and V0 slot IDs "
        "are frozen"
    )
    print(
        "SAM protocol=single-frame SAM3.1 text+positive-points resegmentation; "
        "no temporal propagation"
    )
    print(
        "branches=raw_v0,A_center,C_surface_5,"
        "E_depth_gate_rel_0.15,E_depth_gate_rel_0.20"
    )
    print(
        f"score_fallback=absolute_margin:{args.score_fallback_margin:.3f},"
        f"relative_ratio:{args.score_fallback_ratio:.3f} "
        f"worsened_iou_margin={args.worsened_iou_margin:.3f}"
    )

    point_rows, query_rows, candidate_metadata = _load_causal_candidates(
        candidate_dir,
        arrays=arrays,
        frames=frames,
        track_ids=track_ids,
        payload=payload,
    )
    selected_keys = _select_query_keys(
        query_rows,
        limit=int(args.max_query_events),
    )
    causal_events = _prepare_causal_events(
        point_rows,
        query_rows,
        selected_keys=selected_keys,
        source_size=arrays.image_size,
        target_size=arrays.map_size,
    )
    _write_csv(output_dir / "prompt_candidates_causal.csv", causal_events)
    (output_dir / "candidate_metadata.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "revision": REVISION,
                "source_matrix_revision": candidate_metadata.get("revision", ""),
                "candidate_generation_gt_fields": 0,
                "selected_query_count": len(selected_keys),
                "selected_query_keys": [list(key) for key in selected_keys],
                "branch_specs": [asdict(spec) for spec in REAL_SAM_BRANCHES],
                "source_metadata": dict(candidate_metadata),
            },
            indent=2,
            sort_keys=True,
            allow_nan=True,
        )
        + "\n",
        encoding="utf8",
    )
    print(
        f"candidate artifacts frozen queries={len(selected_keys)} "
        f"max_query_events={args.max_query_events} "
        f"candidate_rows={len(causal_events)} "
        f"estimated_sam_calls<={len(selected_keys) * len(REAL_SAM_BRANCHES)}"
    )

    # ------------------------------
    # Real SAM phase.  No GT is read above or below until this phase finishes.
    # ------------------------------
    recovery = load_config(
        data.recovery_config,
        {"sam3_device": str(args.sam3_device)},
    )
    if recovery.sam3_version != "sam3.1":
        raise ValueError(
            "The real temporal-prompt A/B requires recovery sam3.version=sam3.1."
        )
    print(
        f"loading SAM3.1 checkpoint={recovery.sam3_checkpoint} "
        f"device={recovery.sam3_device}"
    )
    sam3 = SAM3Wrapper(
        repo_path=recovery.sam3_repo,
        checkpoint_path=recovery.sam3_checkpoint,
        device=recovery.sam3_device,
        output_threshold=recovery.sam3_output_threshold,
        # The A/B branches are deliberately points-only.  The recovery config
        # is still reused for the model/checkpoint/session settings.
        prompt_with_box=False,
        version=recovery.sam3_version,
        use_fa3=recovery.sam3_use_fa3,
        max_num_objects=recovery.sam3_max_num_objects,
        multiplex_count=recovery.sam3_multiplex_count,
    ).load()
    variant_masks: dict[str, torch.Tensor] = {
        "raw_v0": arrays.raw_masks.detach().cpu().bool().clone()
    }
    variant_scores: dict[str, torch.Tensor] = {
        "raw_v0": arrays.scores.detach().cpu().float().clone()
    }
    event_rows = _run_real_sam_branches(
        sam3=sam3,
        image_paths=image_paths,
        arrays=arrays,
        track_prompts=track_prompts,
        point_rows=point_rows,
        selected_keys=selected_keys,
        causal_events=causal_events,
        variant_masks=variant_masks,
        variant_scores=variant_scores,
        score_fallback_margin=float(args.score_fallback_margin),
        score_fallback_ratio=float(args.score_fallback_ratio),
        max_positive_points=int(args.max_positive_points),
        progress_interval=int(args.progress_interval),
    )
    _write_csv(output_dir / "sam_events_causal.csv", event_rows)
    mask_artifact = output_dir / "prompted_masks_causal.pt"
    torch.save(
        {
            "schema": 1,
            "revision": REVISION,
            "candidate_generation_gt_fields": 0,
            "sam_generation_gt_fields": 0,
            "clip": str(payload["clip_name"]),
            "scene_id": str(payload["scene_id"]),
            "frame_indices": frames,
            "variant_names": tuple(variant_masks),
            "variant_masks": variant_masks,
            "variant_scores": variant_scores,
            "selected_query_keys": selected_keys,
            "event_count": len(event_rows),
            "score_fallback_margin": float(args.score_fallback_margin),
            "score_fallback_ratio": float(args.score_fallback_ratio),
            "sam_score_semantics": (
                "SAM3 out_probs when exposed by the predictor; the existing "
                "wrapper uses 1.0 as a visibility fallback when out_probs is absent"
            ),
        },
        mask_artifact,
    )
    print(
        f"SAM outputs frozen event_rows={len(event_rows)} "
        f"mask_artifact={mask_artifact}"
    )
    del sam3
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ------------------------------
    # Sealed evaluation phase.  GT is first opened here.
    # ------------------------------
    print("all prompted masks frozen; opening sealed GT evaluation")
    evaluation_run = _load_evaluation_run(config_path)
    ground_truth = load_ground_truth_instances(
        data.manifest,
        scene_id=str(payload["scene_id"]),
        frame_indices=frames,
        output_size=arrays.map_size,
        prompts=tuple(str(value) for value in payload["instance_prompts"]),
        prompt_label_aliases=evaluation_run.prompt_label_aliases,
    )
    gt_masks = load_ground_truth_stream_masks(
        data.manifest,
        scene_id=str(payload["scene_id"]),
        frame_indices=frames,
        instance_ids=ground_truth.instance_ids,
        processed_size=arrays.map_size,
        image_mode=str(payload["image_mode"]),
    )
    gt = GroundTruthInstances(
        masks=gt_masks,
        instance_ids=ground_truth.instance_ids,
        labels=ground_truth.labels,
        all_visible_instance_ids=ground_truth.all_visible_instance_ids,
    )
    tracking = evaluate_tracking_variants(
        scene_id=str(payload["scene_id"]),
        clip_name=str(payload["clip_name"]),
        frame_indices=frames,
        variant_masks=variant_masks,
        variant_scores=variant_scores,
        raw_variant="raw_v0",
        track_ids=track_ids,
        track_prompts=track_prompts,
        ground_truth=gt,
        config=evaluation_run.tracking,
        prompt_label_aliases=evaluation_run.prompt_label_aliases,
    )

    aligned_points = apply_similarity(
        arrays.points,
        scale=float(payload["point_alignment_scale"]),
        rotation=_tensor(payload["point_alignment_rotation"]),
        translation=_tensor(payload["point_alignment_translation"]),
    )
    map_results: list[dict[str, object]] = []
    map_object_rows: list[dict[str, object]] = []
    for variant in variant_masks:
        result = evaluate_semantic_object_map(
            scene_id=str(payload["scene_id"]),
            clip_name=str(payload["clip_name"]),
            variant=variant,
            map_policy="all_visible_observations",
            aligned_world_points=aligned_points,
            target_world_points=_tensor(payload["target_world_points"]),
            confidence=arrays.confidence,
            predicted_masks=variant_masks[variant],
            track_scores=variant_scores[variant],
            gt_masks=gt.masks,
            gt_instance_ids=gt.instance_ids,
            gt_labels=gt.labels,
            assignments=tracking["assignments"],
            config=evaluation_run.map_metrics,
        )
        map_results.append(result["summary"])
        map_object_rows.extend(result["object_rows"])

    tracking_frame_rows = tracking["frame_rows"]
    prompt_summaries = [
        summarize_prompt_events(event_rows, branch=spec.name)
        for spec in REAL_SAM_BRANCHES
    ]
    prompt_summaries.insert(
        0,
        {
            "branch": "raw_v0",
            "query_count": 0,
            "prompt_available_count": 0,
            "sam_call_count": 0,
            "sam_error_count": 0,
            "prompt_applied_count": 0,
            "fallback_count": 0,
            "score_fallback_count": 0,
            "mask_changed_count": 0,
            "prompt_availability": 0.0,
            "sam_success_rate": 1.0,
            "fallback_rate_of_calls": 0.0,
            "fallback_rate_of_available": 0.0,
            "score_fallback_rate": 0.0,
            "mask_changed_rate": 0.0,
            "mean_raw_score": float("nan"),
            "mean_prompted_score": float("nan"),
            "mean_final_score": float("nan"),
        },
    )
    worsening = [
        summarize_worsened_frames(
            tracking_frame_rows,
            branch=variant,
            margin=float(args.worsened_iou_margin),
        )
        for variant in variant_masks
    ]
    summary_rows = _merge_branch_summaries(
        tracking["summary_rows"],
        map_results,
        prompt_summaries,
        worsening,
    )
    _write_csv(output_dir / "tracking_metrics.csv", tracking["summary_rows"])
    _write_csv(output_dir / "tracking_frame_metrics.csv", tracking_frame_rows)
    _write_csv(output_dir / "tracking_object_metrics.csv", tracking["object_rows"])
    _write_csv(output_dir / "map_metrics.csv", map_results)
    _write_csv(output_dir / "map_object_metrics.csv", map_object_rows)
    _write_csv(output_dir / "prompt_branch_metrics.csv", prompt_summaries)
    _write_csv(output_dir / "worsened_frame_metrics.csv", worsening)
    _write_csv(output_dir / "branch_summary.csv", summary_rows)

    summary = _build_summary(
        payload=payload,
        baseline_summary=baseline_summary,
        cache_value=cache_value,
        candidate_dir=candidate_dir,
        output_dir=output_dir,
        arrays=arrays,
        candidate_metadata=candidate_metadata,
        selected_keys=selected_keys,
        event_rows=event_rows,
        summary_rows=summary_rows,
        tracking=tracking,
        map_results=map_results,
        prompt_summaries=prompt_summaries,
        worsening=worsening,
        args=args,
    )
    summary_path = output_dir / "summary.json"
    summary["outputs"] = {
        "summary": str(summary_path),
        "copyable_result": str(output_dir / "copyable_result.txt"),
        "candidate_metadata": str(output_dir / "candidate_metadata.json"),
        "prompt_candidates_causal": str(output_dir / "prompt_candidates_causal.csv"),
        "sam_events_causal": str(output_dir / "sam_events_causal.csv"),
        "prompted_masks_causal": str(mask_artifact),
        "tracking_metrics": str(output_dir / "tracking_metrics.csv"),
        "tracking_frame_metrics": str(output_dir / "tracking_frame_metrics.csv"),
        "tracking_object_metrics": str(output_dir / "tracking_object_metrics.csv"),
        "map_metrics": str(output_dir / "map_metrics.csv"),
        "map_object_metrics": str(output_dir / "map_object_metrics.csv"),
        "prompt_branch_metrics": str(output_dir / "prompt_branch_metrics.csv"),
        "worsened_frame_metrics": str(output_dir / "worsened_frame_metrics.csv"),
        "branch_summary": str(output_dir / "branch_summary.csv"),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=True, default=str)
        + "\n",
        encoding="utf8",
    )
    copyable_path = output_dir / "copyable_result.txt"
    _write_copyable(copyable_path, summary)
    _print_summary(summary, summary_rows, prompt_summaries, worsening)
    print(f"summary={summary_path}")
    print(f"copyable_result={copyable_path}")
    print("decision=DIAGNOSTIC_ONLY")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--candidate-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--sam3-device", default="cuda:0")
    parser.add_argument(
        "--max-query-events",
        type=int,
        default=96,
        help="Common deterministic frame-slot subset; <=0 evaluates all causal queries.",
    )
    parser.add_argument("--max-positive-points", type=int, default=5)
    parser.add_argument("--score-fallback-margin", type=float, default=0.10)
    parser.add_argument("--score-fallback-ratio", type=float, default=0.80)
    parser.add_argument("--worsened-iou-margin", type=float, default=0.05)
    parser.add_argument("--progress-interval", type=int, default=16)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if int(args.max_query_events) == 0:
        # Zero is intentionally accepted as the spelling for "all"; negative
        # values are also accepted for consistency with the helper contract.
        pass
    if int(args.max_positive_points) < 1:
        raise ValueError("--max-positive-points must be positive.")
    if float(args.score_fallback_margin) < 0.0:
        raise ValueError("--score-fallback-margin must be non-negative.")
    if not 0.0 <= float(args.score_fallback_ratio) <= 1.0:
        raise ValueError("--score-fallback-ratio must be in [0,1].")
    if float(args.worsened_iou_margin) < 0.0:
        raise ValueError("--worsened-iou-margin must be non-negative.")
    if int(args.progress_interval) < 1:
        raise ValueError("--progress-interval must be positive.")


def _load_causal_candidates(
    candidate_dir: Path,
    *,
    arrays: V0Arrays,
    frames: Sequence[int],
    track_ids: Sequence[int],
    payload: Mapping[str, object],
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    metadata_path = candidate_dir / "candidate_metadata.json"
    point_path = candidate_dir / "point_candidates_causal.csv"
    query_path = candidate_dir / "query_events_causal.csv"
    for path in (metadata_path, point_path, query_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing frozen A--E candidate artifact {path}; run "
                "commands_run_v0_temporal_prompt_matrix.txt first."
            )
    metadata = json.loads(metadata_path.read_text(encoding="utf8"))
    if int(metadata.get("candidate_generation_gt_fields", -1)) != 0:
        raise ValueError("Candidate metadata indicates GT was used during generation.")
    if str(metadata.get("pose_source", "")) != "raw_streamvggt":
        raise ValueError("Real A/B requires the raw StreamVGGT-pose candidate artifact.")
    point_rows = _read_csv(point_path)
    query_rows = _read_csv(query_path)
    _assert_causal_rows(point_rows, "point candidates")
    _assert_causal_rows(query_rows, "query events")
    sequence, slots = arrays.raw_masks.shape[:2]
    if int(metadata.get("sequence_count", sequence)) != sequence:
        raise ValueError("Candidate sequence count disagrees with the V0 cache.")
    if int(metadata.get("slot_count", slots)) != slots:
        raise ValueError("Candidate slot count disagrees with the V0 cache.")
    expected_frames = tuple(int(value) for value in frames)
    actual_frames = tuple(
        sorted(
            {
                int(row["frame_index"])
                for row in query_rows
                if "frame_index" in row
            }
        )
    )
    if not set(actual_frames).issubset(set(expected_frames)):
        raise ValueError("Candidate rows contain frames outside the V0 cache.")
    for row in point_rows + query_rows:
        frame = int(row["sequence_index"])
        slot = int(row["slot"])
        if not 0 <= frame < sequence or not 0 <= slot < slots:
            raise ValueError("Candidate row points outside V0 sequence/slots.")
        if int(row.get("sam_track_id", track_ids[slot])) != int(track_ids[slot]):
            raise ValueError("Candidate track ID does not match the frozen V0 registry.")
    if str(payload["clip_name"]) != EXPECTED_CLIP:
        raise ValueError("Candidate/cache clip identity mismatch.")
    return point_rows, query_rows, metadata


def _assert_causal_rows(rows: Sequence[Mapping[str, object]], label: str) -> None:
    for row in rows:
        for field in CAUSAL_GT_FIELDS:
            if field not in row:
                continue
            value = str(row[field]).strip().lower()
            if value in {"", "nan", "none"}:
                continue
            try:
                numeric = int(float(value))
            except ValueError as error:
                raise ValueError(f"Invalid {field} in {label}: {value!r}.") from error
            if numeric != -1:
                raise ValueError(
                    f"{label} contains evaluated GT field {field}={numeric}; "
                    "use *_causal.csv artifacts."
                )


def _select_query_keys(
    query_rows: Sequence[Mapping[str, object]],
    *,
    limit: int,
) -> tuple[tuple[int, int], ...]:
    keys = {
        query_key(row)
        for row in query_rows
        if str(row.get("branch", "")) == "A_center"
        and str(row.get("tolerance", "")) == ""
    }
    if not keys:
        raise ValueError("The frozen candidate artifact has no A_center queries.")
    return deterministic_query_subset(tuple(keys), int(limit))


def _prepare_causal_events(
    point_rows: Sequence[Mapping[str, object]],
    query_rows: Sequence[Mapping[str, object]],
    *,
    selected_keys: Sequence[tuple[int, int]],
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> list[dict[str, object]]:
    query_by_key: dict[tuple[int, int], Mapping[str, object]] = {}
    for row in query_rows:
        key = query_key(row)
        query_by_key.setdefault(key, row)
    output: list[dict[str, object]] = []
    for spec in REAL_SAM_BRANCHES:
        for key in selected_keys:
            base = query_by_key.get(key)
            if base is None:
                raise ValueError(f"Missing query metadata for frame-slot {key}.")
            candidates = rows_for_branch(point_rows, key=key, spec=spec)
            mask, coordinates = build_prompt_mask(
                candidates,
                source_size=source_size,
                target_size=target_size,
            )
            row = {
                "branch": spec.name,
                "candidate_branch": spec.candidate_branch,
                "candidate_tolerance": spec.tolerance,
                "sequence_index": int(key[0]),
                "frame_index": int(base["frame_index"]),
                "slot": int(key[1]),
                "sam_track_id": int(base["sam_track_id"]),
                "history_sequence_index": int(base["history_sequence_index"]),
                "history_frame_index": int(base["history_frame_index"]),
                "history_gap": int(base["history_gap"]),
                "candidate_row_count": int(len(candidates)),
                "prompt_pixel_count": int(mask.sum()),
                "prompt_available": int(bool(mask.any())),
                "prompt_coordinates": json.dumps(
                    [[float(u), float(v)] for u, v in coordinates],
                    separators=(",", ":"),
                ),
            }
            output.append(row)
    return output


def _run_real_sam_branches(
    *,
    sam3: SAM3Wrapper,
    image_paths: Sequence[Path],
    arrays: V0Arrays,
    track_prompts: Sequence[str],
    point_rows: Sequence[Mapping[str, object]],
    selected_keys: Sequence[tuple[int, int]],
    causal_events: list[dict[str, object]],
    variant_masks: dict[str, torch.Tensor],
    variant_scores: dict[str, torch.Tensor],
    score_fallback_margin: float,
    score_fallback_ratio: float,
    max_positive_points: int,
    progress_interval: int,
) -> list[dict[str, object]]:
    sequence, slots, height, width = arrays.raw_masks.shape
    event_by_key = {
        (str(row["branch"]), int(row["sequence_index"]), int(row["slot"])): row
        for row in causal_events
    }
    output: list[dict[str, object]] = []
    call_count = 0
    for spec in REAL_SAM_BRANCHES:
        variant_masks[spec.name] = arrays.raw_masks.detach().cpu().bool().clone()
        variant_scores[spec.name] = arrays.scores.detach().cpu().float().clone()
        for frame, slot in selected_keys:
            if not 0 <= int(frame) < sequence or not 0 <= int(slot) < slots:
                raise ValueError("Selected query key is outside the V0 tensor.")
            event = event_by_key[(spec.name, int(frame), int(slot))]
            candidate_rows = rows_for_branch(
                point_rows,
                key=(int(frame), int(slot)),
                spec=spec,
            )
            prompt_mask, coordinates = build_prompt_mask(
                candidate_rows,
                source_size=arrays.image_size,
                target_size=(height, width),
            )
            raw_mask = arrays.raw_masks[frame, slot].detach().cpu().bool()
            raw_score = float(arrays.scores[frame, slot])
            result = dict(event)
            result.update(
                {
                    "sam_called": 0,
                    "sam_error": 0,
                    "sam_error_type": "",
                    "sam_error_message": "",
                    "sam_candidate_count": 0,
                    "sam_obj_id": -1,
                    "raw_score": raw_score,
                    "prompted_score": float("nan"),
                    "raw_mask_pixels": int(raw_mask.sum()),
                    "prompted_mask_pixels": 0,
                    "final_mask_pixels": int(raw_mask.sum()),
                    "final_score": raw_score,
                    "score_fallback": 0,
                    "score_fallback_reason": "",
                    "fallback": 0,
                    "fallback_reason": "",
                    "final_source": "raw",
                    "final_mask_changed": 0,
                    "prompt_coordinates": json.dumps(
                        [[float(u), float(v)] for u, v in coordinates],
                        separators=(",", ":"),
                    ),
                }
            )
            if not bool(prompt_mask.any()):
                result["fallback"] = 1
                result["fallback_reason"] = "no_valid_causal_prompt"
                output.append(result)
                continue

            call_count += 1
            result["sam_called"] = 1
            prompt_text = str(track_prompts[slot]).strip() or "object"
            try:
                with torch.inference_mode():
                    candidates = sam3.propose_geometry_prompted_masks(
                        image_paths[frame],
                        prompt=prompt_text,
                        output_size=(height, width),
                        geometry_prompt=prompt_mask,
                        positive_prompt=prompt_mask,
                        max_positive_points=min(
                            int(max_positive_points), int(spec.max_points)
                        ),
                        use_box=False,
                        use_points=True,
                    )
            except Exception as error:  # noqa: BLE001 - failed call must fallback
                result["sam_error"] = 1
                result["sam_error_type"] = type(error).__name__
                result["sam_error_message"] = _short_error(error)
                result["fallback"] = 1
                result["fallback_reason"] = "sam_call_error"
                output.append(result)
                _print_progress(
                    spec,
                    call_count=call_count,
                    total_calls=len(selected_keys) * len(REAL_SAM_BRANCHES),
                    result=result,
                    interval=progress_interval,
                )
                continue

            result["sam_candidate_count"] = int(len(candidates))
            selected = _select_candidate(candidates)
            if selected is None:
                result["fallback"] = 1
                result["fallback_reason"] = "sam_no_mask_candidate"
                output.append(result)
                _print_progress(
                    spec,
                    call_count=call_count,
                    total_calls=len(selected_keys) * len(REAL_SAM_BRANCHES),
                    result=result,
                    interval=progress_interval,
                )
                continue
            prompted_mask = selected.mask.detach().cpu().bool()
            prompted_score = float(selected.score)
            result["sam_obj_id"] = int(selected.obj_id)
            result["prompted_score"] = prompted_score
            result["prompted_mask_pixels"] = int(prompted_mask.sum())
            if tuple(prompted_mask.shape) != (height, width) or not bool(
                prompted_mask.any()
            ):
                result["fallback"] = 1
                result["fallback_reason"] = (
                    "sam_mask_shape_mismatch" if tuple(prompted_mask.shape) != (height, width)
                    else "sam_empty_mask"
                )
                output.append(result)
                _print_progress(
                    spec,
                    call_count=call_count,
                    total_calls=len(selected_keys) * len(REAL_SAM_BRANCHES),
                    result=result,
                    interval=progress_interval,
                )
                continue

            score_fallback, score_reason = decide_score_fallback(
                raw_score,
                prompted_score,
                absolute_margin=score_fallback_margin,
                relative_ratio=score_fallback_ratio,
            )
            result["score_fallback"] = int(score_fallback)
            result["score_fallback_reason"] = score_reason
            if score_fallback:
                result["fallback"] = 1
                result["fallback_reason"] = score_reason
                output.append(result)
                _print_progress(
                    spec,
                    call_count=call_count,
                    total_calls=len(selected_keys) * len(REAL_SAM_BRANCHES),
                    result=result,
                    interval=progress_interval,
                )
                continue

            final_mask = prompted_mask
            final_score = prompted_score
            variant_masks[spec.name][frame, slot] = final_mask
            variant_scores[spec.name][frame, slot] = final_score
            result["final_mask_pixels"] = int(final_mask.sum())
            result["final_score"] = final_score
            result["final_source"] = "prompt"
            result["final_mask_changed"] = int(not torch.equal(final_mask, raw_mask))
            output.append(result)
            _print_progress(
                spec,
                call_count=call_count,
                total_calls=len(selected_keys) * len(REAL_SAM_BRANCHES),
                result=result,
                interval=progress_interval,
            )
    return output


def _select_candidate(candidates: Sequence[Any]) -> Any | None:
    usable = [candidate for candidate in candidates if candidate is not None]
    if not usable:
        return None
    return max(
        usable,
        key=lambda candidate: (
            float(candidate.score) if math.isfinite(float(candidate.score)) else -math.inf,
            -int(candidate.obj_id),
        ),
    )


def _merge_branch_summaries(
    tracking_rows: Sequence[Mapping[str, object]],
    map_rows: Sequence[Mapping[str, object]],
    prompt_rows: Sequence[Mapping[str, object]],
    worsening_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    tracking = {str(row["variant"]): row for row in tracking_rows}
    maps = {str(row["variant"]): row for row in map_rows}
    prompts = {str(row["branch"]): row for row in prompt_rows}
    worsening = {str(row["branch"]): row for row in worsening_rows}
    variants = tuple(tracking)
    output: list[dict[str, object]] = []
    for variant in variants:
        row: dict[str, object] = {"variant": variant}
        for source in (tracking.get(variant, {}), maps.get(variant, {})):
            row.update(dict(source))
        row.update(
            {
                f"prompt_{key}": value
                for key, value in prompts.get(variant, {}).items()
                if key != "branch"
            }
        )
        row.update(
            {
                f"safety_{key}": value
                for key, value in worsening.get(variant, {}).items()
                if key != "branch"
            }
        )
        output.append(row)
    return output


def _build_summary(
    *,
    payload: Mapping[str, object],
    baseline_summary: Mapping[str, object],
    cache_value: Path,
    candidate_dir: Path,
    output_dir: Path,
    arrays: V0Arrays,
    candidate_metadata: Mapping[str, object],
    selected_keys: Sequence[tuple[int, int]],
    event_rows: Sequence[Mapping[str, object]],
    summary_rows: Sequence[Mapping[str, object]],
    tracking: Mapping[str, object],
    map_results: Sequence[Mapping[str, object]],
    prompt_summaries: Sequence[Mapping[str, object]],
    worsening: Sequence[Mapping[str, object]],
    args: argparse.Namespace,
) -> dict[str, object]:
    return {
        "schema": 1,
        "revision": REVISION,
        "decision": "DIAGNOSTIC_ONLY",
        "diagnostic_only": 1,
        "clip": str(payload["clip_name"]),
        "scene_id": str(payload["scene_id"]),
        "frames": [int(value) for value in payload["frame_indices"]],
        "cache": str(cache_value),
        "candidate_dir": str(candidate_dir),
        "output_dir": str(output_dir),
        "baseline_revision": str(baseline_summary.get("implementation_revision", "")),
        "baseline_selected_pose_branch": str(
            baseline_summary.get("selected_pose_branch", "")
        ),
        "candidate_source": "frozen_v0_temporal_prompt_matrix_causal_csv",
        "candidate_generation_gt_fields": 0,
        "sam_generation_gt_fields": 0,
        "evaluation_gt_fields": 1,
        "models_rerun": 0,
        "streamvggt_rerun": 0,
        "sam_rerun": 1,
        "pose_modified": 0,
        "pointmap_modified": 0,
        "persistent_slot_ids_modified": 0,
        "sam_protocol": "single_frame_points_only_resegmentation",
        "sam_temporal_propagation": 0,
        "sam_score_semantics": (
            "SAM3 out_probs when exposed; wrapper visibility fallback is 1.0 "
            "when out_probs is absent"
        ),
        "parameters": {
            "max_query_events": int(args.max_query_events),
            "selected_query_count": int(len(selected_keys)),
            "max_positive_points": int(args.max_positive_points),
            "score_fallback_margin": float(args.score_fallback_margin),
            "score_fallback_ratio": float(args.score_fallback_ratio),
            "worsened_iou_margin": float(args.worsened_iou_margin),
        },
        "candidate_metadata": dict(candidate_metadata),
        "selected_query_keys": [list(key) for key in selected_keys],
        "sam_event_count": int(len(event_rows)),
        "sam_call_count": int(sum(int(row.get("sam_called", 0)) for row in event_rows)),
        "sam_fallback_count": int(sum(int(row.get("fallback", 0)) for row in event_rows)),
        "score_fallback_count": int(
            sum(int(row.get("score_fallback", 0)) for row in event_rows)
        ),
        "branch_summary": [dict(row) for row in summary_rows],
        "tracking_summary": [dict(row) for row in tracking["summary_rows"]],
        "map_summary": [dict(row) for row in map_results],
        "prompt_summary": [dict(row) for row in prompt_summaries],
        "worsened_frame_summary": [dict(row) for row in worsening],
        "v0_shapes": {
            "world_points": list(arrays.points.shape),
            "raw_masks": list(arrays.raw_masks.shape),
            "image_size": list(arrays.image_size),
            "map_size": list(arrays.map_size),
        },
        "interpretation": {
            "assignment": "raw_v0 global slot-to-GT assignment is frozen for every branch",
            "worsened_frame_ratio": (
                "scene-frame ratio over frames with at least one visible GT object "
                "under the frozen raw assignment; prompted mean IoU is below raw "
                "mean IoU by the declared margin"
            ),
            "worsened_object_frame_ratio": (
                "same threshold measured independently for visible object-frame rows"
            ),
            "fallback": (
                "a missing/invalid SAM result or a prompted score below either fixed "
                "score threshold keeps the raw V0 mask"
            ),
            "scope_caveat": (
                "This is a direct single-frame resegmentation A/B on one V0 scene; "
                "it is not full SAM temporal propagation or a closed-loop retrack."
            ),
        },
        "next_gate": (
            "Do not promote a branch unless its map/tracking gains survive the "
            "worsened-frame safety gate and the experiment is repeated on a new "
            "scene-disjoint validation set."
        ),
        "outputs": {},
    }


def _write_copyable(path: Path, summary: Mapping[str, object]) -> None:
    branch_rows = summary.get("branch_summary", [])
    lines = [
        "===== V0_REAL_SAM_TEMPORAL_PROMPT_AB_BEGIN =====",
        f"revision={summary['revision']}",
        f"decision={summary['decision']}",
        f"clip={summary['clip']}",
        f"scene_id={summary['scene_id']}",
        "candidate_generation_gt_fields=0",
        "sam_generation_gt_fields=0",
        "evaluation_gt_fields=1",
        "streamvggt_rerun=0",
        "sam_rerun=1",
        "sam_protocol=single_frame_points_only_resegmentation",
        f"selected_query_count={summary['parameters']['selected_query_count']}",
        f"sam_call_count={summary['sam_call_count']}",
        f"sam_fallback_count={summary['sam_fallback_count']}",
        f"score_fallback_count={summary['score_fallback_count']}",
        f"score_fallback_margin={summary['parameters']['score_fallback_margin']}",
        f"score_fallback_ratio={summary['parameters']['score_fallback_ratio']}",
        f"worsened_iou_margin={summary['parameters']['worsened_iou_margin']}",
        "",
        "branch_summary_json="
        + json.dumps(branch_rows, sort_keys=True, allow_nan=True, default=str),
        "",
        "scope=single-frame prompted resegmentation; no full temporal propagation",
        "===== V0_REAL_SAM_TEMPORAL_PROMPT_AB_END =====",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf8")


def _print_summary(
    summary: Mapping[str, object],
    branch_rows: Sequence[Mapping[str, object]],
    prompt_rows: Sequence[Mapping[str, object]],
    worsening_rows: Sequence[Mapping[str, object]],
) -> None:
    prompt_by_branch = {str(row["branch"]): row for row in prompt_rows}
    safety_by_branch = {str(row["branch"]): row for row in worsening_rows}
    print("===== V0 REAL SAM A/B SUMMARY =====")
    print(
        "columns=mean_frame_iou frame_IDF1 pixel_IDF1 map_voxelIoU5cm "
        "map_F5cm ghost fallback_rate score_fallback_rate "
        "worsened_frame_ratio worsened_raw_correct_ratio"
    )
    for row in branch_rows:
        variant = str(row.get("variant", ""))
        prompt = prompt_by_branch.get(variant, {})
        safety = safety_by_branch.get(variant, {})
        print(
            f"  variant={variant} "
            f"mean_frame_iou={_display(row.get('mean_frame_iou'))} "
            f"frame_IDF1={_display(row.get('frame_idf1'))} "
            f"pixel_IDF1={_display(row.get('pixel_idf1'))} "
            f"map_voxelIoU5cm={_display(row.get('voxel_iou_5cm'))} "
            f"map_F5cm={_display(row.get('fscore_5cm'))} "
            f"ghost={_display(row.get('ghost_point_ratio'))} "
            f"fallback_rate={_display(prompt.get('fallback_rate_of_calls'))} "
            f"score_fallback_rate={_display(prompt.get('score_fallback_rate'))} "
            f"worsened_frame_ratio={_display(safety.get('worsened_frame_ratio'))} "
            f"worsened_raw_correct_ratio={_display(safety.get('worsened_raw_correct_ratio'))}"
        )
    print(
        "safety_gate=monitor worsened_frame_ratio and worsened_raw_correct_ratio; "
        "15% is a review threshold, not a GT-tuned selection rule"
    )
    print(
        "interpretation=diagnostic_only; raw slot assignment, pose, pointmap, "
        "and unqueried frames are frozen"
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf8")
        return path
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if str(key) not in seen:
                seen.add(str(key))
                fields.append(str(key))
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: _csv_value(row.get(key, ""))
                    for key in fields
                }
            )
    return path


def _read_csv(path: Path) -> list[dict[str, object]]:
    with path.open("r", newline="", encoding="utf8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _validate_image_paths(paths: Sequence[Path], *, expected_count: int) -> None:
    if len(paths) != int(expected_count):
        raise ValueError(
            f"V0 cache has {len(paths)} image paths for {expected_count} frames."
        )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "The frozen cache references missing RGB files; first missing: "
            + missing[0]
        )


def _tensor(value: object) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.detach().cpu().float()
    return torch.as_tensor(value).detach().cpu().float()


def _csv_value(value: object) -> object:
    if torch.is_tensor(value):
        return value.item() if value.ndim == 0 else str(value.tolist())
    return value


def _short_error(error: Exception) -> str:
    message = " ".join(str(error).split())
    return message[:240]


def _print_progress(
    spec: PromptBranchSpec,
    *,
    call_count: int,
    total_calls: int,
    result: Mapping[str, object],
    interval: int,
) -> None:
    if call_count != 1 and call_count % int(interval) != 0 and call_count != total_calls:
        return
    print(
        f"  sam_progress branch={spec.name} calls={call_count}/{total_calls} "
        f"fallback={result.get('fallback', 0)} "
        f"score_fallback={result.get('score_fallback', 0)} "
        f"source={result.get('final_source', 'raw')}"
    )


def _display(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "nan"
    return f"{number:.6f}" if math.isfinite(number) else "nan"


if __name__ == "__main__":
    main()
