#!/usr/bin/env python3
"""Confirm the frozen multi-view gate on an unseen same-scene time holdout."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
import yaml

from streaming_couping.scripts.run_t01_triangulation_reliability import (
    _candidate_metrics,
    _load_and_validate_scores,
    _load_t0_summary,
    _paired_from_rows,
    _resolve,
    _validate_and_select_candidates,
    _write_csv,
    _write_json,
)


REVISION = "t02_same_scene_temporal_holdout_confirmation_r2"
VALIDATION_SCOPE = "same_scene_unseen_temporal_holdout"
BRANCHES = (
    "correct_persistent_id",
    "foreground_union",
    "shuffled_persistent_id",
)
FIXED_MAXIMUM_CONDITION_NUMBER = 100.0
FIXED_MINIMUM_VIEWS = 4


@dataclass(frozen=True)
class T02Run:
    source_path: Path
    discovery_clip: str
    discovery_scene_id: str
    discovery_last_frame: int
    validation_scope: str
    confirmation_clip: str
    confirmation_frames: tuple[int, ...]
    source_t0_output_dir: Path
    output_dir: Path
    gate: "T02Gate"
    minimum_correct_anchor_count: int
    sam_comparison_tolerance_percent: float
    minimum_control_anchor_count: int


@dataclass(frozen=True)
class T02Gate:
    maximum_condition_number_exclusive: float
    minimum_views: int
    require_positive_depth: bool = True


def main() -> None:
    args = _parse_args()
    run = _load_run(args.config)
    artifact_path = run.source_t0_output_dir / "anchors_candidate.pt"
    candidate = torch.load(artifact_path, map_location="cpu", weights_only=False)
    temporal_protocol = validate_temporal_holdout_source(run, candidate)

    anchors_by_branch: dict[str, dict[str, torch.Tensor]] = {}
    metrics_by_branch: dict[str, dict[str, torch.Tensor]] = {}
    masks_by_branch: dict[str, torch.Tensor] = {}
    for branch in BRANCHES:
        anchors = _validate_and_select_candidates(candidate, branch)
        metrics = _candidate_metrics(anchors)
        anchors_by_branch[branch] = anchors
        metrics_by_branch[branch] = metrics
        masks_by_branch[branch] = t02_gate_mask(metrics, run.gate)

    run.output_dir.mkdir(parents=True, exist_ok=True)
    protocol_signature = _protocol_signature(run)
    assignment_signature = _assignment_signature(protocol_signature, artifact_path)
    frozen_path = run.output_dir / "frozen_gate_assignments.pt"
    manifest_path = run.output_dir / "frozen_gate_manifest.json"
    torch.save(
        {
            "schema": 1,
            "revision": REVISION,
            "artifact_role": "temporal_holdout_gt_blind_frozen_gate_assignments",
            "source_candidate_artifact": str(artifact_path),
            "source_candidate_revision": str(candidate["revision"]),
            "validation_scope": run.validation_scope,
            **temporal_protocol,
            "branches": BRANCHES,
            "candidate_generation_raw_pointmap_fields": 0,
            "candidate_generation_gt_fields": 0,
            "positive_depth_source": "inferred_from_t0_acceptance_contract",
            "gate_thresholds_selected_with_holdout_gt": 0,
            "gate": asdict(run.gate),
            "protocol_gate_signature": protocol_signature,
            "assignment_signature": assignment_signature,
            "assignments": {
                branch: {
                    "anchor_index": torch.arange(
                        int(anchors_by_branch[branch]["point_world"].shape[0]),
                        dtype=torch.long,
                    ),
                    "metrics": metrics_by_branch[branch],
                    "gate_mask": masks_by_branch[branch],
                }
                for branch in BRANCHES
            },
        },
        frozen_path,
    )
    manifest = {
        "schema": 1,
        "revision": REVISION,
        "artifact_role": "temporal_holdout_gt_blind_gate_freeze_manifest",
        "discovery_clip": run.discovery_clip,
        "discovery_scene_id": run.discovery_scene_id,
        "discovery_last_frame": run.discovery_last_frame,
        "validation_scope": run.validation_scope,
        **temporal_protocol,
        "source_candidate_artifact": str(artifact_path),
        "source_candidate_revision": str(candidate["revision"]),
        "branches": BRANCHES,
        "candidate_anchor_counts": {
            branch: int(anchors_by_branch[branch]["point_world"].shape[0])
            for branch in BRANCHES
        },
        "candidate_generation_raw_pointmap_fields": 0,
        "candidate_generation_gt_fields": 0,
        "positive_depth_source": "inferred_from_t0_acceptance_contract",
        "gate_thresholds_selected_with_holdout_gt": 0,
        "gate": asdict(run.gate),
        "protocol_gate_signature": protocol_signature,
        "assignment_signature": assignment_signature,
        "frozen_gate_artifact": str(frozen_path),
    }
    _write_json(manifest_path, manifest)
    print("T0.2 SAME-SCENE UNSEEN-TIME MULTI-VIEW CONFIRMATION")
    print("  branches=correct ID, foreground union, shuffled ID")
    print("  fixed gate=positive depth & condition<100 & views>=4")
    print(
        "  gates frozen before temporal-holdout GT-derived scores are opened "
        f"signature={protocol_signature[:12]}"
    )

    # The source summary and score CSV contain evaluation information and are
    # intentionally opened only after all three branch assignments are frozen.
    t0_summary = _load_t0_summary(
        run.source_t0_output_dir / "summary.json", BRANCHES[0]
    )
    _validate_scored_source_metadata(
        t0_summary,
        temporal_protocol=temporal_protocol,
    )
    for branch in BRANCHES:
        if branch not in t0_summary.get("branch_lookup", {}):
            raise ValueError(f"Temporal-holdout T0 summary lacks branch {branch!r}.")
    attempted = {
        int(t0_summary["branch_lookup"][branch]["attempted_query_count"])
        for branch in BRANCHES
    }
    if len(attempted) != 1:
        raise ValueError("T0.2 branches do not share the attempted-query protocol.")

    score_path = run.source_t0_output_dir / "anchor_scores.csv"
    scored_by_branch = {
        branch: _load_and_validate_scores(
            score_path,
            branch=branch,
            anchors=anchors_by_branch[branch],
            metrics=metrics_by_branch[branch],
            gate_masks={"frozen_gate": masks_by_branch[branch]},
        )
        for branch in BRANCHES
    }
    summary, branch_rows, anchor_rows = _score_confirmation(
        run,
        candidate=candidate,
        t0_summary=t0_summary,
        artifact_path=artifact_path,
        frozen_path=frozen_path,
        manifest_path=manifest_path,
        protocol_signature=protocol_signature,
        assignment_signature=assignment_signature,
        temporal_protocol=temporal_protocol,
        scored_by_branch=scored_by_branch,
    )
    _write_outputs(
        run,
        summary=summary,
        branch_rows=branch_rows,
        anchor_rows=anchor_rows,
    )
    correct = summary["branch_lookup"]["correct_persistent_id"]
    print(
        "  correct_id "
        f"anchors={correct['anchor_count']} "
        f"tri_gain={correct['tri_gain_vs_raw_percent']:.4f}% "
        f"P90_gain={correct['tri_p90_gain_vs_raw_percent']:.4f}%"
    )
    print(
        f"  decision={summary['decision']['t02_decision']} "
        f"sam={summary['decision']['sam_geometry_interpretation']}"
    )
    print(f"  result={run.output_dir / 'summary.json'}")


def _score_confirmation(
    run: T02Run,
    *,
    candidate: dict[str, Any],
    t0_summary: dict[str, Any],
    artifact_path: Path,
    frozen_path: Path,
    manifest_path: Path,
    protocol_signature: str,
    assignment_signature: str,
    temporal_protocol: dict[str, Any],
    scored_by_branch: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    branch_rows = []
    anchor_rows = []
    for branch in BRANCHES:
        before = scored_by_branch[branch]
        selected = [row for row in before if int(row["frozen_gate"]) == 1]
        metrics = _paired_from_rows(selected)
        source = t0_summary["branch_lookup"][branch]
        metrics.update(
            {
                "branch": branch,
                "candidate_anchor_count": int(
                    candidate["branches"][branch]["point_world"].shape[0]
                ),
                "evaluated_before_gate_count": len(before),
                "retained_ratio_vs_evaluated": len(selected) / max(len(before), 1),
                "attempted_query_count": int(source["attempted_query_count"]),
                "candidate_valid_anchor_rate": float(
                    source["candidate_valid_anchor_rate"]
                ),
                "tri_p90_gain_vs_raw_percent": _gain(
                    float(metrics["raw_p90"]), float(metrics["tri_p90"])
                ),
                "tri_median_gain_vs_raw_percent": _gain(
                    float(metrics["raw_median"]), float(metrics["tri_median"])
                ),
            }
        )
        branch_rows.append(metrics)
        for row in selected:
            anchor_rows.append({"branch": branch, **row})

    lookup = {row["branch"]: row for row in branch_rows}
    correct_pass = confirmation_gate_pass(
        lookup["correct_persistent_id"],
        minimum_anchor_count=run.minimum_correct_anchor_count,
    )
    sam_interpretation = interpret_sam_geometry(
        lookup,
        tolerance_percent=run.sam_comparison_tolerance_percent,
        minimum_anchor_count=run.minimum_control_anchor_count,
    )
    decision_name = "GO" if correct_pass else "NO_GO"
    decision = {
        "t02_decision": decision_name,
        "correct_id_confirmation_pass": int(correct_pass),
        "sam_geometry_interpretation": sam_interpretation,
        "same_frozen_gate_all_branches": 1,
        "same_scene_unseen_temporal_holdout": 1,
        "same_scene_as_discovery": temporal_protocol["same_scene_as_discovery"],
        "frame_overlap_with_discovery": temporal_protocol["frame_overlap_with_discovery"],
        "all_confirmation_frames_after_discovery": temporal_protocol[
            "all_confirmation_frames_after_discovery"
        ],
        "gate_thresholds_selected_with_holdout_gt": 0,
        "formal_v0_pose_modified": 0,
        "formal_v0_pointmap_modified": 0,
        "formal_v0_semantic_map_modified": 0,
        "next_gate": (
            "conservative_local_pointmap_refinement"
            if correct_pass
            else "stop_training_free_triangulation_pointmap_refinement_keep_v0"
        ),
    }
    summary = {
        "schema": 1,
        "revision": REVISION,
        "experiment": "same_scene_unseen_temporal_holdout_multiview_confirmation",
        "baseline_version": "v0",
        "baseline_status": "frozen_unchanged",
        "validation_scope": run.validation_scope,
        "discovery_clip": run.discovery_clip,
        "discovery_scene_id": run.discovery_scene_id,
        "discovery_last_frame": run.discovery_last_frame,
        "confirmation_clip": t0_summary["clip"],
        "confirmation_scene_id": temporal_protocol["confirmation_scene_id"],
        "frames": t0_summary["frames"],
        "same_scene_unseen_temporal_holdout": 1,
        "same_scene_as_discovery": temporal_protocol["same_scene_as_discovery"],
        "frame_overlap_with_discovery": temporal_protocol["frame_overlap_with_discovery"],
        "all_confirmation_frames_after_discovery": temporal_protocol[
            "all_confirmation_frames_after_discovery"
        ],
        "cross_scene_generalization_claimed": 0,
        "holdout_frame_selection_gt_fields": 0,
        "source_t0_revision": candidate["revision"],
        "source_t0_decision": t0_summary["decision"]["t0_decision"],
        "branches": branch_rows,
        "branch_lookup": lookup,
        "gate": asdict(run.gate),
        "gate_rule": "positive_depth AND condition_number < 100 AND num_views >= 4",
        "condition_boundary_semantics": "strict_less_than_100",
        "go_criteria": {
            "minimum_correct_anchor_count": run.minimum_correct_anchor_count,
            "correct_tri_rmse_less_than_raw_rmse": 1,
            "correct_tri_p90_less_than_raw_p90": 1,
        },
        "sam_comparison": {
            "primary_measure": "tri_gain_vs_paired_raw_percent",
            "comparison_tolerance_percent": run.sam_comparison_tolerance_percent,
            "minimum_control_anchor_count": run.minimum_control_anchor_count,
            "role": "descriptive_branch_evidence_with_branch_specific_anchor_support",
        },
        "candidate_generation_raw_pointmap_fields": 0,
        "candidate_generation_gt_fields": 0,
        "source_temporal_holdout_gt_scores_read_after_gate_freeze": 1,
        "model_loaded_or_run": 0,
        "model_trained": 0,
        "pose_modified": 0,
        "pointmap_modified": 0,
        "protocol_gate_signature": protocol_signature,
        "assignment_signature": assignment_signature,
        "source_candidate_artifact": str(artifact_path),
        "frozen_gate_artifact": str(frozen_path),
        "frozen_gate_manifest": str(manifest_path),
        "decision": decision,
        "claim": (
            "fixed_reliable_multiview_triangulation_confirmed_on_same_scene_unseen_time"
            if correct_pass
            else "fixed_reliable_multiview_triangulation_not_confirmed_on_same_scene_unseen_time"
        ),
    }
    return summary, branch_rows, anchor_rows


def confirmation_gate_pass(
    metrics: dict[str, Any], *, minimum_anchor_count: int
) -> bool:
    """The predeclared T0.2 GO rule for correct persistent identity."""

    return bool(
        int(metrics["anchor_count"]) >= int(minimum_anchor_count)
        and math.isfinite(float(metrics["tri_rmse"]))
        and float(metrics["tri_rmse"]) < float(metrics["raw_rmse"])
        and float(metrics["tri_p90"]) < float(metrics["raw_p90"])
    )


def t02_gate_mask(
    metrics: dict[str, torch.Tensor], gate: T02Gate
) -> torch.Tensor:
    """Apply the frozen T0.2 gate without reprojection or angle filtering."""

    condition = torch.as_tensor(metrics["condition_number"]).float().reshape(-1)
    views = torch.as_tensor(metrics["num_views"]).long().reshape(-1)
    positive = torch.as_tensor(metrics["positive_depth"]).bool().reshape(-1)
    if not (condition.shape == views.shape == positive.shape):
        raise ValueError("T0.2 reliability fields must have equal shape.")
    accepted = (
        torch.isfinite(condition)
        & (condition < gate.maximum_condition_number_exclusive)
        & (views >= gate.minimum_views)
    )
    if gate.require_positive_depth:
        accepted &= positive
    return accepted


def validate_temporal_holdout_source(
    run: T02Run,
    candidate: dict[str, Any],
) -> dict[str, Any]:
    """Validate holdout identity using candidate-only, non-GT metadata."""

    confirmation_clip = str(candidate.get("clip_name", "")).strip()
    confirmation_scene_id = str(candidate.get("scene_id", "")).strip()
    frames = tuple(int(value) for value in candidate.get("frame_indices", ()))
    if not confirmation_clip or not confirmation_scene_id or not frames:
        raise ValueError(
            "T0.2 candidate artifact lacks clip_name, scene_id or frame_indices."
        )
    if confirmation_clip == run.discovery_clip:
        raise ValueError("T0.2 confirmation clip must differ from discovery clip.")
    if confirmation_clip != run.confirmation_clip:
        raise ValueError("T0.2 candidate clip differs from the frozen holdout clip.")
    if confirmation_scene_id != run.discovery_scene_id:
        raise ValueError(
            "T0.2 is a same-scene temporal holdout; confirmation scene differs "
            "from discovery_scene_id."
        )
    if len(frames) != len(set(frames)):
        raise ValueError("T0.2 confirmation frames contain duplicates.")
    if tuple(sorted(frames)) != frames:
        raise ValueError("T0.2 confirmation frames must be time ordered.")
    if frames != run.confirmation_frames:
        raise ValueError("T0.2 candidate frames differ from the frozen holdout frames.")
    if min(frames) <= run.discovery_last_frame:
        raise ValueError(
            "T0.2 requires every confirmation frame to occur after the "
            "discovery interval."
        )
    return {
        "confirmation_clip": confirmation_clip,
        "confirmation_scene_id": confirmation_scene_id,
        "confirmation_frames": frames,
        "same_scene_as_discovery": 1,
        "frame_overlap_with_discovery": 0,
        "all_confirmation_frames_after_discovery": 1,
    }


def _validate_scored_source_metadata(
    summary: dict[str, Any],
    *,
    temporal_protocol: dict[str, Any],
) -> None:
    """Ensure the post-freeze score files describe the frozen candidate."""

    if str(summary.get("clip", "")) != temporal_protocol["confirmation_clip"]:
        raise ValueError("T0 summary clip differs from the frozen candidate artifact.")
    if str(summary.get("scene_id", "")) != temporal_protocol[
        "confirmation_scene_id"
    ]:
        raise ValueError("T0 summary scene differs from the frozen candidate artifact.")
    frames = tuple(int(value) for value in summary.get("frames", ()))
    if frames != tuple(temporal_protocol["confirmation_frames"]):
        raise ValueError("T0 summary frames differ from the frozen candidate artifact.")


def interpret_sam_geometry(
    lookup: dict[str, dict[str, Any]],
    *,
    tolerance_percent: float,
    minimum_anchor_count: int,
) -> str:
    """Give the requested descriptive SAM branch interpretation."""

    if any(
        int(lookup[branch]["anchor_count"]) < int(minimum_anchor_count)
        for branch in BRANCHES
    ):
        return "insufficient_control_anchor_support"
    correct = float(lookup["correct_persistent_id"]["tri_gain_vs_raw_percent"])
    foreground = float(lookup["foreground_union"]["tri_gain_vs_raw_percent"])
    shuffled = float(lookup["shuffled_persistent_id"]["tri_gain_vs_raw_percent"])
    if not all(math.isfinite(value) for value in (correct, foreground, shuffled)):
        return "insufficient_finite_control_metrics"
    tolerance = float(tolerance_percent)
    if correct > foreground + tolerance and foreground > shuffled + tolerance:
        return "persistent_identity_geometry_evidence"
    if abs(correct - foreground) <= tolerance and min(correct, foreground) > (
        shuffled + tolerance
    ):
        return "foreground_gating_geometry_evidence"
    if abs(correct - shuffled) <= tolerance:
        return "no_persistent_identity_geometry_evidence"
    return "inconclusive_branch_ordering"


def _gain(reference: float, candidate: float) -> float:
    if not math.isfinite(reference) or not math.isfinite(candidate):
        return float("nan")
    return 100.0 * (reference - candidate) / max(reference, 1e-12)


def _write_outputs(
    run: T02Run,
    *,
    summary: dict[str, Any],
    branch_rows: list[dict[str, Any]],
    anchor_rows: list[dict[str, Any]],
) -> None:
    _write_json(run.output_dir / "summary.json", summary)
    _write_csv(run.output_dir / "branch_summary.csv", branch_rows)
    _write_csv(run.output_dir / "accepted_anchor_scores.csv", anchor_rows)
    _write_copyable(run.output_dir / "copyable_result.txt", summary)


def _write_copyable(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "===== COPYABLE_T02_MULTIVIEW_CONFIRMATION_BEGIN =====",
        f"revision={summary['revision']}",
        f"validation_scope={summary['validation_scope']}",
        f"discovery_clip={summary['discovery_clip']}",
        f"discovery_scene_id={summary['discovery_scene_id']}",
        f"discovery_last_frame={summary['discovery_last_frame']}",
        f"confirmation_clip={summary['confirmation_clip']}",
        f"confirmation_scene_id={summary['confirmation_scene_id']}",
        f"frames={len(summary['frames'])}",
        "branches=" + ",".join(row["branch"] for row in summary["branches"]),
        f"gate={summary['gate_rule']}",
        f"protocol_gate_signature={summary['protocol_gate_signature']}",
        "same_scene_as_discovery=1",
        "frame_overlap_with_discovery=0",
        "all_confirmation_frames_after_discovery=1",
        "cross_scene_generalization_claimed=0",
        "holdout_frame_selection_gt_fields=0",
        "gate_thresholds_selected_with_holdout_gt=0",
        "candidate_generation_raw_pointmap_fields=0",
        "candidate_generation_gt_fields=0",
        "model_loaded_or_run=0",
        "model_trained=0",
        "formal_v0_modified=0",
        "",
        "branch,candidate_anchor_count,evaluated_before_gate_count,anchor_count,retained_ratio_vs_evaluated,tri_rmse,raw_rmse,tri_gain_vs_raw_percent,tri_p90,raw_p90,tri_p90_gain_vs_raw_percent,tri_median,raw_median,tri_median_gain_vs_raw_percent,improved_anchor_ratio,candidate_valid_anchor_rate",
    ]
    fields = (
        "branch",
        "candidate_anchor_count",
        "evaluated_before_gate_count",
        "anchor_count",
        "retained_ratio_vs_evaluated",
        "tri_rmse",
        "raw_rmse",
        "tri_gain_vs_raw_percent",
        "tri_p90",
        "raw_p90",
        "tri_p90_gain_vs_raw_percent",
        "tri_median",
        "raw_median",
        "tri_median_gain_vs_raw_percent",
        "improved_anchor_ratio",
        "candidate_valid_anchor_rate",
    )
    for row in summary["branches"]:
        lines.append(",".join(str(row[field]) for field in fields))
    lines.extend(
        (
            "",
            "decision=" + json.dumps(summary["decision"], sort_keys=True),
            f"claim={summary['claim']}",
            "outputs:",
            f"summary={path.with_name('summary.json')}",
            f"branch_csv={path.with_name('branch_summary.csv')}",
            f"anchor_csv={path.with_name('accepted_anchor_scores.csv')}",
            f"frozen_gate_artifact={summary['frozen_gate_artifact']}",
            "===== COPYABLE_T02_MULTIVIEW_CONFIRMATION_END =====",
        )
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf8")


def _load_run(path: str | Path) -> T02Run:
    source = Path(path).expanduser().resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf8")) or {}
    gate = raw.get("gate", {})
    evaluation = raw.get("evaluation", {})
    fixed_gate = T02Gate(
        maximum_condition_number_exclusive=float(
            gate.get("maximum_condition_number", FIXED_MAXIMUM_CONDITION_NUMBER)
        ),
        minimum_views=int(gate.get("minimum_views", FIXED_MINIMUM_VIEWS)),
    )
    run = T02Run(
        source_path=source,
        discovery_clip=str(
            raw.get("discovery_clip", "00a231a370_90_525_step15_37_68_54")
        ),
        discovery_scene_id=str(raw.get("discovery_scene_id", "00a231a370")),
        discovery_last_frame=int(raw.get("discovery_last_frame", 525)),
        validation_scope=str(raw.get("validation_scope", VALIDATION_SCOPE)),
        confirmation_clip=str(
            raw.get(
                "confirmation_clip",
                "00a231a370_533_591_step2_t02_holdout",
            )
        ),
        confirmation_frames=tuple(
            int(value)
            for value in raw.get(
                "confirmation_frame_indices",
                tuple(range(533, 592, 2)),
            )
        ),
        source_t0_output_dir=_resolve(
            raw.get(
                "source_t0_output_dir",
                "outputs/streaming_couping_t02_temporal_holdout_anchor_probe",
            )
        ),
        output_dir=_resolve(
            raw.get(
                "output_dir",
                "outputs/streaming_couping_t02_multiview_confirmation",
            )
        ),
        gate=fixed_gate,
        minimum_correct_anchor_count=int(
            evaluation.get("minimum_correct_anchor_count", 30)
        ),
        sam_comparison_tolerance_percent=float(
            evaluation.get("sam_comparison_tolerance_percent", 2.0)
        ),
        minimum_control_anchor_count=int(
            evaluation.get("minimum_control_anchor_count", 10)
        ),
    )
    _validate_run(run)
    return run


def _validate_run(run: T02Run) -> None:
    if not run.discovery_clip.strip():
        raise ValueError("T0.2 discovery clip cannot be empty.")
    if not run.discovery_scene_id.strip():
        raise ValueError("T0.2 discovery scene cannot be empty.")
    if run.discovery_last_frame < 0:
        raise ValueError("T0.2 discovery_last_frame cannot be negative.")
    if run.validation_scope != VALIDATION_SCOPE:
        raise ValueError(
            f"T0.2 validation_scope is frozen at {VALIDATION_SCOPE!r}."
        )
    if not run.confirmation_clip.strip():
        raise ValueError("T0.2 confirmation clip cannot be empty.")
    if run.confirmation_clip == run.discovery_clip:
        raise ValueError("T0.2 confirmation and discovery clips must differ.")
    if len(run.confirmation_frames) != 30:
        raise ValueError("T0.2 confirmation holdout is frozen at 30 frames.")
    if tuple(sorted(set(run.confirmation_frames))) != run.confirmation_frames:
        raise ValueError("T0.2 confirmation frames must be unique and ordered.")
    if min(run.confirmation_frames) <= run.discovery_last_frame:
        raise ValueError("T0.2 confirmation frames must all follow discovery.")
    if run.output_dir == run.source_t0_output_dir:
        raise ValueError("T0.2 cannot overwrite its temporal-holdout T0 source.")
    # Hard rejection prevents changing the post-T0.1 confirmation hypothesis.
    if (
        run.gate.maximum_condition_number_exclusive
        != FIXED_MAXIMUM_CONDITION_NUMBER
    ):
        raise ValueError("T0.2 condition threshold is frozen at 100.")
    if run.gate.minimum_views != FIXED_MINIMUM_VIEWS:
        raise ValueError("T0.2 view threshold is frozen at four views.")
    if run.minimum_correct_anchor_count != 30:
        raise ValueError("T0.2 GO support threshold is frozen at 30 anchors.")
    if run.sam_comparison_tolerance_percent < 0.0:
        raise ValueError("SAM comparison tolerance cannot be negative.")
    if run.minimum_control_anchor_count < 1:
        raise ValueError("Control anchor support must be positive.")


def _protocol_signature(run: T02Run) -> str:
    payload = {
        "revision": REVISION,
        "validation_scope": run.validation_scope,
        "discovery_clip": run.discovery_clip,
        "discovery_scene_id": run.discovery_scene_id,
        "discovery_last_frame": run.discovery_last_frame,
        "confirmation_clip": run.confirmation_clip,
        "confirmation_frames": run.confirmation_frames,
        "branches": BRANCHES,
        "gate_rule": "positive_depth AND condition_number < 100 AND num_views >= 4",
        "condition_threshold": FIXED_MAXIMUM_CONDITION_NUMBER,
        "condition_comparison": "strict_less_than",
        "minimum_views": FIXED_MINIMUM_VIEWS,
        "minimum_correct_anchor_count": run.minimum_correct_anchor_count,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _assignment_signature(protocol_signature: str, artifact_path: Path) -> str:
    source_hash = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    return hashlib.sha256(
        f"{protocol_signature}:{source_hash}".encode()
    ).hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="streaming_couping/configs/t02_multiview_confirmation.yaml",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
