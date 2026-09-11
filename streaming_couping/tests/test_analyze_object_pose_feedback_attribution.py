"""Tests for the stage-2b attribution post-mortem (CPU only, synthetic CSVs).

The column names used here mirror the writers in ``run_object_pose_feedback``;
``REQUIRED_COLUMNS`` in the analysis module is what turns a renamed column into
a loud error instead of an empty table, and
``test_cli_rejects_missing_columns`` covers that guard.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest

from streaming_couping.scripts.analyze_object_pose_feedback_attribution import (
    MAIN_VARIANT,
    category_stratified_correlations,
    consensus_collapse,
    feature_predictiveness,
    main,
    per_frame_effect,
    selection_quality,
)

CONSENSUS_VARIANTS = (
    "single",
    "mean",
    "robust",
    "robust_semantic",
    MAIN_VARIANT,
)


def _write(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _consensus_rows(
    *,
    variant: str,
    frames: int = 40,
    num_reliable: int = 2,
    inlier_count: int | None = None,
    accepted_every: int = 4,
    rejected_gt_recorded: bool = True,
) -> list[dict[str, Any]]:
    inliers = num_reliable if inlier_count is None else inlier_count
    rows: list[dict[str, Any]] = []
    for index in range(frames):
        accepted = index % accepted_every == 0
        recorded = accepted or rejected_gt_recorded
        rows.append(
            {
                "variant": variant,
                "frame": index,
                "num_proposals": num_reliable,
                "num_semantic_reliable": num_reliable,
                "num_reliable": num_reliable,
                "inlier_count": inliers,
                "consensus_translation_norm": 0.05,
                "consensus_rotation_deg": 1.0,
                "aggregate_loss_before": 0.08,
                "aggregate_loss_after": 0.06 if accepted else 0.09,
                "accepted": "True" if accepted else "False",
                "reason": "" if accepted else "no_alignment_improvement",
                "gt_consensus_translation_error": (
                    (0.02 if accepted else 0.09) if recorded else ""
                ),
                "gt_consensus_rotation_error": 0.5 if recorded else "",
            }
        )
    return rows


def _proposal_rows(
    *,
    count: int = 60,
    geometry_confidence_predictive: bool = True,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(count):
        geometry = index / max(1, count - 1)
        semantic = ((index * 7) % count) / max(1, count - 1)
        if geometry_confidence_predictive:
            gt_error = 0.20 * geometry
        else:
            gt_error = 0.10
        rows.append(
            {
                "frame": index,
                "instance_id": index % 3,
                "category": ["bed", "rug", "chair"][index % 3],
                "geometry_type": (
                    "volumetric" if geometry > 0.6 else "planar"
                ),
                "track_length": 20,
                "visibility_ratio": 1.0,
                "point_count": 200,
                "overlap_count": 100,
                "alignment_loss_before": 0.08,
                "alignment_loss_after": 0.03,
                "inlier_ratio": 0.5,
                "eigenvalue_1": 1.0,
                "eigenvalue_2": 0.5,
                "eigenvalue_3": 0.2,
                "degeneracy_factor": 0.2,
                "delta_translation_norm": 0.05,
                "delta_rotation_deg": 1.0,
                "semantic_confidence": semantic,
                "geometry_confidence": geometry,
                "accepted_by_refiner": "True",
                "refiner_reason": "object_loss_accepted",
                "object_reject_reason": (
                    "" if gt_error < 0.10 else "low_geometry_confidence"
                ),
                "consensus_inlier": "True" if index % 2 == 0 else "False",
                "translation_consensus_error": 0.01,
                "rotation_consensus_error": 0.1,
                "gt_translation_correction_error": gt_error,
                "gt_rotation_correction_error": 0.5,
                "gt_translation_saturated": "False",
                "gt_rotation_saturated": "False",
                "reference_frames": "1;2",
            }
        )
    return rows


def _frame_rows(*, variant: str, frames: int = 40) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(frames):
        accepted = index % 4 == 0
        rows.append(
            {
                "variant": variant,
                "frame": index,
                "raw_translation_error": 0.116,
                "feedback_translation_error": 0.100 if accepted else 0.116,
                "raw_rotation_error": 1.7,
                "feedback_rotation_error": 1.6 if accepted else 1.7,
                "num_object_proposals": 2,
                "num_reliable_objects": 2,
                "accepted": "True" if accepted else "False",
                "reject_reason": "" if accepted else "no_alignment_improvement",
                "consensus_translation_norm": 0.05,
                "consensus_rotation_deg": 1.0,
            }
        )
    return rows


# ---------------------------------------------------------------------------


def test_consensus_collapse_flags_single_inlier_regime() -> None:
    rows: list[dict[str, Any]] = []
    for variant in CONSENSUS_VARIANTS:
        rows += _consensus_rows(
            variant=variant,
            inlier_count=1 if variant == MAIN_VARIANT else 2,
        )
    text, payload = consensus_collapse(rows)
    assert "consensus collapse" in text
    baseline = payload["variants"]["robust_semantic"]
    main = payload["variants"][MAIN_VARIANT]
    assert baseline["inlier_count_mean"] == pytest.approx(2.0)
    assert main["inlier_count_mean"] == pytest.approx(1.0)
    assert main["share_inlier_count_at_most_one"] == pytest.approx(1.0)
    assert payload["verdict"].startswith("supported")


def test_consensus_collapse_rejects_healthy_multi_object_regime() -> None:
    rows: list[dict[str, Any]] = []
    for variant in CONSENSUS_VARIANTS:
        rows += _consensus_rows(variant=variant, inlier_count=2)
    _text, payload = consensus_collapse(rows)
    assert payload["verdict"].startswith("not supported")


def test_consensus_collapse_counts_only_frames_with_a_consensus() -> None:
    rows = _consensus_rows(variant=MAIN_VARIANT, frames=10)
    rows += [
        {
            "variant": MAIN_VARIANT,
            "frame": 100 + index,
            "num_proposals": 1,
            "num_semantic_reliable": 1,
            "num_reliable": 1,
            "inlier_count": 0,
            "consensus_translation_norm": "",  # never formed a consensus
            "consensus_rotation_deg": "",
            "aggregate_loss_before": "",
            "aggregate_loss_after": "",
            "accepted": "False",
            "reason": "insufficient_objects",
            "gt_consensus_translation_error": "",
            "gt_consensus_rotation_error": "",
        }
        for index in range(5)
    ]
    _text, payload = consensus_collapse(rows)
    assert payload["variants"][MAIN_VARIANT]["frames_with_consensus"] == 10


def test_feature_predictiveness_separates_signal_from_noise() -> None:
    _text, payload = feature_predictiveness(_proposal_rows())
    predictive = payload["geometry_confidence"]
    assert predictive["spearman_rho"] > 0.9
    assert predictive["permutation_p"] < 0.01
    noisy = payload["semantic_confidence"]
    assert abs(noisy["spearman_rho"]) < 0.5
    assert noisy["permutation_p"] > 0.05


def test_feature_predictiveness_skips_constant_columns() -> None:
    rows = _proposal_rows()
    for row in rows:
        row["alignment_loss_after"] = 0.03  # constant, must not yield nan rho
    _text, payload = feature_predictiveness(rows)
    assert "alignment_loss_after" not in payload


def _confounded_rows(*, per_category: int = 40) -> list[dict[str, Any]]:
    """``track_length`` separates the categories; inside a category its order is
    unrelated to the error's order.  Any marginal correlation is a category
    artifact, and the stratified one is ~0."""

    rows: list[dict[str, Any]] = []
    for label, track_base, error_base in (
        ("bed", 20.0, 0.05),
        ("rug", 200.0, 0.15),
    ):
        for index in range(per_category):
            rows.append(
                {
                    "frame": len(rows),
                    "instance_id": 0,
                    "category": label,
                    "geometry_type": "planar",
                    # feature order is `index`; target order is a different
                    # permutation, so the two are unrelated within a category
                    "track_length": track_base + index * 0.001,
                    "gt_translation_correction_error": (
                        error_base
                        + ((index * 17) % per_category) * 0.0005
                    ),
                    "visibility_ratio": 1.0,
                    "point_count": 200,
                    "overlap_count": 100,
                    "alignment_loss_before": 0.08,
                    "alignment_loss_after": 0.03,
                    "inlier_ratio": 0.5,
                    "eigenvalue_1": 1.0,
                    "eigenvalue_2": 0.5,
                    "eigenvalue_3": 0.2,
                    "degeneracy_factor": 0.2,
                    "delta_translation_norm": 0.05,
                    "delta_rotation_deg": 1.0,
                    "semantic_confidence": 0.5,
                    "geometry_confidence": 0.5,
                    "accepted_by_refiner": "True",
                    "refiner_reason": "object_loss_accepted",
                    "object_reject_reason": "",
                    "consensus_inlier": "True",
                    "translation_consensus_error": 0.01,
                    "rotation_consensus_error": 0.1,
                    "gt_rotation_correction_error": 0.5,
                    "gt_translation_saturated": "False",
                    "gt_rotation_saturated": "False",
                    "reference_frames": "1;2",
                }
            )
    return rows


def test_feature_predictiveness_reports_within_category_rho() -> None:
    """A category proxy keeps its marginal rho but loses its within-category one."""

    _text, payload = feature_predictiveness(_confounded_rows())
    track = payload["track_length"]
    assert track["spearman_rho"] > 0.6  # marginal: picks up the category split
    within = track["spearman_rho_within_category"]
    assert math.isfinite(within)
    assert abs(within) < 0.35  # vanishes once the category is held fixed


def test_category_stratified_correlations_break_out_each_category() -> None:
    _text, payload = category_stratified_correlations(_confounded_rows())
    assert set(payload) == {"bed", "rug"}
    assert payload["bed"]["n"] == 40
    assert abs(payload["bed"]["track_length"]["rho"]) < 0.35


def test_selection_quality_reports_proposal_and_frame_levels() -> None:
    proposals = _proposal_rows()
    consensus = _consensus_rows(variant=MAIN_VARIANT)
    _text, payload = selection_quality(proposals, consensus)
    proposal_level = payload["proposal_level"]
    reliable = next(row for row in proposal_level if row[0].startswith("reliable"))
    assert reliable[2] < reliable[4]  # kept proposals are closer to GT
    frame_level = payload["frame_level"]
    assert frame_level["rejected_gt_recorded"] is True
    assert (
        frame_level["accepted_median_gt_consensus_error_m"]
        < frame_level["rejected_median_gt_consensus_error_m"]
    )


def test_selection_quality_flags_unrecorded_rejected_gt() -> None:
    consensus = _consensus_rows(variant=MAIN_VARIANT, rejected_gt_recorded=False)
    text, payload = selection_quality(_proposal_rows(), consensus)
    assert payload["frame_level"]["rejected_gt_recorded"] is False
    assert "Re-run stage 2b" in text


def test_per_frame_effect_counts_local_improvements() -> None:
    rows = _frame_rows(variant=MAIN_VARIANT)
    _text, payload = per_frame_effect(rows)
    block = payload[MAIN_VARIANT]
    assert block["accepted_frames"] == 10
    assert block["local_improvement_ratio"] == pytest.approx(1.0)
    assert block["median_local_gain_m"] == pytest.approx(0.016, abs=1e-9)


# ---------------------------------------------------------------------------


def test_cli_writes_json_and_rejects_missing_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    consensus = _consensus_rows(variant=MAIN_VARIANT)
    proposals = _proposal_rows()
    frames = _frame_rows(variant=MAIN_VARIANT)
    _write(tmp_path / "object_proposals.csv", proposals)
    _write(tmp_path / "consensus_metrics.csv", consensus)
    _write(tmp_path / "frame_metrics.csv", frames)

    monkeypatch.setattr(
        "sys.argv",
        ["analyze", "--feedback-dir", str(tmp_path)],
    )
    main()
    out = capsys.readouterr().out
    assert "consensus collapse" in out
    assert (tmp_path / "attribution.json").is_file()

    # Dropping a column the analysis reads must fail loudly, not report empty.
    for row in proposals:
        row.pop("geometry_type")
    _write(tmp_path / "object_proposals.csv", proposals)
    monkeypatch.setattr(
        "sys.argv",
        ["analyze", "--feedback-dir", str(tmp_path)],
    )
    with pytest.raises(ValueError, match="geometry_type"):
        main()
