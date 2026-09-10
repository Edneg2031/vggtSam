"""Training-free SAM-object-consensus camera-pose feedback for HorizonStream.

This module consumes the per-instance 6DoF proposals exported by
``ObjectPoseLossRefiner`` (``feedback_diagnostics.pt``) and turns them into one
camera-pose correction hypothesis per frame:

    proposal ΔT_k (c2w left-multiply, frame-0 gauge)
        → semantic reliability S_sem(k)      (SAM track identity quality)
        → geometric reliability S_geo(k)     (alignment quality + observability)
        → robust cross-object consensus over ξ_k = log(ΔT_k)
        → explicit gating (accept / reject with reason)

Design boundaries:

* Proposals are computed once on the raw HorizonStream geometry (two-pass).
  The consensus target is an *absolute* camera pose anchored to the raw-world
  reference clouds, so repeated injections overwrite instead of accumulating.
* Ground truth never enters reliability, consensus, or gating.  The GT
  comparison helpers in this module exist for the offline evaluation stage
  only.
* The pose-accumulator replay itself lives in the stage-2b script because it
  reuses the verified GT-feedback POC helpers (scripts layer).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import torch

from .object_pose_loss_refinement import (
    _huber_distance,
    _mutual_matches,
    _so3_exp,
)


@dataclass(frozen=True)
class ObjectPoseFeedbackConfig:
    """All thresholds for the object-consensus pose feedback baseline."""

    # Frames before this sequence index are refiner anchors and never propose.
    anchor_frame_count: int = 5

    # --- semantic reliability ---
    min_track_length: int = 5
    target_track_length: int = 20
    visibility_window: int = 10
    min_visibility_ratio: float = 0.50
    target_mask_pixels: int = 2000
    min_sam_track_score: float = 0.50
    track_length_weight: float = 0.35
    visibility_weight: float = 0.35
    mask_pixels_weight: float = 0.15
    track_score_weight: float = 0.15

    # --- geometric reliability ---
    min_relative_improvement: float = 0.02
    target_relative_improvement: float = 0.50
    min_inlier_ratio: float = 0.10
    target_inlier_ratio: float = 0.50
    min_overlap_points: int = 16
    target_overlap_points: int = 64
    volumetric_lambda_ratio: float = 0.10
    planar_lambda_ratio: float = 0.10
    degenerate_min_points: int = 8
    improvement_weight: float = 0.40
    inlier_ratio_weight: float = 0.30
    overlap_weight: float = 0.30

    # --- proposal magnitude sanity ---
    max_feedback_translation_m: float = 0.25
    max_feedback_rotation_deg: float = 10.0

    # --- consensus ---
    consensus_irls_iterations: int = 5
    consensus_huber_translation_m: float = 0.05
    consensus_huber_rotation_deg: float = 3.0
    consensus_tolerance_translation_m: float = 0.10
    consensus_tolerance_rotation_deg: float = 6.0
    min_consensus_objects: int = 2
    inlier_weight_threshold: float = 0.50

    # --- gating ---
    max_consensus_translation_m: float = 0.20
    max_consensus_rotation_deg: float = 8.0
    min_aggregate_improvement: float = 0.01

    # --- aggregate alignment-loss re-evaluation (mirrors refiner matching) ---
    aggregate_max_match_distance_m: float = 0.25
    aggregate_trim_ratio: float = 0.70
    aggregate_min_matches_per_pair: int = 8
    aggregate_huber_delta_m: float = 0.05

    # --- GO / NO-GO decision criteria (pose metrics only, never loss) ---
    decision_min_ate_improvement_ratio: float = 0.05
    decision_min_future_gain_positive_ratio: float = 0.60
    decision_max_rpe_degradation_ratio: float = 1.05
    decision_min_accepted_ratio: float = 0.10
    future_gain_window: int = 10

    def validate(self) -> "ObjectPoseFeedbackConfig":
        for name, value in (
            ("anchor_frame_count", self.anchor_frame_count),
            ("min_track_length", self.min_track_length),
            ("visibility_window", self.visibility_window),
            ("min_overlap_points", self.min_overlap_points),
            ("target_overlap_points", self.target_overlap_points),
            ("degenerate_min_points", self.degenerate_min_points),
            ("consensus_irls_iterations", self.consensus_irls_iterations),
            ("min_consensus_objects", self.min_consensus_objects),
            ("future_gain_window", self.future_gain_window),
        ):
            if int(value) < 1:
                raise ValueError(f"object_pose_feedback.{name} must be positive.")
        if int(self.target_track_length) < int(self.min_track_length):
            raise ValueError(
                "object_pose_feedback.target_track_length must be >= "
                "min_track_length."
            )
        if int(self.target_mask_pixels) < 1:
            raise ValueError("object_pose_feedback.target_mask_pixels must be positive.")
        for name, value in (
            ("min_visibility_ratio", self.min_visibility_ratio),
            ("min_sam_track_score", self.min_sam_track_score),
            ("track_length_weight", self.track_length_weight),
            ("visibility_weight", self.visibility_weight),
            ("mask_pixels_weight", self.mask_pixels_weight),
            ("track_score_weight", self.track_score_weight),
            ("min_relative_improvement", self.min_relative_improvement),
            ("target_relative_improvement", self.target_relative_improvement),
            ("min_inlier_ratio", self.min_inlier_ratio),
            ("target_inlier_ratio", self.target_inlier_ratio),
            ("volumetric_lambda_ratio", self.volumetric_lambda_ratio),
            ("planar_lambda_ratio", self.planar_lambda_ratio),
            ("improvement_weight", self.improvement_weight),
            ("inlier_ratio_weight", self.inlier_ratio_weight),
            ("overlap_weight", self.overlap_weight),
            ("consensus_tolerance_translation_m", self.consensus_tolerance_translation_m),
            ("inlier_weight_threshold", self.inlier_weight_threshold),
            ("min_aggregate_improvement", self.min_aggregate_improvement),
            ("aggregate_trim_ratio", self.aggregate_trim_ratio),
            ("decision_min_ate_improvement_ratio", self.decision_min_ate_improvement_ratio),
            ("decision_min_future_gain_positive_ratio", self.decision_min_future_gain_positive_ratio),
            ("decision_min_accepted_ratio", self.decision_min_accepted_ratio),
        ):
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"object_pose_feedback.{name} must be in [0,1].")
        for name, value in (
            ("max_feedback_translation_m", self.max_feedback_translation_m),
            ("max_feedback_rotation_deg", self.max_feedback_rotation_deg),
            ("consensus_huber_translation_m", self.consensus_huber_translation_m),
            ("consensus_huber_rotation_deg", self.consensus_huber_rotation_deg),
            ("consensus_tolerance_rotation_deg", self.consensus_tolerance_rotation_deg),
            ("max_consensus_translation_m", self.max_consensus_translation_m),
            ("max_consensus_rotation_deg", self.max_consensus_rotation_deg),
            ("aggregate_max_match_distance_m", self.aggregate_max_match_distance_m),
            ("aggregate_huber_delta_m", self.aggregate_huber_delta_m),
            ("decision_max_rpe_degradation_ratio", self.decision_max_rpe_degradation_ratio),
        ):
            if float(value) <= 0.0:
                raise ValueError(f"object_pose_feedback.{name} must be positive.")
        if int(self.aggregate_min_matches_per_pair) < 1:
            raise ValueError(
                "object_pose_feedback.aggregate_min_matches_per_pair must be positive."
            )
        if float(self.target_inlier_ratio) < float(self.min_inlier_ratio):
            raise ValueError(
                "object_pose_feedback.target_inlier_ratio must be >= min_inlier_ratio."
            )
        if float(self.target_relative_improvement) < float(self.min_relative_improvement):
            raise ValueError(
                "object_pose_feedback.target_relative_improvement must be >= "
                "min_relative_improvement."
            )
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "anchor_frame_count": int(self.anchor_frame_count),
            "min_track_length": int(self.min_track_length),
            "target_track_length": int(self.target_track_length),
            "visibility_window": int(self.visibility_window),
            "min_visibility_ratio": float(self.min_visibility_ratio),
            "target_mask_pixels": int(self.target_mask_pixels),
            "min_sam_track_score": float(self.min_sam_track_score),
            "track_length_weight": float(self.track_length_weight),
            "visibility_weight": float(self.visibility_weight),
            "mask_pixels_weight": float(self.mask_pixels_weight),
            "track_score_weight": float(self.track_score_weight),
            "min_relative_improvement": float(self.min_relative_improvement),
            "target_relative_improvement": float(self.target_relative_improvement),
            "min_inlier_ratio": float(self.min_inlier_ratio),
            "target_inlier_ratio": float(self.target_inlier_ratio),
            "min_overlap_points": int(self.min_overlap_points),
            "target_overlap_points": int(self.target_overlap_points),
            "volumetric_lambda_ratio": float(self.volumetric_lambda_ratio),
            "planar_lambda_ratio": float(self.planar_lambda_ratio),
            "degenerate_min_points": int(self.degenerate_min_points),
            "improvement_weight": float(self.improvement_weight),
            "inlier_ratio_weight": float(self.inlier_ratio_weight),
            "overlap_weight": float(self.overlap_weight),
            "max_feedback_translation_m": float(self.max_feedback_translation_m),
            "max_feedback_rotation_deg": float(self.max_feedback_rotation_deg),
            "consensus_irls_iterations": int(self.consensus_irls_iterations),
            "consensus_huber_translation_m": float(self.consensus_huber_translation_m),
            "consensus_huber_rotation_deg": float(self.consensus_huber_rotation_deg),
            "consensus_tolerance_translation_m": float(
                self.consensus_tolerance_translation_m
            ),
            "consensus_tolerance_rotation_deg": float(
                self.consensus_tolerance_rotation_deg
            ),
            "min_consensus_objects": int(self.min_consensus_objects),
            "inlier_weight_threshold": float(self.inlier_weight_threshold),
            "max_consensus_translation_m": float(self.max_consensus_translation_m),
            "max_consensus_rotation_deg": float(self.max_consensus_rotation_deg),
            "min_aggregate_improvement": float(self.min_aggregate_improvement),
            "aggregate_max_match_distance_m": float(self.aggregate_max_match_distance_m),
            "aggregate_trim_ratio": float(self.aggregate_trim_ratio),
            "aggregate_min_matches_per_pair": int(self.aggregate_min_matches_per_pair),
            "aggregate_huber_delta_m": float(self.aggregate_huber_delta_m),
            "decision_min_ate_improvement_ratio": float(
                self.decision_min_ate_improvement_ratio
            ),
            "decision_min_future_gain_positive_ratio": float(
                self.decision_min_future_gain_positive_ratio
            ),
            "decision_max_rpe_degradation_ratio": float(
                self.decision_max_rpe_degradation_ratio
            ),
            "decision_min_accepted_ratio": float(self.decision_min_accepted_ratio),
            "future_gain_window": int(self.future_gain_window),
        }


# ---------------------------------------------------------------------------
# Stage-2a refiner -> stage-2b feedback threshold mirroring.
#
# Stage 2b re-evaluates the object alignment loss with its own matcher and
# checks the correction against its own clamps.  Those thresholds only mean
# "re-check the refiner's proposal" if they carry the same values the refiner
# used; otherwise the gate silently measures something else.  The refiner
# exports its settings into the diagnostics so a mismatch is reported instead
# of inferred.
# ---------------------------------------------------------------------------

#: (stage-2a refiner setting, stage-2b feedback setting) pairs that must agree.
REFINER_SETTING_MIRRORS: tuple[tuple[str, str], ...] = (
    ("anchor_frame_count", "anchor_frame_count"),
    ("max_correction_rotation_deg", "max_feedback_rotation_deg"),
    ("max_correction_translation_m", "max_feedback_translation_m"),
    ("max_match_distance_m", "aggregate_max_match_distance_m"),
    ("trim_ratio", "aggregate_trim_ratio"),
    ("min_matches_per_pair", "aggregate_min_matches_per_pair"),
)


def check_refiner_settings(
    refiner_settings: Mapping[str, Any],
    *,
    config: ObjectPoseFeedbackConfig,
) -> dict[str, Any]:
    """Compare exported stage-2a refiner settings against the feedback config."""

    mismatches: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for refiner_name, feedback_name in REFINER_SETTING_MIRRORS:
        if refiner_name not in refiner_settings:
            missing.append(refiner_name)
            continue
        refiner_value = float(refiner_settings[refiner_name])
        feedback_value = float(getattr(config, feedback_name))
        scale = max(1.0, abs(refiner_value), abs(feedback_value))
        if abs(refiner_value - feedback_value) > 1e-9 * scale:
            mismatches[feedback_name] = {
                "refiner_setting": refiner_name,
                "refiner_value": refiner_value,
                "feedback_value": feedback_value,
            }
    return {
        "matches": not mismatches and not missing,
        "mismatches": mismatches,
        "missing_refiner_settings": sorted(missing),
    }


# ---------------------------------------------------------------------------
# Lie algebra helpers (ξ = (rotation vector, translation), small-angle form).
# The per-instance refiner optimizes exactly this parameterization, so using
# it for consensus keeps the aggregation in the proposal's native space.
# ---------------------------------------------------------------------------


def rotation_log(rotation: torch.Tensor) -> torch.Tensor:
    """SO(3) matrix -> rotation vector (axis * angle)."""

    value = torch.as_tensor(rotation).detach().float().cpu()
    cosine = torch.clamp((torch.trace(value) - 1.0) * 0.5, -1.0, 1.0)
    theta = float(torch.acos(cosine))
    skew = 0.5 * (value - value.transpose(0, 1))
    axis_skew = torch.stack((skew[2, 1], skew[0, 2], skew[1, 0]))
    if theta < 1e-8:
        return axis_skew
    if theta > math.pi - 1e-4:
        # Near pi the skew part vanishes; recover the axis from the diagonal.
        diagonal = torch.diagonal(value)
        axis = torch.sqrt(torch.clamp((diagonal + 1.0) * 0.5, min=0.0))
        signs = torch.sign(axis_skew)
        signs[signs == 0] = 1.0
        axis = axis * signs
        norm = float(torch.linalg.vector_norm(axis))
        if norm < 1e-8:
            return torch.zeros(3)
        return axis * (theta / norm)
    return axis_skew * (theta / math.sin(theta))


def pose_to_xi(delta: torch.Tensor) -> np.ndarray:
    """4x4 c2w correction -> 6-vector (rotvec, translation)."""

    value = torch.as_tensor(delta).detach().float().cpu()
    if tuple(value.shape) != (4, 4):
        raise ValueError("pose_to_xi expects a 4x4 transform.")
    rotvec = rotation_log(value[:3, :3])
    translation = value[:3, 3]
    return np.concatenate(
        (rotvec.numpy(), translation.numpy())
    ).astype(np.float64)


def xi_to_pose(xi: np.ndarray) -> torch.Tensor:
    """6-vector (rotvec, translation) -> 4x4 transform."""

    vector = np.asarray(xi, dtype=np.float64)
    if vector.shape != (6,):
        raise ValueError("xi_to_pose expects a 6-vector.")
    rotation = _so3_exp(torch.from_numpy(vector[:3].astype(np.float32)))
    pose = torch.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = torch.from_numpy(vector[3:].astype(np.float32))
    return pose


def rotation_angle_deg(rotation: torch.Tensor | np.ndarray) -> float:
    value = torch.as_tensor(rotation).detach().float().cpu()
    cosine = torch.clamp((torch.trace(value) - 1.0) * 0.5, -1.0, 1.0)
    return math.degrees(float(torch.acos(cosine)))


def translation_norm(delta: torch.Tensor | np.ndarray) -> float:
    value = torch.as_tensor(delta).detach().float().cpu()
    return float(torch.linalg.vector_norm(value[:3, 3]))


def gt_correction_error(
    delta: torch.Tensor | np.ndarray,
    gt_delta: torch.Tensor | np.ndarray,
) -> tuple[float, float]:
    """(translation error m, rotation error deg) between two c2w corrections."""

    predicted = torch.as_tensor(delta).detach().float().cpu()
    target = torch.as_tensor(gt_delta).detach().float().cpu()
    translation_error = float(
        torch.linalg.vector_norm(predicted[:3, 3] - target[:3, 3])
    )
    relative = target[:3, :3].transpose(0, 1) @ predicted[:3, :3]
    rotation_error = rotation_angle_deg(relative)
    return translation_error, rotation_error


# ---------------------------------------------------------------------------
# Geometric degeneracy from weighted covariance eigenvalues.
# ---------------------------------------------------------------------------


def covariance_eigenvalues(
    points: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[float, float, float]:
    """Eigenvalues (lambda1 >= lambda2 >= lambda3) of the weighted covariance."""

    value = torch.as_tensor(points).detach().float().cpu()
    mass = torch.as_tensor(weights).detach().float().cpu().clamp_min(0.0)
    if value.ndim != 2 or value.shape[1] != 3 or value.shape[0] == 0:
        return 0.0, 0.0, 0.0
    if mass.shape[0] != value.shape[0]:
        raise ValueError("covariance_eigenvalues: points/weights length mismatch.")
    total = float(mass.sum())
    if total <= 0.0:
        total = float(value.shape[0])
        mass = torch.full_like(mass, 1.0 / total)
    mean = (value * mass[:, None]).sum(dim=0) / total
    centered = value - mean[None, :]
    covariance = (centered * mass[:, None]).transpose(0, 1) @ centered / total
    eigenvalues = torch.linalg.eigvalsh(covariance)
    eigenvalues, _ = torch.sort(eigenvalues, descending=True)
    values = [max(0.0, float(item)) for item in eigenvalues]
    return values[0], values[1], values[2]


def classify_geometry(
    eigenvalues: tuple[float, float, float],
    *,
    point_count: int,
    config: ObjectPoseFeedbackConfig,
) -> tuple[str, float]:
    """Classify object 3D support as volumetric/planar/linear/degenerate."""

    lambda1, lambda2, lambda3 = eigenvalues
    degeneracy_factor = float(lambda3 / lambda1) if lambda1 > 1e-12 else 0.0
    if int(point_count) < int(config.degenerate_min_points) or lambda1 <= 1e-12:
        return "degenerate", 0.0
    ratio21 = float(lambda2 / lambda1)
    if degeneracy_factor >= float(config.volumetric_lambda_ratio):
        return "volumetric", degeneracy_factor
    if ratio21 >= float(config.planar_lambda_ratio):
        return "planar", degeneracy_factor
    return "linear", degeneracy_factor


# ---------------------------------------------------------------------------
# Semantic reliability (SAM identity quality only, no metric correspondence).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrackStats:
    track_length: int
    visibility_ratio: float


def compute_track_stats(
    observation_frames_by_instance: Mapping[int, Sequence[int]],
    *,
    instance_id: int,
    sequence_index: int,
    config: ObjectPoseFeedbackConfig,
) -> TrackStats:
    frames = sorted(
        int(frame)
        for frame in observation_frames_by_instance.get(int(instance_id), ())
        if int(frame) <= int(sequence_index)
    )
    track_length = len(frames)
    window = int(config.visibility_window)
    start = max(0, int(sequence_index) - window + 1)
    denominator = int(sequence_index) - start + 1
    present = sum(1 for frame in frames if frame >= start)
    visibility_ratio = float(present) / max(1, denominator)
    return TrackStats(
        track_length=track_length,
        visibility_ratio=visibility_ratio,
    )


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def semantic_reliability(
    track: TrackStats,
    *,
    mask_pixels: int,
    track_score: float,
    config: ObjectPoseFeedbackConfig,
) -> tuple[float | None, str | None]:
    """Rule-based SAM identity confidence; returns (score, reject_reason)."""

    if track.track_length < int(config.min_track_length):
        return None, "low_track_confidence"
    if track.visibility_ratio < float(config.min_visibility_ratio):
        return None, "low_track_confidence"
    if float(track_score) < float(config.min_sam_track_score):
        return None, "low_track_confidence"
    track_length_score = _clamp01(
        (track.track_length - int(config.min_track_length))
        / max(
            1,
            int(config.target_track_length) - int(config.min_track_length),
        )
    )
    visibility_score = _clamp01(
        track.visibility_ratio / max(1e-6, float(config.min_visibility_ratio))
        if config.min_visibility_ratio > 0.0
        else 1.0
    )
    visibility_score = min(1.0, visibility_score)
    mask_score = _clamp01(
        float(mask_pixels) / max(1, int(config.target_mask_pixels))
    )
    score_component = _clamp01(
        (float(track_score) - float(config.min_sam_track_score))
        / max(1e-6, 1.0 - float(config.min_sam_track_score))
    )
    score = (
        float(config.track_length_weight) * track_length_score
        + float(config.visibility_weight) * visibility_score
        + float(config.mask_pixels_weight) * mask_score
        + float(config.track_score_weight) * score_component
    )
    return score, None


# ---------------------------------------------------------------------------
# Geometric reliability (alignment quality + observability).
# ---------------------------------------------------------------------------


def geometric_reliability(
    *,
    relative_improvement: float | None,
    inlier_ratio: float,
    overlap_count: int,
    correction: torch.Tensor | None,
    geometry_type: str,
    degeneracy_factor: float,
    config: ObjectPoseFeedbackConfig,
) -> tuple[float | None, str | None]:
    """Rule-based geometry confidence; returns (score, reject_reason)."""

    if geometry_type == "degenerate":
        return None, "degenerate_geometry"
    if correction is not None:
        if translation_norm(correction) > float(config.max_feedback_translation_m):
            return None, "correction_too_large"
        if rotation_angle_deg(correction[:3, :3]) > float(
            config.max_feedback_rotation_deg
        ):
            return None, "correction_too_large"
    improvement = (
        float("inf") if relative_improvement is None else float(relative_improvement)
    )
    if not math.isfinite(improvement) or improvement < float(
        config.min_relative_improvement
    ):
        return None, "low_geometry_confidence"
    if float(inlier_ratio) < float(config.min_inlier_ratio):
        return None, "low_geometry_confidence"
    if int(overlap_count) < int(config.min_overlap_points):
        return None, "low_geometry_confidence"
    improvement_score = _clamp01(
        improvement / max(1e-6, float(config.target_relative_improvement))
    )
    inlier_score = _clamp01(
        float(inlier_ratio) / max(1e-6, float(config.target_inlier_ratio))
    )
    overlap_score = _clamp01(
        int(overlap_count) / max(1, int(config.target_overlap_points))
    )
    degeneracy_damp = _clamp01(
        degeneracy_factor / max(1e-6, float(config.volumetric_lambda_ratio))
    )
    score = (
        float(config.improvement_weight) * improvement_score
        + float(config.inlier_ratio_weight) * inlier_score
        + float(config.overlap_weight) * overlap_score
    ) * degeneracy_damp
    return score, None


# ---------------------------------------------------------------------------
# Proposals.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ObjectPoseProposal:
    frame_id: int
    sequence_index: int
    instance_id: int
    category: str
    correction: torch.Tensor | None
    accepted_by_refiner: bool
    refiner_reason: str
    track_length: int
    visibility_ratio: float
    track_score: float
    mask_pixels: int
    point_count: int
    overlap_count: int
    alignment_loss_before: float | None
    alignment_loss_after: float | None
    relative_improvement: float | None
    inlier_ratio: float
    eigenvalue_1: float
    eigenvalue_2: float
    eigenvalue_3: float
    geometry_type: str
    degeneracy_factor: float
    reference_frames: tuple[int, ...]
    reference_roles: tuple[str, ...]
    pair_weights: tuple[float, ...]
    semantic_confidence: float | None
    semantic_reject_reason: str | None
    geometry_confidence: float | None
    geometry_reject_reason: str | None

    @property
    def object_reject_reason(self) -> str | None:
        if self.correction is None:
            return "no_historical_instance_reference"
        if self.semantic_reject_reason is not None:
            return self.semantic_reject_reason
        if not self.accepted_by_refiner:
            return "refiner_rejected"
        if self.geometry_reject_reason is not None:
            return self.geometry_reject_reason
        return None

    @property
    def is_reliable(self) -> bool:
        return self.object_reject_reason is None


def build_proposals(
    diagnostics: Mapping[str, Any],
    *,
    config: ObjectPoseFeedbackConfig,
) -> list[ObjectPoseProposal]:
    """Convert feedback diagnostics into reliability-scored proposals."""

    observations = list(diagnostics.get("observations", ()))
    observation_by_key: dict[tuple[int, int], Mapping[str, Any]] = {}
    frames_by_instance: dict[int, list[int]] = {}
    for row in observations:
        key = (int(row["sequence_index"]), int(row["instance_id"]))
        if key in observation_by_key:
            raise ValueError(f"Duplicate observation row for {key}.")
        observation_by_key[key] = row
        frames_by_instance.setdefault(int(row["instance_id"]), []).append(
            int(row["sequence_index"])
        )

    proposals: list[ObjectPoseProposal] = []
    for row in diagnostics.get("proposals", ()):
        sequence_index = int(row["sequence_index"])
        instance_id = int(row["instance_id"])
        observation = observation_by_key.get((sequence_index, instance_id))
        if observation is None:
            raise ValueError(
                "Proposal row without a retained observation: "
                f"frame={sequence_index} instance={instance_id}"
            )
        points = observation["points_camera"]
        weights = observation["weights"]
        point_count = int(points.shape[0])
        eigenvalues = covariance_eigenvalues(points, weights)
        geometry_type, degeneracy_factor = classify_geometry(
            eigenvalues,
            point_count=point_count,
            config=config,
        )
        track = compute_track_stats(
            frames_by_instance,
            instance_id=instance_id,
            sequence_index=sequence_index,
            config=config,
        )
        track_score = float(observation["track_score"])
        mask_pixels = int(observation["mask_pixels"])
        semantic_confidence, semantic_reject = semantic_reliability(
            track,
            mask_pixels=mask_pixels,
            track_score=track_score,
            config=config,
        )
        correction = row.get("correction")
        correction = (
            torch.as_tensor(correction).detach().float().cpu()
            if correction is not None
            else None
        )
        overlap_raw = row.get("final_match_count")
        overlap_count = 0 if overlap_raw is None else int(overlap_raw)
        inlier_ratio = float(overlap_count) / max(1, point_count)
        relative_improvement = row.get("relative_loss_improvement")
        relative_improvement = (
            None
            if relative_improvement is None
            else float(relative_improvement)
        )
        geometry_confidence, geometry_reject = geometric_reliability(
            relative_improvement=relative_improvement,
            inlier_ratio=inlier_ratio,
            overlap_count=overlap_count,
            correction=correction,
            geometry_type=geometry_type,
            degeneracy_factor=degeneracy_factor,
            config=config,
        )
        loss_before = row.get("initial_loss_m")
        loss_after = row.get("final_loss_m")
        proposals.append(
            ObjectPoseProposal(
                frame_id=int(row["frame_id"]),
                sequence_index=sequence_index,
                instance_id=instance_id,
                category=str(row["category"]),
                correction=correction,
                accepted_by_refiner=bool(row["accepted"]),
                refiner_reason=str(row.get("reason", "unknown")),
                track_length=track.track_length,
                visibility_ratio=track.visibility_ratio,
                track_score=track_score,
                mask_pixels=mask_pixels,
                point_count=point_count,
                overlap_count=overlap_count,
                alignment_loss_before=(
                    None if loss_before is None else float(loss_before)
                ),
                alignment_loss_after=(
                    None if loss_after is None else float(loss_after)
                ),
                relative_improvement=relative_improvement,
                inlier_ratio=inlier_ratio,
                eigenvalue_1=eigenvalues[0],
                eigenvalue_2=eigenvalues[1],
                eigenvalue_3=eigenvalues[2],
                geometry_type=geometry_type,
                degeneracy_factor=degeneracy_factor,
                reference_frames=tuple(
                    int(value) for value in row.get("reference_frames", ())
                ),
                reference_roles=tuple(
                    str(value) for value in row.get("reference_roles", ())
                ),
                pair_weights=tuple(
                    float(value) for value in row.get("pair_weights", ())
                ),
                semantic_confidence=semantic_confidence,
                semantic_reject_reason=semantic_reject,
                geometry_confidence=geometry_confidence,
                geometry_reject_reason=geometry_reject,
            )
        )
    proposals.sort(key=lambda item: (item.sequence_index, item.instance_id))
    return proposals


# ---------------------------------------------------------------------------
# Cross-object consensus.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConsensusVariant:
    name: str
    mode: str  # "single" | "mean" | "robust"
    weight_mode: str  # "uniform" | "semantic" | "semantic_geometric"


CONSENSUS_VARIANTS: tuple[ConsensusVariant, ...] = (
    ConsensusVariant(name="single", mode="single", weight_mode="semantic_geometric"),
    ConsensusVariant(name="mean", mode="mean", weight_mode="uniform"),
    ConsensusVariant(name="robust", mode="robust", weight_mode="uniform"),
    ConsensusVariant(name="robust_semantic", mode="robust", weight_mode="semantic"),
    ConsensusVariant(
        name="robust_semantic_geometric",
        mode="robust",
        weight_mode="semantic_geometric",
    ),
)

MAIN_VARIANT_NAME = "robust_semantic_geometric"


@dataclass(frozen=True)
class ConsensusResult:
    delta: torch.Tensor | None
    xi: np.ndarray | None
    weights: tuple[float, ...]
    translation_errors: tuple[float, ...]
    rotation_errors: tuple[float, ...]
    inlier_flags: tuple[bool, ...]


def consensus_weight(
    proposal: ObjectPoseProposal,
    variant: ConsensusVariant,
) -> float:
    if variant.weight_mode == "uniform":
        return 1.0
    semantic = (
        0.0
        if proposal.semantic_confidence is None
        else float(proposal.semantic_confidence)
    )
    if variant.weight_mode == "semantic":
        return max(1e-6, semantic)
    geometric = (
        0.0
        if proposal.geometry_confidence is None
        else float(proposal.geometry_confidence)
    )
    return max(1e-6, semantic * geometric)


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    output = np.empty(values.shape[1], dtype=np.float64)
    for column in range(values.shape[1]):
        order = np.argsort(values[:, column], kind="stable")
        sorted_values = values[order, column]
        sorted_weights = weights[order]
        cumulative = np.cumsum(sorted_weights)
        threshold = 0.5 * float(sorted_weights.sum())
        index = int(np.searchsorted(cumulative, threshold, side="left"))
        index = min(index, sorted_values.shape[0] - 1)
        output[column] = sorted_values[index]
    return output


def robust_consensus(
    proposals: Sequence[ObjectPoseProposal],
    variant: ConsensusVariant,
    *,
    config: ObjectPoseFeedbackConfig,
) -> ConsensusResult:
    """Aggregate reliable per-object corrections into one camera correction."""

    if not proposals:
        return ConsensusResult(
            delta=None,
            xi=None,
            weights=(),
            translation_errors=(),
            rotation_errors=(),
            inlier_flags=(),
        )
    corrections = [
        proposal for proposal in proposals if proposal.correction is not None
    ]
    if not corrections:
        return ConsensusResult(
            delta=None,
            xi=None,
            weights=(),
            translation_errors=(),
            rotation_errors=(),
            inlier_flags=(),
        )
    xi_matrix = np.stack(
        [pose_to_xi(proposal.correction) for proposal in corrections], axis=0
    )
    weights = np.asarray(
        [consensus_weight(proposal, variant) for proposal in corrections],
        dtype=np.float64,
    )
    rotation_scale = math.radians(float(config.consensus_huber_rotation_deg))
    translation_scale = float(config.consensus_huber_translation_m)

    def scaled_residual(xi: np.ndarray) -> np.ndarray:
        rotation_part = np.linalg.norm(xi[:, :3] / rotation_scale, axis=1)
        translation_part = np.linalg.norm(xi[:, 3:] / translation_scale, axis=1)
        return np.sqrt(rotation_part**2 + translation_part**2)

    if variant.mode == "single":
        best_index = int(np.argmax(weights))
        xi = xi_matrix[best_index].copy()
    elif variant.mode == "mean":
        xi = (weights[:, None] * xi_matrix).sum(axis=0) / weights.sum()
    else:
        xi = _weighted_median(xi_matrix, weights)
        for _ in range(int(config.consensus_irls_iterations)):
            residual = scaled_residual(xi_matrix - xi[None, :])
            huber = np.where(residual > 1.0, 1.0 / np.maximum(residual, 1e-12), 1.0)
            combined = weights * huber
            if float(combined.sum()) <= 0.0:
                break
            xi = (combined[:, None] * xi_matrix).sum(axis=0) / float(
                combined.sum()
            )

    delta = xi_to_pose(xi)
    translation_errors: list[float] = []
    rotation_errors: list[float] = []
    inlier_flags: list[bool] = []
    for proposal, proposal_xi in zip(corrections, xi_matrix):
        t_error, r_error = gt_correction_error(proposal.correction, delta)
        translation_errors.append(t_error)
        rotation_errors.append(r_error)
        if variant.mode == "robust":
            residual = float(
                np.linalg.norm(
                    np.concatenate(
                        (
                            (proposal_xi[:3] - xi[:3]) / rotation_scale,
                            (proposal_xi[3:] - xi[3:]) / translation_scale,
                        )
                    )
                )
            )
            huber = 1.0 if residual <= 1.0 else 1.0 / max(residual, 1e-12)
            inlier = huber >= float(config.inlier_weight_threshold)
        else:
            inlier = (
                t_error <= float(config.consensus_tolerance_translation_m)
                and r_error <= float(config.consensus_tolerance_rotation_deg)
            )
        inlier_flags.append(bool(inlier))
    return ConsensusResult(
        delta=delta,
        xi=xi,
        weights=tuple(float(value) for value in weights),
        translation_errors=tuple(translation_errors),
        rotation_errors=tuple(rotation_errors),
        inlier_flags=tuple(inlier_flags),
    )


# ---------------------------------------------------------------------------
# Aggregate alignment loss (gate evidence, reuses the refiner's matcher).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReferenceCloud:
    frame_id: int
    sequence_index: int
    role: str
    quality: float
    points_world: torch.Tensor
    weights: torch.Tensor


@dataclass(frozen=True)
class FrameAlignmentContext:
    sequence_index: int
    raw_pose: torch.Tensor  # 4x4 c2w, frame-0 gauge
    points_camera: Mapping[int, torch.Tensor]
    weights: Mapping[int, torch.Tensor]
    references: Mapping[int, tuple[ReferenceCloud, ...]]


def _transform_points(points: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
    value = torch.as_tensor(points).detach().float().cpu()
    transform = torch.as_tensor(pose).detach().float().cpu()
    if tuple(transform.shape) == (3, 4):
        bottom = torch.zeros((1, 4), dtype=transform.dtype)
        bottom[0, 3] = 1.0
        transform = torch.cat((transform, bottom), dim=0)
    return value @ transform[:3, :3].transpose(0, 1) + transform[:3, 3]


def _match_loss(
    current_world: torch.Tensor,
    current_weights: torch.Tensor,
    reference: ReferenceCloud,
    *,
    config: ObjectPoseFeedbackConfig,
) -> float | None:
    current_indices, reference_indices, match_weights = _mutual_matches(
        current_world,
        reference.points_world,
        current_weights,
        reference.weights,
        max_distance=float(config.aggregate_max_match_distance_m),
        trim_ratio=float(config.aggregate_trim_ratio),
        min_matches=int(config.aggregate_min_matches_per_pair),
        pair_weight=1.0,
    )
    if current_indices.numel() == 0:
        return None
    with torch.no_grad():
        residual = current_world.index_select(0, current_indices) - (
            reference.points_world.index_select(0, reference_indices)
        )
        distance = torch.linalg.vector_norm(residual, dim=-1)
        values = _huber_distance(distance, float(config.aggregate_huber_delta_m))
        mass = match_weights.clamp_min(1e-8)
        return float((values * mass).sum() / mass.sum())


def aggregate_alignment_loss(
    context: FrameAlignmentContext,
    proposals: Sequence[ObjectPoseProposal],
    correction: torch.Tensor | None,
    *,
    config: ObjectPoseFeedbackConfig,
) -> tuple[float, float]:
    """Weighted object alignment residual (before, after) under ``correction``.

    ``before`` places the current points with the raw pose; ``after`` left
    multiplies the consensus correction on those world points.  Pairs without
    mutual matches are penalized with the match-distance cap on that side.
    """

    penalty = float(config.aggregate_max_match_distance_m)
    before_total = 0.0
    after_total = 0.0
    weight_total = 0.0
    for proposal in proposals:
        instance_id = int(proposal.instance_id)
        points = context.points_camera.get(instance_id)
        weights = context.weights.get(instance_id)
        references = context.references.get(instance_id, ())
        if points is None or weights is None:
            continue
        if proposal.pair_weights:
            pair_weights = proposal.pair_weights
        else:
            pair_weights = (1.0,) * max(1, len(references))
        base_world = _transform_points(points, context.raw_pose)
        corrected_world = (
            _transform_points(base_world, correction)
            if correction is not None
            else base_world
        )
        for reference_index, reference in enumerate(references):
            pair_weight = float(
                pair_weights[min(reference_index, len(pair_weights) - 1)]
            )
            loss_before = _match_loss(
                base_world, weights, reference, config=config
            )
            loss_after = _match_loss(
                corrected_world, weights, reference, config=config
            )
            before_total += pair_weight * (
                penalty if loss_before is None else loss_before
            )
            after_total += pair_weight * (
                penalty if loss_after is None else loss_after
            )
            weight_total += pair_weight
    if weight_total <= 0.0:
        return float("inf"), float("inf")
    return before_total / weight_total, after_total / weight_total


# ---------------------------------------------------------------------------
# Per-frame gating.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FrameGateResult:
    variant: str
    sequence_index: int
    frame_id: int
    accepted: bool
    reason: str | None
    delta: torch.Tensor | None
    target_c2w: torch.Tensor | None
    num_proposals: int
    num_semantic_reliable: int
    num_reliable: int
    inlier_count: int
    consensus_translation_norm: float | None
    consensus_rotation_deg: float | None
    aggregate_loss_before: float | None
    aggregate_loss_after: float | None


REJECT_INSUFFICIENT_OBJECTS = "insufficient_objects"
REJECT_LOW_TRACK_CONFIDENCE = "low_track_confidence"
REJECT_LOW_GEOMETRY_CONFIDENCE = "low_geometry_confidence"
REJECT_DEGENERATE_GEOMETRY = "degenerate_geometry"
REJECT_CORRECTION_TOO_LARGE = "correction_too_large"
REJECT_NO_CONSENSUS = "no_consensus"
REJECT_NO_ALIGNMENT_IMPROVEMENT = "no_alignment_improvement"


def gate_frame(
    proposals: Sequence[ObjectPoseProposal],
    variant: ConsensusVariant,
    *,
    context: FrameAlignmentContext,
    config: ObjectPoseFeedbackConfig,
) -> FrameGateResult:
    """Run one variant's reliability -> consensus -> gate chain on one frame."""

    sequence_index = context.sequence_index
    frame_id = proposals[0].frame_id if proposals else sequence_index
    usable = [
        proposal
        for proposal in proposals
        if proposal.correction is not None
    ]

    def result(
        *,
        accepted: bool,
        reason: str | None,
        consensus: ConsensusResult | None,
        num_semantic_reliable: int,
        num_reliable: int,
        aggregate: tuple[float, float] | None,
    ) -> FrameGateResult:
        delta = consensus.delta if accepted and consensus is not None else None
        return FrameGateResult(
            variant=variant.name,
            sequence_index=sequence_index,
            frame_id=int(frame_id),
            accepted=accepted,
            reason=reason,
            delta=delta,
            target_c2w=None,
            num_proposals=len(usable),
            num_semantic_reliable=num_semantic_reliable,
            num_reliable=num_reliable,
            inlier_count=(
                sum(1 for flag in consensus.inlier_flags if flag)
                if consensus is not None
                else 0
            ),
            consensus_translation_norm=(
                translation_norm(consensus.delta)
                if consensus is not None and consensus.delta is not None
                else None
            ),
            consensus_rotation_deg=(
                rotation_angle_deg(consensus.delta[:3, :3])
                if consensus is not None and consensus.delta is not None
                else None
            ),
            aggregate_loss_before=aggregate[0] if aggregate else None,
            aggregate_loss_after=aggregate[1] if aggregate else None,
        )

    if not usable:
        return result(
            accepted=False,
            reason=REJECT_INSUFFICIENT_OBJECTS,
            consensus=None,
            num_semantic_reliable=0,
            num_reliable=0,
            aggregate=None,
        )

    semantic_reliable = [
        proposal
        for proposal in usable
        if proposal.semantic_reject_reason is None
    ]
    if not semantic_reliable:
        return result(
            accepted=False,
            reason=REJECT_LOW_TRACK_CONFIDENCE,
            consensus=None,
            num_semantic_reliable=0,
            num_reliable=0,
            aggregate=None,
        )

    reliable = [
        proposal
        for proposal in semantic_reliable
        if proposal.accepted_by_refiner and proposal.geometry_reject_reason is None
    ]
    if not reliable:
        failures = {
            (
                proposal.geometry_reject_reason
                if proposal.geometry_reject_reason is not None
                else "refiner_rejected"
            )
            for proposal in semantic_reliable
        }
        if failures == {"degenerate_geometry"}:
            reason = REJECT_DEGENERATE_GEOMETRY
        elif failures == {"correction_too_large"}:
            reason = REJECT_CORRECTION_TOO_LARGE
        else:
            reason = REJECT_LOW_GEOMETRY_CONFIDENCE
        return result(
            accepted=False,
            reason=reason,
            consensus=None,
            num_semantic_reliable=len(semantic_reliable),
            num_reliable=0,
            aggregate=None,
        )

    min_objects = (
        1 if variant.mode == "single" else int(config.min_consensus_objects)
    )
    if len(reliable) < min_objects:
        return result(
            accepted=False,
            reason=REJECT_INSUFFICIENT_OBJECTS,
            consensus=None,
            num_semantic_reliable=len(semantic_reliable),
            num_reliable=len(reliable),
            aggregate=None,
        )

    consensus = robust_consensus(reliable, variant, config=config)
    inlier_indices = [
        index
        for index, flag in enumerate(consensus.inlier_flags)
        if flag
    ]
    if len(inlier_indices) < min_objects:
        return result(
            accepted=False,
            reason=REJECT_NO_CONSENSUS,
            consensus=consensus,
            num_semantic_reliable=len(semantic_reliable),
            num_reliable=len(reliable),
            aggregate=None,
        )
    inlier_weights = [consensus.weights[index] for index in inlier_indices]
    weight_sum = sum(inlier_weights)
    mean_inlier_translation = (
        sum(
            consensus.weights[index] * consensus.translation_errors[index]
            for index in inlier_indices
        )
        / weight_sum
        if weight_sum > 0.0
        else float("inf")
    )
    mean_inlier_rotation = (
        sum(
            consensus.weights[index] * consensus.rotation_errors[index]
            for index in inlier_indices
        )
        / weight_sum
        if weight_sum > 0.0
        else float("inf")
    )
    if (
        mean_inlier_translation > float(config.consensus_tolerance_translation_m)
        or mean_inlier_rotation > float(config.consensus_tolerance_rotation_deg)
    ):
        return result(
            accepted=False,
            reason=REJECT_NO_CONSENSUS,
            consensus=consensus,
            num_semantic_reliable=len(semantic_reliable),
            num_reliable=len(reliable),
            aggregate=None,
        )

    delta = consensus.delta
    if (
        translation_norm(delta) > float(config.max_consensus_translation_m)
        or rotation_angle_deg(delta[:3, :3])
        > float(config.max_consensus_rotation_deg)
    ):
        return result(
            accepted=False,
            reason=REJECT_CORRECTION_TOO_LARGE,
            consensus=consensus,
            num_semantic_reliable=len(semantic_reliable),
            num_reliable=len(reliable),
            aggregate=None,
        )

    before, after = aggregate_alignment_loss(
        context, reliable, delta, config=config
    )
    if (
        not math.isfinite(before)
        or not math.isfinite(after)
        or not after < before * (1.0 - float(config.min_aggregate_improvement))
    ):
        return result(
            accepted=False,
            reason=REJECT_NO_ALIGNMENT_IMPROVEMENT,
            consensus=consensus,
            num_semantic_reliable=len(semantic_reliable),
            num_reliable=len(reliable),
            aggregate=(before, after),
        )

    target_c2w = delta @ torch.as_tensor(context.raw_pose).detach().float().cpu()
    gate = FrameGateResult(
        variant=variant.name,
        sequence_index=sequence_index,
        frame_id=int(frame_id),
        accepted=True,
        reason=None,
        delta=delta,
        target_c2w=target_c2w,
        num_proposals=len(usable),
        num_semantic_reliable=len(semantic_reliable),
        num_reliable=len(reliable),
        inlier_count=len(inlier_indices),
        consensus_translation_norm=translation_norm(delta),
        consensus_rotation_deg=rotation_angle_deg(delta[:3, :3]),
        aggregate_loss_before=before,
        aggregate_loss_after=after,
    )
    return gate


def build_frame_contexts(
    diagnostics: Mapping[str, Any],
    *,
    config: ObjectPoseFeedbackConfig,
) -> dict[int, FrameAlignmentContext]:
    """Per-frame alignment contexts from feedback diagnostics."""

    del config  # thresholds are only used inside the loss itself
    raw_poses = torch.as_tensor(
        diagnostics["raw_camera_to_world"]
    ).detach().float().cpu()
    frame_ids = [int(value) for value in diagnostics.get("frame_ids", ())]
    if raw_poses.shape[0] != len(frame_ids):
        raise ValueError("raw_camera_to_world does not match frame_ids.")
    points_camera: dict[int, dict[int, torch.Tensor]] = {}
    weights_by_frame: dict[int, dict[int, torch.Tensor]] = {}
    for row in diagnostics.get("observations", ()):
        sequence_index = int(row["sequence_index"])
        instance_id = int(row["instance_id"])
        points_camera.setdefault(sequence_index, {})[instance_id] = (
            torch.as_tensor(row["points_camera"]).detach().float().cpu()
        )
        weights_by_frame.setdefault(sequence_index, {})[instance_id] = (
            torch.as_tensor(row["weights"]).detach().float().cpu()
        )
    references: dict[int, dict[int, tuple[ReferenceCloud, ...]]] = {}
    for row in diagnostics.get("pairing_snapshots", ()):
        sequence_index = int(row["sequence_index"])
        instance_id = int(row["instance_id"])
        clouds = tuple(
            ReferenceCloud(
                frame_id=int(reference["frame_id"]),
                sequence_index=int(reference["sequence_index"]),
                role=str(reference["role"]),
                quality=float(reference["quality"]),
                points_world=torch.as_tensor(reference["points_world"])
                .detach()
                .float()
                .cpu(),
                weights=torch.as_tensor(reference["weights"])
                .detach()
                .float()
                .cpu(),
            )
            for reference in row.get("references", ())
        )
        references.setdefault(sequence_index, {})[instance_id] = clouds

    contexts: dict[int, FrameAlignmentContext] = {}
    for sequence_index, frame_id in enumerate(frame_ids):
        contexts[sequence_index] = FrameAlignmentContext(
            sequence_index=sequence_index,
            raw_pose=raw_poses[sequence_index],
            points_camera=points_camera.get(sequence_index, {}),
            weights=weights_by_frame.get(sequence_index, {}),
            references=references.get(sequence_index, {}),
        )
    return contexts


# ---------------------------------------------------------------------------
# Offline GT evaluation and GO/NO-GO decision (never used for feedback).
# ---------------------------------------------------------------------------


#: Metric keys written by the stage-2b replay and read by the decision.
#: Defined once here so the producer and the consumer cannot drift apart.
RPE_TRANSLATION_BOUNDARY_EXCLUDED_KEY = (
    "rpe_translation_rmse_excluding_correction_boundaries_m"
)
RPE_ROTATION_BOUNDARY_EXCLUDED_KEY = (
    "rpe_rotation_rmse_excluding_correction_boundaries_deg"
)
RPE_EXCLUDED_BOUNDARY_PAIR_COUNT_KEY = (
    "rpe_excluded_correction_boundary_pair_count"
)


@dataclass(frozen=True)
class TrajectoryComparison:
    ate_improvement_ratio: float | None
    rpe_translation_ratio: float | None
    rpe_rotation_ratio: float | None


def compare_trajectories(
    raw_metrics: Mapping[str, Any],
    feedback_metrics: Mapping[str, Any],
    *,
    raw_rpe_translation_key: str,
    feedback_rpe_translation_key: str,
    raw_rpe_rotation_key: str,
    feedback_rpe_rotation_key: str,
) -> TrajectoryComparison:
    """Relative changes of feedback vs raw (positive = feedback better).

    The rotation key is passed explicitly rather than derived from the
    translation key: the metric suffixes differ (``_m`` vs ``_deg``), so
    string substitution silently looks up a key that does not exist and
    reports ``None`` for every frame.
    """

    def ratio(raw: Any, feedback: Any) -> float | None:
        if raw is None or feedback is None:
            return None
        raw_value = float(raw)
        feedback_value = float(feedback)
        if raw_value <= 0.0:
            return None
        return (raw_value - feedback_value) / raw_value

    return TrajectoryComparison(
        ate_improvement_ratio=ratio(
            raw_metrics.get("ate_rmse_m"), feedback_metrics.get("ate_rmse_m")
        ),
        rpe_translation_ratio=ratio(
            raw_metrics.get(raw_rpe_translation_key),
            feedback_metrics.get(feedback_rpe_translation_key),
        ),
        rpe_rotation_ratio=ratio(
            raw_metrics.get(raw_rpe_rotation_key),
            feedback_metrics.get(feedback_rpe_rotation_key),
        ),
    )


def decide_object_feedback(
    *,
    variant_name: str,
    raw_metrics: Mapping[str, Any],
    feedback_metrics: Mapping[str, Any],
    future_translation_gains: Sequence[float],
    future_rotation_gains: Sequence[float],
    accepted_ratio: float | None,
    config: ObjectPoseFeedbackConfig,
) -> dict[str, Any]:
    """Pose-metric-only GO / NO-GO decision for one feedback variant."""

    comparison = compare_trajectories(
        raw_metrics,
        feedback_metrics,
        raw_rpe_translation_key=RPE_TRANSLATION_BOUNDARY_EXCLUDED_KEY,
        feedback_rpe_translation_key=RPE_TRANSLATION_BOUNDARY_EXCLUDED_KEY,
        raw_rpe_rotation_key=RPE_ROTATION_BOUNDARY_EXCLUDED_KEY,
        feedback_rpe_rotation_key=RPE_ROTATION_BOUNDARY_EXCLUDED_KEY,
    )
    finite_gains = [
        float(value) for value in future_translation_gains if math.isfinite(value)
    ]
    positive_ratio = (
        sum(1 for value in finite_gains if value > 1e-9) / len(finite_gains)
        if finite_gains
        else 0.0
    )
    median_gain = (
        float(np.median(finite_gains)) if finite_gains else float("-inf")
    )
    ate_ok = (
        comparison.ate_improvement_ratio is not None
        and comparison.ate_improvement_ratio
        >= float(config.decision_min_ate_improvement_ratio)
    )
    future_ok = (
        median_gain > 0.0
        and positive_ratio >= float(config.decision_min_future_gain_positive_ratio)
    )
    rpe_ok = (
        comparison.rpe_translation_ratio is not None
        and comparison.rpe_translation_ratio
        >= -(
            float(config.decision_max_rpe_degradation_ratio) - 1.0
        )  # degradation tolerance
    )
    accepted_ok = accepted_ratio is not None and (
        float(accepted_ratio) >= float(config.decision_min_accepted_ratio)
    )
    criteria = {
        "direct_ate_improvement_ratio": {
            "value": comparison.ate_improvement_ratio,
            "threshold": float(config.decision_min_ate_improvement_ratio),
            "passed": bool(ate_ok),
        },
        "future_translation_gain_median_m": {
            "value": None if median_gain == float("-inf") else median_gain,
            "threshold": "> 0",
            "passed": bool(median_gain > 0.0),
        },
        "future_translation_gain_positive_ratio": {
            "value": positive_ratio,
            "threshold": float(config.decision_min_future_gain_positive_ratio),
            "passed": bool(future_ok),
        },
        "rpe_translation_ratio_boundary_excluded": {
            "value": comparison.rpe_translation_ratio,
            "threshold": f">= {-(config.decision_max_rpe_degradation_ratio - 1.0):.3f}",
            "passed": bool(rpe_ok),
        },
        "accepted_ratio": {
            "value": accepted_ratio,
            "threshold": float(config.decision_min_accepted_ratio),
            "passed": bool(accepted_ok),
        },
    }
    decision = (
        "OBJECT_FEEDBACK_GO"
        if all(item["passed"] for item in criteria.values())
        else "OBJECT_FEEDBACK_NO_GO"
    )
    return {
        "variant": variant_name,
        "decision": decision,
        "decision_basis": "GT pose metrics and future trajectory only; "
        "alignment loss is not a decision input.",
        "criteria": criteria,
        # Reported for audit only; deliberately outside ``criteria`` so it
        # cannot change the decision (the protocol gates rotation RPE through
        # the future-gain criterion, not through a direct RPE ratio).
        "rpe_rotation_ratio_boundary_excluded_informational": (
            comparison.rpe_rotation_ratio
        ),
        "future_rotation_gain_median_deg": (
            float(np.median([v for v in future_rotation_gains if math.isfinite(v)]))
            if any(math.isfinite(v) for v in future_rotation_gains)
            else None
        ),
    }


def assess_bottleneck(
    *,
    proposal_gt_translation_errors: Sequence[float],
    proposal_gt_rotation_errors: Sequence[float],
    proposal_semantic_confidences: Sequence[float | None],
    consensus_gt_translation_errors: Sequence[float],
    consensus_gt_rotation_errors: Sequence[float],
    gate_reason_counts: Mapping[str, int],
    accepted_ratio: float | None,
    feedback_ate_improvement_ratio: float | None,
) -> dict[str, Any]:
    """Heuristic attribution of the dominant failure mode (auditable labels)."""

    def median(values: Sequence[float]) -> float | None:
        finite = [float(v) for v in values if math.isfinite(float(v))]
        return float(np.median(finite)) if finite else None

    proposal_translation = median(proposal_gt_translation_errors)
    proposal_rotation = median(proposal_gt_rotation_errors)
    consensus_translation = median(consensus_gt_translation_errors)
    best_consensus = min(
        (float(v) for v in consensus_gt_translation_errors if math.isfinite(v)),
        default=None,
    )
    best_proposal = min(
        (float(v) for v in proposal_gt_translation_errors if math.isfinite(v)),
        default=None,
    )
    evidence = {
        "proposal_gt_translation_error_median_m": proposal_translation,
        "proposal_gt_rotation_error_median_deg": proposal_rotation,
        "consensus_gt_translation_error_median_m": consensus_translation,
        "consensus_gt_rotation_error_median_deg": median(
            consensus_gt_rotation_errors
        ),
        "best_consensus_gt_translation_error_m": best_consensus,
        "best_proposal_gt_translation_error_m": best_proposal,
        "gate_reason_counts": {str(k): int(v) for k, v in gate_reason_counts.items()},
        "accepted_ratio": accepted_ratio,
        "feedback_ate_improvement_ratio": feedback_ate_improvement_ratio,
    }
    labels: list[str] = []
    if proposal_translation is not None and proposal_translation > 0.15:
        weak_semantic = [
            float(value)
            for value in proposal_semantic_confidences
            if value is not None
        ]
        mean_semantic = (
            float(np.mean(weak_semantic)) if weak_semantic else None
        )
        labels.append(
            "sam_tracking"
            if mean_semantic is not None and mean_semantic < 0.5
            else "nn_correspondence_or_object_geometry"
        )
    if (
        proposal_translation is not None
        and consensus_translation is not None
        and best_proposal is not None
        and consensus_translation > 1.5 * max(best_proposal, 1e-6)
    ):
        labels.append("consensus")
    if accepted_ratio is not None and accepted_ratio < 0.10:
        labels.append("gating")
    if (
        not labels
        and feedback_ate_improvement_ratio is not None
        and feedback_ate_improvement_ratio < 0.05
    ):
        labels.append("pose_feedback")
    if not labels:
        labels.append("none_observed")
    evidence["primary_bottleneck"] = labels[0]
    evidence["bottleneck_labels"] = labels
    return evidence


__all__ = [
    "CONSENSUS_VARIANTS",
    "FrameAlignmentContext",
    "FrameGateResult",
    "MAIN_VARIANT_NAME",
    "ObjectPoseFeedbackConfig",
    "ObjectPoseProposal",
    "REFINER_SETTING_MIRRORS",
    "RPE_EXCLUDED_BOUNDARY_PAIR_COUNT_KEY",
    "RPE_ROTATION_BOUNDARY_EXCLUDED_KEY",
    "RPE_TRANSLATION_BOUNDARY_EXCLUDED_KEY",
    "ReferenceCloud",
    "TrackStats",
    "aggregate_alignment_loss",
    "assess_bottleneck",
    "build_frame_contexts",
    "build_proposals",
    "check_refiner_settings",
    "classify_geometry",
    "compare_trajectories",
    "compute_track_stats",
    "consensus_weight",
    "covariance_eigenvalues",
    "decide_object_feedback",
    "gate_frame",
    "geometric_reliability",
    "gt_correction_error",
    "pose_to_xi",
    "robust_consensus",
    "rotation_angle_deg",
    "rotation_log",
    "semantic_reliability",
    "translation_norm",
    "xi_to_pose",
]
