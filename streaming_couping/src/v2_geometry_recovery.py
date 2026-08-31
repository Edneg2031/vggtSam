"""Conditional geometry-guided recovery for the V2 semantic-map ablation.

V2 deliberately keeps the V0 SAM3 slot identity and normal mask untouched.
Geometry is a recovery signal only: a historical object point set is projected
when a slot is weak, missing, or re-enters after a gap; the projection is then
given to frozen SAM3.1 as a box/positive-point prompt.  A candidate that fails
the geometry gate falls back to the original V0 observation.

The module is model-agnostic.  ``process_v2_sequence`` receives callbacks for
proposal construction and SAM refinement, which makes the causal policy
testable without loading either foundation model.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import torch

from .aggregation.point_map_fusion import sample_masked_observation
from .recovery import output_mask_to_stream
from .types import RevisitCandidate, SAM3MaskCandidate


@dataclass(frozen=True)
class V2RecoveryConfig:
    """Causal trigger, proposal and acceptance thresholds."""

    low_score_threshold: float = 0.50
    min_mask_pixels: int = 32
    min_area_ratio: float = 0.35
    max_area_ratio: float = 2.85
    reentry_gap: int = 3
    min_history_points: int = 16
    max_points_per_object: int = 4096
    min_memory_update_score: float = 0.50
    min_geometry_confidence: float = 0.30

    max_positive_points: int = 6
    min_candidate_score: float = 0.50
    min_candidate_support_recall: float = 0.15
    min_candidate_iou: float = 0.02
    min_candidate_area_ratio: float = 0.20
    max_candidate_area_ratio: float = 5.00

    proposal_box_quantile: float = 0.02
    proposal_box_padding_ratio: float = 0.12
    min_projected_points: int = 8
    min_projected_fraction: float = 0.005
    min_supported_points: int = 8
    min_support_ratio: float = 0.02
    support_abs_distance: float = 0.15
    support_relative_distance: float = 0.10


@dataclass
class V2ObjectMemoryState:
    """Causal memory for one existing V0 SAM slot.

    The tensors stay outside JSON and are serialized by the exporter in the
    semantic-map artifact.  ``history`` contains compact, JSON-safe metadata
    for each accepted observation, including the camera matrix and mask bbox.
    """

    object_id: int
    prompt: str
    sam_track_id: int
    max_points: int
    world_points: torch.Tensor = field(
        default_factory=lambda: torch.empty(0, 3, dtype=torch.float32)
    )
    world_weights: torch.Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.float32)
    )
    history: list[dict[str, Any]] = field(default_factory=list)
    last_seen_sequence_index: int = -1
    last_seen_frame_index: int = -1
    mask_area_ema: float = 0.0
    confidence_ema: float = 0.0

    @property
    def observation_count(self) -> int:
        return len(self.history)

    @property
    def has_history(self) -> bool:
        return bool(
            self.last_seen_sequence_index >= 0
            and int(self.world_points.shape[0]) > 0
        )

    def update(
        self,
        *,
        points: torch.Tensor,
        weights: torch.Tensor,
        mask: torch.Tensor,
        score: float,
        sequence_index: int,
        frame_index: int,
        camera: torch.Tensor,
        source: str,
        trigger_reason: str,
        recovery_applied: bool,
    ) -> None:
        points = points.detach().float().cpu().reshape(-1, 3)
        weights = weights.detach().float().cpu().reshape(-1)
        valid = torch.isfinite(points).all(dim=-1) & torch.isfinite(weights)
        points = points[valid]
        weights = weights[valid].clamp_min(0.0)
        if points.numel():
            previous_points = self.world_points.detach().float().cpu().reshape(-1, 3)
            previous_weights = self.world_weights.detach().float().cpu().reshape(-1)
            points = torch.cat([previous_points, points], dim=0)
            weights = torch.cat([previous_weights, weights], dim=0)
            points, weights = _keep_high_confidence(
                points,
                weights,
                int(self.max_points),
            )
            self.world_points = points
            self.world_weights = weights

        mask = mask.detach().cpu().bool()
        area = int(mask.sum())
        alpha = 0.25
        self.mask_area_ema = (
            float(area)
            if not self.history
            else (1.0 - alpha) * self.mask_area_ema + alpha * float(area)
        )
        value = float(score)
        self.confidence_ema = (
            value
            if not self.history
            else (1.0 - alpha) * self.confidence_ema + alpha * value
        )
        self.last_seen_sequence_index = int(sequence_index)
        self.last_seen_frame_index = int(frame_index)
        self.history.append(
            {
                "sequence_index": int(sequence_index),
                "frame_index": int(frame_index),
                "mask_pixels": area,
                "mask_bbox_xyxy": _mask_bbox(mask),
                "score": value,
                "source": str(source),
                "trigger_reason": str(trigger_reason),
                "recovery_applied": int(bool(recovery_applied)),
                "camera_world_to_camera": (
                    camera.detach().float().cpu().reshape(3, 4).tolist()
                ),
                "world_point_count_after_update": int(self.world_points.shape[0]),
            }
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "object_id": int(self.object_id),
            "prompt": str(self.prompt),
            "sam_track_id": int(self.sam_track_id),
            "observation_count": int(self.observation_count),
            "last_seen_sequence_index": int(self.last_seen_sequence_index),
            "last_seen_frame_index": int(self.last_seen_frame_index),
            "mask_area_ema": float(self.mask_area_ema),
            "confidence_ema": float(self.confidence_ema),
            "world_point_count": int(self.world_points.shape[0]),
            "history": list(self.history),
        }


class V2ObjectMemoryBank:
    """One non-merging memory entry per pre-existing V0 slot."""

    def __init__(
        self,
        *,
        track_ids: Sequence[int],
        prompts: Sequence[str],
        max_points_per_object: int,
    ) -> None:
        if len(track_ids) != len(prompts):
            raise ValueError("V2 track IDs and prompts must have equal length.")
        self.objects = {
            int(slot): V2ObjectMemoryState(
                object_id=int(slot),
                prompt=str(prompts[slot]),
                sam_track_id=int(track_ids[slot]),
                max_points=int(max_points_per_object),
            )
            for slot in range(len(track_ids))
        }

    def get(self, slot: int) -> V2ObjectMemoryState:
        try:
            return self.objects[int(slot)]
        except KeyError as exc:
            raise KeyError(f"Unknown V2 SAM slot {slot}.") from exc

    def metadata(self) -> list[dict[str, Any]]:
        return [self.objects[key].metadata() for key in sorted(self.objects)]

    def point_tensors(self) -> dict[int, dict[str, torch.Tensor]]:
        return {
            int(slot): {
                "world_points": state.world_points.detach().float().cpu(),
                "world_weights": state.world_weights.detach().float().cpu(),
            }
            for slot, state in sorted(self.objects.items())
        }


def decide_v2_trigger(
    *,
    mask: torch.Tensor,
    score: float,
    sequence_index: int,
    memory: V2ObjectMemoryState,
    config: V2RecoveryConfig,
) -> dict[str, Any]:
    """Return a causal trigger decision without consulting GT or future frames."""

    pixels = int(mask.detach().bool().sum())
    gap = (
        int(sequence_index) - int(memory.last_seen_sequence_index) - 1
        if memory.last_seen_sequence_index >= 0
        else -1
    )
    base = {
        "triggered": 0,
        "reason": "no_history",
        "raw_mask_pixels": pixels,
        "raw_score": float(score),
        "gap_since_last_seen": int(gap),
        "raw_area_ratio_vs_memory": float("nan"),
    }
    if (
        not memory.has_history
        or int(memory.world_points.shape[0]) < int(config.min_history_points)
    ):
        return base

    if pixels == 0:
        base.update(triggered=1, reason="sam_track_lost")
        return base
    if pixels < int(config.min_mask_pixels):
        base.update(triggered=1, reason="mask_quality_too_small")
        return base
    if float(score) < float(config.low_score_threshold):
        base.update(triggered=1, reason="low_sam_confidence")
        return base
    if memory.mask_area_ema > 0.0:
        ratio = pixels / max(memory.mask_area_ema, 1.0)
        base["raw_area_ratio_vs_memory"] = float(ratio)
        if ratio < float(config.min_area_ratio) or ratio > float(config.max_area_ratio):
            base.update(triggered=1, reason="mask_quality_area_outlier")
            return base
    if gap >= int(config.reentry_gap):
        base.update(triggered=1, reason="reentry_after_gap")
        return base
    base["reason"] = "normal_sam_observation"
    return base


def evaluate_recovery_candidate(
    candidate: SAM3MaskCandidate,
    *,
    proposal: RevisitCandidate,
    raw_mask: torch.Tensor,
    config: V2RecoveryConfig,
) -> dict[str, Any]:
    """Apply a conservative, GT-free acceptance gate to one SAM candidate."""

    mask = candidate.mask.detach().cpu().bool()
    proposal_mask = proposal.mask.detach().cpu().bool()
    support = proposal.supported_mask.detach().cpu().bool()
    if not support.any():
        support = proposal.projected_mask.detach().cpu().bool()
    intersection = int((mask & proposal_mask).sum())
    union = int((mask | proposal_mask).sum())
    proposal_pixels = int(proposal_mask.sum())
    support_pixels = int(support.sum())
    candidate_pixels = int(mask.sum())
    iou = intersection / union if union else 0.0
    support_recall = (
        int((mask & support).sum()) / support_pixels
        if support_pixels
        else 0.0
    )
    support_precision = (
        int((mask & support).sum()) / candidate_pixels
        if candidate_pixels
        else 0.0
    )
    area_ratio = candidate_pixels / max(proposal_pixels, 1)
    raw_pixels = int(raw_mask.detach().bool().sum())
    if raw_pixels:
        raw_area_ratio = candidate_pixels / raw_pixels
    else:
        raw_area_ratio = float("nan")
    checks = {
        "nonempty": candidate_pixels > 0,
        "score": float(candidate.score) >= float(config.min_candidate_score),
        "support_recall": support_recall
        >= float(config.min_candidate_support_recall),
        "proposal_iou": iou >= float(config.min_candidate_iou),
        "area_ratio": (
            float(config.min_candidate_area_ratio)
            <= area_ratio
            <= float(config.max_candidate_area_ratio)
        ),
    }
    accepted = all(checks.values())
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "candidate": candidate,
        "candidate_pixels": candidate_pixels,
        "candidate_score": float(candidate.score),
        "proposal_pixels": proposal_pixels,
        "support_pixels": support_pixels,
        "proposal_iou": float(iou),
        "support_recall": float(support_recall),
        "support_precision": float(support_precision),
        "candidate_area_ratio": float(area_ratio),
        "candidate_area_ratio_vs_raw": float(raw_area_ratio),
        "accepted": int(accepted),
        "reason": "accepted" if accepted else "rejected:" + ",".join(failed),
    }


def select_recovery_candidate(
    candidates: Sequence[SAM3MaskCandidate],
    *,
    proposal: RevisitCandidate,
    raw_mask: torch.Tensor,
    config: V2RecoveryConfig,
) -> dict[str, Any] | None:
    """Choose the best accepted candidate, preserving raw fallback otherwise."""

    evaluated = [
        evaluate_recovery_candidate(
            candidate,
            proposal=proposal,
            raw_mask=raw_mask,
            config=config,
        )
        for candidate in candidates
    ]
    accepted = [row for row in evaluated if int(row["accepted"])]
    if not accepted:
        return {
            "selected": None,
            "evaluated": evaluated,
            "reason": (
                "no_candidate"
                if not evaluated
                else "all_candidates_rejected"
            ),
        }
    selected = max(
        accepted,
        key=lambda row: (
            float(row["support_recall"]),
            float(row["proposal_iou"]),
            float(row["candidate_score"]),
            -int(row["candidate_pixels"]),
        ),
    )
    return {
        "selected": selected,
        "evaluated": evaluated,
        "reason": "accepted",
    }


def process_v2_sequence(
    *,
    world_points: torch.Tensor,
    confidence: torch.Tensor,
    raw_masks_output: torch.Tensor,
    raw_scores: torch.Tensor,
    raw_masks_stream: torch.Tensor,
    frame_indices: Sequence[int],
    track_ids: Sequence[int],
    track_prompts: Sequence[str],
    world_to_camera: torch.Tensor,
    intrinsics: torch.Tensor,
    source_sizes: Sequence[Sequence[int]],
    processed_size: Sequence[int],
    image_mode: str,
    config: V2RecoveryConfig,
    proposal_builder: Callable[
        [int, int, V2ObjectMemoryState], RevisitCandidate
    ],
    refine_callback: Callable[
        [int, int, str, RevisitCandidate], Sequence[SAM3MaskCandidate]
    ],
) -> dict[str, Any]:
    """Run the causal V2 policy over a frozen V0 cache.

    ``proposal_builder`` and ``refine_callback`` are called only after a
    trigger.  Consequently, a normal frame is byte-for-byte identical at both
    the output-mask and StreamVGGT-grid levels.
    """

    raw_masks_output = raw_masks_output.detach().cpu().bool()
    raw_scores = raw_scores.detach().cpu().float()
    raw_masks_stream = raw_masks_stream.detach().cpu().bool()
    world_points = world_points.detach().float().cpu()
    confidence = confidence.detach().float().cpu()
    world_to_camera = _normalize_pose_tensor(world_to_camera)
    intrinsics = intrinsics.detach().float().cpu()
    sequence, tracks = raw_masks_output.shape[:2]
    if raw_masks_output.ndim != 4:
        raise ValueError("V2 output masks must have shape [S,K,H,W].")
    if tuple(raw_scores.shape) != (sequence, tracks):
        raise ValueError("V2 raw scores do not match output masks.")
    if tuple(raw_masks_stream.shape[:2]) != (sequence, tracks):
        raise ValueError("V2 stream masks do not match output masks.")
    if world_points.ndim != 4 or world_points.shape[-1] != 3:
        raise ValueError("V2 world_points must have shape [S,H,W,3].")
    if confidence.shape != world_points.shape[:3]:
        raise ValueError("V2 confidence does not match world_points.")
    if world_points.shape[0] != sequence:
        raise ValueError("V2 pointmap/frame count mismatch.")
    if world_to_camera.shape != (sequence, 3, 4):
        raise ValueError("V2 world_to_camera must have shape [S,3,4].")
    if intrinsics.shape != (sequence, 3, 3):
        raise ValueError("V2 intrinsics must have shape [S,3,3].")
    if len(frame_indices) != sequence or len(source_sizes) != sequence:
        raise ValueError("V2 frame metadata length mismatch.")
    if len(track_ids) != tracks or len(track_prompts) != tracks:
        raise ValueError("V2 track metadata length mismatch.")
    if tuple(raw_masks_stream.shape[-2:]) != tuple(world_points.shape[1:3]):
        raise ValueError("V2 stream masks and pointmap grids disagree.")

    bank = V2ObjectMemoryBank(
        track_ids=track_ids,
        prompts=track_prompts,
        max_points_per_object=config.max_points_per_object,
    )
    final_output = raw_masks_output.clone()
    final_scores = raw_scores.clone()
    final_stream = raw_masks_stream.clone()
    history_masks_output = torch.zeros_like(final_output)
    events: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    applied_by_slot: Counter[int] = Counter()
    trigger_by_slot: Counter[int] = Counter()
    reentry_by_slot: Counter[int] = Counter()
    reentry_applied_by_slot: Counter[int] = Counter()

    for frame in range(sequence):
        for slot in range(tracks):
            memory = bank.get(slot)
            raw_mask = raw_masks_output[frame, slot]
            raw_score = float(raw_scores[frame, slot])
            decision = decide_v2_trigger(
                mask=raw_mask,
                score=raw_score,
                sequence_index=frame,
                memory=memory,
                config=config,
            )
            triggered = bool(decision["triggered"])
            if triggered:
                trigger_by_slot[slot] += 1
                reason_counts[str(decision["reason"])] += 1
                if str(decision["reason"]) == "reentry_after_gap":
                    reentry_by_slot[slot] += 1
            event: dict[str, Any] = {
                "sequence_index": int(frame),
                "frame_index": int(frame_indices[frame]),
                "slot": int(slot),
                "object_id": int(slot),
                "sam_track_id": int(track_ids[slot]),
                "prompt": str(track_prompts[slot]),
                **decision,
                "recovery_applied": 0,
                "recovery_rejected": 0,
                "proposal_available": 0,
                "proposal_reason": "not_triggered",
                "candidate_count": 0,
                "candidate_score": float("nan"),
                "proposal_iou": float("nan"),
                "support_recall": float("nan"),
                "support_precision": float("nan"),
                "candidate_area_ratio": float("nan"),
                "candidate_area_ratio_vs_raw": float("nan"),
                "recovery_reason": str(decision["reason"]),
                "normal_frame_unchanged": int(not triggered),
            }
            source = "raw_sam"
            trigger_reason = str(decision["reason"])
            if triggered:
                try:
                    proposal = proposal_builder(frame, slot, memory)
                    event["proposal_available"] = int(
                        bool(proposal.accepted and proposal.mask.any())
                    )
                    event["proposal_reason"] = str(proposal.reason)
                    event["projected_points"] = int(proposal.projected_points)
                    event["supported_points"] = int(proposal.supported_points)
                    event["projected_fraction"] = float(proposal.projected_fraction)
                    event["support_ratio"] = float(proposal.support_ratio)
                except Exception as exc:  # keep V0 fallback if a proposal fails
                    proposal = None
                    event["proposal_reason"] = f"proposal_error:{type(exc).__name__}"
                    event["recovery_reason"] = event["proposal_reason"]
                if proposal is not None and bool(proposal.accepted and proposal.mask.any()):
                    try:
                        prompt = str(track_prompts[slot]) or "object"
                        candidates = list(
                            refine_callback(frame, slot, prompt, proposal)
                        )
                        event["candidate_count"] = int(len(candidates))
                        selected = select_recovery_candidate(
                            candidates,
                            proposal=proposal,
                            raw_mask=raw_mask,
                            config=config,
                        )
                        if selected is not None and selected.get("selected") is not None:
                            row = selected["selected"]
                            candidate = row["candidate"]
                            final_output[frame, slot] = candidate.mask.detach().cpu().bool()
                            final_scores[frame, slot] = float(candidate.score)
                            final_stream[frame, slot] = output_mask_to_stream(
                                candidate.mask,
                                source_size=tuple(
                                    int(value) for value in source_sizes[frame]
                                ),
                                processed_size=tuple(
                                    int(value) for value in processed_size
                                ),
                                image_mode=str(image_mode),
                            )
                            source = "geometry_recovery"
                            applied_by_slot[slot] += 1
                            if trigger_reason == "reentry_after_gap":
                                reentry_applied_by_slot[slot] += 1
                            event.update(
                                {
                                    "recovery_applied": 1,
                                    "recovery_reason": "accepted_sam_refinement",
                                    "candidate_score": float(row["candidate_score"]),
                                    "proposal_iou": float(row["proposal_iou"]),
                                    "support_recall": float(row["support_recall"]),
                                    "support_precision": float(row["support_precision"]),
                                    "candidate_area_ratio": float(
                                        row["candidate_area_ratio"]
                                    ),
                                    "candidate_area_ratio_vs_raw": float(
                                        row["candidate_area_ratio_vs_raw"]
                                    ),
                                    "selected_gate_reason": str(row["reason"]),
                                    "normal_frame_unchanged": 0,
                                }
                            )
                        else:
                            event["recovery_rejected"] = 1
                            event["recovery_reason"] = str(
                                selected.get("reason", "all_candidates_rejected")
                                if selected is not None
                                else "no_candidate"
                            )
                    except Exception as exc:  # model/runtime fallback is raw V0
                        event["recovery_rejected"] = 1
                        event["recovery_reason"] = (
                            f"sam_refinement_error:{type(exc).__name__}"
                        )
                else:
                    event["recovery_rejected"] = 1

            final_mask = final_output[frame, slot]
            final_score = float(final_scores[frame, slot])
            history_masks_output[frame, slot] = final_mask
            unsafe_raw_trigger = (
                triggered
                and source != "geometry_recovery"
                and trigger_reason in {
                    "sam_track_lost",
                    "low_sam_confidence",
                    "mask_quality_too_small",
                    "mask_quality_area_outlier",
                }
            )
            can_update = bool(final_mask.any()) and final_score >= float(
                config.min_memory_update_score
            ) and not unsafe_raw_trigger
            memory_updated = False
            if can_update:
                points, weights = sample_masked_observation(
                    world_points[frame],
                    confidence[frame],
                    final_stream[frame, slot]
                    & (confidence[frame] >= float(config.min_geometry_confidence)),
                    max_points=int(config.max_points_per_object),
                )
                if points.numel():
                    memory.update(
                        points=points,
                        weights=weights,
                        mask=final_mask,
                        score=final_score,
                        sequence_index=frame,
                        frame_index=int(frame_indices[frame]),
                        camera=world_to_camera[frame],
                        source=source,
                        trigger_reason=trigger_reason,
                        recovery_applied=bool(event["recovery_applied"]),
                    )
                    memory_updated = True
            event["memory_updated"] = int(memory_updated)
            event["memory_point_count"] = int(memory.world_points.shape[0])
            event["memory_observation_count"] = int(memory.observation_count)
            event["last_seen_sequence_index"] = int(
                memory.last_seen_sequence_index
            )
            events.append(event)

    stats = {
        "frame_count": int(sequence),
        "slot_count": int(tracks),
        "event_count": int(len(events)),
        "recovery_trigger_count": int(sum(trigger_by_slot.values())),
        "recovery_success_count": int(sum(applied_by_slot.values())),
        "recovery_reject_count": int(
            sum(int(row["recovery_rejected"]) for row in events)
        ),
        "normal_frame_count": int(
            sum(int(not row["triggered"]) for row in events)
        ),
        "normal_frame_unchanged_count": int(
            sum(
                int(row["normal_frame_unchanged"])
                for row in events
                if not bool(row["triggered"])
            )
        ),
        "normal_frame_unchanged_ratio": _ratio(
            sum(
                int(row["normal_frame_unchanged"])
                for row in events
                if not bool(row["triggered"])
            ),
            sum(int(not row["triggered"]) for row in events),
        ),
        "trigger_reason_counts": dict(sorted(reason_counts.items())),
        "per_object_trigger_count": {
            str(slot): int(trigger_by_slot.get(slot, 0))
            for slot in range(tracks)
        },
        "per_object_recovery_count": {
            str(slot): int(applied_by_slot.get(slot, 0))
            for slot in range(tracks)
        },
        "per_object_reentry_count": {
            str(slot): int(reentry_by_slot.get(slot, 0))
            for slot in range(tracks)
        },
        "per_object_reentry_recovery_count": {
            str(slot): int(reentry_applied_by_slot.get(slot, 0))
            for slot in range(tracks)
        },
    }
    return {
        "masks_output": final_output,
        "scores": final_scores,
        "masks_stream": final_stream,
        "history_masks_output": history_masks_output,
        "events": events,
        "objects": bank.metadata(),
        "object_tensors": bank.point_tensors(),
        "stats": stats,
    }


def _normalize_pose_tensor(value: torch.Tensor) -> torch.Tensor:
    pose = value.detach().float().cpu()
    if pose.ndim == 4 and pose.shape[0] == 1:
        pose = pose[0]
    if pose.ndim != 3 or tuple(pose.shape[-2:]) != (3, 4):
        raise ValueError(f"Expected pose [S,3,4] or [1,S,3,4], got {tuple(pose.shape)}")
    return pose


def _keep_high_confidence(
    points: torch.Tensor,
    weights: torch.Tensor,
    limit: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if points.shape[0] <= int(limit):
        return points, weights
    indices = torch.topk(weights, k=int(limit), sorted=False).indices
    return points.index_select(0, indices), weights.index_select(0, indices)


def _mask_bbox(mask: torch.Tensor) -> list[int] | None:
    if not bool(mask.any()):
        return None
    ys, xs = mask.nonzero(as_tuple=True)
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _ratio(numerator: int, denominator: int) -> float:
    # A sequence with no normal frames satisfies the exact-copy invariant
    # vacuously; keeping this finite also makes the shell preflight robust.
    return float(numerator) / float(denominator) if int(denominator) else 1.0


__all__ = [
    "V2RecoveryConfig",
    "V2ObjectMemoryState",
    "V2ObjectMemoryBank",
    "decide_v2_trigger",
    "evaluate_recovery_candidate",
    "select_recovery_candidate",
    "process_v2_sequence",
]
