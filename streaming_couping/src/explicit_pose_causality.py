"""No-training instance geometry to camera-centre refinement.

The module intentionally contains no learnable parameters.  SAM masks choose
regions and persistent identities; StreamVGGT supplies world points, point
confidence, raw rotation and intrinsics; the retained V5 angular ray solver
produces the final bounded camera-centre correction.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch

from .instance_observations import (
    InstanceRefinementConfig,
    TranslationProposal,
    deterministic_limit,
    merge_map_points,
    proposal_consensus,
    translation_icp,
)
from .learned_pose.config import RayPoseConfig
from .learned_pose.ray_pose import _fit_angular_huber_center
from .pose_evaluation import _prepare_ray_inputs, _ray_residuals
from .v75_explicit_protocol import ExplicitVariant


@dataclass(frozen=True)
class ExplicitPoseConfig:
    point_confidence_threshold: float
    track_confidence_threshold: float
    matched_point_budget: int
    max_center_shift: float
    center_blend: float
    ray: RayPoseConfig
    refinement: InstanceRefinementConfig


@dataclass(frozen=True)
class ExplicitPoseResult:
    world_to_camera: torch.Tensor
    active_frames: torch.Tensor
    diagnostics: tuple[dict[str, object], ...]


def run_explicit_pose_variant(
    *,
    variant: ExplicitVariant,
    memory_mode: str,
    frame_indices: Sequence[int],
    history_indices: Sequence[int],
    test_indices: Sequence[int],
    baseline_world_to_camera: torch.Tensor,
    intrinsics: torch.Tensor,
    world_points: torch.Tensor,
    confidence: torch.Tensor,
    sam_masks: torch.Tensor,
    sam_scores: torch.Tensor,
    gt_masks: torch.Tensor | None,
    config: ExplicitPoseConfig,
) -> ExplicitPoseResult:
    """Run one strictly chronological explicit-pose variant.

    Persistent tensors and the ray solve stay on the host.  Only bounded ICP
    temporarily uses ``config.refinement.compute_device``.
    """

    _validate_inputs(
        memory_mode=memory_mode,
        frame_indices=frame_indices,
        history_indices=history_indices,
        test_indices=test_indices,
        baseline_world_to_camera=baseline_world_to_camera,
        intrinsics=intrinsics,
        world_points=world_points,
        confidence=confidence,
        sam_masks=sam_masks,
        sam_scores=sam_scores,
        gt_masks=gt_masks,
        variant=variant,
    )
    baseline = baseline_world_to_camera.detach().double().cpu()
    intrinsics = intrinsics.detach().double().cpu()
    world_points = world_points.detach().float().cpu()
    confidence = _squeeze_confidence(confidence).detach().float().cpu()
    sam_masks = sam_masks.detach().bool().cpu()
    sam_scores = sam_scores.detach().float().cpu()
    gt_masks = None if gt_masks is None else gt_masks.detach().bool().cpu()
    output = baseline.clone()
    active = torch.zeros(len(frame_indices), dtype=torch.bool)
    history_set = {int(value) for value in history_indices}
    test_set = {int(value) for value in test_indices}
    last_required = max(history_set | test_set)
    object_maps: dict[int, torch.Tensor] = {}
    diagnostics: list[dict[str, object]] = []

    for sequence_index in range(last_required + 1):
        raw_masks, raw_scores = _region_masks(
            variant.region,
            sequence_index=sequence_index,
            sam_masks=sam_masks,
            sam_scores=sam_scores,
            gt_masks=gt_masks,
        )
        current_points = _instance_points(
            world_points[sequence_index],
            confidence[sequence_index],
            raw_masks,
            threshold=config.point_confidence_threshold,
            limit=config.refinement.icp_max_points,
        )
        needs_memory = variant.history != "none"
        correct_proposals = (
            _proposals(
                current_points=current_points,
                scores=raw_scores,
                object_maps=object_maps,
                history_source="correct",
                config=config,
            )
            if needs_memory
            else []
        )
        if variant.history == "correct":
            # The proposal used for prediction is exactly the proposal needed
            # for the subsequent causal map update.  Reusing it avoids a
            # duplicate GPU cdist without changing any result.
            pose_proposals = correct_proposals
        elif variant.history == "wrong_id":
            pose_proposals = _proposals(
                current_points=current_points,
                scores=raw_scores,
                object_maps=object_maps,
                history_source="wrong_id",
                config=config,
            )
        else:
            pose_proposals = []
        shared, participating, disagreement = _shared_translation(
            pose_proposals,
            consensus_distance=config.refinement.consensus_distance,
        )
        is_test = sequence_index in test_set
        budget_mask = (
            raw_masks.any(dim=0)
            if variant.region == "gt"
            else sam_masks[sequence_index].any(dim=0)
        )
        budget = _matched_budget(
            world_points[sequence_index],
            confidence[sequence_index],
            budget_mask,
            threshold=config.point_confidence_threshold,
            maximum=config.matched_point_budget,
        )
        row = _base_diagnostic(
            variant=variant,
            memory_mode=memory_mode,
            sequence_index=sequence_index,
            frame_index=int(frame_indices[sequence_index]),
            is_history=sequence_index in history_set,
            is_test=is_test,
            raw_masks=raw_masks,
            object_maps=object_maps,
            pose_proposals=pose_proposals,
            shared=shared,
            participating=participating,
            disagreement=disagreement,
            budget=budget,
        )

        should_solve = is_test and variant.region != "none"
        if variant.history != "none" and shared is None:
            should_solve = False
            row["fit_status"] = "fallback_no_history_consensus"
        if should_solve:
            center, accepted, fit = _solve_frame_center(
                baseline_world_to_camera=baseline[sequence_index],
                intrinsics=intrinsics[sequence_index],
                world_points=world_points[sequence_index],
                confidence=confidence[sequence_index],
                masks=raw_masks,
                translation=(
                    torch.zeros(3, dtype=torch.float64)
                    if shared is None
                    else shared
                ),
                matched_budget=budget,
                config=config,
            )
            row.update(fit)
            if accepted:
                rotation = baseline[sequence_index, :3, :3]
                output[sequence_index, :3, :3] = rotation
                output[sequence_index, :3, 3] = -(rotation @ center)
                active[sequence_index] = True
        elif is_test and variant.region == "none":
            row["fit_status"] = "raw_exact_fallback"

        can_update = (
            sequence_index in history_set
            or (
                sequence_index in test_set
                and memory_mode == "online"
            )
        )
        if variant.history != "none" and can_update:
            births, updates = _update_maps(
                object_maps=object_maps,
                current_points=current_points,
                scores=raw_scores,
                correct_proposals=correct_proposals,
                config=config,
            )
            row["map_birth_slots"] = _int_text(births)
            row["map_update_slots"] = _int_text(updates)
        diagnostics.append(row)

    return ExplicitPoseResult(
        world_to_camera=output,
        active_frames=active,
        diagnostics=tuple(diagnostics),
    )


def _solve_frame_center(
    *,
    baseline_world_to_camera: torch.Tensor,
    intrinsics: torch.Tensor,
    world_points: torch.Tensor,
    confidence: torch.Tensor,
    masks: torch.Tensor,
    translation: torch.Tensor,
    matched_budget: int,
    config: ExplicitPoseConfig,
) -> tuple[torch.Tensor, bool, dict[str, object]]:
    rotation = baseline_world_to_camera[:3, :3].double()
    fallback_center = -(rotation.T @ baseline_world_to_camera[:3, 3].double())
    if matched_budget < int(config.ray.min_points):
        return fallback_center, False, {
            "fit_status": (
                f"fallback_matched_budget:{matched_budget}"
                f"<{int(config.ray.min_points)}"
            ),
            "candidate_points": matched_budget,
            "sampled_points": 0,
            "retained_points": 0,
            "solve_iterations": 0,
            "condition_number": float("nan"),
            "initial_ray_rmse": float("nan"),
            "fitted_ray_rmse": float("nan"),
            "proposed_center_shift": 0.0,
            "applied_center_shift": 0.0,
        }
    union = masks.any(dim=0)
    masked_confidence = torch.where(
        union,
        confidence,
        torch.full_like(confidence, float("-inf")),
    )
    points, directions, weights, candidate_points = _prepare_ray_inputs(
        world_points,
        masked_confidence,
        intrinsics,
        rotation.T,
        confidence_threshold=config.point_confidence_threshold,
        max_points=max(int(config.ray.min_points), int(matched_budget)),
    )
    points = points + translation.double()[None]
    initial = _rmse(_ray_residuals(points, directions, fallback_center))
    fit = _fit_angular_huber_center(
        points,
        directions,
        weights,
        candidate_points=candidate_points,
        fallback_center=fallback_center,
        config=config.ray,
    )
    proposed = fit["center"].double()
    proposed_shift = float(torch.linalg.vector_norm(proposed - fallback_center))
    fitted_rmse = float(fit["point_residual_rmse"])
    accepted = (
        bool(fit["solver_accepted"])
        and math.isfinite(fitted_rmse)
        and fitted_rmse <= float(config.ray.max_residual_rmse)
        and proposed_shift <= float(config.max_center_shift)
    )
    if accepted:
        applied = fallback_center + float(config.center_blend) * (
            proposed - fallback_center
        )
        status = "accepted"
    else:
        applied = fallback_center
        reasons = [str(fit["status"])]
        if not math.isfinite(fitted_rmse):
            reasons.append("non_finite_residual")
        elif fitted_rmse > float(config.ray.max_residual_rmse):
            reasons.append("ray_residual_above_limit")
        if proposed_shift > float(config.max_center_shift):
            reasons.append("center_shift_above_limit")
        status = ";".join(dict.fromkeys(reasons))
    return applied, accepted, {
        "fit_status": status,
        "candidate_points": int(candidate_points),
        "sampled_points": int(points.shape[0]),
        "retained_points": int(fit["retained_points"]),
        "solve_iterations": int(fit["solve_iterations"]),
        "condition_number": float(fit["condition_number"]),
        "initial_ray_rmse": initial,
        "fitted_ray_rmse": fitted_rmse,
        "proposed_center_shift": proposed_shift,
        "applied_center_shift": float(
            torch.linalg.vector_norm(applied - fallback_center)
        ),
    }


def _region_masks(
    region: str,
    *,
    sequence_index: int,
    sam_masks: torch.Tensor,
    sam_scores: torch.Tensor,
    gt_masks: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if region == "none":
        return (
            torch.zeros(
                1,
                *sam_masks.shape[-2:],
                dtype=torch.bool,
            ),
            torch.zeros(1),
        )
    if region == "sam":
        return sam_masks[sequence_index].clone(), sam_scores[sequence_index].clone()
    if region == "bbox":
        return _bbox_masks(sam_masks[sequence_index]), sam_scores[sequence_index].clone()
    if region == "random":
        return (
            _random_shift_masks(sam_masks[sequence_index]),
            sam_scores[sequence_index].clone(),
        )
    if region == "stale":
        stale_index = max(0, int(sequence_index) - 1)
        return sam_masks[stale_index].clone(), sam_scores[stale_index].clone()
    if region == "full":
        full = torch.ones(1, *sam_masks.shape[-2:], dtype=torch.bool)
        return full, torch.ones(1)
    if region == "gt":
        if gt_masks is None:
            raise ValueError("GT oracle variant requires gt_masks.")
        return gt_masks[sequence_index].clone(), torch.ones(gt_masks.shape[1])
    raise ValueError(f"Unknown explicit-pose region={region!r}.")


def _bbox_masks(masks: torch.Tensor) -> torch.Tensor:
    output = torch.zeros_like(masks)
    for slot, mask in enumerate(masks):
        positions = torch.nonzero(mask, as_tuple=False)
        if not positions.numel():
            continue
        y0, x0 = positions.min(dim=0).values.tolist()
        y1, x1 = positions.max(dim=0).values.tolist()
        output[slot, int(y0) : int(y1) + 1, int(x0) : int(x1) + 1] = True
    return output


def _random_shift_masks(masks: torch.Tensor) -> torch.Tensor:
    """Preserve exact area while selecting the least-overlapping fixed roll."""

    height, width = masks.shape[-2:]
    candidates = (
        (max(1, height // 3), 0),
        (0, max(1, width // 3)),
        (max(1, height // 3), max(1, width // 3)),
        (max(1, height // 2), max(1, width // 2)),
    )
    output = torch.zeros_like(masks)
    for slot, mask in enumerate(masks):
        if not bool(mask.any()):
            continue
        shifted = [torch.roll(mask, shifts=value, dims=(-2, -1)) for value in candidates]
        output[slot] = min(
            shifted,
            key=lambda value: int((value & mask).sum()),
        )
    return output


def _instance_points(
    world_points: torch.Tensor,
    confidence: torch.Tensor,
    masks: torch.Tensor,
    *,
    threshold: float,
    limit: int,
) -> list[torch.Tensor]:
    output = []
    finite = torch.isfinite(world_points).all(dim=-1) & torch.isfinite(confidence)
    for mask in masks:
        valid = mask & finite & confidence.ge(float(threshold))
        points = world_points[valid]
        output.append(deterministic_limit(points, int(limit)).float().cpu())
    return output


def _proposals(
    *,
    current_points: Sequence[torch.Tensor],
    scores: torch.Tensor,
    object_maps: dict[int, torch.Tensor],
    history_source: str,
    config: ExplicitPoseConfig,
) -> list[TranslationProposal]:
    if history_source == "none" or not object_maps:
        return []
    keys = sorted(object_maps)
    proposals = []
    for slot, points in enumerate(current_points):
        if slot >= scores.numel() or float(scores[slot]) < float(
            config.track_confidence_threshold
        ):
            continue
        map_slot = slot
        if history_source == "wrong_id":
            if len(keys) < 2 or slot not in keys:
                continue
            position = keys.index(slot)
            map_slot = keys[(position + 1) % len(keys)]
        elif history_source != "correct":
            raise ValueError(f"Unknown history source={history_source!r}.")
        if map_slot not in object_maps:
            continue
        proposals.append(
            translation_icp(
                points,
                object_maps[map_slot],
                instance_id=slot,
                config=config.refinement,
            )
        )
    return proposals


def _shared_translation(
    proposals: Sequence[TranslationProposal],
    *,
    consensus_distance: float,
) -> tuple[torch.Tensor | None, tuple[int, ...], float]:
    accepted = [proposal for proposal in proposals if proposal.accepted]
    if not accepted:
        return None, (), float("nan")
    if len(accepted) >= 2:
        shared, participating, disagreement = proposal_consensus(
            accepted,
            min_instances=2,
            max_distance=float(consensus_distance),
        )
        if shared is not None:
            return shared.double(), participating, float(disagreement)
    best = max(
        accepted,
        key=lambda value: (
            float(value.fitness),
            -float(value.rmse) if math.isfinite(value.rmse) else -1e30,
        ),
    )
    return best.translation.double(), (int(best.instance_id),), float("nan")


def _update_maps(
    *,
    object_maps: dict[int, torch.Tensor],
    current_points: Sequence[torch.Tensor],
    scores: torch.Tensor,
    correct_proposals: Sequence[TranslationProposal],
    config: ExplicitPoseConfig,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    by_slot = {int(proposal.instance_id): proposal for proposal in correct_proposals}
    births = []
    updates = []
    for slot, points in enumerate(current_points):
        if slot >= scores.numel() or float(scores[slot]) < float(
            config.track_confidence_threshold
        ):
            continue
        if int(points.shape[0]) < int(config.refinement.min_instance_points):
            continue
        if slot not in object_maps:
            object_maps[slot] = deterministic_limit(
                points,
                config.refinement.map_max_points,
            )
            births.append(slot)
            continue
        proposal = by_slot.get(slot)
        if proposal is None or not proposal.accepted:
            continue
        corrected = points + proposal.translation.float()[None]
        object_maps[slot] = merge_map_points(
            object_maps[slot],
            corrected,
            max_points=config.refinement.map_max_points,
        )
        updates.append(slot)
    return tuple(births), tuple(updates)


def _matched_budget(
    world_points: torch.Tensor,
    confidence: torch.Tensor,
    sam_union: torch.Tensor,
    *,
    threshold: float,
    maximum: int,
) -> int:
    valid = (
        sam_union
        & torch.isfinite(world_points).all(dim=-1)
        & torch.isfinite(confidence)
        & confidence.ge(float(threshold))
    )
    return min(int(valid.sum()), int(maximum))


def _base_diagnostic(
    *,
    variant: ExplicitVariant,
    memory_mode: str,
    sequence_index: int,
    frame_index: int,
    is_history: bool,
    is_test: bool,
    raw_masks: torch.Tensor,
    object_maps: dict[int, torch.Tensor],
    pose_proposals: Sequence[TranslationProposal],
    shared: torch.Tensor | None,
    participating: Sequence[int],
    disagreement: float,
    budget: int,
) -> dict[str, object]:
    accepted = [proposal for proposal in pose_proposals if proposal.accepted]
    fitness = [float(proposal.fitness) for proposal in accepted]
    rmse = [float(proposal.rmse) for proposal in accepted if math.isfinite(proposal.rmse)]
    return {
        "variant": variant.name,
        "memory_mode": memory_mode,
        "sequence_index": sequence_index,
        "frame_index": frame_index,
        "is_history": int(is_history),
        "is_test": int(is_test),
        "observed_instances": int(raw_masks.flatten(1).any(dim=1).sum()),
        "map_instances_before": len(object_maps),
        "proposal_instances": len(pose_proposals),
        "accepted_proposals": len(accepted),
        "participating_instances": len(participating),
        "participating_slots": _int_text(participating),
        "mean_icp_fitness": _mean(fitness),
        "mean_icp_rmse": _mean(rmse),
        "consensus_disagreement": disagreement,
        "shared_translation_norm": (
            float(torch.linalg.vector_norm(shared))
            if shared is not None
            else 0.0
        ),
        "matched_point_budget": budget,
        "candidate_points": 0,
        "sampled_points": 0,
        "retained_points": 0,
        "solve_iterations": 0,
        "condition_number": float("nan"),
        "initial_ray_rmse": float("nan"),
        "fitted_ray_rmse": float("nan"),
        "proposed_center_shift": 0.0,
        "applied_center_shift": 0.0,
        "fit_status": "not_evaluated",
        "map_birth_slots": "",
        "map_update_slots": "",
    }


def _validate_inputs(
    *,
    memory_mode: str,
    frame_indices: Sequence[int],
    history_indices: Sequence[int],
    test_indices: Sequence[int],
    baseline_world_to_camera: torch.Tensor,
    intrinsics: torch.Tensor,
    world_points: torch.Tensor,
    confidence: torch.Tensor,
    sam_masks: torch.Tensor,
    sam_scores: torch.Tensor,
    gt_masks: torch.Tensor | None,
    variant: ExplicitVariant,
) -> None:
    sequence = len(frame_indices)
    if memory_mode not in {"none", "current_only", "frozen", "online"}:
        raise ValueError(f"Unknown memory_mode={memory_mode!r}.")
    if variant.history == "none":
        expected_mode = "none" if variant.region == "none" else "current_only"
        if memory_mode != expected_mode:
            raise ValueError(
                f"Variant {variant.name!r} requires memory_mode={expected_mode!r}."
            )
    elif memory_mode not in {"frozen", "online"}:
        raise ValueError(
            f"History variant {variant.name!r} requires frozen or online memory."
        )
    if baseline_world_to_camera.shape != (sequence, 3, 4):
        raise ValueError("baseline_world_to_camera must be [S,3,4].")
    if intrinsics.shape != (sequence, 3, 3):
        raise ValueError("intrinsics must be [S,3,3].")
    if world_points.ndim != 4 or world_points.shape[0] != sequence or world_points.shape[-1] != 3:
        raise ValueError("world_points must be [S,H,W,3].")
    if _squeeze_confidence(confidence).shape != world_points.shape[:3]:
        raise ValueError("confidence/world point shapes disagree.")
    if sam_masks.ndim != 4 or sam_masks.shape[0] != sequence:
        raise ValueError("sam_masks must be [S,K,H,W].")
    if sam_masks.shape[-2:] != world_points.shape[1:3]:
        raise ValueError("SAM mask/world point grids disagree.")
    if sam_scores.shape != sam_masks.shape[:2]:
        raise ValueError("sam_scores must be [S,K].")
    if variant.region == "gt" and gt_masks is None:
        raise ValueError("GT oracle variant requires gt_masks.")
    if gt_masks is not None and (
        gt_masks.ndim != 4
        or gt_masks.shape[0] != sequence
        or gt_masks.shape[-2:] != world_points.shape[1:3]
    ):
        raise ValueError("gt_masks must be [S,Kgt,H,W].")
    if not history_indices or not test_indices:
        raise ValueError("history/test indices must be non-empty.")
    history_values = tuple(int(value) for value in history_indices)
    test_values = tuple(int(value) for value in test_indices)
    if history_values != tuple(sorted(set(history_values))):
        raise ValueError("history indices must be sorted and unique.")
    if test_values != tuple(sorted(set(test_values))):
        raise ValueError("test indices must be sorted and unique.")
    valid_indices = set(range(sequence))
    if not set(history_values).issubset(valid_indices):
        raise ValueError("history indices are outside the sequence.")
    if not set(test_values).issubset(valid_indices):
        raise ValueError("test indices are outside the sequence.")
    if max(history_indices) >= min(test_indices):
        raise ValueError("history must end before test begins.")
    if not torch.isfinite(baseline_world_to_camera).all():
        raise ValueError("baseline camera poses contain non-finite values.")
    if not torch.isfinite(intrinsics).all():
        raise ValueError("camera intrinsics contain non-finite values.")


def _squeeze_confidence(value: torch.Tensor) -> torch.Tensor:
    if value.ndim == 4 and value.shape[-1] == 1:
        return value[..., 0]
    return value


def _rmse(values: torch.Tensor) -> float:
    finite = values[torch.isfinite(values)]
    return (
        float(torch.sqrt(finite.square().mean()))
        if finite.numel()
        else float("nan")
    )


def _mean(values: Sequence[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return sum(finite) / len(finite) if finite else float("nan")


def _int_text(values: Sequence[int]) -> str:
    return " ".join(str(int(value)) for value in values)
