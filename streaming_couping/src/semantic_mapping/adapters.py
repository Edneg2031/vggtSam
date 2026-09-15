"""Adapters from the current V0 artifacts and model wrappers to the contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np
import torch
from PIL import Image

from ..horizonstream_cache import validate_horizonstream_cache
from ..types import GeometrySequence
from .contracts import GeometryFrame, ObjectObservation, SegmentationFrame
from .geometry_guidance import (
    CausalObjectGeometryMemory,
    GeometryGuidanceConfig,
    build_geometry_prompt,
    choose_geometry_candidate,
    evaluate_geometry_mask,
    gate_geometry_replacement,
    summarize_diagnostics,
)

if TYPE_CHECKING:
    from ..backbones.sam3_wrapper import SAM3Wrapper
    from ..backbones.streamvggt_wrapper import StreamVGGTWrapper


class StreamVGGTGeometryAdapter:
    """Expose the existing StreamVGGT wrapper through ``GeometryProvider``."""

    backend_name = "streamvggt"

    def __init__(
        self,
        wrapper: StreamVGGTWrapper,
        *,
        scale_type: str = "unknown",
    ) -> None:
        self.wrapper = wrapper
        self.scale_type = str(scale_type)

    def infer(
        self,
        image_paths: Sequence[str | Path],
    ) -> tuple[GeometryFrame, ...]:
        paths = tuple(Path(path) for path in image_paths)
        sequence = self.wrapper.extract(paths)
        return geometry_frames_from_sequence(
            sequence,
            paths,
            backend=self.backend_name,
            scale_type=self.scale_type,
        )


class HorizonStreamGeometryCacheAdapter:
    """Expose isolated HorizonStream depth and poses as canonical geometry."""

    backend_name = "horizonstream"

    def __init__(self, payload: Mapping[str, Any]) -> None:
        validate_horizonstream_cache(payload)
        self.payload = payload
        self.depth = torch.as_tensor(payload["depth"]).detach().float().cpu()
        if self.depth.ndim == 4 and self.depth.shape[-1] == 1:
            self.depth = self.depth[..., 0]
        self.confidence = (
            torch.as_tensor(payload["confidence"]).detach().float().cpu()
        )
        if self.confidence.ndim == 4 and self.confidence.shape[-1] == 1:
            self.confidence = self.confidence[..., 0]
        self.world_to_camera = (
            torch.as_tensor(payload["world_to_camera"]).detach().float().cpu()
        )
        self.intrinsics = (
            torch.as_tensor(payload["intrinsics"]).detach().float().cpu()
        )
        self.rgb = (
            torch.as_tensor(payload["processed_rgb"])
            .detach()
            .float()
            .cpu()
            .div(255.0)
        )
        self.image_paths = tuple(Path(path) for path in payload["image_paths"])
        self.source_sizes = tuple(
            tuple(int(value) for value in size)
            for size in payload["source_sizes"]
        )
        self.frame_ids = tuple(
            int(value)
            for value in payload.get("frame_ids", range(int(self.depth.shape[0])))
        )
        if self.frame_ids != tuple(range(int(self.depth.shape[0]))):
            raise ValueError(
                "HorizonStream cache frame_ids must be zero-based and contiguous."
            )

    @property
    def image_size(self) -> tuple[int, int]:
        return int(self.depth.shape[1]), int(self.depth.shape[2])

    def infer(
        self,
        image_paths: Sequence[str | Path],
    ) -> tuple[GeometryFrame, ...]:
        if len(image_paths) != len(self.frame_ids):
            raise ValueError(
                "Image count differs from HorizonStream geometry cache frame count."
            )
        height, width = self.image_size
        output = []
        for index, frame_id in enumerate(self.frame_ids):
            depth = self.depth[index]
            output.append(
                GeometryFrame(
                    frame_id=frame_id,
                    image_size=(height, width),
                    depth=depth,
                    intrinsics=self.intrinsics[index],
                    camera_to_world=_invert_world_to_camera(
                        self.world_to_camera[index]
                    ),
                    confidence=self.confidence[index],
                    valid=torch.isfinite(depth) & (depth > 0.0),
                    rgb=self.rgb[index],
                    scale_type="metric",
                    backend=self.backend_name,
                    metadata={
                        "source_image": str(self.image_paths[index]),
                        "source_size": self.source_sizes[index],
                        "processed_size": (height, width),
                        "pose_convention_source": "world_to_camera",
                        "pose_source": str(
                            self.payload.get("pose_source", "online_motion_averaged")
                        ),
                        "cache_schema_version": int(
                            self.payload.get("schema_version", 0)
                        ),
                    },
                )
            )
        return tuple(output)


def geometry_frames_from_sequence(
    sequence: GeometrySequence,
    image_paths: Sequence[str | Path],
    *,
    backend: str = "streamvggt",
    scale_type: str = "unknown",
) -> tuple[GeometryFrame, ...]:
    """Convert the repository's legacy sequence object to canonical frames."""

    points = sequence.world_points.detach().float().cpu()
    confidence = sequence.confidence.detach().float().cpu()
    poses = sequence.world_to_camera.detach().float().cpu()
    intrinsics = sequence.intrinsics.detach().float().cpu()
    paths = tuple(Path(path) for path in image_paths)
    if points.ndim != 4 or points.shape[-1] != 3:
        raise ValueError("GeometrySequence.world_points must have shape [S,H,W,3].")
    frame_count, height, width, _ = points.shape
    if len(paths) != frame_count:
        raise ValueError(
            f"Geometry frame count {frame_count} differs from image count {len(paths)}."
        )
    if tuple(confidence.shape) != (frame_count, height, width):
        raise ValueError("GeometrySequence.confidence is not aligned to world_points.")
    if tuple(poses.shape) != (frame_count, 3, 4):
        raise ValueError("GeometrySequence.world_to_camera must have shape [S,3,4].")
    if tuple(intrinsics.shape) != (frame_count, 3, 3):
        raise ValueError("GeometrySequence.intrinsics must have shape [S,3,3].")

    frames = []
    for frame_id in range(frame_count):
        frame_points = points[frame_id]
        frames.append(
            GeometryFrame(
                frame_id=frame_id,
                image_size=(height, width),
                world_points=frame_points,
                camera_to_world=_invert_world_to_camera(poses[frame_id]),
                intrinsics=intrinsics[frame_id],
                confidence=confidence[frame_id],
                valid=torch.isfinite(frame_points).all(dim=-1),
                rgb=_load_rgb(paths[frame_id], (height, width)),
                scale_type=scale_type,
                backend=backend,
                metadata={
                    "source_image": str(paths[frame_id]),
                    "source_size": tuple(
                        int(value) for value in sequence.source_sizes[frame_id]
                    ),
                    "processed_size": tuple(
                        int(value) for value in sequence.processed_size
                    ),
                    "pose_convention_source": "world_to_camera",
                },
            )
        )
    return tuple(frames)


class SAM31SegmentationAdapter:
    """Run causal SAM3.1 text tracking and emit canonical observations.

    Each prompt is run in its own forward-only SAM3.1 session.  Tracks from
    different prompts are assigned permanent IDs in first-seen order, with a
    conservative birth-time overlap deduplication pass.  The map layer does
    not depend on SAM-specific object IDs.
    """

    backend_name = "sam3.1_forward_causal"

    def __init__(
        self,
        wrapper: SAM3Wrapper,
        *,
        output_size: tuple[int, int],
        max_objects_per_prompt: int = 16,
        max_total_objects: int = 16,
        min_birth_pixels: int = 128,
        duplicate_iou: float = 0.80,
    ) -> None:
        self.wrapper = wrapper
        self.output_size = _size(output_size)
        self.max_objects_per_prompt = int(max_objects_per_prompt)
        self.max_total_objects = int(max_total_objects)
        self.min_birth_pixels = int(min_birth_pixels)
        self.duplicate_iou = float(duplicate_iou)
        #: What every prompt's tracks became, filled in by ``infer``.  Empty
        #: until a run happens, so readers can always getattr it.
        self.candidate_ledger: list[dict[str, Any]] = []
        self.candidate_ledger_summary: dict[str, Any] = ledger_summary([])
        if self.max_objects_per_prompt < 1 or self.max_total_objects < 1:
            raise ValueError("SAM object limits must be positive.")
        if self.min_birth_pixels < 1:
            raise ValueError("min_birth_pixels must be positive.")
        if not 0.0 <= self.duplicate_iou <= 1.0:
            raise ValueError("duplicate_iou must be in [0,1].")

    def infer(
        self,
        image_paths: Sequence[str | Path],
        prompts: Sequence[str] | None = None,
    ) -> tuple[SegmentationFrame, ...]:
        paths = tuple(Path(path) for path in image_paths)
        if not paths:
            raise ValueError("SAM31SegmentationAdapter requires image paths.")
        normalized_prompts = _normalize_prompts(prompts)
        if not normalized_prompts:
            raise ValueError("At least one text prompt is required for SAM3.1.")

        candidates: list[dict[str, Any]] = []
        # Every track a prompt produced, and what became of it.  A prompt can
        # return masks on every frame and still contribute nothing -- dropped
        # as too small at birth, or as a duplicate of an earlier-born track, or
        # beyond the object cap.  All three are silent, and the surviving
        # observations cannot tell them apart from "the prompt returned
        # nothing", which are opposite diagnoses with opposite fixes.
        ledger: list[dict[str, Any]] = []
        for prompt_index, prompt in enumerate(normalized_prompts):
            tracked = self.wrapper.track_all_forward(
                paths,
                prompt=prompt,
                output_size=self.output_size,
                max_objects=self.max_objects_per_prompt,
            )
            masks = tracked.masks.detach().cpu().bool()
            scores = tracked.scores.detach().cpu().float()
            if masks.ndim != 4 or masks.shape[0] != len(paths):
                raise ValueError("SAM3.1 returned masks with an invalid shape.")
            for source_slot, source_obj_id in enumerate(tracked.obj_ids):
                birth = int(tracked.birth_indices[source_slot])
                entry: dict[str, Any] = {
                    "prompt": prompt,
                    "prompt_index": int(prompt_index),
                    "source_obj_id": int(source_obj_id),
                    "birth": birth,
                }
                if birth < 0 or birth >= len(paths):
                    ledger.append({**entry, "outcome": "birth_out_of_range"})
                    continue
                candidate_masks = masks[:, source_slot]
                birth_pixels = int(candidate_masks[birth].sum())
                birth_ratio = float(candidate_masks[birth].float().mean())
                entry["birth_pixels"] = birth_pixels
                entry["birth_area_ratio"] = birth_ratio
                if birth_pixels < self.min_birth_pixels or birth_ratio > 0.90:
                    ledger.append({**entry, "outcome": "birth_mask_rejected"})
                    continue
                candidates.append(
                    {
                        "prompt_index": int(prompt_index),
                        "prompt": prompt,
                        "source_obj_id": int(source_obj_id),
                        "birth": birth,
                        "masks": candidate_masks,
                        "scores": scores[:, source_slot],
                        "ledger_index": len(ledger),
                    }
                )
                ledger.append({**entry, "outcome": "candidate"})

        candidates.sort(
            key=lambda item: (
                int(item["birth"]),
                int(item["prompt_index"]),
                int(item["source_obj_id"]),
            )
        )
        accepted: list[dict[str, Any]] = []
        for candidate in candidates:
            duplicate_of = next(
                (
                    previous
                    for previous in accepted
                    if _tracks_duplicate(candidate, previous, self.duplicate_iou)
                ),
                None,
            )
            if duplicate_of is not None:
                # Naming the track it lost to is the whole point: which of two
                # overlapping tracks survives is decided by birth frame, so a
                # category can disappear because another prompt's track for the
                # same pixels was born earlier.
                ledger[candidate["ledger_index"]].update(
                    {
                        "outcome": "duplicate",
                        "duplicate_of_prompt": duplicate_of["prompt"],
                        "duplicate_of_source_obj_id": int(duplicate_of["source_obj_id"]),
                        "duplicate_of_birth": int(duplicate_of["birth"]),
                    }
                )
                continue
            accepted.append(candidate)
            ledger[candidate["ledger_index"]]["outcome"] = "accepted"
            if len(accepted) >= self.max_total_objects:
                break
        # The cap breaks out of the loop, so the candidates behind the cut were
        # never examined and would otherwise keep the provisional outcome.
        for entry in ledger:
            if entry["outcome"] == "candidate":
                entry["outcome"] = "over_object_cap"

        # Run-level, so it hangs off the adapter rather than being repeated on
        # every frame -- the pipeline already collects adapter summaries this
        # way for the geometry guidance layer.
        self.candidate_ledger = ledger
        self.candidate_ledger_summary = ledger_summary(ledger, normalized_prompts)

        observations_by_frame: list[list[ObjectObservation]] = [
            [] for _ in paths
        ]
        for instance_id, candidate in enumerate(accepted):
            masks = candidate["masks"]
            scores = candidate["scores"]
            for frame_id in range(len(paths)):
                mask = masks[frame_id]
                if not bool(mask.any()):
                    continue
                score = float(torch.nan_to_num(scores[frame_id], nan=0.0).clamp(0.0, 1.0))
                observations_by_frame[frame_id].append(
                    ObjectObservation(
                        category=str(candidate["prompt"]),
                        instance_id=instance_id,
                        mask=mask,
                        score=score,
                        source_track_id=int(candidate["source_obj_id"]),
                        birth_index=int(candidate["birth"]),
                        metadata={
                            "prompt_index": int(candidate["prompt_index"]),
                            "source_obj_id": int(candidate["source_obj_id"]),
                        },
                    )
                )

        return tuple(
            SegmentationFrame(
                frame_id=frame_id,
                image_size=self.output_size,
                observations=tuple(observations_by_frame[frame_id]),
                backend=self.backend_name,
                metadata={
                    "prompts": list(normalized_prompts),
                    "track_count": len(accepted),
                    "causal": True,
                },
            )
            for frame_id in range(len(paths))
        )


class GeometryAwareSAM31SegmentationAdapter(SAM31SegmentationAdapter):
    """Add an opt-in causal geometry competition layer to SAM3.1.

    ``infer`` remains exactly the raw baseline.  The pipeline calls
    ``infer_with_geometry`` only when this adapter is explicitly selected.
    Persistent IDs and raw tracking scores come from the baseline tracker;
    geometry is allowed to replace a mask only after the candidate wins the
    image-space gate.  All history is updated after the current decision, so a
    frame never contributes geometry prompts to itself or an earlier frame.
    """

    backend_name = "sam3.1_forward_causal_geometry_gated"

    def __init__(
        self,
        wrapper: SAM3Wrapper,
        *,
        output_size: tuple[int, int],
        max_objects_per_prompt: int = 16,
        max_total_objects: int = 16,
        min_birth_pixels: int = 128,
        duplicate_iou: float = 0.80,
        geometry_config: GeometryGuidanceConfig | None = None,
    ) -> None:
        super().__init__(
            wrapper,
            output_size=output_size,
            max_objects_per_prompt=max_objects_per_prompt,
            max_total_objects=max_total_objects,
            min_birth_pixels=min_birth_pixels,
            duplicate_iou=duplicate_iou,
        )
        self.geometry_config = (
            geometry_config or GeometryGuidanceConfig()
        ).validate()
        self.last_diagnostics: list[dict[str, object]] = []
        self.last_summary: dict[str, object] = {
            "enabled": True,
            "observation_count": 0,
        }

    def infer_with_geometry(
        self,
        image_paths: Sequence[str | Path],
        geometry_frames: Sequence[GeometryFrame],
        prompts: Sequence[str] | None = None,
    ) -> tuple[SegmentationFrame, ...]:
        paths = tuple(Path(path) for path in image_paths)
        geometry = tuple(geometry_frames)
        raw_frames = tuple(self.infer(paths, prompts))
        if len(geometry) != len(paths):
            raise ValueError(
                "Geometry-aware SAM received a different number of geometry "
                f"frames: {len(geometry)} vs {len(paths)}."
            )
        if len(raw_frames) != len(geometry):
            raise ValueError("Raw SAM and geometry frame counts do not match.")
        geometry_by_id = {int(frame.frame_id): frame for frame in geometry}
        if len(geometry_by_id) != len(geometry):
            raise ValueError("Geometry-aware SAM received duplicate frame IDs.")
        if {int(frame.frame_id) for frame in raw_frames} != set(geometry_by_id):
            raise ValueError(
                "Raw SAM and geometry frame IDs do not match for geometry guidance."
            )

        memory = CausalObjectGeometryMemory(self.geometry_config)
        output_frames: list[SegmentationFrame] = []
        diagnostics: list[dict[str, object]] = []
        applied_count = 0
        attempted_count = 0
        for path, raw_frame in zip(paths, raw_frames):
            frame_id = int(raw_frame.frame_id)
            current_geometry = geometry_by_id[frame_id]
            output_observations: list[ObjectObservation] = []
            frame_applied = 0
            frame_attempted = 0
            for observation in raw_frame.observations:
                raw_mask = observation.mask.detach().cpu().bool()
                history = memory.get(
                    observation.instance_id,
                    current_frame_id=frame_id,
                )
                prompt = build_geometry_prompt(
                    history,
                    current_geometry,
                    config=self.geometry_config,
                    output_size=self.output_size,
                )
                row: dict[str, object] = {
                    "frame_id": frame_id,
                    "category": str(observation.category),
                    "instance_id": int(observation.instance_id),
                    "raw_mask_pixels": int(raw_mask.sum()),
                    "history_available": int(history is not None),
                    "history_point_count": (
                        0 if history is None else int(history.points.shape[0])
                    ),
                    "history_last_frame_id": (
                        None if history is None else int(history.last_frame_id)
                    ),
                    "history_source_frame_ids": (
                        []
                        if history is None
                        else [int(value) for value in history.source_frame_ids]
                    ),
                    "prompt_available": int(prompt is not None),
                    "prompt_attempted": 0,
                    "correction_applied": 0,
                    "candidate_mask_pixels": 0,
                }
                selected_mask = raw_mask
                if prompt is None:
                    row["reason"] = (
                        "keep_raw:no_history"
                        if history is None
                        else "keep_raw:projection_unavailable"
                    )
                else:
                    raw_row = evaluate_geometry_mask(
                        raw_mask,
                        score=float(observation.score),
                        prompt=prompt,
                        support_dilation=int(self.geometry_config.support_dilation),
                    )
                    row.update(
                        {
                            "raw_support_recall": float(raw_row["support_recall"]),
                            "raw_box_precision": float(raw_row["box_precision"]),
                            "raw_support_precision": float(
                                raw_row["support_precision"]
                            ),
                            "raw_geometry_score": float(raw_row["geometry_score"]),
                            "prompt_projected_points": int(prompt.projected_points),
                            "prompt_source_point_count": int(
                                prompt.source_point_count
                            ),
                        }
                    )
                    row["prompt_attempted"] = 1
                    attempted_count += 1
                    frame_attempted += 1
                    try:
                        candidates = self.wrapper.propose_geometry_prompted_masks(
                            path,
                            prompt=str(observation.category),
                            output_size=self.output_size,
                            geometry_prompt=prompt.box_mask,
                            positive_prompt=prompt.positive_mask,
                            max_positive_points=int(
                                self.geometry_config.max_positive_points
                            ),
                            use_box=(
                                str(self.geometry_config.prompt_mode).lower()
                                in {"box_points", "box_only"}
                            ),
                            use_points=(
                                str(self.geometry_config.prompt_mode).lower()
                                in {"box_points", "points_only"}
                            ),
                        )
                        candidate_row = choose_geometry_candidate(
                            candidates,
                            prompt=prompt,
                            config=self.geometry_config,
                        )
                        selected, reason = gate_geometry_replacement(
                            raw_row=raw_row,
                            candidate_row=candidate_row,
                            config=self.geometry_config,
                        )
                        if selected is not None:
                            selected_mask = torch.as_tensor(
                                selected["mask"]
                            ).detach().cpu().bool()
                            frame_applied += 1
                            applied_count += 1
                            row["correction_applied"] = 1
                            row["candidate_mask_pixels"] = int(
                                selected["mask_pixels"]
                            )
                            row.update(
                                {
                                    "candidate_score": float(selected["score"]),
                                    "candidate_support_recall": float(
                                        selected["support_recall"]
                                    ),
                                    "candidate_box_precision": float(
                                        selected["box_precision"]
                                    ),
                                    "candidate_support_precision": float(
                                        selected["support_precision"]
                                    ),
                                    "candidate_geometry_score": float(
                                        selected["geometry_score"]
                                    ),
                                }
                            )
                        else:
                            if candidate_row is not None:
                                row["candidate_mask_pixels"] = int(
                                    candidate_row["mask_pixels"]
                                )
                                row.update(
                                    {
                                        "candidate_score": float(
                                            candidate_row["score"]
                                        ),
                                        "candidate_support_recall": float(
                                            candidate_row["support_recall"]
                                        ),
                                        "candidate_box_precision": float(
                                            candidate_row["box_precision"]
                                        ),
                                        "candidate_support_precision": float(
                                            candidate_row["support_precision"]
                                        ),
                                        "candidate_geometry_score": float(
                                            candidate_row["geometry_score"]
                                        ),
                                    }
                                )
                            row["reason"] = reason
                    except Exception as exc:
                        # A geometry prompt is an optional refinement.  A
                        # backend failure must never remove the raw track.
                        row["reason"] = "keep_raw:prompt_error"
                        row["error_type"] = type(exc).__name__

                memory_update = memory.update(
                    observation.instance_id,
                    current_geometry,
                    selected_mask,
                    frame_id=frame_id,
                    score=float(observation.score),
                )
                row["memory_updated"] = int(bool(memory_update["updated"]))
                row["memory_update_reason"] = str(memory_update["reason"])
                row["memory_point_count"] = int(memory_update["point_count"])
                row.setdefault("reason", "keep_raw:unknown")
                row["geometry_guidance"] = True
                metadata = dict(observation.metadata)
                metadata["geometry_guidance"] = dict(row)
                output_observations.append(
                    replace(observation, mask=selected_mask, metadata=metadata)
                )
                diagnostics.append(row)
            output_frames.append(
                SegmentationFrame(
                    frame_id=frame_id,
                    image_size=raw_frame.image_size,
                    observations=tuple(output_observations),
                    backend=self.backend_name,
                    metadata={
                        **dict(raw_frame.metadata),
                        "geometry_guidance": True,
                        "geometry_guidance_attempted": int(frame_attempted),
                        "geometry_guidance_applied": int(frame_applied),
                    },
                )
            )
        self.last_diagnostics = diagnostics
        self.last_summary = {
            "enabled": True,
            "config": self.geometry_config.to_dict(),
            "observation_count": int(len(diagnostics)),
            "attempted_count": int(attempted_count),
            "applied_count": int(applied_count),
            "fallback_count": int(len(diagnostics) - applied_count),
            "memory": memory.summary(),
            **summarize_diagnostics(diagnostics),
        }
        return tuple(output_frames)


class V0CacheGeometryAdapter:
    """Read frozen V0 geometry fields without exposing cache names to mapping."""

    backend_name = "streamvggt_v0_cache"

    def __init__(self, payload: Mapping[str, Any], *, scale_type: str = "unknown") -> None:
        self.payload = payload
        self.scale_type = str(scale_type)
        self.points = _payload_tensor(
            payload,
            ("baseline_world_points", "world_points"),
            "world points",
        ).detach().float().cpu()
        self.confidence = _payload_tensor(
            payload,
            ("baseline_world_confidence", "confidence"),
            "world confidence",
        ).detach().float().cpu()
        if self.points.ndim != 4 or self.points.shape[-1] != 3:
            raise ValueError("V0 cache world points must have shape [S,H,W,3].")
        if tuple(self.confidence.shape) != tuple(self.points.shape[:3]):
            raise ValueError("V0 cache world confidence is not aligned to points.")
        self.frame_ids = tuple(
            int(value)
            for value in payload.get(
                "frame_indices",
                range(int(self.points.shape[0])),
            )
        )
        if len(self.frame_ids) != self.points.shape[0]:
            raise ValueError("V0 cache frame_indices do not match point count.")
        self.rgb = _payload_rgb(payload, self.points.shape[0], self.points.shape[1:3])

    def infer(
        self,
        image_paths: Sequence[str | Path],
    ) -> tuple[GeometryFrame, ...]:
        if image_paths and len(image_paths) != len(self.frame_ids):
            raise ValueError("Image count differs from V0 cache frame count.")
        height, width = (int(value) for value in self.points.shape[1:3])
        return tuple(
            GeometryFrame(
                frame_id=int(self.frame_ids[index]),
                image_size=(height, width),
                world_points=self.points[index],
                confidence=self.confidence[index],
                valid=torch.isfinite(self.points[index]).all(dim=-1),
                rgb=None if self.rgb is None else self.rgb[index],
                scale_type=self.scale_type,
                backend=self.backend_name,
                metadata={
                    "clip_name": str(self.payload.get("clip_name", "")),
                    "cache_version": self.payload.get("cache_version"),
                    "frame_index": int(self.frame_ids[index]),
                    "source": "frozen_v0_cache",
                },
            )
            for index in range(len(self.frame_ids))
        )


class V0CacheSegmentationAdapter:
    """Expose frozen V0 SAM masks/IDs as canonical object observations."""

    backend_name = "sam31_v0_cache"

    def __init__(
        self,
        payload: Mapping[str, Any],
        *,
        mask_field: str = "tracking_masks_stream",
    ) -> None:
        self.payload = payload
        self.mask_field = str(mask_field)
        if self.mask_field not in payload:
            raise KeyError(f"V0 cache does not contain mask field {self.mask_field!r}.")
        self.masks = torch.as_tensor(payload[self.mask_field]).detach().bool().cpu()
        if self.masks.ndim != 4:
            raise ValueError("V0 cache masks must have shape [S,I,H,W].")
        self.scores = _optional_tensor(payload.get("tracking_scores"))
        if self.scores is None:
            self.scores = torch.ones(self.masks.shape[:2], dtype=torch.float32)
        if tuple(self.scores.shape) != tuple(self.masks.shape[:2]):
            raise ValueError("V0 cache tracking scores do not match masks.")
        quality = _optional_tensor(payload.get("quality"))
        self.static_scores = None
        if quality is not None and quality.ndim == 3 and quality.shape[-1] >= 3:
            if tuple(quality.shape[:2]) == tuple(self.masks.shape[:2]):
                self.static_scores = torch.nan_to_num(
                    quality[..., 2],
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                ).clamp(0.0, 1.0)
        self.frame_ids = tuple(
            int(value)
            for value in payload.get(
                "frame_indices",
                range(int(self.masks.shape[0])),
            )
        )
        if len(self.frame_ids) != self.masks.shape[0]:
            raise ValueError("V0 cache frame_indices do not match masks.")
        self.instance_ids = tuple(
            int(value)
            for value in payload.get(
                "instance_ids",
                range(int(self.masks.shape[1])),
            )
        )
        if len(self.instance_ids) != self.masks.shape[1]:
            raise ValueError("V0 cache instance_ids do not match masks.")
        self.track_ids = tuple(
            int(value)
            for value in payload.get(
                "sam_track_ids",
                [-1] * int(self.masks.shape[1]),
            )
        )
        if len(self.track_ids) != self.masks.shape[1]:
            raise ValueError("V0 cache sam_track_ids do not match masks.")
        self.labels = _cache_labels(payload, self.masks.shape[1])
        self.birth_indices = tuple(
            int(value)
            for value in payload.get(
                "sam_birth_indices",
                [-1] * int(self.masks.shape[1]),
            )
        )
        if len(self.birth_indices) != self.masks.shape[1]:
            raise ValueError("V0 cache sam_birth_indices do not match masks.")

    def infer(
        self,
        image_paths: Sequence[str | Path],
        prompts: Sequence[str] | None = None,
    ) -> tuple[SegmentationFrame, ...]:
        if image_paths and len(image_paths) != len(self.frame_ids):
            raise ValueError("Image count differs from V0 cache frame count.")
        if prompts and not _has_prompt_labels(self.payload):
            normalized = _normalize_prompts(prompts)
            if normalized:
                self.labels = tuple(
                    normalized[index % len(normalized)]
                    for index in range(self.masks.shape[1])
                )
        height, width = (int(value) for value in self.masks.shape[-2:])
        output = []
        for frame_index, frame_id in enumerate(self.frame_ids):
            observations = []
            for slot in range(self.masks.shape[1]):
                mask = self.masks[frame_index, slot]
                if not bool(mask.any()) or self.instance_ids[slot] < 0:
                    continue
                score = float(
                    torch.nan_to_num(self.scores[frame_index, slot], nan=0.0)
                    .clamp(0.0, 1.0)
                )
                static_score = (
                    None
                    if self.static_scores is None
                    else float(self.static_scores[frame_index, slot])
                )
                birth = self.birth_indices[slot]
                observations.append(
                    ObjectObservation(
                        category=self.labels[slot],
                        instance_id=self.instance_ids[slot],
                        mask=mask,
                        score=score,
                        static_score=static_score,
                        source_track_id=(
                            None if self.track_ids[slot] < 0 else self.track_ids[slot]
                        ),
                        birth_index=(None if birth < 0 else birth),
                        metadata={"slot": slot, "mask_field": self.mask_field},
                    )
                )
            output.append(
                SegmentationFrame(
                    frame_id=frame_id,
                    image_size=(height, width),
                    observations=tuple(observations),
                    backend=self.backend_name,
                    metadata={
                        "clip_name": str(self.payload.get("clip_name", "")),
                        "mask_field": self.mask_field,
                        "source": "frozen_v0_cache",
                    },
                )
            )
        return tuple(output)


def _invert_world_to_camera(pose: torch.Tensor) -> torch.Tensor:
    value = pose.detach().float().cpu()
    if tuple(value.shape) == (4, 4):
        value = value[:3]
    if tuple(value.shape) != (3, 4):
        raise ValueError("world_to_camera must have shape [3,4] or [4,4].")
    rotation = value[:, :3]
    translation = value[:, 3]
    inverse = torch.eye(4, dtype=value.dtype)
    inverse[:3, :3] = rotation.transpose(0, 1)
    inverse[:3, 3] = -rotation.transpose(0, 1) @ translation
    return inverse


def _load_rgb(path: Path, output_size: tuple[int, int]) -> torch.Tensor:
    height, width = output_size
    with Image.open(path) as image:
        resized = image.convert("RGB").resize(
            (width, height),
            resample=Image.Resampling.BILINEAR,
        )
        array = np.asarray(resized, dtype=np.float32).copy() / 255.0
    return torch.from_numpy(array)


#: Outcomes that mean the prompt DID return a track, whatever became of it.
TRACK_PRODUCING_OUTCOMES = ("accepted", "duplicate", "over_object_cap")


def ledger_summary(
    ledger: Sequence[Mapping[str, Any]],
    prompts: Sequence[str] = (),
) -> dict[str, Any]:
    """Per prompt, how many tracks it produced and what became of each.

    This is the readable half of the ledger.  "Produced nothing" and "produced
    something that was dropped" look the same from the observations that
    survive, and they call for opposite fixes: a prompt that returns no masks
    needs different words, while a prompt whose tracks are dropped as
    duplicates needs the competition between prompts addressed instead.

    ``prompts`` is the full request list.  It has to be given, because a prompt
    that returned no tracks contributes no ledger entries and would otherwise
    be indistinguishable from a prompt that was never asked for.
    """

    per_prompt: dict[str, dict[str, int]] = {
        str(prompt): {} for prompt in prompts
    }
    for entry in ledger:
        prompt = str(entry.get("prompt", ""))
        outcome = str(entry.get("outcome", "unknown"))
        counts = per_prompt.setdefault(prompt, {})
        counts[outcome] = counts.get(outcome, 0) + 1
    # Two different failures that a single "no track" count would merge, and
    # they call for opposite work.  A prompt absent from the ledger returned
    # nothing at all -- the text grounding found no object -- which is a word to
    # replace.  A prompt present with outcomes but none of them producing means
    # the object WAS found and every track died on the way in, which is a
    # threshold or a competition to look at instead.
    returned_nothing = sorted(
        name for name, counts in per_prompt.items() if not counts
    )
    found_but_unusable = sorted(
        name
        for name, counts in per_prompt.items()
        if counts
        and not any(counts.get(outcome) for outcome in TRACK_PRODUCING_OUTCOMES)
    )
    return {
        "total": len(ledger),
        "prompt_count": len(per_prompt),
        "per_prompt": {
            name: dict(counts) for name, counts in sorted(per_prompt.items())
        },
        "prompts_returning_nothing": returned_nothing,
        "prompts_found_but_unusable": found_but_unusable,
        "prompts_with_no_track": sorted(returned_nothing + found_but_unusable),
    }


def _tracks_duplicate(
    candidate: Mapping[str, Any],
    previous: Mapping[str, Any],
    threshold: float,
) -> bool:
    first = torch.as_tensor(candidate["masks"]).bool()
    second = torch.as_tensor(previous["masks"]).bool()
    common = first.flatten(1).any(dim=1) & second.flatten(1).any(dim=1)
    for frame_index in torch.nonzero(common, as_tuple=False).flatten().tolist():
        left = first[frame_index]
        right = second[frame_index]
        union = (left | right).sum()
        if int(union) == 0:
            continue
        iou = float((left & right).sum()) / float(union)
        if iou >= float(threshold):
            return True
    return False


def _payload_tensor(
    payload: Mapping[str, Any],
    names: Sequence[str],
    description: str,
) -> torch.Tensor:
    for name in names:
        if name in payload:
            value = payload[name]
            if torch.is_tensor(value):
                return value
            return torch.as_tensor(value)
    raise KeyError(f"V0 cache lacks {description}; tried fields={tuple(names)}.")


def _optional_tensor(value: Any) -> torch.Tensor | None:
    if value is None:
        return None
    return torch.as_tensor(value).detach().float().cpu()


def _payload_rgb(
    payload: Mapping[str, Any],
    frame_count: int,
    image_size: Sequence[int],
) -> torch.Tensor | None:
    value = payload.get("stream_images", payload.get("images"))
    if value is None:
        return None
    rgb = torch.as_tensor(value).detach().float().cpu()
    height, width = (int(item) for item in image_size)
    if rgb.ndim == 4 and tuple(rgb.shape) == (frame_count, 3, height, width):
        rgb = rgb.permute(0, 2, 3, 1)
    if tuple(rgb.shape) != (frame_count, height, width, 3):
        return None
    if float(rgb.max()) > 1.0:
        rgb = rgb / 255.0
    return rgb.clamp(0.0, 1.0)


def _cache_labels(payload: Mapping[str, Any], count: int) -> tuple[str, ...]:
    values = payload.get("sam_track_prompts", ())
    labels = [str(value).strip() for value in values]
    configured = [
        str(value).strip()
        for value in payload.get("instance_prompts", ())
        if str(value).strip()
    ]
    output = []
    for index in range(int(count)):
        label = labels[index] if index < len(labels) else ""
        if not label:
            label = configured[index % len(configured)] if configured else "object"
        output.append(label)
    return tuple(output)


def _has_prompt_labels(payload: Mapping[str, Any]) -> bool:
    values = payload.get("sam_track_prompts", ())
    return any(str(value).strip() for value in values)


def _normalize_prompts(prompts: Sequence[str] | None) -> tuple[str, ...]:
    if prompts is None:
        return ()
    output = []
    for value in prompts:
        for piece in str(value).split(","):
            piece = piece.strip()
            if piece:
                output.append(piece)
    return tuple(output)


def _size(value: Sequence[int]) -> tuple[int, int]:
    if len(value) != 2:
        raise ValueError(f"Expected (height, width), got {value!r}.")
    height, width = int(value[0]), int(value[1])
    if height <= 0 or width <= 0:
        raise ValueError("Image dimensions must be positive.")
    return height, width
