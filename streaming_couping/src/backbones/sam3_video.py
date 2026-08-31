"""SAM3 video tracker adapter.

This adapter uses SAM3's original video predictor and memory propagation. It is
kept separate from the SAM3 intermediate-feature adapter because the first use
case is validation and visualization, not training loss.
"""

from __future__ import annotations

import contextlib
import inspect
import logging
import os
import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ..external_repos import maybe_add_repo_to_path

_ORIGINAL_TENSOR_ARGSORT = torch.Tensor.argsort
_BOOL_ARGSORT_COMPATIBILITY_INSTALLED = False


@dataclass
class SAM3TrackOutput:
    masks: torch.Tensor
    selected_obj_id: Optional[int]
    prompt_frame_idx: int
    prompt_box_xywh: Optional[tuple[float, float, float, float]]
    scores: Optional[torch.Tensor] = None
    frame_objects: Dict[int, Dict[int, torch.Tensor]] = field(default_factory=dict)
    aux: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SAM3MultiTrackOutput:
    """Fixed-slot view of every object discovered by a forward SAM3.1 run."""

    masks: torch.Tensor
    scores: torch.Tensor
    obj_ids: tuple[int, ...]
    birth_indices: tuple[int, ...]
    frame_objects: Dict[int, Dict[int, torch.Tensor]] = field(default_factory=dict)
    aux: Dict[str, Any] = field(default_factory=dict)


def load_sam3_video_predictor(
    *,
    repo_path: Optional[str | Path],
    checkpoint_path: str | Path,
    device: str,
    version: str = "sam3",
    use_fa3: bool = False,
    max_num_objects: int = 16,
    multiplex_count: int = 16,
    grounding_batch_size: int = 16,
    async_loading_frames: bool = False,
    quiet: bool = True,
):
    repo = maybe_add_repo_to_path(repo_path)
    if repo_path is not None:
        expected = Path(repo_path).expanduser()
        if repo is None:
            raise RuntimeError(
                f"SAM3 repo path does not exist: {expected}\n"
                "Run `git submodule update --init --recursive`, or pass the correct repo path."
            )
        if not ((repo / "sam3").is_dir() or (repo / "src" / "sam3").is_dir()):
            raise RuntimeError(
                f"SAM3 repo at {repo} does not look initialized; missing package `sam3`."
            )
    if not str(device).startswith("cuda"):
        raise RuntimeError("SAM3 video predictor requires a CUDA device.")
    version = str(version).strip().lower()
    if version not in {"sam3", "sam3.1"}:
        raise ValueError(
            f"Unsupported SAM video version {version!r}; "
            "expected 'sam3' or 'sam3.1'."
        )
    if int(grounding_batch_size) < 1:
        raise ValueError("grounding_batch_size must be positive.")

    with quiet_sam3_output(quiet):
        try:
            from sam3.model_builder import build_sam3_video_predictor
        except ModuleNotFoundError as exc:
            if exc.name == "sam3":
                raise RuntimeError(
                    "Could not import `sam3`. Run `git submodule update --init --recursive` "
                    "or pass `--sam3-repo` to a SAM3 repo."
                ) from exc
            raise

        gpu_id = parse_cuda_device_index(device)
        if version == "sam3.1":
            _install_bool_argsort_compatibility()
            try:
                from sam3.model_builder import build_sam3_predictor
            except ImportError as exc:
                raise RuntimeError(
                    "The configured SAM repository does not support SAM3.1. "
                    "Update externals/sam3 to the release containing "
                    "build_sam3_predictor."
                ) from exc
            # The multiplex implementation still creates some tensors on bare
            # ``cuda``. Keep its process-wide default on the requested device,
            # not only during checkpoint construction.
            torch.cuda.set_device(gpu_id)
            predictor = build_sam3_predictor(
                checkpoint_path=str(checkpoint_path),
                version="sam3.1",
                compile=False,
                warm_up=False,
                use_fa3=bool(use_fa3),
                max_num_objects=int(max_num_objects),
                multiplex_count=int(multiplex_count),
                async_loading_frames=async_loading_frames,
            )
            _set_sam31_grounding_batch_size(
                predictor,
                grounding_batch_size=int(grounding_batch_size),
            )
            _filter_init_state_kwargs(predictor)
            return predictor
        return build_sam3_video_predictor(
            checkpoint_path=str(checkpoint_path),
            gpus_to_use=[gpu_id],
            compile=False,
            async_loading_frames=async_loading_frames,
        )


def _filter_init_state_kwargs(predictor) -> None:
    """Bridge SAM3.1 predictor/session signatures without editing externals."""

    model = predictor.model
    original = model.init_state
    signature = inspect.signature(original)
    if any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return
    accepted = set(signature.parameters)

    def compatible_init_state(*args, **kwargs):
        filtered = {
            key: value
            for key, value in kwargs.items()
            if key in accepted
        }
        return original(*args, **filtered)

    model.init_state = compatible_init_state


def _set_sam31_grounding_batch_size(
    predictor,
    *,
    grounding_batch_size: int,
) -> None:
    """Limit the frame batch used by SAM3.1's high-resolution detector."""

    model = getattr(predictor, "model", None)
    if model is None or not hasattr(model, "batched_grounding_batch_size"):
        raise RuntimeError("SAM3.1 predictor lacks batched grounding controls.")
    model.batched_grounding_batch_size = int(grounding_batch_size)


def _bool_compatible_argsort(
    tensor: torch.Tensor,
    *args,
    **kwargs,
) -> torch.Tensor:
    """Preserve False<True ordering where PyTorch rejects bool argsort."""

    sortable = tensor.to(torch.uint8) if tensor.dtype == torch.bool else tensor
    return _ORIGINAL_TENSOR_ARGSORT(sortable, *args, **kwargs)


def _install_bool_argsort_compatibility() -> None:
    """Install the narrow compatibility needed by SAM3.1 multiplex."""

    global _BOOL_ARGSORT_COMPATIBILITY_INSTALLED
    if _BOOL_ARGSORT_COMPATIBILITY_INSTALLED:
        return
    torch.Tensor.argsort = _bool_compatible_argsort
    _BOOL_ARGSORT_COMPATIBILITY_INSTALLED = True


class SAM3VideoTrackerAdapter:
    """Track a prompted object with SAM3's original video memory."""

    def __init__(
        self,
        predictor,
        *,
        output_prob_thresh: float = 0.5,
        prompt_with_box: bool = True,
        offload_video_to_cpu: bool = False,
    ) -> None:
        self.predictor = predictor
        self.output_prob_thresh = float(output_prob_thresh)
        self.prompt_with_box = bool(prompt_with_box)
        self.offload_video_to_cpu = bool(offload_video_to_cpu)

    @torch.no_grad()
    def track_from_paths(
        self,
        image_paths: Sequence[str | Path],
        *,
        prompt: str,
        output_size: tuple[int, int],
        prompt_frame_idx: int = 0,
        reference_mask: torch.Tensor | np.ndarray | None = None,
        propagation_direction: str = "both",
        quiet: bool = True,
    ) -> SAM3TrackOutput:
        if not image_paths:
            raise ValueError("At least one image path is required for SAM3 tracking.")
        prompt_frame_idx = int(prompt_frame_idx)
        if prompt_frame_idx < 0 or prompt_frame_idx >= len(image_paths):
            raise ValueError(
                f"prompt_frame_idx={prompt_frame_idx} is out of range for "
                f"{len(image_paths)} frames."
            )
        propagation_direction = str(propagation_direction).strip().lower()
        if propagation_direction not in {"forward", "both"}:
            raise ValueError(
                "propagation_direction must be 'forward' or 'both'."
            )

        with tempfile.TemporaryDirectory(prefix="sam3_track_") as tmp:
            tmp_dir = Path(tmp)
            materialize_video_dir(image_paths, tmp_dir)
            with quiet_sam3_output(quiet):
                session = self.predictor.start_session(
                    resource_path=str(tmp_dir),
                    offload_video_to_cpu=self.offload_video_to_cpu,
                )
                session_id = session["session_id"] if isinstance(session, dict) else session
                try:
                    prompt_box = None
                    reference_mask_out = normalize_reference_mask(
                        reference_mask,
                        output_size=output_size,
                    )
                    if self.prompt_with_box and reference_mask_out is not None:
                        prompt_box = mask_to_normalized_box(
                            reference_mask_out,
                            image_path=Path(image_paths[prompt_frame_idx]),
                        )

                    add_kwargs: Dict[str, Any] = {
                        "session_id": session_id,
                        "frame_idx": prompt_frame_idx,
                        "text": prompt,
                        "output_prob_thresh": self.output_prob_thresh,
                    }
                    if prompt_box is not None:
                        add_kwargs["bounding_boxes"] = [prompt_box]
                        add_kwargs["bounding_box_labels"] = [1]
                    prompted = self.predictor.add_prompt(**add_kwargs)
                    propagated = list(
                        self.predictor.propagate_in_video(
                            session_id=session_id,
                            propagation_direction=propagation_direction,
                            start_frame_idx=prompt_frame_idx,
                            max_frame_num_to_track=(
                                len(image_paths) - prompt_frame_idx
                                if propagation_direction == "forward"
                                else len(image_paths)
                            ),
                            output_prob_thresh=self.output_prob_thresh,
                        )
                    )
                finally:
                    self.predictor.close_session(session_id)

        frame_objects = collect_frame_objects(
            [prompted, *propagated],
            output_size=output_size,
        )
        frame_scores = collect_frame_scores([prompted, *propagated])
        selected_obj_id = select_tracked_object_id(
            frame_objects,
            prompt_frame_idx=prompt_frame_idx,
            reference_mask=reference_mask_out,
        )
        masks = masks_for_selected_object(
            frame_objects,
            selected_obj_id=selected_obj_id,
            num_frames=len(image_paths),
            output_size=output_size,
        )
        scores = scores_for_selected_object(
            frame_scores,
            selected_obj_id=selected_obj_id,
            num_frames=len(image_paths),
        )
        return SAM3TrackOutput(
            masks=masks,
            selected_obj_id=selected_obj_id,
            prompt_frame_idx=prompt_frame_idx,
            prompt_box_xywh=tuple(prompt_box) if prompt_box is not None else None,
            scores=scores,
            frame_objects=frame_objects,
            aux={
                "prompt": prompt,
                "num_frames": len(image_paths),
                "propagation_direction": propagation_direction,
                "frame_object_counts": {
                    int(frame_idx): len(objects)
                    for frame_idx, objects in frame_objects.items()
                },
                "score_semantics": (
                    "SAM3 out_probs for the selected ID; propagation keeps the "
                    "initial detector score, while a missing ID is assigned zero"
                ),
            },
        )

    @torch.no_grad()
    def track_all_forward_from_paths(
        self,
        image_paths: Sequence[str | Path],
        *,
        prompt: str,
        output_size: tuple[int, int],
        max_objects: int,
        quiet: bool = True,
    ) -> SAM3MultiTrackOutput:
        """Discover and track object IDs in temporal order without backtracking.

        SAM3.1's text prompt applies to every frame.  Its multiplex tracker can
        therefore create a persistent object ID when an object first becomes
        visible after frame zero.  The detector/tracker is forced into a
        no-lookahead mode here: outputs are propagated only forward, hot-start
        buffering is disabled, and one detection is sufficient to confirm a
        new masklet.  Slots are assigned by first appearance and never reused.
        """

        if not image_paths:
            raise ValueError("At least one image path is required for SAM3 tracking.")
        if int(max_objects) < 1:
            raise ValueError("max_objects must be positive.")
        if len(output_size) != 2 or any(int(value) < 1 for value in output_size):
            raise ValueError("output_size must contain two positive dimensions.")

        with tempfile.TemporaryDirectory(prefix="sam31_track_all_") as tmp:
            tmp_dir = Path(tmp)
            materialize_video_dir(image_paths, tmp_dir)
            with quiet_sam3_output(quiet):
                session = self.predictor.start_session(
                    resource_path=str(tmp_dir),
                    offload_video_to_cpu=self.offload_video_to_cpu,
                )
                session_id = session["session_id"] if isinstance(session, dict) else session
                causal_settings = _set_causal_multiplex_settings(self.predictor)
                try:
                    prompted = self.predictor.add_prompt(
                        session_id=session_id,
                        frame_idx=0,
                        text=prompt,
                        output_prob_thresh=self.output_prob_thresh,
                    )
                    propagated = list(
                        self.predictor.propagate_in_video(
                            session_id=session_id,
                            propagation_direction="forward",
                            start_frame_idx=0,
                            max_frame_num_to_track=len(image_paths),
                            output_prob_thresh=self.output_prob_thresh,
                        )
                    )
                finally:
                    _restore_multiplex_settings(self.predictor, causal_settings)
                    self.predictor.close_session(session_id)

        results = [prompted, *propagated]
        frame_objects = collect_frame_objects(results, output_size=output_size)
        frame_scores = collect_frame_scores(results)
        first_seen: Dict[int, int] = {}
        for frame_idx in range(len(image_paths)):
            for obj_id, mask in sorted(frame_objects.get(frame_idx, {}).items()):
                if bool(mask.any()):
                    first_seen.setdefault(int(obj_id), int(frame_idx))
        ordered = sorted(first_seen, key=lambda obj_id: (first_seen[obj_id], obj_id))
        selected = tuple(ordered[: int(max_objects)])
        sequence = len(image_paths)
        height, width = (int(value) for value in output_size)
        masks = torch.zeros(sequence, len(selected), height, width, dtype=torch.bool)
        scores = torch.zeros(sequence, len(selected), dtype=torch.float32)
        for slot, obj_id in enumerate(selected):
            for frame_idx in range(sequence):
                mask = frame_objects.get(frame_idx, {}).get(obj_id)
                if mask is None or not bool(mask.any()):
                    continue
                masks[frame_idx, slot] = mask.detach().cpu().bool()
                score = frame_scores.get(frame_idx, {}).get(obj_id)
                scores[frame_idx, slot] = 1.0 if score is None else float(score)
        return SAM3MultiTrackOutput(
            masks=masks,
            scores=scores,
            obj_ids=selected,
            birth_indices=tuple(first_seen[obj_id] for obj_id in selected),
            frame_objects=frame_objects,
            aux={
                "prompt": str(prompt),
                "propagation_direction": "forward",
                "causal_confirmation": True,
                "detected_object_count": len(ordered),
                "retained_object_count": len(selected),
                "dropped_object_ids": tuple(ordered[int(max_objects) :]),
            },
        )

    @torch.no_grad()
    def track_auto_points_forward_from_paths(
        self,
        image_paths: Sequence[str | Path],
        *,
        output_size: tuple[int, int],
        max_objects: int,
        discovery_stride: int = 5,
        grid_rows: int = 8,
        grid_columns: int = 12,
        points_per_discovery: int = 24,
        min_mask_pixels: int = 128,
        max_mask_area_ratio: float = 0.35,
        duplicate_iou: float = 0.80,
        duplicate_intersection_over_smaller: float = 0.90,
        quiet: bool = True,
    ) -> SAM3MultiTrackOutput:
        """Discover objects from visual point prompts without category text.

        Discovery is causal.  At fixed frames, deterministic grid points not
        already covered by a retained mask are submitted as positive visual
        prompts.  A proposal is admitted immediately from its birth-frame
        mask, after area filtering and mask NMS, and its slot is never reused.
        Ground truth, future masks, and semantic noun phrases are not used.
        """

        if not image_paths:
            raise ValueError("At least one image path is required for SAM3 tracking.")
        if int(max_objects) < 1:
            raise ValueError("max_objects must be positive.")
        if len(output_size) != 2 or any(int(value) < 1 for value in output_size):
            raise ValueError("output_size must contain two positive dimensions.")
        if int(discovery_stride) < 1:
            raise ValueError("discovery_stride must be positive.")
        if int(points_per_discovery) < 1:
            raise ValueError("points_per_discovery must be positive.")
        _validate_auto_proposal_thresholds(
            grid_rows=grid_rows,
            grid_columns=grid_columns,
            min_mask_pixels=min_mask_pixels,
            max_mask_area_ratio=max_mask_area_ratio,
            duplicate_iou=duplicate_iou,
            duplicate_intersection_over_smaller=(
                duplicate_intersection_over_smaller
            ),
        )

        sequence = len(image_paths)
        discovery_frames = list(range(0, sequence, int(discovery_stride)))
        grid = auto_proposal_point_grid(grid_rows, grid_columns)
        retained_ids: list[int] = []
        birth_indices: dict[int, int] = {}
        diagnostics: list[dict[str, Any]] = []
        results: list[Dict[str, Any]] = []
        next_obj_id = 1

        with tempfile.TemporaryDirectory(prefix="sam31_auto_points_") as tmp:
            video_dir = Path(tmp)
            materialize_video_dir(image_paths, video_dir)
            with quiet_sam3_output(quiet):
                session = self.predictor.start_session(
                    resource_path=str(video_dir),
                    offload_video_to_cpu=self.offload_video_to_cpu,
                )
                session_id = (
                    session["session_id"]
                    if isinstance(session, dict)
                    else session
                )
                causal_settings = _set_causal_multiplex_settings(
                    self.predictor
                )
                try:
                    for discovery_index, frame_idx in enumerate(
                        discovery_frames
                    ):
                        frame_objects = collect_frame_objects(
                            results,
                            output_size=output_size,
                        ).get(frame_idx, {})
                        retained_masks = [
                            frame_objects[obj_id]
                            for obj_id in retained_ids
                            if obj_id in frame_objects
                            and bool(frame_objects[obj_id].any())
                        ]
                        ordered_points = auto_proposal_points_for_discovery(
                            grid,
                            discovery_index=discovery_index,
                            limit=points_per_discovery,
                        )
                        for grid_index, point_x, point_y in ordered_points:
                            if len(retained_ids) >= int(max_objects):
                                break
                            if _point_is_covered(
                                retained_masks,
                                point_x=point_x,
                                point_y=point_y,
                            ):
                                diagnostics.append(
                                    {
                                        "sequence_index": int(frame_idx),
                                        "grid_index": int(grid_index),
                                        "point_x": float(point_x),
                                        "point_y": float(point_y),
                                        "obj_id": -1,
                                        "accepted": 0,
                                        "reason": "covered_by_retained_mask",
                                        "mask_pixels": 0,
                                        "mask_area_ratio": 0.0,
                                        "maximum_iou": 0.0,
                                        "maximum_intersection_over_smaller": 0.0,
                                    }
                                )
                                continue

                            obj_id = int(next_obj_id)
                            next_obj_id += 1
                            prompted = self.predictor.add_prompt(
                                session_id=session_id,
                                frame_idx=int(frame_idx),
                                text=None,
                                points=torch.tensor(
                                    [[point_x, point_y]],
                                    dtype=torch.float32,
                                ),
                                point_labels=torch.tensor(
                                    [1],
                                    dtype=torch.int32,
                                ),
                                obj_id=obj_id,
                                rel_coordinates=True,
                                output_prob_thresh=self.output_prob_thresh,
                            )
                            results.append(prompted)
                            prompted_objects = collect_frame_objects(
                                [prompted],
                                output_size=output_size,
                            ).get(frame_idx, {})
                            candidate = prompted_objects.get(obj_id)
                            assessment = assess_auto_proposal_mask(
                                candidate,
                                retained_masks=retained_masks,
                                point_x=point_x,
                                point_y=point_y,
                                output_size=output_size,
                                min_mask_pixels=min_mask_pixels,
                                max_mask_area_ratio=max_mask_area_ratio,
                                duplicate_iou=duplicate_iou,
                                duplicate_intersection_over_smaller=(
                                    duplicate_intersection_over_smaller
                                ),
                            )
                            diagnostics.append(
                                {
                                    "sequence_index": int(frame_idx),
                                    "grid_index": int(grid_index),
                                    "point_x": float(point_x),
                                    "point_y": float(point_y),
                                    "obj_id": obj_id,
                                    **assessment,
                                }
                            )
                            if not bool(assessment["accepted"]):
                                # A point prompt can register an object even
                                # when postprocessing returns no usable mask.
                                # Always remove rejected IDs so failed
                                # proposals cannot consume multiplex capacity.
                                removed = self.predictor.remove_object(
                                    session_id=session_id,
                                    frame_idx=int(frame_idx),
                                    obj_id=obj_id,
                                    is_user_action=False,
                                )
                                results.append(removed)
                                continue
                            retained_ids.append(obj_id)
                            birth_indices[obj_id] = int(frame_idx)
                            retained_masks.append(candidate.bool())

                        next_frame = (
                            discovery_frames[discovery_index + 1]
                            if discovery_index + 1 < len(discovery_frames)
                            and len(retained_ids) < int(max_objects)
                            else sequence - 1
                        )
                        if retained_ids and next_frame >= frame_idx:
                            results.extend(
                                self.predictor.propagate_in_video(
                                    session_id=session_id,
                                    propagation_direction="forward",
                                    start_frame_idx=int(frame_idx),
                                    max_frame_num_to_track=int(
                                        next_frame - frame_idx
                                    ),
                                    output_prob_thresh=(
                                        self.output_prob_thresh
                                    ),
                                )
                            )
                        if len(retained_ids) >= int(max_objects):
                            break
                finally:
                    _restore_multiplex_settings(
                        self.predictor,
                        causal_settings,
                    )
                    self.predictor.close_session(session_id)

        frame_objects = collect_frame_objects(
            results,
            output_size=output_size,
        )
        frame_scores = collect_frame_scores(results)
        height, width = (int(value) for value in output_size)
        masks = torch.zeros(
            sequence,
            len(retained_ids),
            height,
            width,
            dtype=torch.bool,
        )
        scores = torch.zeros(
            sequence,
            len(retained_ids),
            dtype=torch.float32,
        )
        for slot, obj_id in enumerate(retained_ids):
            for frame_idx in range(sequence):
                mask = frame_objects.get(frame_idx, {}).get(obj_id)
                if mask is None or not bool(mask.any()):
                    continue
                masks[frame_idx, slot] = mask.bool()
                # Point-prompt tracking does not consistently expose a
                # detector probability.  Prompted visibility is therefore
                # the score fallback, matching the existing wrapper policy.
                scores[frame_idx, slot] = float(
                    frame_scores.get(frame_idx, {}).get(obj_id, 1.0)
                )
        return SAM3MultiTrackOutput(
            masks=masks,
            scores=scores,
            obj_ids=tuple(retained_ids),
            birth_indices=tuple(
                birth_indices[obj_id] for obj_id in retained_ids
            ),
            frame_objects=frame_objects,
            aux={
                "proposal_source": "class_agnostic_visual_point_grid",
                "semantic_prompt": None,
                "propagation_direction": "forward",
                "causal_confirmation": True,
                "discovery_frames": tuple(discovery_frames),
                "grid_rows": int(grid_rows),
                "grid_columns": int(grid_columns),
                "points_per_discovery": int(points_per_discovery),
                "proposal_diagnostics": diagnostics,
                "retained_object_count": len(retained_ids),
                "next_unused_obj_id": int(next_obj_id),
            },
        )


def auto_proposal_point_grid(
    rows: int,
    columns: int,
) -> tuple[tuple[int, float, float], ...]:
    """Return deterministic normalized cell-centre prompts."""

    rows = int(rows)
    columns = int(columns)
    if rows < 1 or columns < 1:
        raise ValueError("Auto-proposal grid dimensions must be positive.")
    values = []
    for row in range(rows):
        for column in range(columns):
            index = row * columns + column
            values.append(
                (
                    index,
                    (column + 0.5) / float(columns),
                    (row + 0.5) / float(rows),
                )
            )
    # A centre-out order usually finds bounded foreground objects before
    # image-edge background surfaces.  The ordering remains fixed and GT-free.
    return tuple(
        sorted(
            values,
            key=lambda value: (
                (value[1] - 0.5) ** 2 + (value[2] - 0.5) ** 2,
                value[0],
            ),
        )
    )


def auto_proposal_points_for_discovery(
    grid: Sequence[tuple[int, float, float]],
    *,
    discovery_index: int,
    limit: int,
) -> tuple[tuple[int, float, float], ...]:
    """Rotate a fixed grid so later discovery frames inspect new cells."""

    values = tuple(grid)
    if not values:
        return ()
    limit = min(max(1, int(limit)), len(values))
    start = (max(0, int(discovery_index)) * limit) % len(values)
    return tuple(
        values[(start + offset) % len(values)]
        for offset in range(limit)
    )


def assess_auto_proposal_mask(
    mask: torch.Tensor | None,
    *,
    retained_masks: Sequence[torch.Tensor],
    point_x: float,
    point_y: float,
    output_size: tuple[int, int],
    min_mask_pixels: int,
    max_mask_area_ratio: float,
    duplicate_iou: float,
    duplicate_intersection_over_smaller: float,
) -> dict[str, object]:
    """Apply birth-frame-only admission tests to one visual proposal."""

    height, width = (int(value) for value in output_size)
    if mask is None:
        return _auto_assessment(False, "no_mask")
    candidate = mask.detach().cpu().bool()
    if tuple(candidate.shape) != (height, width):
        raise ValueError(
            "Auto-proposal mask does not match configured output size."
        )
    pixels = int(candidate.sum())
    area_ratio = pixels / float(height * width)
    if pixels < int(min_mask_pixels):
        return _auto_assessment(
            False,
            "mask_too_small",
            pixels=pixels,
            area_ratio=area_ratio,
        )
    if area_ratio > float(max_mask_area_ratio):
        return _auto_assessment(
            False,
            "mask_too_large",
            pixels=pixels,
            area_ratio=area_ratio,
        )
    point_column = min(width - 1, max(0, int(float(point_x) * width)))
    point_row = min(height - 1, max(0, int(float(point_y) * height)))
    if not bool(candidate[point_row, point_column]):
        return _auto_assessment(
            False,
            "prompt_point_outside_mask",
            pixels=pixels,
            area_ratio=area_ratio,
        )

    maximum_iou = 0.0
    maximum_ios = 0.0
    for retained in retained_masks:
        current = retained.detach().cpu().bool()
        if tuple(current.shape) != (height, width):
            raise ValueError("Retained auto-proposal masks have mixed sizes.")
        retained_pixels = int(current.sum())
        if not retained_pixels:
            continue
        intersection = int((candidate & current).sum())
        union = pixels + retained_pixels - intersection
        maximum_iou = max(
            maximum_iou,
            intersection / float(union) if union else 0.0,
        )
        maximum_ios = max(
            maximum_ios,
            intersection / float(min(pixels, retained_pixels)),
        )
    if maximum_iou >= float(duplicate_iou):
        return _auto_assessment(
            False,
            "duplicate_iou",
            pixels=pixels,
            area_ratio=area_ratio,
            maximum_iou=maximum_iou,
            maximum_ios=maximum_ios,
        )
    if maximum_ios >= float(duplicate_intersection_over_smaller):
        return _auto_assessment(
            False,
            "duplicate_containment",
            pixels=pixels,
            area_ratio=area_ratio,
            maximum_iou=maximum_iou,
            maximum_ios=maximum_ios,
        )
    return _auto_assessment(
        True,
        "accepted",
        pixels=pixels,
        area_ratio=area_ratio,
        maximum_iou=maximum_iou,
        maximum_ios=maximum_ios,
    )


def _auto_assessment(
    accepted: bool,
    reason: str,
    *,
    pixels: int = 0,
    area_ratio: float = 0.0,
    maximum_iou: float = 0.0,
    maximum_ios: float = 0.0,
) -> dict[str, object]:
    return {
        "accepted": int(bool(accepted)),
        "reason": str(reason),
        "mask_pixels": int(pixels),
        "mask_area_ratio": float(area_ratio),
        "maximum_iou": float(maximum_iou),
        "maximum_intersection_over_smaller": float(maximum_ios),
    }


def _point_is_covered(
    masks: Sequence[torch.Tensor],
    *,
    point_x: float,
    point_y: float,
) -> bool:
    for mask in masks:
        current = mask.detach().cpu().bool()
        if current.ndim != 2 or not current.numel():
            continue
        height, width = current.shape
        column = min(width - 1, max(0, int(float(point_x) * width)))
        row = min(height - 1, max(0, int(float(point_y) * height)))
        if bool(current[row, column]):
            return True
    return False


def _validate_auto_proposal_thresholds(
    *,
    grid_rows: int,
    grid_columns: int,
    min_mask_pixels: int,
    max_mask_area_ratio: float,
    duplicate_iou: float,
    duplicate_intersection_over_smaller: float,
) -> None:
    if int(grid_rows) < 1 or int(grid_columns) < 1:
        raise ValueError("Auto-proposal grid dimensions must be positive.")
    if int(min_mask_pixels) < 1:
        raise ValueError("min_mask_pixels must be positive.")
    for name, value in (
        ("max_mask_area_ratio", max_mask_area_ratio),
        ("duplicate_iou", duplicate_iou),
        (
            "duplicate_intersection_over_smaller",
            duplicate_intersection_over_smaller,
        ),
    ):
        if not 0.0 < float(value) <= 1.0:
            raise ValueError(f"{name} must be in (0,1].")


@contextlib.contextmanager
def quiet_sam3_output(enabled: bool = True):
    if not enabled:
        yield
        return

    previous_levels: Dict[str, int] = {}
    set_sam3_loggers(logging.WARNING, previous_levels)
    with open(os.devnull, "w", encoding="utf8") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            try:
                yield
            finally:
                restore_loggers(previous_levels)


def _set_causal_multiplex_settings(predictor) -> dict[str, Any]:
    """Temporarily remove SAM3.1 output buffering that consults future frames."""

    model = getattr(predictor, "model", None)
    saved: dict[str, Any] = {}
    if model is None:
        return saved
    replacements = {
        "hotstart_delay": 0,
        "masklet_confirmation_consecutive_det_thresh": 1,
        "postprocess_batch_size": 1,
    }
    for name, value in replacements.items():
        if hasattr(model, name):
            saved[name] = getattr(model, name)
            setattr(model, name, value)
    return saved


def _restore_multiplex_settings(predictor, saved: Dict[str, Any]) -> None:
    model = getattr(predictor, "model", None)
    if model is None:
        return
    for name, value in saved.items():
        setattr(model, name, value)


def set_sam3_loggers(level: int, previous_levels: Dict[str, int]) -> None:
    names = {"", "sam3"}
    names.update(
        name
        for name in logging.Logger.manager.loggerDict.keys()
        if str(name).startswith("sam3")
    )
    for name in names:
        logger = logging.getLogger(name)
        previous_levels.setdefault(name, logger.level)
        logger.setLevel(level)


def restore_loggers(previous_levels: Dict[str, int]) -> None:
    for name, level in previous_levels.items():
        logging.getLogger(name).setLevel(level)


def parse_cuda_device_index(device: str) -> int:
    if device == "cuda":
        return int(torch.cuda.current_device())
    if device.startswith("cuda:"):
        return int(device.split(":", 1)[1])
    raise ValueError(f"Expected CUDA device string, got {device!r}")


def materialize_video_dir(image_paths: Sequence[str | Path], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir.chmod(0o755)
    for idx, image_path in enumerate(image_paths):
        src = Path(image_path).expanduser().resolve()
        dst = output_dir / f"{idx:05d}.jpg"
        # SAM3 may load frames in worker processes. A real, world-readable copy
        # avoids permission failures caused by symlinks into protected NAS trees
        # or TemporaryDirectory's default owner-only permissions.
        shutil.copyfile(src, dst)
        dst.chmod(0o644)


def normalize_reference_mask(
    reference_mask: torch.Tensor | np.ndarray | None,
    *,
    output_size: tuple[int, int],
) -> torch.Tensor | None:
    if reference_mask is None:
        return None
    mask = torch.as_tensor(reference_mask).detach().cpu()
    if mask.ndim != 2:
        raise ValueError(f"Expected reference mask [H, W], got {tuple(mask.shape)}")
    if tuple(mask.shape) != tuple(output_size):
        mask = resize_bool_mask(mask, output_size)
    return mask.bool()


def mask_to_normalized_box(
    mask: torch.Tensor,
    *,
    image_path: Path,
) -> list[float] | None:
    if not mask.any():
        return None
    width, height = Image.open(image_path).size
    full = resize_bool_mask(mask, (height, width))
    ys, xs = full.nonzero(as_tuple=True)
    if xs.numel() == 0:
        return None
    x0 = float(xs.min().item())
    x1 = float(xs.max().item() + 1)
    y0 = float(ys.min().item())
    y1 = float(ys.max().item() + 1)
    return [
        max(0.0, min(1.0, x0 / float(width))),
        max(0.0, min(1.0, y0 / float(height))),
        max(1.0 / float(width), min(1.0, (x1 - x0) / float(width))),
        max(1.0 / float(height), min(1.0, (y1 - y0) / float(height))),
    ]


def collect_frame_objects(
    results: Sequence[Dict[str, Any]],
    *,
    output_size: tuple[int, int],
) -> Dict[int, Dict[int, torch.Tensor]]:
    frame_objects: Dict[int, Dict[int, torch.Tensor]] = {}
    for result in results:
        if result is None:
            continue
        frame_idx = int(result.get("frame_index", -1))
        if frame_idx < 0:
            continue
        outputs = result.get("outputs", {}) or {}
        obj_ids = np.asarray(outputs.get("out_obj_ids", []), dtype=np.int64).reshape(-1)
        raw_masks = outputs.get("out_binary_masks", [])
        masks_np = np.asarray(raw_masks)
        if obj_ids.size == 0 or masks_np.size == 0:
            frame_objects.setdefault(frame_idx, {})
            continue
        if masks_np.ndim == 4 and masks_np.shape[1] == 1:
            masks_np = masks_np[:, 0]
        if masks_np.ndim == 2:
            masks_np = masks_np[None]
        objects = frame_objects.setdefault(frame_idx, {})
        for obj_id, mask_np in zip(obj_ids.tolist(), masks_np):
            mask = torch.from_numpy(np.asarray(mask_np).astype(bool))
            objects[int(obj_id)] = resize_bool_mask(mask, output_size)
    return frame_objects


def collect_frame_scores(
    results: Sequence[Dict[str, Any]],
) -> Dict[int, Dict[int, float]]:
    """Collect SAM3 detector/tracker probabilities without changing mask selection."""

    frame_scores: Dict[int, Dict[int, float]] = {}
    for result in results:
        if result is None:
            continue
        frame_idx = int(result.get("frame_index", -1))
        if frame_idx < 0:
            continue
        outputs = result.get("outputs", {}) or {}
        obj_ids = np.asarray(outputs.get("out_obj_ids", []), dtype=np.int64).reshape(-1)
        probabilities = np.asarray(outputs.get("out_probs", []), dtype=np.float32).reshape(-1)
        scores = frame_scores.setdefault(frame_idx, {})
        for obj_id, probability in zip(obj_ids.tolist(), probabilities.tolist()):
            scores[int(obj_id)] = float(probability)
    return frame_scores


def scores_for_selected_object(
    frame_scores: Dict[int, Dict[int, float]],
    *,
    selected_obj_id: Optional[int],
    num_frames: int,
) -> torch.Tensor:
    scores = torch.zeros(int(num_frames), dtype=torch.float32)
    if selected_obj_id is None:
        return scores
    for frame_idx in range(int(num_frames)):
        scores[frame_idx] = float(
            frame_scores.get(frame_idx, {}).get(int(selected_obj_id), 0.0)
        )
    return scores


def select_tracked_object_id(
    frame_objects: Dict[int, Dict[int, torch.Tensor]],
    *,
    prompt_frame_idx: int,
    reference_mask: torch.Tensor | None,
) -> Optional[int]:
    objects = frame_objects.get(int(prompt_frame_idx), {})
    if not objects:
        return None
    if reference_mask is not None and reference_mask.any():
        best_obj_id = None
        best_iou = -1.0
        for obj_id, mask in objects.items():
            iou = binary_iou(mask, reference_mask)
            if iou > best_iou:
                best_iou = iou
                best_obj_id = obj_id
        if best_obj_id is not None:
            return int(best_obj_id)
    return int(max(objects.items(), key=lambda item: int(item[1].sum().item()))[0])


def masks_for_selected_object(
    frame_objects: Dict[int, Dict[int, torch.Tensor]],
    *,
    selected_obj_id: Optional[int],
    num_frames: int,
    output_size: tuple[int, int],
) -> torch.Tensor:
    masks = torch.zeros(
        int(num_frames),
        int(output_size[0]),
        int(output_size[1]),
        dtype=torch.bool,
    )
    if selected_obj_id is None:
        return masks
    for frame_idx in range(num_frames):
        mask = frame_objects.get(frame_idx, {}).get(int(selected_obj_id))
        if mask is not None:
            masks[frame_idx] = mask.bool()
    return masks


def resize_bool_mask(mask: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
    if tuple(mask.shape[-2:]) == tuple(output_size):
        return mask.bool()
    resized = F.interpolate(
        mask.float()[None, None],
        size=output_size,
        mode="nearest",
    )
    return resized[0, 0].bool()


def binary_iou(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred = pred.bool()
    target = target.bool()
    union = (pred | target).sum().item()
    if union == 0:
        return 1.0
    return float((pred & target).sum().item() / union)
