"""Minimal SAM3 interface for tracking, re-detection, and memory writeback."""

from __future__ import annotations

import tempfile
from collections.abc import Sequence
from pathlib import Path

import torch

from ..types import SAM3MaskCandidate, TrackingSequence
from .sam3_video import (
    SAM3MultiTrackOutput,
    SAM3VideoTrackerAdapter,
    collect_frame_objects,
    collect_frame_scores,
    load_sam3_video_predictor,
    mask_to_normalized_box,
    masks_for_selected_object,
    materialize_video_dir,
    quiet_sam3_output,
    scores_for_selected_object,
    select_tracked_object_id,
)


class SAM3Wrapper:
    """Expose only the SAM3 operations used by the final tracking pipeline."""

    def __init__(
        self,
        *,
        repo_path: str | Path,
        checkpoint_path: str | Path,
        device: str,
        output_threshold: float,
        prompt_with_box: bool,
        version: str = "sam3",
        use_fa3: bool = False,
        max_num_objects: int = 16,
        multiplex_count: int = 16,
    ) -> None:
        self.repo_path = Path(repo_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.device = str(device)
        self.output_threshold = float(output_threshold)
        self.prompt_with_box = bool(prompt_with_box)
        self.version = str(version).strip().lower()
        self.use_fa3 = bool(use_fa3)
        self.max_num_objects = int(max_num_objects)
        self.multiplex_count = int(multiplex_count)
        self.predictor = None
        self.adapter = None

    def load(self) -> SAM3Wrapper:
        self.predictor = load_sam3_video_predictor(
            repo_path=self.repo_path,
            checkpoint_path=self.checkpoint_path,
            device=self.device,
            version=self.version,
            use_fa3=self.use_fa3,
            max_num_objects=self.max_num_objects,
            multiplex_count=self.multiplex_count,
            quiet=True,
        )
        self.adapter = SAM3VideoTrackerAdapter(
            self.predictor,
            output_prob_thresh=self.output_threshold,
            prompt_with_box=self.prompt_with_box,
        )
        return self

    def track(
        self,
        image_paths: Sequence[str | Path],
        *,
        prompt: str,
        output_size: tuple[int, int],
        reference_frame_idx: int,
        reference_mask: torch.Tensor,
        propagation_direction: str = "both",
    ) -> TrackingSequence:
        """Run the frozen original SAM3 tracker."""

        adapter = self._require_adapter()
        output = adapter.track_from_paths(
            image_paths,
            prompt=prompt,
            output_size=output_size,
            prompt_frame_idx=reference_frame_idx,
            reference_mask=reference_mask,
            propagation_direction=propagation_direction,
            quiet=True,
        )
        scores = output.scores
        if scores is None:
            scores = output.masks.flatten(1).any(dim=1).float()
        return TrackingSequence(
            masks=output.masks.detach().cpu().bool(),
            scores=scores.detach().cpu().float(),
            selected_obj_id=output.selected_obj_id,
        )

    def track_all_forward(
        self,
        image_paths: Sequence[str | Path],
        *,
        prompt: str,
        output_size: tuple[int, int],
        max_objects: int,
    ) -> SAM3MultiTrackOutput:
        """Return dynamically born SAM3.1 tracks from one forward-only run."""

        if self.version != "sam3.1":
            raise RuntimeError("Dynamic forward discovery requires SAM3.1.")
        return self._require_adapter().track_all_forward_from_paths(
            image_paths,
            prompt=prompt,
            output_size=output_size,
            max_objects=max_objects,
            quiet=True,
        )

    def propose_text_masks(
        self,
        image_path: str | Path,
        *,
        prompt: str,
        output_size: tuple[int, int],
    ) -> list[SAM3MaskCandidate]:
        """Return every full mask from one global-text query."""

        return self._propose_masks(
            image_path,
            prompt=prompt,
            output_size=output_size,
        )

    def propose_geometry_point_refined_masks(
        self,
        image_path: str | Path,
        *,
        prompt: str,
        output_size: tuple[int, int],
        geometry_prompt: torch.Tensor,
        positive_prompt: torch.Tensor,
        max_positive_points: int = 6,
    ) -> list[SAM3MaskCandidate]:
        """Run SAM3.1 text+box detection, then geometry-point refinement."""

        return self.propose_geometry_prompted_masks(
            image_path,
            prompt=prompt,
            output_size=output_size,
            geometry_prompt=geometry_prompt,
            positive_prompt=positive_prompt,
            max_positive_points=max_positive_points,
            use_box=True,
            use_points=True,
        )

    def propose_geometry_prompted_masks(
        self,
        image_path: str | Path,
        *,
        prompt: str,
        output_size: tuple[int, int],
        geometry_prompt: torch.Tensor,
        positive_prompt: torch.Tensor,
        max_positive_points: int = 6,
        use_box: bool = True,
        use_points: bool = True,
    ) -> list[SAM3MaskCandidate]:
        """Run a controlled text/box/point SAM3.1 prompt ablation.

        The geometry masks are always retained for candidate scoring, while
        ``use_box`` and ``use_points`` control only which prompts SAM3.1 sees.
        This keeps the candidate budget and post-prompt geometry gate matched
        across box-only, points-only, and box+points variants.
        """

        if self.version != "sam3.1":
            raise RuntimeError(
                "Geometry point refinement requires a SAM3.1 predictor."
            )
        if not bool(use_box) and not bool(use_points):
            raise ValueError("At least one geometry prompt type is required.")
        prompt_box = mask_to_normalized_box(
            geometry_prompt.detach().cpu().bool(),
            image_path=Path(image_path),
        )
        positive = _sample_normalized_points(
            positive_prompt,
            limit=max_positive_points,
        )
        if (bool(use_box) and prompt_box is None) or (
            bool(use_points) and not positive
        ):
            return []
        predictor = self._require_predictor()
        with tempfile.TemporaryDirectory(
            prefix="sam31_geometry_points_"
        ) as tmp:
            video_dir = Path(tmp)
            materialize_video_dir([image_path], video_dir)
            with quiet_sam3_output(True):
                session = predictor.start_session(
                    resource_path=str(video_dir)
                )
                session_id = _session_id(session)
                try:
                    detection_kwargs = {
                        "session_id": session_id,
                        "frame_idx": 0,
                        "text": prompt,
                        "output_prob_thresh": self.output_threshold,
                    }
                    if bool(use_box):
                        detection_kwargs["bounding_boxes"] = [prompt_box]
                        detection_kwargs["bounding_box_labels"] = [1]
                    detected = predictor.add_prompt(**detection_kwargs)
                    detected_objects = collect_frame_objects(
                        [detected],
                        output_size=output_size,
                    ).get(0, {})
                    selected_obj_id = _select_geometry_prompted_object(
                        detected_objects,
                        positive_prompt=positive_prompt,
                        geometry_prompt=geometry_prompt,
                    )
                    if selected_obj_id is None and bool(use_points):
                        selected_obj_id = (
                            max(detected_objects, default=0) + 1
                        )
                    if selected_obj_id is None:
                        return []
                    if bool(use_points):
                        refined = predictor.add_prompt(
                            session_id=session_id,
                            frame_idx=0,
                            points=positive,
                            point_labels=[1] * len(positive),
                            clear_old_points=True,
                            obj_id=int(selected_obj_id),
                            rel_coordinates=True,
                            output_prob_thresh=self.output_threshold,
                        )
                    else:
                        refined = detected
                finally:
                    predictor.close_session(session_id)

        refined_objects = collect_frame_objects(
            [refined],
            output_size=output_size,
        ).get(0, {})
        mask = refined_objects.get(int(selected_obj_id))
        if mask is None:
            return []
        refined_scores = collect_frame_scores([refined]).get(0, {})
        detected_scores = collect_frame_scores([detected]).get(0, {})
        score = float(
            refined_scores.get(
                int(selected_obj_id),
                detected_scores.get(int(selected_obj_id), 0.0),
            )
        )
        if score == 0.0 and mask.any():
            score = 1.0
        return [
            SAM3MaskCandidate(
                obj_id=int(selected_obj_id),
                mask=mask.detach().cpu().bool(),
                score=score,
            )
        ]

    def _propose_masks(
        self,
        image_path: str | Path,
        *,
        prompt: str,
        output_size: tuple[int, int],
    ) -> list[SAM3MaskCandidate]:
        predictor = self._require_predictor()
        with tempfile.TemporaryDirectory(prefix="sam3_text_candidates_") as tmp:
            video_dir = Path(tmp)
            materialize_video_dir([image_path], video_dir)
            with quiet_sam3_output(True):
                session = predictor.start_session(
                    resource_path=str(video_dir)
                )
                session_id = _session_id(session)
                try:
                    detected = predictor.add_prompt(
                        session_id=session_id,
                        frame_idx=0,
                        text=prompt,
                        output_prob_thresh=self.output_threshold,
                    )
                finally:
                    predictor.close_session(session_id)

        objects = collect_frame_objects(
            [detected],
            output_size=output_size,
        ).get(0, {})
        scores = collect_frame_scores([detected]).get(0, {})
        candidates = []
        for obj_id, mask in sorted(objects.items()):
            score = float(scores.get(int(obj_id), 0.0))
            if score == 0.0 and mask.any():
                score = 1.0
            candidates.append(
                SAM3MaskCandidate(
                    obj_id=int(obj_id),
                    mask=mask.detach().cpu().bool(),
                    score=score,
                )
            )
        return candidates

    def track_with_recovery_mask_memory(
        self,
        image_paths: Sequence[str | Path],
        *,
        prompt: str,
        output_size: tuple[int, int],
        reference_frame_idx: int,
        reference_mask: torch.Tensor,
        recovery_frame_idx: int,
        recovery_mask: torch.Tensor,
    ) -> TrackingSequence:
        """Write one full mask into the original persistent SAM3 object."""

        predictor = self._require_predictor()
        reference_frame_idx, recovery_frame_idx = _validate_split(
            image_paths,
            reference_frame_idx=reference_frame_idx,
            split_frame_idx=recovery_frame_idx,
        )
        if not recovery_mask.any():
            raise ValueError("The recovery mask is empty.")

        with tempfile.TemporaryDirectory(
            prefix="sam3_memory_writeback_"
        ) as tmp:
            video_dir = Path(tmp)
            materialize_video_dir(image_paths, video_dir)
            with quiet_sam3_output(True):
                session = predictor.start_session(
                    resource_path=str(video_dir)
                )
                session_id = _session_id(session)
                try:
                    prompted = predictor.add_prompt(
                        **self._reference_prompt_kwargs(
                            session_id=session_id,
                            image_paths=image_paths,
                            prompt=prompt,
                            reference_frame_idx=reference_frame_idx,
                            reference_mask=reference_mask,
                        )
                    )
                    before_recovery = list(
                        predictor.propagate_in_video(
                            session_id=session_id,
                            propagation_direction="forward",
                            start_frame_idx=reference_frame_idx,
                            max_frame_num_to_track=(
                                recovery_frame_idx - reference_frame_idx
                            ),
                            output_prob_thresh=self.output_threshold,
                        )
                    )
                    selected_obj_id = _select_persistent_id(
                        [prompted, *before_recovery],
                        output_size=output_size,
                        reference_frame_idx=reference_frame_idx,
                        reference_mask=reference_mask,
                    )
                    self._write_mask_to_existing_memory(
                        session_id=session_id,
                        frame_idx=recovery_frame_idx,
                        obj_id=selected_obj_id,
                        mask=recovery_mask,
                    )
                    after_recovery = self._propagate_future(
                        session_id=session_id,
                        image_paths=image_paths,
                        start_frame_idx=recovery_frame_idx + 1,
                    )
                finally:
                    predictor.close_session(session_id)

        tracking = _collect_tracking(
            [prompted, *before_recovery, *after_recovery],
            selected_obj_id=selected_obj_id,
            num_frames=len(image_paths),
            output_size=output_size,
        )
        masks = tracking.masks.clone()
        scores = tracking.scores.clone()
        masks[recovery_frame_idx] = recovery_mask.detach().cpu().bool()
        scores[recovery_frame_idx] = 1.0
        return TrackingSequence(
            masks=masks,
            scores=scores,
            selected_obj_id=tracking.selected_obj_id,
        )

    def _reference_prompt_kwargs(
        self,
        *,
        session_id: str,
        image_paths: Sequence[str | Path],
        prompt: str,
        reference_frame_idx: int,
        reference_mask: torch.Tensor,
    ) -> dict:
        kwargs = {
            "session_id": session_id,
            "frame_idx": reference_frame_idx,
            "text": prompt,
            "output_prob_thresh": self.output_threshold,
        }
        if self.prompt_with_box:
            box = mask_to_normalized_box(
                reference_mask,
                image_path=Path(image_paths[reference_frame_idx]),
            )
            if box is not None:
                kwargs["bounding_boxes"] = [box]
                kwargs["bounding_box_labels"] = [1]
        return kwargs

    def _propagate_future(
        self,
        *,
        session_id: str,
        image_paths: Sequence[str | Path],
        start_frame_idx: int,
    ) -> list:
        if start_frame_idx >= len(image_paths):
            return []
        predictor = self._require_predictor()
        return list(
            predictor.propagate_in_video(
                session_id=session_id,
                propagation_direction="forward",
                start_frame_idx=start_frame_idx,
                max_frame_num_to_track=(
                    len(image_paths) - 1 - start_frame_idx
                ),
                output_prob_thresh=self.output_threshold,
            )
        )

    def _write_mask_to_existing_memory(
        self,
        *,
        session_id: str,
        frame_idx: int,
        obj_id: int,
        mask: torch.Tensor,
    ) -> None:
        predictor = self._require_predictor()
        inference_state = predictor._get_session(session_id)["state"]
        model = predictor.model
        obj_rank = model._get_gpu_id_by_obj_id(inference_state, obj_id)
        if obj_rank is None:
            raise RuntimeError(f"SAM3 object {obj_id} is unavailable.")
        if int(obj_rank) != int(model.rank):
            raise RuntimeError(
                "Memory writeback requires the object to be local to the "
                "predictor rank."
            )
        tracker_states = model._get_tracker_inference_states_by_obj_ids(
            inference_state,
            [obj_id],
        )
        if len(tracker_states) != 1:
            raise RuntimeError(
                f"Expected one tracker state for object {obj_id}, got "
                f"{len(tracker_states)}."
            )
        self._reactivate_existing_object(
            inference_state=inference_state,
            model=model,
            frame_idx=frame_idx,
            obj_id=obj_id,
        )
        tracker_state = tracker_states[0]
        model.tracker.add_new_mask(
            inference_state=tracker_state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            mask=mask.detach().bool(),
        )
        model.tracker.propagate_in_video_preflight(
            tracker_state,
            run_mem_encoder=True,
        )

    @staticmethod
    def _reactivate_existing_object(
        *,
        inference_state,
        model,
        frame_idx: int,
        obj_id: int,
    ) -> None:
        """Mirror SAM3's native existing-object refinement bookkeeping."""

        metadata = inference_state["tracker_metadata"]
        model.add_action_history(
            inference_state,
            "refine",
            frame_idx=frame_idx,
            obj_ids=[obj_id],
        )
        metadata["obj_id_to_score"][obj_id] = 1.0
        metadata["obj_id_to_tracker_score_frame_wise"][frame_idx][obj_id] = 1.0
        if int(model.rank) != 0:
            return

        rank0 = metadata.get("rank0_metadata", {})
        rank0.get("removed_obj_ids", set()).discard(obj_id)
        for suppressed in rank0.get("suppressed_obj_ids", {}).values():
            suppressed.discard(obj_id)
        confirmation = rank0.get("masklet_confirmation")
        if confirmation is None:
            return
        matching = [
            index
            for index, candidate_id in enumerate(metadata["obj_ids_all_gpu"])
            if int(candidate_id) == obj_id
        ]
        if not matching or matching[0] >= len(confirmation["status"]):
            return
        obj_index = matching[0]
        confirmation["status"][obj_index] = 1
        confirmation["consecutive_det_num"][obj_index] = (
            model.masklet_confirmation_consecutive_det_thresh
        )

    def _require_adapter(self) -> SAM3VideoTrackerAdapter:
        if self.adapter is None:
            raise RuntimeError("Call SAM3Wrapper.load() before inference.")
        return self.adapter

    def _require_predictor(self):
        if self.predictor is None:
            raise RuntimeError("Call SAM3Wrapper.load() before inference.")
        return self.predictor


def _sample_normalized_points(
    mask: torch.Tensor,
    *,
    limit: int,
) -> list[list[float]]:
    if int(limit) <= 0:
        return []
    mask = mask.detach().cpu().bool()
    if mask.ndim != 2 or not mask.any():
        return []
    ys, xs = mask.nonzero(as_tuple=True)
    coordinates = torch.stack([xs, ys], dim=-1).float()
    count = min(int(limit), int(coordinates.shape[0]))
    center = coordinates.mean(dim=0)
    first = int(
        torch.argmin(((coordinates - center) ** 2).sum(dim=-1))
    )
    selected = [first]
    minimum_distance = (
        (coordinates - coordinates[first]) ** 2
    ).sum(dim=-1)
    while len(selected) < count:
        index = int(torch.argmax(minimum_distance))
        selected.append(index)
        current = ((coordinates - coordinates[index]) ** 2).sum(dim=-1)
        minimum_distance = torch.minimum(minimum_distance, current)
        minimum_distance[selected] = -1.0
    height, width = mask.shape
    return [
        [
            float((coordinates[index, 0] + 0.5) / width),
            float((coordinates[index, 1] + 0.5) / height),
        ]
        for index in selected
    ]


def _select_geometry_prompted_object(
    objects: dict[int, torch.Tensor],
    *,
    positive_prompt: torch.Tensor,
    geometry_prompt: torch.Tensor,
) -> int | None:
    positive = positive_prompt.detach().cpu().bool()
    geometry = geometry_prompt.detach().cpu().bool()
    if not objects or not positive.any():
        return None

    def score(item: tuple[int, torch.Tensor]) -> tuple[float, float, int]:
        obj_id, mask = item
        mask = mask.detach().cpu().bool()
        support_recall = float((positive & mask).sum()) / max(
            int(positive.sum()),
            1,
        )
        box_precision = float((geometry & mask).sum()) / max(
            int(mask.sum()),
            1,
        )
        return (
            0.70 * support_recall + 0.30 * box_precision,
            box_precision,
            -int(obj_id),
        )

    return int(max(objects.items(), key=score)[0])


def _session_id(session) -> str:
    return session["session_id"] if isinstance(session, dict) else session


def _validate_split(
    image_paths: Sequence[str | Path],
    *,
    reference_frame_idx: int,
    split_frame_idx: int,
) -> tuple[int, int]:
    if not image_paths:
        raise ValueError("At least one image is required.")
    reference_frame_idx = int(reference_frame_idx)
    split_frame_idx = int(split_frame_idx)
    if split_frame_idx <= reference_frame_idx:
        raise ValueError("Recovery must happen after the reference frame.")
    if split_frame_idx >= len(image_paths):
        raise ValueError("Recovery frame is outside the input sequence.")
    return reference_frame_idx, split_frame_idx


def _select_persistent_id(
    results: list,
    *,
    output_size: tuple[int, int],
    reference_frame_idx: int,
    reference_mask: torch.Tensor,
) -> int:
    objects = collect_frame_objects(results, output_size=output_size)
    obj_id = select_tracked_object_id(
        objects,
        prompt_frame_idx=reference_frame_idx,
        reference_mask=reference_mask,
    )
    if obj_id is None:
        raise RuntimeError(
            "SAM3 did not produce an object ID on the reference frame."
        )
    return int(obj_id)


def _collect_tracking(
    results: list,
    *,
    selected_obj_id: int,
    num_frames: int,
    output_size: tuple[int, int],
) -> TrackingSequence:
    objects = collect_frame_objects(results, output_size=output_size)
    frame_scores = collect_frame_scores(results)
    return TrackingSequence(
        masks=masks_for_selected_object(
            objects,
            selected_obj_id=selected_obj_id,
            num_frames=num_frames,
            output_size=output_size,
        ).bool(),
        scores=scores_for_selected_object(
            frame_scores,
            selected_obj_id=selected_obj_id,
            num_frames=num_frames,
        ).float(),
        selected_obj_id=selected_obj_id,
    )
