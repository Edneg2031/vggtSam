"""SAM3 wrapper backed by the repository's tested video adapter."""

from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Sequence

import torch

from vggtsam.adapters.sam3_video import (
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

from ..types import TrackingSequence


class SAM3Wrapper:
    """Run original SAM3 tracking and same-instance geometry correction."""

    def __init__(
        self,
        *,
        repo_path: str | Path,
        checkpoint_path: str | Path,
        device: str,
        output_threshold: float,
        prompt_with_box: bool,
    ) -> None:
        self.repo_path = Path(repo_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.device = str(device)
        self.output_threshold = float(output_threshold)
        self.prompt_with_box = bool(prompt_with_box)
        self.predictor = None
        self.adapter = None

    def load(self) -> "SAM3Wrapper":
        self.predictor = load_sam3_video_predictor(
            repo_path=self.repo_path,
            checkpoint_path=self.checkpoint_path,
            device=self.device,
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
    ) -> TrackingSequence:
        adapter = self._require_adapter()
        output = adapter.track_from_paths(
            image_paths,
            prompt=prompt,
            output_size=output_size,
            prompt_frame_idx=reference_frame_idx,
            reference_mask=reference_mask,
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

    def track_with_memory_position_warp(
        self,
        image_paths: Sequence[str | Path],
        *,
        prompt: str,
        output_size: tuple[int, int],
        reference_frame_idx: int,
        reference_mask: torch.Tensor,
        warper,
    ) -> TrackingSequence:
        """Run original SAM3 while changing only historical memory positions."""

        predictor = self.predictor
        if predictor is None:
            raise RuntimeError("Call SAM3Wrapper.load() before inference.")
        tracker = getattr(getattr(predictor, "model", None), "tracker", None)
        if tracker is None or not hasattr(
            tracker, "_prepare_memory_conditioned_features"
        ):
            raise RuntimeError(
                "The loaded SAM3 tracker does not expose its memory-conditioning path."
            )
        from ..bridge.memory_warp import install_memory_position_warp

        with install_memory_position_warp(tracker, warper):
            return self.track(
                image_paths,
                prompt=prompt,
                output_size=output_size,
                reference_frame_idx=reference_frame_idx,
                reference_mask=reference_mask,
            )

    def recover_mask_with_text_geometry(
        self,
        image_path: str | Path,
        *,
        prompt: str,
        output_size: tuple[int, int],
        candidate_mask: torch.Tensor,
        supported_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        """Generate one dense recovery mask from text, geometry box, and points."""

        predictor = self.predictor
        if predictor is None:
            raise RuntimeError("Call SAM3Wrapper.load() before inference.")
        if not self.prompt_with_box:
            raise RuntimeError("Text-guided recovery requires sam3.prompt_with_box=true.")
        points = _positive_points(supported_mask)
        if not points:
            raise ValueError("The geometry-supported recovery points are empty.")

        with tempfile.TemporaryDirectory(prefix="sam3_text_geometry_recovery_") as tmp:
            tmp_dir = Path(tmp)
            materialize_video_dir([image_path], tmp_dir)
            with quiet_sam3_output(True):
                session = predictor.start_session(resource_path=str(tmp_dir))
                session_id = session["session_id"] if isinstance(session, dict) else session
                try:
                    box = mask_to_normalized_box(
                        candidate_mask,
                        image_path=Path(image_path),
                    )
                    if box is None:
                        raise RuntimeError("The geometry recovery box is empty.")
                    detected = predictor.add_prompt(
                        session_id=session_id,
                        frame_idx=0,
                        text=prompt,
                        bounding_boxes=[box],
                        bounding_box_labels=[1],
                        output_prob_thresh=self.output_threshold,
                    )
                    detected_objects = collect_frame_objects(
                        [detected],
                        output_size=output_size,
                    )
                    temporary_obj_id = select_tracked_object_id(
                        detected_objects,
                        prompt_frame_idx=0,
                        reference_mask=(
                            supported_mask if supported_mask.any() else candidate_mask
                        ),
                    )
                    if temporary_obj_id is None:
                        raise RuntimeError(
                            "Text-guided SAM3 recovery found no matching object."
                        )
                    refined = predictor.add_prompt(
                        session_id=session_id,
                        frame_idx=0,
                        points=points,
                        point_labels=[1] * len(points),
                        obj_id=int(temporary_obj_id),
                        output_prob_thresh=self.output_threshold,
                    )
                finally:
                    predictor.close_session(session_id)

        results = [detected, refined]
        frame_objects = collect_frame_objects(results, output_size=output_size)
        masks = masks_for_selected_object(
            frame_objects,
            selected_obj_id=temporary_obj_id,
            num_frames=1,
            output_size=output_size,
        )
        frame_scores = collect_frame_scores(results)
        scores = scores_for_selected_object(
            frame_scores,
            selected_obj_id=temporary_obj_id,
            num_frames=1,
        )
        score = float(scores[0]) if scores.numel() else 0.0
        if score == 0.0 and masks[0].any():
            score = 1.0
        return masks[0].detach().cpu().bool(), score

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
        """Write a shared dense recovery mask into the persistent SAM3 object."""

        predictor = self.predictor
        if predictor is None:
            raise RuntimeError("Call SAM3Wrapper.load() before inference.")
        if not image_paths:
            raise ValueError("At least one image is required for memory writeback.")
        recovery_frame_idx = int(recovery_frame_idx)
        reference_frame_idx = int(reference_frame_idx)
        if recovery_frame_idx <= reference_frame_idx:
            raise ValueError("Recovery must happen after the reference frame.")
        if recovery_frame_idx >= len(image_paths):
            raise ValueError("Recovery frame is outside the input sequence.")
        if not recovery_mask.any():
            raise ValueError("The recovery mask is empty; it cannot update memory.")
        with tempfile.TemporaryDirectory(prefix="sam3_memory_writeback_") as tmp:
            tmp_dir = Path(tmp)
            materialize_video_dir(image_paths, tmp_dir)
            with quiet_sam3_output(True):
                session = predictor.start_session(resource_path=str(tmp_dir))
                session_id = session["session_id"] if isinstance(session, dict) else session
                try:
                    add_kwargs = {
                        "session_id": session_id,
                        "frame_idx": reference_frame_idx,
                        "text": prompt,
                        "output_prob_thresh": self.output_threshold,
                    }
                    if self.prompt_with_box:
                        reference_box = mask_to_normalized_box(
                            reference_mask,
                            image_path=Path(image_paths[reference_frame_idx]),
                        )
                        if reference_box is not None:
                            add_kwargs["bounding_boxes"] = [reference_box]
                            add_kwargs["bounding_box_labels"] = [1]
                    prompted = predictor.add_prompt(**add_kwargs)
                    # Stop at the recovery frame. Running the full sequence and
                    # then correcting an earlier frame would leak future tracker
                    # state into what is intended to be a causal memory test.
                    pre_recovery_results = list(
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
                    original_objects = collect_frame_objects(
                        [prompted, *pre_recovery_results],
                        output_size=output_size,
                    )
                    selected_obj_id = select_tracked_object_id(
                        original_objects,
                        prompt_frame_idx=reference_frame_idx,
                        reference_mask=reference_mask,
                    )
                    if selected_obj_id is None:
                        raise RuntimeError(
                            "SAM3 did not produce an object ID on the reference frame."
                        )

                    self._write_mask_to_existing_memory(
                        session_id=session_id,
                        frame_idx=recovery_frame_idx,
                        obj_id=int(selected_obj_id),
                        mask=recovery_mask,
                    )
                    future_results = []
                    future_start = recovery_frame_idx + 1
                    if future_start < len(image_paths):
                        future_results = list(
                            predictor.propagate_in_video(
                                session_id=session_id,
                                propagation_direction="forward",
                                start_frame_idx=future_start,
                                max_frame_num_to_track=(
                                    len(image_paths) - 1 - future_start
                                ),
                                output_prob_thresh=self.output_threshold,
                            )
                        )
                finally:
                    predictor.close_session(session_id)

        results = [prompted, *pre_recovery_results, *future_results]
        frame_objects = collect_frame_objects(results, output_size=output_size)
        frame_scores = collect_frame_scores(results)
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
        masks[recovery_frame_idx] = recovery_mask.detach().cpu().bool()
        scores[recovery_frame_idx] = 1.0
        return TrackingSequence(
            masks=masks.bool(),
            scores=scores.float(),
            selected_obj_id=int(selected_obj_id),
        )

    def track_split_without_memory(
        self,
        image_paths: Sequence[str | Path],
        *,
        prompt: str,
        output_size: tuple[int, int],
        reference_frame_idx: int,
        reference_mask: torch.Tensor,
        split_frame_idx: int,
    ) -> TrackingSequence:
        """Run the same causal split as memory writeback, without correction."""

        predictor = self.predictor
        if predictor is None:
            raise RuntimeError("Call SAM3Wrapper.load() before inference.")
        split_frame_idx = int(split_frame_idx)
        reference_frame_idx = int(reference_frame_idx)
        if split_frame_idx <= reference_frame_idx:
            raise ValueError("The causal split must happen after the reference frame.")
        if split_frame_idx >= len(image_paths):
            raise ValueError("The causal split is outside the input sequence.")

        with tempfile.TemporaryDirectory(prefix="sam3_no_memory_") as tmp:
            tmp_dir = Path(tmp)
            materialize_video_dir(image_paths, tmp_dir)
            with quiet_sam3_output(True):
                session = predictor.start_session(resource_path=str(tmp_dir))
                session_id = session["session_id"] if isinstance(session, dict) else session
                try:
                    add_kwargs = {
                        "session_id": session_id,
                        "frame_idx": reference_frame_idx,
                        "text": prompt,
                        "output_prob_thresh": self.output_threshold,
                    }
                    if self.prompt_with_box:
                        reference_box = mask_to_normalized_box(
                            reference_mask,
                            image_path=Path(image_paths[reference_frame_idx]),
                        )
                        if reference_box is not None:
                            add_kwargs["bounding_boxes"] = [reference_box]
                            add_kwargs["bounding_box_labels"] = [1]
                    prompted = predictor.add_prompt(**add_kwargs)
                    before_split = list(
                        predictor.propagate_in_video(
                            session_id=session_id,
                            propagation_direction="forward",
                            start_frame_idx=reference_frame_idx,
                            max_frame_num_to_track=(
                                split_frame_idx - reference_frame_idx
                            ),
                            output_prob_thresh=self.output_threshold,
                        )
                    )
                    frame_objects = collect_frame_objects(
                        [prompted, *before_split],
                        output_size=output_size,
                    )
                    selected_obj_id = select_tracked_object_id(
                        frame_objects,
                        prompt_frame_idx=reference_frame_idx,
                        reference_mask=reference_mask,
                    )
                    if selected_obj_id is None:
                        raise RuntimeError(
                            "SAM3 did not produce an object ID on the reference frame."
                        )
                    after_split = []
                    future_start = split_frame_idx + 1
                    if future_start < len(image_paths):
                        after_split = list(
                            predictor.propagate_in_video(
                                session_id=session_id,
                                propagation_direction="forward",
                                start_frame_idx=future_start,
                                max_frame_num_to_track=(
                                    len(image_paths) - 1 - future_start
                                ),
                                output_prob_thresh=self.output_threshold,
                            )
                        )
                finally:
                    predictor.close_session(session_id)

        results = [prompted, *before_split, *after_split]
        frame_objects = collect_frame_objects(results, output_size=output_size)
        frame_scores = collect_frame_scores(results)
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
        return TrackingSequence(
            masks=masks.bool(),
            scores=scores.float(),
            selected_obj_id=int(selected_obj_id),
        )

    def _write_mask_to_existing_memory(
        self,
        *,
        session_id: str,
        frame_idx: int,
        obj_id: int,
        mask: torch.Tensor,
    ) -> None:
        """Encode a full mask into one existing ID without recreating it."""

        predictor = self.predictor
        if predictor is None:
            raise RuntimeError("Call SAM3Wrapper.load() before inference.")
        session = predictor._get_session(session_id)
        inference_state = session["state"]
        model = predictor.model
        obj_rank = model._get_gpu_id_by_obj_id(inference_state, obj_id)
        if obj_rank is None:
            raise RuntimeError(f"SAM3 tracker object {obj_id} is unavailable.")
        if int(obj_rank) != int(model.rank):
            raise RuntimeError(
                "Mask memory writeback currently requires the selected object "
                "to be local to the predictor rank."
            )
        tracker_states = model._get_tracker_inference_states_by_obj_ids(
            inference_state,
            [obj_id],
        )
        if len(tracker_states) != 1:
            raise RuntimeError(
                f"Expected one SAM3 tracker state for object {obj_id}, got "
                f"{len(tracker_states)}."
            )
        tracker_state = tracker_states[0]
        self._reactivate_existing_object(
            inference_state=inference_state,
            model=model,
            frame_idx=frame_idx,
            obj_id=obj_id,
        )
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

        tracker_metadata = inference_state["tracker_metadata"]
        model.add_action_history(
            inference_state,
            "refine",
            frame_idx=frame_idx,
            obj_ids=[obj_id],
        )
        tracker_metadata["obj_id_to_score"][obj_id] = 1.0
        tracker_metadata["obj_id_to_tracker_score_frame_wise"][frame_idx][obj_id] = 1.0

        if int(model.rank) != 0:
            return
        rank0_metadata = tracker_metadata.get("rank0_metadata", {})
        rank0_metadata.get("removed_obj_ids", set()).discard(obj_id)
        for suppressed_ids in rank0_metadata.get("suppressed_obj_ids", {}).values():
            suppressed_ids.discard(obj_id)

        confirmation = rank0_metadata.get("masklet_confirmation")
        if confirmation is None:
            return
        # obj_ids_all_gpu is a NumPy array in SAM3's tracker metadata.
        matching = [
            index
            for index, candidate_id in enumerate(tracker_metadata["obj_ids_all_gpu"])
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


def _positive_points(mask: torch.Tensor, *, max_points: int = 3) -> list[list[float]]:
    """Choose stable positive prompts from the geometry-supported pixels."""

    coordinates = mask.bool().nonzero().float()
    if coordinates.numel() == 0:
        return []
    height, width = mask.shape

    # Start at the supported-pixel medoid, then spread the remaining prompts
    # with farthest-point sampling. Every returned point is an observed support
    # pixel rather than the center of a potentially hollow candidate box.
    centroid = coordinates.mean(dim=0, keepdim=True)
    selected_indices = [int(torch.argmin(torch.cdist(coordinates, centroid)))]
    while len(selected_indices) < min(max_points, coordinates.shape[0]):
        selected = coordinates[selected_indices]
        distance = torch.cdist(coordinates, selected).amin(dim=1)
        distance[selected_indices] = -1
        selected_indices.append(int(torch.argmax(distance)))

    return [
        [
            float((coordinates[index, 1] + 0.5) / width),
            float((coordinates[index, 0] + 0.5) / height),
        ]
        for index in selected_indices
    ]
