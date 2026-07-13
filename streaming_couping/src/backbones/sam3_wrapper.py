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
    """Run original SAM3 tracking and geometry-prompt re-segmentation.

    The fallback call starts a fresh one-frame SAM3 session. It is deliberately
    named re-segmentation rather than tracking: the geometry bridge proposes a
    current-frame box and SAM3 converts that coarse proposal into a dense mask.
    """

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

    def segment_candidate(
        self,
        image_path: str | Path,
        *,
        prompt: str,
        output_size: tuple[int, int],
        candidate_mask: torch.Tensor,
        supported_mask: torch.Tensor,
        prompt_mode: str,
    ) -> tuple[torch.Tensor, float]:
        """Refine one geometry proposal with a real SAM3 prompt path."""

        if prompt_mode not in {"box", "point", "box_point"}:
            raise ValueError(f"Unsupported fallback prompt mode: {prompt_mode}")
        if prompt_mode == "box":
            return self._segment_with_box(
                image_path,
                prompt=prompt,
                output_size=output_size,
                candidate_mask=candidate_mask,
            )
        return self._segment_with_points(
            image_path,
            prompt=prompt,
            output_size=output_size,
            candidate_mask=candidate_mask,
            supported_mask=supported_mask,
            use_box_first=prompt_mode == "box_point",
        )

    def track_with_memory_writeback(
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
        """Recover one frame, write it to SAM3 memory, then re-track the future.

        Geometry is used exactly once to obtain ``recovery_mask``. This method
        converts that mask to positive tracker prompts for the already selected
        object ID. SAM3's point-refinement path runs its memory encoder, so all
        later masks come from the original tracker with the corrected memory.
        """

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
        points = _positive_points(recovery_mask)
        if not points:
            raise ValueError("Recovery mask is empty; it cannot update SAM3 memory.")

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

                    corrected = predictor.add_prompt(
                        session_id=session_id,
                        frame_idx=recovery_frame_idx,
                        points=points,
                        point_labels=[1] * len(points),
                        obj_id=int(selected_obj_id),
                        output_prob_thresh=self.output_threshold,
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

        results = [prompted, *pre_recovery_results, corrected, *future_results]
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

    def _segment_with_box(
        self,
        image_path: str | Path,
        *,
        prompt: str,
        output_size: tuple[int, int],
        candidate_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        if not self.prompt_with_box:
            raise RuntimeError(
                "The box fallback requires sam3.prompt_with_box=true."
            )
        adapter = self._require_adapter()
        output = adapter.track_from_paths(
            [image_path],
            prompt=prompt,
            output_size=output_size,
            prompt_frame_idx=0,
            reference_mask=candidate_mask,
            quiet=True,
        )
        score = 0.0
        if output.scores is not None and output.scores.numel() > 0:
            score = float(output.scores[0].detach().cpu())
        return output.masks[0].detach().cpu().bool(), score

    def _segment_with_points(
        self,
        image_path: str | Path,
        *,
        prompt: str,
        output_size: tuple[int, int],
        candidate_mask: torch.Tensor,
        supported_mask: torch.Tensor,
        use_box_first: bool,
    ) -> tuple[torch.Tensor, float]:
        predictor = self.predictor
        if predictor is None:
            raise RuntimeError("Call SAM3Wrapper.load() before inference.")
        points = _positive_points(supported_mask)
        if not points:
            return torch.zeros(output_size, dtype=torch.bool), 0.0

        with tempfile.TemporaryDirectory(prefix="sam3_geometry_prompt_") as tmp:
            tmp_dir = Path(tmp)
            materialize_video_dir([image_path], tmp_dir)
            with quiet_sam3_output(True):
                session = predictor.start_session(resource_path=str(tmp_dir))
                session_id = session["session_id"] if isinstance(session, dict) else session
                try:
                    results = []
                    selected_obj_id = 0
                    if use_box_first:
                        box = mask_to_normalized_box(
                            candidate_mask,
                            image_path=Path(image_path),
                        )
                        if box is None:
                            return torch.zeros(output_size, dtype=torch.bool), 0.0
                        detected = predictor.add_prompt(
                            session_id=session_id,
                            frame_idx=0,
                            text=prompt,
                            bounding_boxes=[box],
                            bounding_box_labels=[1],
                            output_prob_thresh=self.output_threshold,
                        )
                        results.append(detected)
                        detected_objects = collect_frame_objects(
                            [detected],
                            output_size=output_size,
                        )
                        detected_id = select_tracked_object_id(
                            detected_objects,
                            prompt_frame_idx=0,
                            reference_mask=(
                                supported_mask
                                if supported_mask.any()
                                else candidate_mask
                            ),
                        )
                        if detected_id is None:
                            return torch.zeros(output_size, dtype=torch.bool), 0.0
                        selected_obj_id = int(detected_id)

                    refined = predictor.add_prompt(
                        session_id=session_id,
                        frame_idx=0,
                        points=points,
                        point_labels=[1] * len(points),
                        obj_id=selected_obj_id,
                        output_prob_thresh=self.output_threshold,
                    )
                    results.append(refined)
                finally:
                    predictor.close_session(session_id)

        frame_objects = collect_frame_objects(results, output_size=output_size)
        masks = masks_for_selected_object(
            frame_objects,
            selected_obj_id=selected_obj_id,
            num_frames=1,
            output_size=output_size,
        )
        frame_scores = collect_frame_scores(results)
        scores = scores_for_selected_object(
            frame_scores,
            selected_obj_id=selected_obj_id,
            num_frames=1,
        )
        score = float(scores[0]) if scores.numel() else 0.0
        if score == 0.0 and masks[0].any():
            score = 1.0
        return masks[0].bool(), score

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
