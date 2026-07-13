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
