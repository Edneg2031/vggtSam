"""SAM3 wrapper backed by the repository's tested video adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch

from vggtsam.adapters.sam3_video import (
    SAM3VideoTrackerAdapter,
    load_sam3_video_predictor,
)

from ..types import TrackingSequence


class SAM3Wrapper:
    """Run original SAM3 tracking and geometry-box re-segmentation.

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
    ) -> tuple[torch.Tensor, float]:
        """Refine one geometry proposal with SAM3 text + box prompting."""

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

    def _require_adapter(self) -> SAM3VideoTrackerAdapter:
        if self.adapter is None:
            raise RuntimeError("Call SAM3Wrapper.load() before inference.")
        return self.adapter

