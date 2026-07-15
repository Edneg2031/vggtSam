"""Capture SAM3 decoder outputs before the hard object-presence gate."""

from __future__ import annotations

from contextlib import contextmanager
from types import MethodType
from typing import Iterator

import torch
import torch.nn.functional as F

from ..types import SAM3SoftSequence


class SAM3SoftOutputCapture:
    def __init__(self, *, output_size: tuple[int, int]) -> None:
        self.output_size = tuple(int(value) for value in output_size)
        self._active_frame: int | None = None
        self._capture_enabled = False
        self._records: dict[int, list[tuple[torch.Tensor, float]]] = {}

    def set_active(self, frame_idx: int | None, *, enabled: bool) -> None:
        self._active_frame = None if frame_idx is None else int(frame_idx)
        self._capture_enabled = bool(enabled)

    def record(self, decoder_output) -> None:
        if not self._capture_enabled or self._active_frame is None:
            return
        if not isinstance(decoder_output, (tuple, list)) or len(decoder_output) < 4:
            raise RuntimeError("Unexpected SAM3 mask decoder output signature.")
        masks, ious, _, object_score_logits = decoder_output[:4]
        if masks.ndim != 4 or ious.ndim != 2:
            raise RuntimeError(
                "Expected raw SAM3 masks [B,M,H,W] and IoUs [B,M], got "
                f"{tuple(masks.shape)} and {tuple(ious.shape)}."
            )
        records = self._records.setdefault(self._active_frame, [])
        for batch_index in range(masks.shape[0]):
            mask_index = int(ious[batch_index].argmax().item())
            raw_logit = masks[batch_index, mask_index].detach().float()
            resized_logit = F.interpolate(
                raw_logit[None, None],
                size=self.output_size,
                mode="bilinear",
                align_corners=False,
            )[0, 0]
            presence_logit = float(
                object_score_logits[batch_index].detach().float().reshape(-1)[0].item()
            )
            records.append((resized_logit.cpu(), presence_logit))

    def finalize(
        self,
        *,
        num_frames: int,
        selected_obj_id: int | None,
    ) -> SAM3SoftSequence:
        probabilities = torch.zeros(
            num_frames,
            self.output_size[0],
            self.output_size[1],
            dtype=torch.float32,
        )
        presence_logits = torch.full((num_frames,), float("nan"))
        captures_per_frame = torch.zeros(num_frames, dtype=torch.int64)
        for frame_idx in range(num_frames):
            records = self._records.get(frame_idx, [])
            captures_per_frame[frame_idx] = len(records)
            if not records:
                continue
            record_index = int(selected_obj_id or 0)
            if record_index < 0 or record_index >= len(records):
                record_index = 0
            raw_logit, presence_logit = records[record_index]
            probabilities[frame_idx] = raw_logit.sigmoid()
            presence_logits[frame_idx] = presence_logit
        return SAM3SoftSequence(
            probabilities=probabilities,
            presence_logits=presence_logits,
            captures_per_frame=captures_per_frame,
        )


@contextmanager
def capture_sam3_soft_outputs(tracker, capture: SAM3SoftOutputCapture) -> Iterator[None]:
    """Capture decoder logits while leaving SAM3 inference behavior unchanged."""

    original_track_step = tracker.track_step
    decoder = tracker.sam_mask_decoder
    original_decoder_forward = decoder.forward

    def wrapped_track_step(tracker_self, *args, **kwargs):
        frame_idx = kwargs.get("frame_idx", args[0] if args else None)
        mask_inputs = kwargs.get("mask_inputs", args[7] if len(args) > 7 else None)
        previous_frame = capture._active_frame
        previous_enabled = capture._capture_enabled
        capture.set_active(frame_idx, enabled=mask_inputs is None)
        try:
            return original_track_step(*args, **kwargs)
        finally:
            capture.set_active(previous_frame, enabled=previous_enabled)

    def wrapped_decoder_forward(decoder_self, *args, **kwargs):
        output = original_decoder_forward(*args, **kwargs)
        capture.record(output)
        return output

    tracker.track_step = MethodType(wrapped_track_step, tracker)
    decoder.forward = MethodType(wrapped_decoder_forward, decoder)
    try:
        yield
    finally:
        decoder.forward = original_decoder_forward
        tracker.track_step = original_track_step
