"""Causal SAM3.1 video-memory feature capture for V9.8.

The multiplex tracker first builds a propagation feature for the current
image and then lets that feature read earlier mask-memory/object-pointer
tokens.  V9.8 captures both sides of that exact boundary.  It deliberately
does not use ``image_features`` or ``maskmem_features`` as a proxy: the former
has not read history and the latter is the value written for future frames.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import torch
from torch.nn import functional as F

from streaming_couping.src.v90_explicit_matcher import (
    canonicalize_descriptor_channels,
)


TEMPORAL_ARCHITECTURES = (
    "sam_memory",
    "memory_off_raw",
    "sam_memory_train_off",
)

TEMPORAL_VARIANTS = (
    "normal",
    "memory_off",
    "channel_permute",
    "shuffle_memory_time",
)


def resolve_sam31_memory_model(predictor_model: Any) -> Any:
    """Return the concrete tracker module that executes memory attention.

    The public predictor owns a detector/tracker orchestration model.  Its
    ``tracker`` is itself a proxy whose ``model`` is the concrete
    ``VideoTrackingMultiplex`` instance.  Patching the orchestrator or proxy
    would not affect internal ``self._prepare_memory_conditioned_features``
    calls, so require the class hierarchy to define the method directly.
    """

    candidates = []
    tracker = getattr(predictor_model, "tracker", None)
    if tracker is not None:
        concrete = getattr(tracker, "model", None)
        if concrete is not None:
            candidates.append(concrete)
        candidates.append(tracker)
    candidates.append(predictor_model)
    method = "_prepare_memory_conditioned_features"
    for candidate in candidates:
        if any(method in cls.__dict__ for cls in type(candidate).mro()):
            return candidate
    raise AttributeError(
        "V9.8 could not locate the concrete SAM3.1 multiplex memory module."
    )


@dataclass(frozen=True)
class CapturedTemporalFrame:
    frame_index: int
    memory_features: torch.Tensor
    raw_features: torch.Tensor
    bucket_count: int
    memory_delta_l2: float
    past_memory_frame_count: int
    same_frame_memory_present: bool


class SAM31MemoryReadCapture:
    """Temporarily capture the tracker's real history-read feature.

    SAM3.1 multiplex returns one feature map per multiplex bucket, not one map
    per object.  V9.8 therefore averages buckets only after the tracker has
    jointly conditioned them on all objects in that bucket.  Calling
    ``MultiplexState.demux`` here would be a shape/semantic error.
    """

    def __init__(
        self,
        model: Any,
        *,
        grid_size: tuple[int, int] = (72, 72),
        canonical_dim: int = 256,
    ) -> None:
        self.model = model
        self.grid_size = tuple(int(value) for value in grid_size)
        self.canonical_dim = int(canonical_dim)
        self._original = None
        self._had_instance_override = False
        self._instance_override = None
        self._records: dict[int, list[CapturedTemporalFrame]] = defaultdict(list)

    def __enter__(self) -> "SAM31MemoryReadCapture":
        if self._original is not None:
            raise RuntimeError("V9.8 SAM memory capture is already installed.")
        original = getattr(self.model, "_prepare_memory_conditioned_features", None)
        if original is None:
            raise AttributeError(
                "V9.8 requires SAM3.1 _prepare_memory_conditioned_features."
            )
        self._original = original
        self._had_instance_override = (
            "_prepare_memory_conditioned_features" in getattr(self.model, "__dict__", {})
        )
        if self._had_instance_override:
            self._instance_override = self.model.__dict__["_prepare_memory_conditioned_features"]

        def wrapped(*args, **kwargs):
            output = original(*args, **kwargs)
            self._capture(output, kwargs)
            return output

        self.model._prepare_memory_conditioned_features = wrapped
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self._original is not None:
            if self._had_instance_override:
                self.model._prepare_memory_conditioned_features = self._instance_override
            else:
                delattr(self.model, "_prepare_memory_conditioned_features")
            self._original = None
            self._had_instance_override = False
            self._instance_override = None

    @torch.no_grad()
    def _capture(self, memory: torch.Tensor, kwargs: dict[str, Any]) -> None:
        if memory.ndim != 4:
            raise ValueError(
                "V9.8 expected memory-conditioned features [bucket,C,H,W]."
            )
        frame_index = int(kwargs["frame_idx"])
        if bool(kwargs.get("track_in_reverse", False)):
            raise ValueError("V9.8 refuses reverse/future memory propagation.")
        output_dict = kwargs.get("output_dict", {})
        available_memory_frames = []
        for storage_key in ("cond_frame_outputs", "non_cond_frame_outputs"):
            for index, output in output_dict.get(storage_key, {}).items():
                if isinstance(output, dict) and output.get("maskmem_features") is not None:
                    available_memory_frames.append(int(index))
        future = [index for index in available_memory_frames if index > frame_index]
        if future:
            raise ValueError(
                f"V9.8 causal violation: frame={frame_index} sees future memory={future}."
            )
        past_count = len({index for index in available_memory_frames if index < frame_index})
        same_frame = frame_index in available_memory_frames
        feature_levels = kwargs["current_vision_feats"]
        feature_sizes = kwargs["feat_sizes"]
        if not feature_levels or not feature_sizes:
            raise ValueError("V9.8 capture received no propagation feature level.")
        source = feature_levels[-1]
        if source.ndim != 3:
            raise ValueError("V9.8 raw propagation feature must be [HW,B,C].")
        height, width = (int(value) for value in feature_sizes[-1])
        raw = source.permute(1, 2, 0).reshape(source.shape[1], source.shape[2], height, width)
        if raw.shape[0] == 1 and memory.shape[0] != 1:
            raw = raw.expand(memory.shape[0], -1, -1, -1)
        if raw.shape != memory.shape:
            raise ValueError(
                "V9.8 raw/memory feature shapes disagree: "
                f"{tuple(raw.shape)} vs {tuple(memory.shape)}."
            )
        memory_grid = _canonical_grid(memory, self.grid_size, self.canonical_dim)
        raw_grid = _canonical_grid(raw, self.grid_size, self.canonical_dim)
        memory_grid = memory_grid.mean(dim=0).detach().to(torch.float16).cpu()
        raw_grid = raw_grid.mean(dim=0).detach().to(torch.float16).cpu()
        delta = float(
            torch.linalg.vector_norm(memory_grid.float() - raw_grid.float())
            / max(memory_grid.numel() ** 0.5, 1.0)
        )
        self._records[frame_index].append(
            CapturedTemporalFrame(
                frame_index=frame_index,
                memory_features=memory_grid,
                raw_features=raw_grid,
                bucket_count=int(memory.shape[0]),
                memory_delta_l2=delta,
                past_memory_frame_count=past_count,
                same_frame_memory_present=same_frame,
            )
        )

    def finalized(self, frame_count: int) -> dict[str, torch.Tensor]:
        """Use the last causal tracker state for each processed frame."""

        tokens = self.grid_size[0] * self.grid_size[1]
        memory = torch.zeros(
            frame_count, tokens, self.canonical_dim, dtype=torch.float16
        )
        raw = torch.zeros_like(memory)
        valid = torch.zeros(frame_count, dtype=torch.bool)
        bucket_count = torch.zeros(frame_count, dtype=torch.long)
        delta = torch.zeros(frame_count)
        call_count = torch.zeros(frame_count, dtype=torch.long)
        past_count = torch.zeros(frame_count, dtype=torch.long)
        same_frame = torch.zeros(frame_count, dtype=torch.bool)
        for frame_index, values in self._records.items():
            if frame_index < 0 or frame_index >= frame_count:
                raise ValueError(f"V9.8 captured out-of-range frame={frame_index}.")
            # Dynamic discovery can revisit a frame. Prefer the last call that
            # reads strictly earlier memory but has not yet written/read the
            # current detection as a conditioning frame. This isolates past
            # video memory from same-frame detector conditioning.
            past_only = [
                value
                for value in values
                if value.past_memory_frame_count > 0
                and not value.same_frame_memory_present
            ]
            if not past_only:
                # Do not silently relabel a same-frame/fallback call as a
                # temporal feature. Another prompt may still provide a valid
                # past-memory read for this frame during fixed prompt merging.
                continue
            selected = past_only[-1]
            memory[frame_index] = selected.memory_features
            raw[frame_index] = selected.raw_features
            valid[frame_index] = True
            bucket_count[frame_index] = selected.bucket_count
            delta[frame_index] = selected.memory_delta_l2
            call_count[frame_index] = len(values)
            past_count[frame_index] = selected.past_memory_frame_count
            same_frame[frame_index] = selected.same_frame_memory_present
        return {
            "memory": memory,
            "raw": raw,
            "valid": valid,
            "bucket_count": bucket_count,
            "memory_delta_l2": delta,
            "call_count": call_count,
            "past_memory_frame_count": past_count,
            "same_frame_memory_present": same_frame,
        }


def combine_prompt_feature_runs(
    memory_runs: torch.Tensor,
    raw_runs: torch.Tensor,
    valid_runs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Average independent text-prompt sessions without changing support.

    Args:
        memory_runs: ``[P,T,N,C]`` memory-conditioned maps.
        raw_runs: matching pre-memory propagation maps.
        valid_runs: ``[P,T]`` capture availability.
    """

    if memory_runs.ndim != 4 or raw_runs.shape != memory_runs.shape:
        raise ValueError("V9.8 prompt features must be matching [P,T,N,C] tensors.")
    if valid_runs.shape != memory_runs.shape[:2]:
        raise ValueError("V9.8 prompt validity must be [P,T].")
    weights = valid_runs.to(memory_runs.dtype)[..., None, None]
    denominator = weights.sum(dim=0).clamp_min(1.0)
    memory = (memory_runs * weights).sum(dim=0) / denominator
    raw = (raw_runs * weights).sum(dim=0) / denominator
    valid = valid_runs.any(dim=0)
    return memory, raw, valid


def _canonical_grid(
    value: torch.Tensor,
    grid_size: tuple[int, int],
    canonical_dim: int,
) -> torch.Tensor:
    value = F.interpolate(
        value.float(), size=grid_size, mode="bilinear", align_corners=False
    )
    value = value.permute(0, 2, 3, 1).reshape(value.shape[0], -1, value.shape[1])
    return canonicalize_descriptor_channels(value, canonical_dim).contiguous()
