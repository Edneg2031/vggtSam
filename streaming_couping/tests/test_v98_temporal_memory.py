from __future__ import annotations

import torch

from streaming_couping.src.v98_temporal_memory import (
    SAM31MemoryReadCapture,
    combine_prompt_feature_runs,
    resolve_sam31_memory_model,
)


class DummyTracker:
    def _prepare_memory_conditioned_features(self, **kwargs):
        source = kwargs["current_vision_feats"][-1]
        height, width = kwargs["feat_sizes"][-1]
        raw = source.permute(1, 2, 0).reshape(1, source.shape[-1], height, width)
        return raw + float(kwargs.get("memory_delta", 1.0))


class DummyProxy:
    def __init__(self, model):
        self.model = model

    def __getattr__(self, name):
        return getattr(self.model, name)


class DummyPredictorModel:
    def __init__(self, concrete):
        self.tracker = DummyProxy(concrete)


def test_resolve_selects_concrete_tracker_not_proxy_or_orchestrator() -> None:
    concrete = DummyTracker()
    assert resolve_sam31_memory_model(DummyPredictorModel(concrete)) is concrete


def test_capture_records_actual_return_and_same_call_raw_boundary() -> None:
    tracker = DummyTracker()
    source = torch.randn(6, 1, 5)
    with SAM31MemoryReadCapture(
        tracker, grid_size=(4, 4), canonical_dim=8
    ) as capture:
        returned = tracker._prepare_memory_conditioned_features(
            frame_idx=2,
            current_vision_feats=[source],
            feat_sizes=[(2, 3)],
            memory_delta=0.5,
            output_dict={
                "cond_frame_outputs": {1: {"maskmem_features": torch.ones(1)}},
                "non_cond_frame_outputs": {},
            },
        )
    result = capture.finalized(4)
    assert returned.shape == (1, 5, 2, 3)
    assert result["memory"].shape == (4, 16, 8)
    assert result["valid"].tolist() == [False, False, True, False]
    assert float(result["memory_delta_l2"][2]) > 0.0
    assert torch.count_nonzero(result["raw"][2]) > 0


def test_capture_keeps_last_same_frame_causal_state() -> None:
    tracker = DummyTracker()
    source = torch.ones(4, 1, 4)
    with SAM31MemoryReadCapture(
        tracker, grid_size=(2, 2), canonical_dim=4
    ) as capture:
        for delta in (0.25, 0.75):
            tracker._prepare_memory_conditioned_features(
                frame_idx=1,
                current_vision_feats=[source],
                feat_sizes=[(2, 2)],
                memory_delta=delta,
                output_dict={
                    "cond_frame_outputs": {0: {"maskmem_features": torch.ones(1)}},
                    "non_cond_frame_outputs": {},
                },
            )
    result = capture.finalized(2)
    assert int(result["call_count"][1]) == 2
    difference = result["memory"][1] - result["raw"][1]
    assert torch.allclose(difference, torch.full_like(difference, 0.75))


def test_prompt_merge_uses_only_available_sessions() -> None:
    memory = torch.tensor(
        [[[[1.0]], [[3.0]]], [[[9.0]], [[5.0]]]]
    )
    raw = memory - 1.0
    valid = torch.tensor([[True, True], [False, True]])
    merged_memory, merged_raw, merged_valid = combine_prompt_feature_runs(
        memory, raw, valid
    )
    assert merged_memory.flatten().tolist() == [1.0, 4.0]
    assert merged_raw.flatten().tolist() == [0.0, 3.0]
    assert merged_valid.tolist() == [True, True]
