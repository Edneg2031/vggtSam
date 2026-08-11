"""Tests for the retained geometry-assisted SAM mask component."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from streaming_couping.src.backbones.sam3_intermediate import (
    _load_sam31_image_checkpoint,
)
from streaming_couping.src.backbones.sam3_video import (
    _bool_compatible_argsort,
    _filter_init_state_kwargs,
)
from streaming_couping.src.backbones.sam3_wrapper import SAM3Wrapper
from streaming_couping.src.types import SAM3MaskCandidate
from streaming_couping.src.geometry_segmentation import (
    GeometrySegmentationPrompt,
    V6_DEPLOYED_VARIANT,
    V6_SEGMENTATION_VARIANTS,
    V6GeometrySegmentationConfig,
    select_adaptive_correction,
    select_geometry_prompt_candidate,
)


def test_sam31_session_filters_unsupported_init_state_kwargs() -> None:
    class Model:
        def init_state(
            self,
            resource_path,
            offload_video_to_cpu=False,
            async_loading_frames=False,
        ):
            return {
                "resource_path": resource_path,
                "offload_video_to_cpu": offload_video_to_cpu,
                "async_loading_frames": async_loading_frames,
            }

    class Predictor:
        def __init__(self) -> None:
            self.model = Model()

    predictor = Predictor()
    _filter_init_state_kwargs(predictor)
    state = predictor.model.init_state(
        resource_path="/tmp/frames",
        offload_video_to_cpu=True,
        offload_state_to_cpu=True,
        async_loading_frames=True,
        video_loader_type="images",
    )

    assert state == {
        "resource_path": "/tmp/frames",
        "offload_video_to_cpu": True,
        "async_loading_frames": True,
    }


def test_sam31_bool_argsort_compatibility_keeps_true_first() -> None:
    values = torch.tensor([False, True, False, True])

    indices = _bool_compatible_argsort(values, descending=True)

    assert values[indices].tolist() == [True, True, False, False]
    assert sorted(indices.tolist()) == [0, 1, 2, 3]
    assert values.dtype == torch.bool


def test_candidate_selection_rejects_large_low_purity_lookalike() -> None:
    mask = torch.ones(8, 8, dtype=torch.bool)
    config = V6GeometrySegmentationConfig(
        min_candidate_support_recall=0.25,
    )
    wrong = {
        "candidate": SAM3MaskCandidate(1, mask, 0.99),
        "support_recall": 1.0,
        "box_precision": 0.20,
        "support_precision": 0.10,
        "geometry_score": 0.7585,
        "mask_pixels": 64,
    }
    correct = {
        "candidate": SAM3MaskCandidate(2, mask, 0.80),
        "support_recall": 0.80,
        "box_precision": 0.90,
        "support_precision": 0.70,
        "geometry_score": 0.83,
        "mask_pixels": 64,
    }

    selected = select_geometry_prompt_candidate(
        [wrong, correct],
        config=config,
    )

    assert selected is correct


def test_adaptive_policy_keeps_geometry_reliable_raw_mask() -> None:
    mask = torch.ones(8, 8, dtype=torch.bool)
    raw = {
        "candidate": SAM3MaskCandidate(-1, mask, 0.90),
        "support_recall": 0.90,
        "box_precision": 0.50,
        "support_precision": 0.80,
        "geometry_score": 0.82,
        "mask_pixels": 64,
        "box_pixels": 64,
    }
    prompted = {
        **raw,
        "candidate": SAM3MaskCandidate(2, mask, 0.95),
        "support_recall": 0.98,
        "geometry_score": 0.90,
        "box_pixels": 64,
    }

    selected, reason = select_adaptive_correction(
        raw_row=raw,
        prompted_row=prompted,
        config=V6GeometrySegmentationConfig(),
    )

    assert selected is None
    assert reason == "keep_raw:insufficient_support_gain"


def test_adaptive_policy_applies_clear_geometry_improvement() -> None:
    raw_mask = torch.ones(8, 8, dtype=torch.bool)
    prompted_mask = torch.ones(10, 10, dtype=torch.bool)
    raw = {
        "candidate": SAM3MaskCandidate(-1, raw_mask, 0.90),
        "support_recall": 0.30,
        "box_precision": 0.50,
        "support_precision": 0.20,
        "geometry_score": 0.40,
        "mask_pixels": 64,
        "box_pixels": 100,
    }
    prompted = {
        "candidate": SAM3MaskCandidate(2, prompted_mask, 0.90),
        "support_recall": 0.80,
        "box_precision": 0.60,
        "support_precision": 0.50,
        "geometry_score": 0.77,
        "mask_pixels": 100,
        "box_pixels": 100,
    }

    selected, reason = select_adaptive_correction(
        raw_row=raw,
        prompted_row=prompted,
        config=V6GeometrySegmentationConfig(),
    )

    assert selected is prompted
    assert reason == "apply_prompt:raw_unreliable_geometry_improved"


def test_competitive_policy_can_replace_reliable_raw_on_clear_gain() -> None:
    mask = torch.ones(8, 8, dtype=torch.bool)
    raw = {
        "candidate": SAM3MaskCandidate(-1, mask, 0.90),
        "support_recall": 0.80,
        "box_precision": 0.50,
        "support_precision": 0.60,
        "geometry_score": 0.70,
        "mask_pixels": 64,
        "box_pixels": 64,
    }
    prompted = {
        **raw,
        "candidate": SAM3MaskCandidate(2, mask, 0.95),
        "support_recall": 0.92,
        "geometry_score": 0.82,
    }

    selected, reason = select_adaptive_correction(
        raw_row=raw,
        prompted_row=prompted,
        config=V6GeometrySegmentationConfig(),
    )

    assert selected is prompted
    assert reason == "apply_prompt:competitive_geometry_improved"


def test_competitive_policy_keeps_reliable_raw_below_margin() -> None:
    mask = torch.ones(8, 8, dtype=torch.bool)
    raw = {
        "candidate": SAM3MaskCandidate(-1, mask, 0.90),
        "support_recall": 0.80,
        "box_precision": 0.50,
        "support_precision": 0.60,
        "geometry_score": 0.70,
        "mask_pixels": 64,
        "box_pixels": 64,
    }
    prompted = {
        **raw,
        "candidate": SAM3MaskCandidate(2, mask, 0.95),
        "support_recall": 0.88,
        "geometry_score": 0.78,
    }

    selected, reason = select_adaptive_correction(
        raw_row=raw,
        prompted_row=prompted,
        config=V6GeometrySegmentationConfig(),
    )

    assert selected is None
    assert reason == "keep_raw:insufficient_support_gain"


def test_adaptive_policy_rejects_huge_prompt_when_raw_is_missing() -> None:
    empty = torch.zeros(8, 8, dtype=torch.bool)
    huge = torch.ones(32, 32, dtype=torch.bool)
    raw = {
        "candidate": SAM3MaskCandidate(-1, empty, 0.0),
        "support_recall": 0.0,
        "box_precision": 0.0,
        "support_precision": 0.0,
        "geometry_score": 0.0,
        "mask_pixels": 0,
        "box_pixels": 64,
    }
    prompted = {
        "candidate": SAM3MaskCandidate(2, huge, 0.9),
        "support_recall": 1.0,
        "box_precision": 0.1,
        "support_precision": 0.1,
        "geometry_score": 0.715,
        "mask_pixels": 1024,
        "box_pixels": 64,
    }

    selected, reason = select_adaptive_correction(
        raw_row=raw,
        prompted_row=prompted,
        config=V6GeometrySegmentationConfig(),
    )

    assert selected is None
    assert reason == "keep_raw:prompt_too_large_for_box"


def test_geometry_backend_contract_is_two_image_space_masks() -> None:
    prompt = GeometrySegmentationPrompt(
        box_mask=torch.ones(8, 8, dtype=torch.bool),
        positive_mask=torch.zeros(8, 8, dtype=torch.bool),
    )
    prompt.positive_mask[1, 1] = True

    assert prompt.box_mask.shape == (8, 8)
    assert prompt.positive_mask[1, 1]
    assert V6_SEGMENTATION_VARIANTS == (
        "raw_sam31",
        V6_DEPLOYED_VARIANT,
    )


def test_sam31_geometry_points_refine_the_box_selected_object(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "frame.jpg"
    Image.new("RGB", (100, 50), "black").save(image_path)

    class Predictor:
        def __init__(self) -> None:
            self.calls = []

        def start_session(self, *, resource_path):
            assert Path(resource_path).is_dir()
            return {"session_id": "test"}

        def add_prompt(self, **kwargs):
            self.calls.append(kwargs)
            mask = np.zeros((1, 1, 50, 100), dtype=bool)
            mask[:, :, 10:30, 20:60] = True
            return {
                "frame_index": 0,
                "outputs": {
                    "out_obj_ids": np.asarray([7]),
                    "out_binary_masks": mask,
                    "out_probs": np.asarray([0.8]),
                },
            }

        def close_session(self, session_id):
            assert session_id == "test"

    wrapper = SAM3Wrapper(
        repo_path=tmp_path,
        checkpoint_path=tmp_path / "unused.pt",
        device="cuda:0",
        output_threshold=0.5,
        prompt_with_box=True,
        version="sam3.1",
    )
    predictor = Predictor()
    wrapper.predictor = predictor
    geometry = torch.zeros(50, 100, dtype=torch.bool)
    geometry[10:30, 20:60] = True
    positive = torch.zeros_like(geometry)
    positive[15, 25] = True
    positive[25, 50] = True
    candidates = wrapper.propose_geometry_point_refined_masks(
        image_path,
        prompt="bed",
        output_size=(50, 100),
        geometry_prompt=geometry,
        positive_prompt=positive,
    )

    assert len(candidates) == 1
    assert len(predictor.calls) == 2
    assert predictor.calls[0]["text"] == "bed"
    assert "points" not in predictor.calls[0]
    assert "text" not in predictor.calls[1]
    assert predictor.calls[1]["obj_id"] == 7
    assert predictor.calls[1]["point_labels"] == [1, 1]
    assert len(predictor.calls[1]["points"]) == 2


def test_sam31_image_checkpoint_accepts_internal_key_layout(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "sam31.pt"
    torch.save(
        {
            "model": {
                "sam3_model.backbone.weight": torch.ones(2),
                "sam2_predictor.unused": torch.zeros(1),
            }
        },
        checkpoint,
    )

    class Model:
        def __init__(self) -> None:
            self.state = None

        def load_state_dict(self, state, *, strict):
            assert not strict
            self.state = state

            class Incompatible:
                missing_keys = []
                unexpected_keys = []

            return Incompatible()

    model = Model()
    _load_sam31_image_checkpoint(model, checkpoint)

    assert list(model.state) == ["backbone.weight"]
    assert torch.equal(model.state["backbone.weight"], torch.ones(2))
