from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from streaming_couping.src.backbones.sam3_video import (
    _bool_compatible_argsort,
    _filter_init_state_kwargs,
)
from streaming_couping.src.backbones.sam3_wrapper import SAM3Wrapper
from streaming_couping.src.types import SAM3MaskCandidate
from streaming_couping.src.v6_geometry_segmentation import (
    GeometrySegmentationPrompt,
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
    assert reason == "keep_raw:raw_geometry_reliable"


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
    assert reason == "apply_prompt:geometry_improved"


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


def test_geometry_backend_contract_is_three_image_space_masks() -> None:
    prompt = GeometrySegmentationPrompt(
        box_mask=torch.ones(8, 8, dtype=torch.bool),
        positive_mask=torch.zeros(8, 8, dtype=torch.bool),
        negative_mask=torch.zeros(8, 8, dtype=torch.bool),
    )
    prompt.positive_mask[1, 1] = True
    prompt.negative_mask[6, 6] = True

    assert prompt.box_mask.shape == (8, 8)
    assert prompt.positive_mask[1, 1]
    assert prompt.negative_mask[6, 6]
    assert not prompt.negative_mask[1, 1]


def test_sam3_geometry_prompt_is_passed_as_text_plus_box(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.jpg"
    Image.new("RGB", (100, 50), "black").save(image_path)

    class Predictor:
        def __init__(self) -> None:
            self.kwargs = None

        def start_session(self, *, resource_path):
            assert Path(resource_path).is_dir()
            return {"session_id": "test"}

        def add_prompt(self, **kwargs):
            self.kwargs = kwargs
            return {
                "frame_index": 0,
                "outputs": {
                    "out_obj_ids": np.asarray([7]),
                    "out_binary_masks": np.ones((1, 1, 50, 100), dtype=bool),
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
    )
    predictor = Predictor()
    wrapper.predictor = predictor
    geometry = torch.zeros(50, 100, dtype=torch.bool)
    geometry[10:30, 20:60] = True

    candidates = wrapper.propose_geometry_prompt_masks(
        image_path,
        prompt="bed",
        output_size=(50, 100),
        geometry_prompt=geometry,
    )

    assert len(candidates) == 1
    assert predictor.kwargs["text"] == "bed"
    assert predictor.kwargs["bounding_box_labels"] == [1]
    box = predictor.kwargs["bounding_boxes"][0]
    np.testing.assert_allclose(box, [0.2, 0.2, 0.4, 0.4])


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
    negative = torch.zeros_like(geometry)
    negative[5, 10] = True

    candidates = wrapper.propose_geometry_point_refined_masks(
        image_path,
        prompt="bed",
        output_size=(50, 100),
        geometry_prompt=geometry,
        positive_prompt=positive,
        negative_prompt=negative,
    )

    assert len(candidates) == 1
    assert len(predictor.calls) == 2
    assert predictor.calls[0]["text"] == "bed"
    assert "points" not in predictor.calls[0]
    assert "text" not in predictor.calls[1]
    assert predictor.calls[1]["obj_id"] == 7
    assert predictor.calls[1]["point_labels"] == [1, 1, 0]
    assert len(predictor.calls[1]["points"]) == 3


def test_v6_segmentation_command_and_config_are_retained() -> None:
    root = Path(__file__).resolve().parents[2]
    command = (
        root / "streaming_couping/commands_v6_geometry_segmentation.txt"
    ).read_text()
    config = (
        root / "streaming_couping/configs/v6_geometry_segmentation.yaml"
    ).read_text()

    assert "run_v6_geometry_segmentation" in command
    assert "v6_segmentation_summary.csv" in command
    assert "sam3.1_multiplex.pt" in command
    assert "version: sam3.1" in config
    assert "type: streamvggt_reference_projection" in config
    assert "min_candidate_support_recall: 0.25" in config
    assert "adaptive_raw_support_recall: 0.70" in config
    assert "adaptive_score_margin: 0.05" in config
    assert "point_positive_samples: 6" in config
