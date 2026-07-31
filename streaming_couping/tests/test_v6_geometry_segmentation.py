from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from streaming_couping.src.backbones.sam3_video import (
    _filter_init_state_kwargs,
)
from streaming_couping.src.backbones.sam3_wrapper import SAM3Wrapper
from streaming_couping.src.instance_observations import TranslationProposal
from streaming_couping.src.types import SAM3MaskCandidate
from streaming_couping.src.v6_geometry_segmentation import (
    V6GeometrySegmentationConfig,
    select_geometry_prompt_candidate,
)


def _proposal(*, accepted: bool, fitness: float) -> TranslationProposal:
    return TranslationProposal(
        instance_id=54,
        translation=torch.zeros(3),
        accepted=accepted,
        reason="test",
        current_points=256,
        map_points=256,
        correspondences=128 if accepted else 0,
        fitness=fitness,
        rmse=0.01 if accepted else float("nan"),
        correspondence_distance=0.05,
        object_scale=1.0,
        iterations=1,
        initialization="zero",
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


def test_candidate_selection_rejects_large_low_purity_lookalike() -> None:
    mask = torch.ones(8, 8, dtype=torch.bool)
    config = V6GeometrySegmentationConfig(
        min_support_recall=0.25,
        min_box_precision=0.50,
    )
    wrong = {
        "candidate": SAM3MaskCandidate(1, mask, 0.99),
        "support_recall": 1.0,
        "box_precision": 0.20,
        "support_precision": 0.10,
        "registration": _proposal(accepted=True, fitness=0.9),
        "two_d_score": 0.95,
        "three_d_score": 0.95,
    }
    correct = {
        "candidate": SAM3MaskCandidate(2, mask, 0.80),
        "support_recall": 0.80,
        "box_precision": 0.90,
        "support_precision": 0.70,
        "registration": _proposal(accepted=True, fitness=0.8),
        "two_d_score": 0.80,
        "three_d_score": 0.80,
    }

    selected = select_geometry_prompt_candidate(
        [wrong, correct],
        require_registration=True,
        config=config,
    )

    assert selected is correct


def test_3d_variant_falls_back_when_registration_is_rejected() -> None:
    mask = torch.ones(8, 8, dtype=torch.bool)
    row = {
        "candidate": SAM3MaskCandidate(2, mask, 0.90),
        "support_recall": 0.90,
        "box_precision": 0.90,
        "support_precision": 0.80,
        "registration": _proposal(accepted=False, fitness=0.0),
        "two_d_score": 0.90,
        "three_d_score": 0.60,
    }
    config = V6GeometrySegmentationConfig()

    assert (
        select_geometry_prompt_candidate(
            [row],
            require_registration=False,
            config=config,
        )
        is row
    )
    assert (
        select_geometry_prompt_candidate(
            [row],
            require_registration=True,
            config=config,
        )
        is None
    )


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
    assert "min_support_recall: 0.25" in config
    assert "min_box_precision: 0.50" in config
    assert "point_positive_samples: 6" in config
