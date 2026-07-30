import csv
import json

import numpy as np
import torch
from PIL import Image

from streaming_couping.src.external_repos import maybe_add_repo_to_path
from streaming_couping.src.learned_pose.export import (
    _align_camera_pose,
    _camera_matrices_from_world_to_camera,
    _export_gt_object_comparison,
    _export_tracking_mask_visualizations,
    _load_gt_instance_masks,
    _paired_distance_statistics,
    _world_confidence,
)


def test_align_camera_pose_matches_pointmap_similarity() -> None:
    native = torch.eye(4, dtype=torch.float64).repeat(2, 1, 1)
    native[0, :3, 3] = torch.tensor([1.0, 0.0, 0.0])
    native[1, :3, 3] = torch.tensor([0.0, 2.0, 0.0])
    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    translation = torch.tensor([0.5, -0.25, 2.0], dtype=torch.float64)

    c2w, w2c = _align_camera_pose(
        native,
        scale=2.0,
        rotation=rotation,
        translation=translation,
    )

    expected_centers = 2.0 * (native[:, :3, 3] @ rotation.T) + translation
    assert torch.allclose(c2w[:, :3, 3], expected_centers)
    assert torch.allclose(c2w[:, :3, :3], rotation.expand(2, -1, -1))
    assert torch.allclose(w2c @ c2w, torch.eye(4).double().expand(2, -1, -1))


def test_world_confidence_preserves_single_frame_axis() -> None:
    points = torch.zeros(1, 2, 3, 3)
    confidence = torch.ones(1, 1, 2, 3, 1)

    normalized = _world_confidence(confidence, points)

    assert normalized.shape == (1, 2, 3)


def test_repo_path_registration_adds_src_layout(tmp_path, monkeypatch) -> None:
    repo = tmp_path / "streamvggt"
    source = repo / "src"
    source.mkdir(parents=True)
    monkeypatch.setattr("sys.path", [])

    resolved = maybe_add_repo_to_path(repo)

    assert resolved == repo.resolve()
    assert str(source.resolve()) in __import__("sys").path
    assert str(repo.resolve()) in __import__("sys").path


def test_ground_truth_world_to_camera_conversion() -> None:
    w2c = torch.eye(4, dtype=torch.float64).repeat(2, 1, 1)
    w2c[1, 0, 3] = -2.0

    c2w, recovered_w2c = _camera_matrices_from_world_to_camera(
        w2c,
        frame_indices=(10, 20),
    )

    assert torch.allclose(
        c2w[1, :3, 3],
        torch.tensor([2.0, 0.0, 0.0], dtype=torch.float64),
    )
    assert torch.allclose(recovered_w2c, w2c)


def test_paired_distance_statistics_are_metric() -> None:
    predicted = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    target = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    statistics = _paired_distance_statistics(predicted, target)

    assert statistics["paired_distance_mean"] == 1.0
    assert torch.isclose(
        torch.tensor(statistics["paired_distance_rmse"]),
        torch.sqrt(torch.tensor(2.0)),
    )


def test_tracking_masks_are_exported_as_binary_and_overlay_images(tmp_path) -> None:
    image_paths = []
    for index in range(2):
        path = tmp_path / f"rgb_{index}.png"
        Image.new("RGB", (6, 4), color=(20 + index, 30, 40)).save(path)
        image_paths.append(path)
    masks = torch.zeros(2, 2, 2, 3, dtype=torch.bool)
    masks[0, 0, 0, 0] = True
    masks[0, 1, 1, 2] = True
    masks[1, 0, :, 1] = True
    scores = torch.tensor([[1.0, 0.9], [0.8, 0.0]])
    matched = torch.tensor([[True, True], [False, False]])
    unknown = torch.tensor([[False, False], [True, False]])
    mismatch = torch.tensor([[False, False], [False, True]])
    root = tmp_path / "segmentation_masks"

    _export_tracking_mask_visualizations(
        root,
        frame_indices=(105, 254),
        instance_ids=(37, 68),
        image_paths=image_paths,
        masks=masks,
        scores=scores,
        identity_valid=matched,
        identity_unknown=unknown,
        identity_mismatch=mismatch,
        reference_sequence_index=0,
    )

    first = "seq_000_frame_000105.png"
    second = "seq_001_frame_000254.png"
    assert (root / "overlays" / first).is_file()
    assert (root / "binary/instance_37" / first).is_file()
    assert (root / "binary/instance_68" / second).is_file()
    assert (root / "binary/union" / first).is_file()
    assert (root / "sequence_overview.png").is_file()
    binary = np.asarray(Image.open(root / "binary/instance_37" / first))
    assert binary.shape == (4, 6)
    assert set(np.unique(binary)).issubset({0, 255})
    with (root / "mask_summary.csv").open(newline="", encoding="utf8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert rows[0]["frame_index"] == "105"
    assert rows[0]["instance_id"] == "37"
    assert [row["identity_state"] for row in rows] == [
        "MATCH",
        "MATCH",
        "UNKNOWN",
        "MISMATCH",
    ]


def test_gt_instance_masks_are_loaded_independently_of_tracking(tmp_path) -> None:
    image_paths = []
    mask_paths = []
    for index in range(2):
        image_path = tmp_path / f"rgb_{index}.png"
        mask_path = tmp_path / f"mask_{index}.png"
        Image.new("RGB", (6, 4), color=(20, 30, 40)).save(image_path)
        labels = np.zeros((4, 6), dtype=np.uint16)
        labels[:, :3] = 37
        labels[:, 3:] = 68
        Image.fromarray(labels).save(mask_path)
        image_paths.append(image_path)
        mask_paths.append(mask_path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "scenes": [
                    {
                        "scene_id": "scene",
                        "objects": {"37": "chair", "68": "wardrobe"},
                        "frames": [
                            {
                                "image_path": str(image_path),
                                "instance_mask": str(mask_path),
                            }
                            for image_path, mask_path in zip(image_paths, mask_paths)
                        ],
                    }
                ]
            }
        ),
        encoding="utf8",
    )

    masks = _load_gt_instance_masks(
        manifest,
        scene_id="scene",
        frame_indices=(0, 1),
        instance_ids=(37, 68),
        processed_size=(2, 3),
        image_mode="crop",
    )

    assert masks.shape == (2, 2, 2, 3)
    assert masks[:, 0].any()
    assert masks[:, 1].any()
    assert not (masks[:, 0] & masks[:, 1]).any()


def test_gt_object_export_separates_gt_and_predicted_masks(tmp_path) -> None:
    target = torch.zeros(1, 2, 3, 3)
    target[..., 0] = torch.arange(6).reshape(1, 2, 3)
    raw = target + torch.tensor([1.0, 0.0, 0.0])
    ours = target + torch.tensor([0.5, 0.0, 0.0])
    colors = torch.full((1, 2, 3, 3), 128, dtype=torch.uint8)
    gt_masks = torch.zeros(1, 1, 2, 3, dtype=torch.bool)
    gt_masks[0, 0, 0, :2] = True
    predicted_masks = torch.zeros_like(gt_masks)
    predicted_masks[0, 0, 0, 1:] = True

    _export_gt_object_comparison(
        tmp_path,
        clip_name="clip",
        scene_id="scene",
        instance_ids=(37,),
        reference_sequence_index=0,
        raw_points=raw,
        refined_points=ours,
        target_points=target,
        predicted_masks=predicted_masks,
        gt_masks=gt_masks,
        colors=colors,
        scale=1.0,
        rotation=torch.eye(3),
        translation=torch.zeros(3),
        max_instance_points=100,
    )

    with (tmp_path / "object_comparison_metrics.csv").open(
        newline="", encoding="utf8"
    ) as handle:
        row = next(csv.DictReader(handle))
    assert float(row["mask_iou"]) == 1.0 / 3.0
    assert float(row["mask_precision"]) == 0.5
    assert float(row["mask_recall"]) == 0.5
    assert float(row["raw_gt_mask_rmse"]) == 1.0
    assert float(row["ours_gt_mask_rmse"]) == 0.5
    with (tmp_path / "object_comparison_short.csv").open(
        newline="", encoding="utf8"
    ) as handle:
        short_row = next(csv.DictReader(handle))
    assert float(short_row["ours_minus_raw_gt_mask_rmse"]) == -0.5
    instance_root = tmp_path / "instance_37"
    assert (instance_root / "gt_object.ply").is_file()
    assert (instance_root / "ours_predicted_object.ply").is_file()
    assert (instance_root / "ours_on_gt_mask.ply").is_file()
