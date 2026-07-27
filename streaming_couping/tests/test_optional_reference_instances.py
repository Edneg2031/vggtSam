import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from streaming_couping.src.instance_observations import load_instance_sequences


def _write_manifest(tmp_path):
    frames = []
    masks = (
        np.array([[0, 68], [0, 68]], dtype=np.uint16),
        np.array([[37, 68], [37, 68]], dtype=np.uint16),
    )
    for index, mask in enumerate(masks):
        image_path = tmp_path / f"{index}.png"
        mask_path = tmp_path / f"{index}_instance.png"
        Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8)).save(image_path)
        Image.fromarray(mask).save(mask_path)
        frames.append(
            {
                "image_path": str(image_path),
                "instance_mask": str(mask_path),
            }
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "scenes": [
                    {
                        "scene_id": "scene",
                        "frames": frames,
                        "objects": {"37": "bed", "68": "wardrobe"},
                    }
                ]
            }
        ),
        encoding="utf8",
    )
    return manifest


def _config(manifest):
    return SimpleNamespace(
        manifest=manifest,
        scene_id="scene",
        frame_indices=(0, 1),
        min_pixels=1,
        max_area_ratio=1.0,
        excluded_labels=(),
        output_size=(2, 2),
    )


def test_visible_subset_keeps_missing_reference_instance_as_empty_slot(tmp_path):
    config = _config(_write_manifest(tmp_path))
    sequences, masks = load_instance_sequences(
        config,
        instance_ids=(37, 68, 54),
        reference_sequence_index=0,
        allow_missing_reference=True,
    )

    assert tuple(sequences) == (37, 68, 54)
    assert not bool(masks[37][0].any())
    assert bool(masks[37][1].any())
    assert bool(masks[68][0].any())
    assert not bool(masks[54].any())


def test_missing_reference_instance_remains_an_error_by_default(tmp_path):
    config = _config(_write_manifest(tmp_path))
    with pytest.raises(ValueError, match="absent at reference frame"):
        load_instance_sequences(
            config,
            instance_ids=(37, 68),
            reference_sequence_index=0,
        )
