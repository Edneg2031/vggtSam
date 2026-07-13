"""CPU-only unit checks for image/grid coordinate transforms."""

from __future__ import annotations

import math

import numpy as np

from test_sam.coordinates import (
    compose,
    processed_to_grid_transform,
    sam3_resize_transform,
    streamvggt_label_to_grid,
    streamvggt_image_transform,
)


def main() -> None:
    source = (350, 518)
    sam = sam3_resize_transform(source)
    assert sam.target_size == (1008, 1008)
    _check_round_trip(sam, 127.25, 211.75)

    stream = streamvggt_image_transform(source, mode="crop")
    assert stream.target_size == (350, 518)
    grid = processed_to_grid_transform(
        stream.target_size,
        (25, 37),
        description="StreamVGGT patch grid",
    )
    composed = compose(stream, grid)
    _check_round_trip(composed, 127.25, 211.75)

    portrait = streamvggt_image_transform((900, 600), mode="crop")
    assert portrait.target_size == (518, 518)
    assert portrait.offset_xy[1] < 0

    padded = streamvggt_image_transform((350, 518), mode="pad")
    assert padded.target_size == (518, 518)
    assert padded.offset_xy[1] > 0

    labels = np.zeros((350, 518), dtype=np.uint16)
    labels[100:200, 150:300] = 37
    grid_labels = streamvggt_label_to_grid(labels, (25, 37), mode="crop")
    assert grid_labels.shape == (25, 37)
    assert 37 in np.unique(grid_labels)
    print("coordinate transform checks passed")


def _check_round_trip(transform, x: float, y: float) -> None:
    mapped = transform.map_xy(x, y)
    restored = transform.inverse_xy(*mapped)
    assert math.isclose(restored[0], x, abs_tol=1e-5)
    assert math.isclose(restored[1], y, abs_tol=1e-5)


if __name__ == "__main__":
    main()
