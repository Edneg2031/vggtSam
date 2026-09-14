"""The object-cloud figure.

The claim the picture makes is that only the pose changed -- the same points,
placed differently.  So the thing worth testing is that the two predicted
clouds come from one observation set, and that a correction moving away from
ground truth is drawn as moving away rather than auto-scaled out of sight.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from streaming_couping.scripts.plot_object_cloud_comparison import (
    _spread,
    _subsample,
    plot_objects,
)


def _cloud(count: int, offset: float = 0.0) -> torch.Tensor:
    points = torch.zeros(count, 3)
    points[:, 0] = torch.linspace(-0.5, 0.5, count)
    points[:, 2] = torch.linspace(-0.5, 0.5, count) + offset
    return points


def test_subsample_is_deterministic_and_spans_the_cloud() -> None:
    points = torch.arange(100, dtype=torch.float32).reshape(-1, 1).repeat(1, 3)
    first = _subsample(points, 10)
    assert torch.equal(first, _subsample(points, 10))
    assert first.shape[0] == 10
    # the ends are kept, so a cloud does not lose its extent when thinned
    assert float(first[0, 0]) == 0.0
    assert float(first[-1, 0]) == 99.0


def test_subsample_leaves_a_small_cloud_alone() -> None:
    points = _cloud(5)
    assert torch.equal(_subsample(points, 100), points)


def test_spread_ignores_a_stray_point() -> None:
    """One far outlier must not decide the view's scale."""

    points = _cloud(200)
    points = torch.cat([points, torch.tensor([[50.0, 0.0, 50.0]])], dim=0)
    extent = _spread(points)
    assert float(extent[0]) < 2.0


def test_plot_needs_an_object_with_both_clouds() -> None:
    with pytest.raises(ValueError, match="reference cloud and a prediction"):
        plot_objects(
            reference={"bed": _cloud(10)},
            raw={},
            corrected={},
            variant="robust_semantic",
            title="t",
            out_path=Path("/tmp/unused.png"),
            max_objects=5,
            points_per_cloud=10,
        )


def test_plot_draws_every_object_that_has_both(tmp_path: Path) -> None:
    reference = {"bed": _cloud(50), "chair": _cloud(40), "lamp": _cloud(30)}
    drawn = plot_objects(
        reference=reference,
        raw={"bed": _cloud(50), "chair": _cloud(40)},
        corrected={"bed": _cloud(50), "chair": _cloud(40)},
        variant="robust_semantic",
        title="t",
        out_path=tmp_path / "objects.png",
        max_objects=5,
        points_per_cloud=20,
    )
    # lamp has no prediction, so it is not drawn -- rather than drawn empty
    assert drawn == ["bed", "chair"]
    assert (tmp_path / "objects.png").is_file()


def test_plot_caps_the_number_of_objects(tmp_path: Path) -> None:
    reference = {f"o{i}": _cloud(50) for i in range(8)}
    predictions = {name: _cloud(50) for name in reference}
    drawn = plot_objects(
        reference=reference,
        raw=predictions,
        corrected=predictions,
        variant="robust_semantic",
        title="t",
        out_path=tmp_path / "objects.png",
        max_objects=3,
        points_per_cloud=10,
    )
    assert len(drawn) == 3


def test_mean_distance_is_to_the_nearest_reference_point() -> None:
    """The mean, matching object_accuracy_m -- a median would be a different
    statistic, and torch.median returns the lower middle value on top of it."""

    from streaming_couping.scripts.plot_object_cloud_comparison import (
        _mean_distance,
    )

    reference = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    points = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]])
    assert _mean_distance(points, reference) == pytest.approx(1.0)


def test_mean_distance_is_nan_without_points() -> None:
    from streaming_couping.scripts.plot_object_cloud_comparison import (
        _mean_distance,
    )

    assert math.isnan(_mean_distance(torch.zeros(0, 3), torch.zeros(3, 3)))
