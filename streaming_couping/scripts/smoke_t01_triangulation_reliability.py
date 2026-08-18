#!/usr/bin/env python3
"""CPU smoke checks for fixed T0.1 reliability diagnostics."""

from __future__ import annotations

import torch

from streaming_couping.src.triangulation_reliability import (
    ReliabilityGate,
    fixed_bin_label,
    paired_error_metrics,
    reliability_gate_mask,
    reliability_gate_pass,
)


def main() -> None:
    test_fixed_non_gt_gate()
    test_fixed_bins()
    test_paired_metrics_and_go_rule()
    print("T0.1 fixed non-GT reliability gate smoke passed")


def test_fixed_non_gt_gate() -> None:
    metrics = {
        "mean_reprojection_error_px": torch.tensor([1.0, 3.0, 1.0, 1.0, 1.0]),
        "maximum_ray_angle_degrees": torch.tensor([3.0, 3.0, 1.0, 3.0, 3.0]),
        "condition_number": torch.tensor([100.0, 100.0, 100.0, 2000.0, 100.0]),
        "num_views": torch.tensor([2, 3, 3, 3, 3]),
        "positive_depth": torch.tensor([True, True, True, True, False]),
    }
    primary = ReliabilityGate("primary", 2.0, 2.0, 1000.0, 2)
    strict = ReliabilityGate("strict", 2.0, 2.0, 1000.0, 3)
    assert reliability_gate_mask(metrics, primary).tolist() == [
        True,
        False,
        False,
        False,
        False,
    ]
    assert not bool(reliability_gate_mask(metrics, strict).any())


def test_fixed_bins() -> None:
    assert fixed_bin_label("mean_reprojection_error_px", 0.99) == "0_to_1_px"
    assert fixed_bin_label("mean_reprojection_error_px", 1.0) == "1_to_2_px"
    assert fixed_bin_label("maximum_ray_angle_degrees", 10.0) == "10_plus_deg"
    assert fixed_bin_label("condition_number", 1000.0) == "1e3_to_3e3"
    assert fixed_bin_label("num_views", 2) == "2_views"
    assert fixed_bin_label("num_views", 3) == "3_views"
    assert fixed_bin_label("num_views", 4) == "4_plus_views"


def test_paired_metrics_and_go_rule() -> None:
    metrics = paired_error_metrics(
        torch.tensor([0.1, 0.2, 0.3]),
        torch.tensor([0.2, 0.3, 0.4]),
    )
    metrics["retained_ratio"] = 0.5
    assert metrics["anchor_count"] == 3
    assert metrics["tri_rmse"] < metrics["raw_rmse"]
    assert metrics["tri_p90"] < metrics["raw_p90"]
    assert reliability_gate_pass(
        metrics, minimum_anchor_count=3, minimum_retained_ratio=0.1
    )
    assert not reliability_gate_pass(
        metrics, minimum_anchor_count=4, minimum_retained_ratio=0.1
    )


if __name__ == "__main__":
    main()
