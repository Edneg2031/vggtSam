import csv

import torch

from streaming_couping.scripts.run_v5_adaptive import (
    ADAPTIVE_FINAL,
    CONTROL_FIXED,
    CONTROL_RAW,
    POINT_BLEND_VALUES,
    SUMMARY_FIELDS,
    _blend_confidence,
    _blend_pointmap,
    _compact_summary,
    _select_adaptive_blend,
    _write_csv,
    point_blend_name,
)


def test_compact_summary_keeps_only_retained_candidates() -> None:
    variants = (
        CONTROL_RAW,
        CONTROL_FIXED,
        *(point_blend_name(value) for value in POINT_BLEND_VALUES),
        ADAPTIVE_FINAL,
    )
    pose_rows = []
    pointmap_rows = []
    diagnostics = []
    for index, variant in enumerate(variants):
        pose_rows.append(
            {
                "clip": "clip",
                "perturbation": variant,
                "evaluation_protocol": "held_out_clip",
                "ate_rmse": 0.4 - 0.01 * index,
                "rotation_error_mean_degrees": 2.0 - 0.1 * index,
            }
        )
        pointmap_rows.append(
            {
                "clip": "clip",
                "perturbation": variant,
                "spatial_region": "full_scene",
                "geometry_source": "point_head",
                "group": "all_frames",
                "mean_frame_paired_distance_mean": 0.2 - 0.01 * index,
            }
        )
        point_blend = ""
        if variant == point_blend_name(0.0):
            point_blend = 0.0
        elif variant in {point_blend_name(1.0), ADAPTIVE_FINAL}:
            point_blend = 1.0
        diagnostics.append(
            {
                "clip": "clip",
                "variant": variant,
                "module_off_exact": 1,
                "status": "control" if index < 2 else "ray_fit",
                "point_blend": point_blend,
                "pose_pointmap_coupled": int(index >= 2),
                "fit_accepted": 1,
                "support_ratio": 1.0,
                "mean_pose_center_shift_native": 0.01,
                "raw_reference_exact": 0,
                "learned_reference_preserved": 1,
                "a100_pose_matches_fixed": (
                    1 if variant == point_blend_name(1.0) else ""
                ),
            }
        )

    rows = _compact_summary(pose_rows, pointmap_rows, diagnostics)

    assert len(rows) == 5
    assert tuple(rows[0]) == SUMMARY_FIELDS
    assert [row["variant"] for row in rows] == list(variants)
    assert rows[-1]["ate_delta_from_raw"] == "-0.04"
    assert rows[-1]["pointmap_delta_from_raw"] == "-0.04"
    assert rows[-2]["a100_pose_matches_fixed"] == 1


def test_csv_writer_unions_control_and_adaptive_fields(tmp_path) -> None:
    path = tmp_path / "diagnostics.csv"
    _write_csv(
        path,
        [
            {"clip": "clip", "variant": CONTROL_RAW, "status": "control"},
            {
                "clip": "clip",
                "variant": point_blend_name(1.0),
                "status": "ray_fit",
                "fit_accepted": 2,
            },
        ],
    )

    with path.open(newline="", encoding="utf8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["fit_accepted"] == ""
    assert rows[1]["fit_accepted"] == "2"


def test_pointmap_blend_is_exact_at_endpoints_and_preserves_reference() -> None:
    raw = torch.zeros(2, 2, 2, 3)
    learned = torch.full_like(raw, 2.0)
    learned[1, 0, 0] = torch.nan

    zero = _blend_pointmap(raw, learned, blend=0.0, reference_index=0)
    one = _blend_pointmap(raw, learned, blend=1.0, reference_index=0)

    assert torch.equal(zero, raw)
    assert torch.equal(one[0], raw[0])
    assert torch.equal(one[1, 1, 1], learned[1, 1, 1])
    assert torch.equal(one[1, 0, 0], raw[1, 0, 0])


def test_confidence_blend_accepts_singleton_channel() -> None:
    raw = torch.zeros(2, 2, 2, 1)
    learned = torch.ones(2, 2, 2)
    value = _blend_confidence(
        raw,
        learned,
        blend=1.0,
        reference_index=0,
    )

    assert value.shape == (2, 2, 2)
    assert torch.equal(value[0], torch.zeros(2, 2))
    assert torch.equal(value[1], torch.ones(2, 2))


def test_adaptive_blend_uses_documented_support_threshold() -> None:
    assert _select_adaptive_blend(0.7499) == 0.0
    assert _select_adaptive_blend(0.75) == 1.0
    assert _select_adaptive_blend(1.0) == 1.0
