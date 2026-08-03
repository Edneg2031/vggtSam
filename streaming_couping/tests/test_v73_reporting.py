import csv

import pytest

from streaming_couping.scripts.analyze_v73_results import CORE_FIELDS, analyze
from streaming_couping.scripts.aggregate_v73_seeds import aggregate


def _row(name, family, score, *, uses_sam=0, causal=0):
    row = {field: "0" for field in CORE_FIELDS}
    row.update(
        {
            "architecture": name,
            "architecture_family": family,
            "token_count": "8" if family == "v73_correspondence" else "0",
            "uses_sam": str(uses_sam),
            "development_score": str(score),
            "development_loss": str(score),
            "validation_loss": str(score),
            "future_loss": str(score + 0.1),
            "cross_loss": str(score + 0.2),
            "sam_perturbation_hurt_count": "16" if causal else "10",
            "sam_perturbation_total": "16" if uses_sam else "0",
            "causal_sam_pass": str(causal),
        }
    )
    return row


def _write(path, rows):
    fields = ["architecture_family", *CORE_FIELDS]
    with path.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_analyzer_selects_only_v73_development_candidates(tmp_path):
    path = tmp_path / "result.csv"
    rows = [
        _row("raw_streamvggt", "control", 0.1),
        _row("frozen_l0", "control", 0.2),
        _row("camera_extra_all", "retained_v72_camera_control", 0.3),
        _row("camera_extra_common_gate", "retained_v72_camera_control", 0.4),
        _row("geometry_transport_k08", "v73_correspondence", 0.8),
        _row(
            "sam_geometry_transport_k08",
            "v73_correspondence",
            0.7,
            uses_sam=1,
            causal=1,
        ),
    ]
    _write(path, rows)
    compact, summary = analyze(path)
    assert len(compact) == len(rows) - 1
    assert "`sam_geometry_transport_k08`" in summary
    assert "passes the predeclared" in summary


def test_analyzer_rejects_duplicate_architectures(tmp_path):
    path = tmp_path / "duplicates.csv"
    row = _row("raw_streamvggt", "control", 1.0)
    _write(path, [row, row])
    with pytest.raises(ValueError, match="duplicate"):
        analyze(path)


def test_seed_aggregate_keeps_architectures_separate(tmp_path):
    for seed, offset in ((0, 0.0), (1, 0.2)):
        directory = tmp_path / f"seed_{seed}"
        directory.mkdir()
        rows = [
            _row("geometry_transport_k08", "v73_correspondence", 1.0 + offset),
            _row(
                "sam_geometry_transport_k08",
                "v73_correspondence",
                0.8 + offset,
                uses_sam=1,
                causal=int(seed == 0),
            ),
        ]
        _write(directory / "v73_correspondence_ablation.csv", rows)
    rows = aggregate(tmp_path, (0, 1))
    by_name = {row["architecture"]: row for row in rows}
    sam = by_name["sam_geometry_transport_k08"]
    assert sam["seed_count"] == 2
    assert float(sam["development_score_mean"]) == pytest.approx(0.9)
    assert sam["causal_sam_pass_count"] == 1
