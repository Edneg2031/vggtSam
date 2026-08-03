import csv

from streaming_couping.scripts.aggregate_v71_seeds import aggregate
from streaming_couping.scripts.analyze_v71_results import (
    REQUIRED_ARCHITECTURES,
    analyze,
)


def _write_result(path, *, offset: float = 0.0):
    rows = []
    for index, architecture in enumerate(sorted(REQUIRED_ARCHITECTURES)):
        instance_content = int(
            architecture
            in {
                "appearance_pool",
                "geometry_pool",
                "decoupled_global",
                "local32_decoupled",
            }
        )
        score = 1.0 + 0.01 * index + offset
        if architecture == "geometry_pool":
            score = 0.8 + offset
        rows.append(
            {
                "architecture": architecture,
                "evidence": "test",
                "instance_content": instance_content,
                "development_score": score,
                "development_best": int(architecture == "geometry_pool"),
                "causal_instance_pass": int(
                    architecture == "geometry_pool"
                ),
                "development_loss": score,
                "development_gain_vs_frozen_l0_percent": 2.0,
                "validation_loss": score,
                "validation_gain_vs_frozen_l0_percent": 2.0,
                "future_loss": score,
                "future_gain_vs_frozen_l0_percent": 3.0,
                "cross_loss": score,
                "cross_gain_vs_frozen_l0_percent": 4.0,
                "future_wrong_geometry_damage_percent": (
                    5.0 if instance_content else ""
                ),
                "future_shuffle_time_damage_percent": (
                    6.0 if instance_content else ""
                ),
            }
        )
    with path.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_v71_analysis_and_seed_aggregation(tmp_path) -> None:
    first = tmp_path / "seed_0.csv"
    second = tmp_path / "seed_1.csv"
    _write_result(first)
    _write_result(second, offset=0.01)

    report = analyze(first, tmp_path / "summary.md")
    text = report.read_text(encoding="utf8")
    assert "Development best: `geometry_pool`" in text
    assert "Causal instance evidence passed" in text

    aggregate_path = aggregate(
        [first, second],
        tmp_path / "aggregate.csv",
    )
    with aggregate_path.open("r", encoding="utf8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    geometry = next(
        row for row in rows if row["architecture"] == "geometry_pool"
    )
    assert geometry["runs"] == "2"
    assert geometry["causal_instance_pass_count"] == "2"
    assert float(geometry["development_score_mean"]) == 0.805
