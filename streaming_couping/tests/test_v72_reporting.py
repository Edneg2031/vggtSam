import csv
import tarfile

from streaming_couping.scripts.aggregate_v72_seeds import aggregate
from streaming_couping.scripts.build_v7_report_bundle import build_report_bundle


def _write(path, rows):
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _v71_rows():
    common = {
        "spatial_tokens": "0",
        "development_loss": "1.0",
        "validation_loss": "1.0",
        "future_loss": "1.0",
        "cross_loss": "1.0",
        "future_gain_vs_frozen_l0_percent": "0",
        "cross_gain_vs_frozen_l0_percent": "0",
        "causal_instance_pass": "0",
    }
    return [
        {"architecture": "raw_streamvggt", "development_score": "1.2", **common},
        {"architecture": "frozen_l0", "development_score": "1.0", **common},
        {"architecture": "camera_extra_all", "development_score": "0.99", **common},
    ]


def _v72_rows(offset=0.0):
    rows = []
    for architecture, token_count, score in (
        ("raw_streamvggt", 0, 1.2),
        ("frozen_l0", 0, 1.0),
        ("camera_extra_all", 0, 0.99),
        ("camera_extra_common_gate", 0, 1.01),
        ("sam_local_match_k08", 8, 0.90 + offset),
    ):
        row = {
            "architecture": architecture,
            "token_count": str(token_count),
            "mechanism": "test",
            "instance_content": str(int(token_count > 0)),
            "development_score": str(score),
            "development_best": str(int(architecture == "sam_local_match_k08")),
            "beats_camera_controls_report_only": str(int(token_count > 0)),
            "causal_local_pass": str(int(token_count > 0)),
        }
        for split in ("development", "validation", "future", "cross"):
            row[f"{split}_loss"] = str(score)
            row[f"{split}_gain_vs_l0_percent"] = str(100 * (1 - score))
            for variant in (
                "local_off",
                "wrong_local_identity",
                "shuffle_local_time",
            ):
                row[f"{split}_{variant}_loss"] = "1.0"
        rows.append(row)
    return rows


def test_report_bundle_contains_tables_manifest_and_archive(tmp_path) -> None:
    v71 = tmp_path / "v71.csv"
    v72 = tmp_path / "v72.csv"
    _write(v71, _v71_rows())
    _write(v72, _v72_rows())
    missing = tmp_path / "missing.csv"
    output = tmp_path / "report"
    result = build_report_bundle(
        {
            "v71": v71,
            "v71_frames": missing,
            "v72": v72,
            "v72_frames": missing,
        },
        output_dir=output,
        require_v72=True,
    )
    assert (output / "v7_result_summary.md").is_file()
    assert (output / "v7_paper_tables.tex").is_file()
    assert (output / "bundle_manifest.json").is_file()
    with tarfile.open(result["archive"], "r:gz") as archive:
        names = archive.getnames()
    assert any(name.endswith("v7_result_summary.md") for name in names)


def test_v72_seed_aggregate_reports_mean_std_and_pass_counts(tmp_path) -> None:
    first = tmp_path / "seed0.csv"
    second = tmp_path / "seed1.csv"
    _write(first, _v72_rows(0.0))
    _write(second, _v72_rows(0.04))
    output = aggregate([first, second], tmp_path / "aggregate.csv")
    with output.open(newline="", encoding="utf8") as handle:
        rows = {row["architecture"]: row for row in csv.DictReader(handle)}
    local = rows["sam_local_match_k08"]
    assert local["runs"] == "2"
    assert local["causal_local_pass_count"] == "2"
    assert float(local["development_score_mean"]) == 0.92
    assert float(local["development_score_std"]) > 0


def test_report_bundle_can_be_v72_only(tmp_path) -> None:
    v72 = tmp_path / "v72.csv"
    _write(v72, _v72_rows())
    missing = tmp_path / "missing.csv"
    output = tmp_path / "v72_only"
    build_report_bundle(
        {
            "v71": missing,
            "v71_frames": missing,
            "v72": v72,
            "v72_frames": missing,
        },
        output_dir=output,
        require_v72=True,
    )
    text = (output / "v7_result_summary.md").read_text(encoding="utf8")
    assert "V7.2 only" in text
