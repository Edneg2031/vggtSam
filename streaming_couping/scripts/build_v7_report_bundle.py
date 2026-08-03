#!/usr/bin/env python3
"""Build plots, Markdown/LaTeX tables and a portable V7 result bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import tarfile
from pathlib import Path
from typing import Any


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    inputs = {
        "v71": Path(args.v71_csv).expanduser().resolve(),
        "v71_frames": Path(args.v71_frames).expanduser().resolve(),
        "v72": Path(args.v72_csv).expanduser().resolve(),
        "v72_frames": Path(args.v72_frames).expanduser().resolve(),
    }
    result = build_report_bundle(
        inputs,
        output_dir=output_dir,
        require_v72=bool(args.require_v72),
    )
    print(f"V7 report={result['markdown']}")
    print(f"V7 portable bundle={result['archive']}")


def build_report_bundle(
    inputs: dict[str, Path],
    *,
    output_dir: Path,
    require_v72: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    available = {name: path for name, path in inputs.items() if path.is_file()}
    if require_v72 and "v72" not in available:
        raise FileNotFoundError(f"Missing required V7.2 CSV: {inputs['v72']}")
    if "v71" not in available and "v72" not in available:
        raise FileNotFoundError(
            "Neither a V7.1 nor V7.2 result CSV is available."
        )

    source_dir = output_dir / "source_csv"
    source_dir.mkdir(exist_ok=True)
    copied: dict[str, Path] = {}
    for name, path in available.items():
        destination = source_dir / path.name
        shutil.copy2(path, destination)
        copied[name] = destination

    v71_rows = _read_csv(copied["v71"]) if "v71" in copied else []
    v72_rows = _read_csv(copied["v72"]) if "v72" in copied else []
    v71_frames = (
        _read_csv(copied["v71_frames"]) if "v71_frames" in copied else []
    )
    v72_frames = (
        _read_csv(copied["v72_frames"]) if "v72_frames" in copied else []
    )

    v71_ranked = _rank(v71_rows, "development_score")
    v72_ranked = _rank(v72_rows, "development_score")
    if v71_ranked:
        _write_paper_csv(
            output_dir / "v71_paper_table.csv", v71_ranked, version="v71"
        )
    if v72_ranked:
        _write_paper_csv(output_dir / "v72_paper_table.csv", v72_ranked, version="v72")
    (output_dir / "v7_paper_tables.tex").write_text(
        _latex_tables(v71_ranked, v72_ranked), encoding="utf8"
    )

    plot_status: list[str] = []
    try:
        if v71_rows:
            _plot_split_ratios(
                v71_rows,
                output_dir / "v71_split_loss_ratio.png",
                title="V7.1 loss relative to frozen camera L0",
            )
            plot_status.append("v71_split_loss_ratio.png")
        if v71_frames:
            _plot_frame_gain(
                v71_frames,
                output_dir / "v71_report_only_frame_gain.png",
                title="V7.1 per-frame gain on report-only splits",
            )
            plot_status.append("v71_report_only_frame_gain.png")
        if v72_rows:
            _plot_v72_token_sweep(
                v72_rows, output_dir / "v72_token_count_sweep.png"
            )
            plot_status.append("v72_token_count_sweep.png")
        if v72_frames:
            _plot_local_coverage(
                v72_frames, output_dir / "v72_local_token_coverage.png"
            )
            plot_status.append("v72_local_token_coverage.png")
            sam_candidates = [
                row
                for row in v72_rows
                if int(_float(row.get("token_count"), 0)) > 0
                and "sam31" in row.get("identity_source", "")
            ]
            if (
                sam_candidates
                and "sam_attention_entropy_normalized" in v72_frames[0]
            ):
                selected = min(
                    sam_candidates,
                    key=lambda row: _float(row.get("development_score"), float("inf")),
                )["architecture"]
                _plot_attention_health(
                    v72_frames,
                    output_dir / "v72_attention_health.png",
                    architecture=selected,
                )
                plot_status.append("v72_attention_health.png")
    except ImportError as exc:
        plot_status.append(f"plots_skipped:{exc}")

    markdown_path = output_dir / "v7_result_summary.md"
    markdown_path.write_text(
        _markdown_report(v71_ranked, v72_ranked, plot_status),
        encoding="utf8",
    )
    manifest_path = output_dir / "bundle_manifest.json"
    archive_path = output_dir.parent / f"{output_dir.name}.tar.gz"
    manifest = _manifest(output_dir, exclude={manifest_path, archive_path})
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf8",
    )
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(output_dir, arcname=output_dir.name)
    return {
        "markdown": str(markdown_path),
        "archive": str(archive_path),
        "files": len(manifest["files"]),
        "plots": plot_status,
    }


def _rank(rows: list[dict[str, str]], key: str) -> list[dict[str, str]]:
    return sorted(
        rows,
        key=lambda row: (_float(row.get(key), float("inf")), row.get("architecture", "")),
    )


def _write_paper_csv(
    path: Path,
    rows: list[dict[str, str]],
    *,
    version: str,
) -> None:
    fields = [
        "architecture",
        "token_count",
        "development_score",
        "development_best",
        "development_loss",
        "validation_loss",
        "future_loss",
        "cross_loss",
        "future_gain",
        "cross_gain",
        "strict_pass",
    ]
    compact = []
    for row in rows:
        compact.append(
            {
                "architecture": row.get("architecture", ""),
                "token_count": row.get("token_count", row.get("spatial_tokens", "")),
                "development_score": row.get("development_score", ""),
                "development_best": row.get("development_best", ""),
                "development_loss": row.get("development_loss", ""),
                "validation_loss": row.get("validation_loss", ""),
                "future_loss": row.get("future_loss", ""),
                "cross_loss": row.get("cross_loss", ""),
                "future_gain": row.get(
                    "future_gain_vs_l0_percent",
                    row.get("future_gain_vs_frozen_l0_percent", ""),
                ),
                "cross_gain": row.get(
                    "cross_gain_vs_l0_percent",
                    row.get("cross_gain_vs_frozen_l0_percent", ""),
                ),
                "strict_pass": row.get(
                    "causal_local_pass",
                    row.get("causal_instance_pass", ""),
                ),
            }
        )
    _write_csv(path, compact, fields=fields)


def _markdown_report(
    v71: list[dict[str, str]],
    v72: list[dict[str, str]],
    plots: list[str],
) -> str:
    lines = [
        "# V7 causal fusion experiment report",
        "",
        "This report is generated directly from the experiment CSV files. "
        "Development/validation select architectures; future and cross are "
        "report-only evidence.",
        "",
    ]
    if v71:
        lines.extend(
            [
                "## V7.1 global/pseudo-local residuals",
                "",
                _markdown_table(v71, version="v71"),
                "",
            ]
        )
        v71_pass = [
            row for row in v71 if row.get("causal_instance_pass") == "1"
        ]
        if v71_pass:
            lines.append(
                "Strict V7.1 pass: "
                + ", ".join(row["architecture"] for row in v71_pass)
                + "."
            )
        else:
            lines.append(
                "No V7.1 instance-content branch satisfies the strict causal "
                "criterion in this run."
            )
        lines.append("")
    else:
        lines.append(
            "V7.1 CSV was not available; this bundle reports V7.2 only."
        )
        lines.append("")
    if v72:
        lines.extend(
            [
                "## V7.2 true SAM3.1 mask-local descriptors",
                "",
                _markdown_table(v72, version="v72"),
                "",
            ]
        )
        v72_pass = [row for row in v72 if row.get("causal_local_pass") == "1"]
        if v72_pass:
            lines.append(
                "Strict V7.2 pass: "
                + ", ".join(row["architecture"] for row in v72_pass)
                + "."
            )
        else:
            lines.append(
                "No true local-token branch satisfies the strict causal "
                "criterion in this run."
            )
        lines.append("")
    lines.extend(
        [
            "## Generated figures",
            "",
            *[f"- `{value}`" for value in plots],
            "",
            "## Interpretation rule",
            "",
            "A lower development score alone is not sufficient. A positive "
            "claim requires improvement on development, validation, future "
            "and cross; superiority to both camera-capacity controls on "
            "future/cross; and exact fallback to frozen L0 for instance-off.",
            "",
        ]
    )
    return "\n".join(lines)


def _markdown_table(rows: list[dict[str, str]], *, version: str) -> str:
    if not rows:
        return "No result CSV was available."
    pass_field = "causal_local_pass" if version == "v72" else "causal_instance_pass"
    token_field = "token_count" if version == "v72" else "spatial_tokens"
    lines = [
        "| architecture | K | dev score | dev loss | val loss | future loss | cross loss | strict pass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {architecture} | {tokens} | {score} | {dev} | {val} | {future} | {cross} | {passed} |".format(
                architecture=row.get("architecture", ""),
                tokens=row.get(token_field, ""),
                score=row.get("development_score", ""),
                dev=row.get("development_loss", ""),
                val=row.get("validation_loss", ""),
                future=row.get("future_loss", ""),
                cross=row.get("cross_loss", ""),
                passed=row.get(pass_field, ""),
            )
        )
    return "\n".join(lines)


def _latex_tables(
    v71: list[dict[str, str]], v72: list[dict[str, str]]
) -> str:
    return "\n\n".join(
        [
            _latex_table(v71, caption="V7.1 causal residual ablation", label="tab:v71")
            if v71
            else "% V7.1 results were not available.",
            _latex_table(v72, caption="V7.2 local-token ablation", label="tab:v72")
            if v72
            else "% V7.2 results were not available.",
        ]
    ) + "\n"


def _latex_table(rows: list[dict[str, str]], *, caption: str, label: str) -> str:
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Method & Dev. score & Dev. loss & Val. loss & Future loss & Cross loss & Pass \\\\",
        "\\midrule",
    ]
    for row in rows:
        name = row.get("architecture", "").replace("_", "\\_")
        passed = row.get("causal_local_pass", row.get("causal_instance_pass", "0"))
        lines.append(
            f"{name} & {row.get('development_score', '')} & "
            f"{row.get('development_loss', '')} & {row.get('validation_loss', '')} & "
            f"{row.get('future_loss', '')} & {row.get('cross_loss', '')} & {passed} \\\\" 
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            "\\end{table*}",
        ]
    )
    return "\n".join(lines)


def _plot_split_ratios(rows: list[dict[str, str]], path: Path, *, title: str) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    base = next(row for row in rows if row.get("architecture") == "frozen_l0")
    candidates = [row for row in rows if row.get("architecture") != "raw_streamvggt"]
    splits = ("development", "validation", "future", "cross")
    values = [
        [
            _float(row.get(f"{split}_loss")) / _float(base.get(f"{split}_loss"))
            for split in splits
        ]
        for row in candidates
    ]
    figure, axis = plt.subplots(figsize=(max(10, len(candidates) * 0.8), 5.2))
    image = axis.imshow(np.asarray(values).T, aspect="auto", cmap="RdYlGn_r", vmin=0.7, vmax=1.3)
    axis.set_xticks(range(len(candidates)), [row["architecture"] for row in candidates], rotation=45, ha="right")
    axis.set_yticks(range(len(splits)), splits)
    axis.axhline(1.5, color="black", linewidth=1.0)
    axis.set_title(title)
    figure.colorbar(image, ax=axis, label="loss / frozen-L0 loss")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_v72_token_sweep(rows: list[dict[str, str]], path: Path) -> None:
    import matplotlib.pyplot as plt

    candidates = [row for row in rows if int(_float(row.get("token_count"), 0)) > 0]
    groups: dict[str, list[tuple[int, float]]] = {}
    for row in candidates:
        name = row["architecture"].rsplit("_k", 1)[0]
        groups.setdefault(name, []).append(
            (int(_float(row.get("token_count"), 0)), _float(row.get("development_score")))
        )
    figure, axis = plt.subplots(figsize=(9, 5.5))
    for name, values in sorted(groups.items()):
        values.sort()
        axis.plot([v[0] for v in values], [v[1] for v in values], marker="o", label=name)
    axis.axhline(1.0, color="black", linestyle="--", linewidth=1)
    axis.set_xlabel("SAM local tokens per instance (K)")
    axis.set_ylabel("development loss ratio vs frozen L0")
    axis.set_title("V7.2 local-token count sweep")
    axis.legend(fontsize=8, ncol=2)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_frame_gain(rows: list[dict[str, str]], path: Path, *, title: str) -> None:
    import matplotlib.pyplot as plt

    candidates = sorted(
        {row["architecture"] for row in rows if row.get("split") in {"future", "cross"}}
    )
    figure, axes = plt.subplots(2, 1, figsize=(11, 7), sharey=True)
    for axis, split in zip(axes, ("future", "cross")):
        for architecture in candidates:
            current = [
                row for row in rows
                if row.get("split") == split and row.get("architecture") == architecture
            ]
            if not current:
                continue
            axis.plot(
                [int(row["frame"]) for row in current],
                [_float(row.get("gain_vs_frozen_l0_percent")) for row in current],
                marker=".",
                label=architecture,
            )
        axis.axhline(0, color="black", linewidth=1)
        axis.set_title(split)
        axis.grid(alpha=0.25)
    axes[-1].set_xlabel("frame")
    figure.supylabel("gain vs frozen L0 (%)")
    axes[0].legend(fontsize=7, ncol=3)
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_local_coverage(rows: list[dict[str, str]], path: Path) -> None:
    import matplotlib.pyplot as plt

    selected = [
        row for row in rows
        if row.get("architecture", "").endswith("_k32")
        and row.get("split") in {"future", "cross"}
    ]
    figure, axis = plt.subplots(figsize=(10, 4.8))
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in selected:
        grouped.setdefault(row["split"], []).append(row)
    offset = 0
    for split, current in grouped.items():
        # Coverage is cache-dependent and therefore identical across models;
        # use the first row per frame.
        by_frame = {int(row["frame"]): row for row in current}
        frames = sorted(by_frame)
        x = list(range(offset, offset + len(frames)))
        axis.plot(
            x,
            [_float(by_frame[frame].get("sam_local_tokens")) for frame in frames],
            marker="o",
            label=split,
        )
        offset += len(frames) + 1
    axis.set_ylabel("valid cached SAM local tokens")
    axis.set_xlabel("report-only frame order")
    axis.set_title("V7.2 local-token coverage")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_attention_health(
    rows: list[dict[str, str]],
    path: Path,
    *,
    architecture: str,
) -> None:
    import matplotlib.pyplot as plt

    selected = [
        row
        for row in rows
        if row.get("architecture") == architecture
        and row.get("split") in {"development", "validation", "future", "cross"}
    ]
    figure, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    offset = 0
    for split in ("development", "validation", "future", "cross"):
        current = [row for row in selected if row.get("split") == split]
        if not current:
            continue
        x = list(range(offset, offset + len(current)))
        axes[0].plot(
            x,
            [_float(row.get("sam_attention_entropy_normalized")) for row in current],
            marker="o",
            label=split,
        )
        axes[1].plot(
            x,
            [_float(row.get("sam_attention_max_probability")) for row in current],
            marker="o",
            label=split,
        )
        offset += len(current) + 1
    axes[0].set_ylabel("normalized entropy")
    axes[1].set_ylabel("maximum probability")
    axes[1].set_xlabel("evaluation frame order")
    axes[0].set_ylim(-0.02, 1.02)
    axes[1].set_ylim(-0.02, 1.02)
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8, ncol=4)
    figure.suptitle(f"Local-attention health: {architecture}")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _manifest(root: Path, *, exclude: set[Path]) -> dict[str, Any]:
    files = []
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        if path in exclude:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        files.append(
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": digest,
            }
        )
    return {"schema": 1, "root": str(root), "files": files}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], *, fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--v71-csv",
        default="outputs/streaming_couping_v71_instance_causality/v71_instance_causality.csv",
    )
    parser.add_argument(
        "--v71-frames",
        default="outputs/streaming_couping_v71_instance_causality/v71_frame_diagnostics.csv",
    )
    parser.add_argument(
        "--v72-csv",
        default="outputs/streaming_couping_v72_local_token_ablation/v72_local_token_ablation.csv",
    )
    parser.add_argument(
        "--v72-frames",
        default="outputs/streaming_couping_v72_local_token_ablation/v72_frame_diagnostics.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/streaming_couping_v7_report_bundle",
    )
    parser.add_argument("--require-v72", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
