#!/usr/bin/env python3
"""Why did a prompt contribute nothing?  (CPU only)

The observations that reach the refiner are the ones that survived every
filter, so from them alone "this word returned no masks" and "this word's masks
were dropped" are the same observation -- and they call for opposite changes.
The adapter now records what became of every track a prompt produced; this
prints that.

Reads ``<run>/<branch>/semantic_map.pt``, whose metadata carries the ledger
from the run.  Runs made before the ledger existed simply do not have it, which
this says rather than reporting an empty result.

    python -m streaming_couping.scripts.show_sam3_candidate_ledger \
        --run-dir <run>.baseline
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from streaming_couping.scripts.summarize_feedback_branches import run_label

#: How each outcome reads, and what it says to do about it.
OUTCOME_MEANING = {
    "accepted": "tracked, and its observations reached the refiner",
    "duplicate": "dropped: overlaps a track that was born earlier",
    "over_object_cap": "dropped: behind the --max-objects cut",
    "birth_mask_rejected": "dropped: birth mask below the pixel floor or too large",
    "birth_out_of_range": "dropped: birth frame outside the sequence",
}


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return "nan"
    return f"{float(value):.4f}"


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    rendered = [[_fmt(cell) for cell in row] for row in rows]
    widths = [
        max(len(str(header)), *(len(row[index]) for row in rendered))
        for index, header in enumerate(headers)
    ]
    lines = [
        "  ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers)),
        "  ".join("-" * width for width in widths),
    ]
    for row in rendered:
        lines.append("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))
    return "\n".join(lines)


def load_ledger(artifact: Path) -> dict[str, Any] | None:
    """The ledger a run exported, or None if it predates the ledger."""

    payload = torch.load(
        artifact.expanduser().resolve(), map_location="cpu", weights_only=False
    )
    metadata = payload.get("metadata") if isinstance(payload, Mapping) else None
    if not isinstance(metadata, Mapping):
        return None
    if "sam3_candidate_ledger" not in metadata:
        return None
    return {
        "ledger": list(metadata.get("sam3_candidate_ledger") or ()),
        "summary": dict(metadata.get("sam3_candidate_ledger_summary") or {}),
        "prompts": list(metadata.get("prompts") or ()),
    }


def outcome_matrix(summary: Mapping[str, Any]) -> tuple[list[str], list[str], list[list[Any]]]:
    """Rows = prompts in request order, columns = outcomes seen."""

    per_prompt = dict(summary.get("per_prompt") or {})
    outcomes = sorted(
        {outcome for counts in per_prompt.values() for outcome in counts}
    )
    prompts = [str(p) for p in (summary.get("prompts") or sorted(per_prompt))]
    # A prompt missing from per_prompt was never requested; one present with no
    # counts was requested and returned nothing.  Keep both visible.
    for prompt in sorted(per_prompt):
        if prompt not in prompts:
            prompts.append(prompt)
    rows = [
        [prompt, *[per_prompt.get(prompt, {}).get(outcome, 0) for outcome in outcomes]]
        for prompt in prompts
    ]
    return prompts, outcomes, rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        action="append",
        required=True,
        dest="run_dirs",
        help="Repeat once per run directory, in the order to print.",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument(
        "--report-out",
        type=Path,
        default=None,
        help=(
            "Write the per-prompt tables here and print only one line per run "
            "to stdout.  Without it everything goes to stdout."
        ),
    )
    return parser.parse_args()


def digest_line(run_name: str, summary: Mapping[str, Any] | None, note: str | None) -> str:
    """One line per run: what the prompt set produced, and what did not.

    The label is the same ``<generation>/<branch>`` the other tables use, so it
    does not print the whole run-directory name and push the line off screen.
    """

    run_name = run_label(Path(run_name))
    if summary is None:
        return f"  {run_name:58s} {note}"
    prompts = summary.get("prompts") or []
    per_prompt = summary.get("per_prompt") or {}
    no_track = summary.get("prompts_with_no_track") or []
    dropped = sorted(
        {
            outcome
            for counts in per_prompt.values()
            for outcome in counts
            if outcome != "accepted"
        }
    )
    detail = f"{summary.get('total', 0)} tracks / {len(prompts)} prompts"
    nothing = summary.get("prompts_returning_nothing") or []
    unusable = summary.get("prompts_found_but_unusable") or []
    if nothing:
        detail += f"; returned nothing: {', '.join(map(str, nothing))}"
    if unusable:
        detail += f"; found but unusable: {', '.join(map(str, unusable))}"
    if not nothing and not unusable and no_track:
        detail += f"; no track: {', '.join(map(str, no_track))}"
    if dropped:
        detail += f"; dropped as: {', '.join(dropped)}"
    return f"  {run_name:58s} {detail}"


def main() -> None:
    args = _parse_args()
    reports: dict[str, Any] = {}
    sections: list[str] = []
    stdout_lines: list[str] = []
    outcomes_seen: set[str] = set()

    for run_dir in args.run_dirs:
        run_dir = run_dir.expanduser().resolve()
        artifact = run_dir / "raw_pose" / "semantic_map.pt"
        sections.append(f"=== {run_dir.name} ===")
        if not artifact.is_file():
            sections.append(f"  missing: {artifact}")
            stdout_lines.append(digest_line(run_dir.name, None, f"missing {artifact.name}"))
            continue
        loaded = load_ledger(artifact)
        if loaded is None:
            note = (
                "predates the ledger -- re-run stage 2a to record what its "
                "prompts contributed"
            )
            sections.append("  " + note)
            stdout_lines.append(digest_line(run_dir.name, None, note))
            continue
        summary = {**loaded["summary"], "prompts": loaded["prompts"]}
        prompts, outcomes, rows = outcome_matrix(summary)
        outcomes_seen.update(outcomes)
        nothing_count = len(summary.get("prompts_returning_nothing") or [])
        unusable_count = len(summary.get("prompts_found_but_unusable") or [])
        sections.append(
            f"  {summary.get('total')} track(s) from {len(prompts)} prompt(s); "
            f"{nothing_count} returned nothing, {unusable_count} found but unusable"
            if (nothing_count or unusable_count or "prompts_returning_nothing" in summary)
            else f"  {summary.get('total')} track(s) from {len(prompts)} prompt(s)"
        )
        sections.append("")
        sections.append(_table(["prompt", *outcomes], rows))
        nothing = summary.get("prompts_returning_nothing") or []
        unusable = summary.get("prompts_found_but_unusable") or []
        if nothing or unusable:
            sections.append("")
        if nothing:
            sections.append(
                f"  returned NOTHING -- the text grounding found no object, so "
                f"nothing was offered to any filter: {', '.join(map(str, nothing))}"
            )
        if unusable:
            sections.append(
                f"  found but unusable -- every track died on the way in: "
                f"{', '.join(map(str, unusable))}"
            )
        # The legend is printed once for the whole report, after every run, so
        # a run does not repeat the same three lines; it used to appear both
        # inside each run's block and again at the end.
        sections.append("")
        stdout_lines.append(digest_line(run_dir.name, summary, None))
        reports[run_dir.name] = summary

    if args.report_out is not None:
        report_path = args.report_out.expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        legend = [
            "",
            "outcome meanings:",
            *[
                f"  {outcome:22s} {OUTCOME_MEANING[outcome]}"
                for outcome in sorted(outcomes_seen, key=str)
                if outcome in OUTCOME_MEANING
            ],
        ]
        report_path.write_text("\n".join([*sections, *legend]) + "\n", encoding="utf-8")
        print("per prompt, tracks returned and what became of them")
        print("\n".join(stdout_lines))
        print(f"  full report: {report_path}")
    else:
        print("\n".join(sections))
        for outcome in sorted(outcomes_seen, key=str):
            if outcome in OUTCOME_MEANING:
                print(f"  {outcome:22s} {OUTCOME_MEANING[outcome]}")

    if args.json_out is not None:
        args.json_out.expanduser().resolve().write_text(
            json.dumps(reports, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"json_out={args.json_out.expanduser().resolve()}")


if __name__ == "__main__":
    main()
