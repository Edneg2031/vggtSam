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

#: Outcomes that mean the prompt DID return a track, whatever became of it.
TRACK_PRODUCING_OUTCOMES = ("accepted", "duplicate", "over_object_cap")

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
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    reports: dict[str, Any] = {}
    for run_dir in args.run_dirs:
        run_dir = run_dir.expanduser().resolve()
        artifact = run_dir / "raw_pose" / "semantic_map.pt"
        print(f"=== {run_dir.name} ===")
        if not artifact.is_file():
            print(f"  missing: {artifact}")
            print()
            continue
        loaded = load_ledger(artifact)
        if loaded is None:
            print("  this run predates the candidate ledger, so it cannot say")
            print("  which prompts were dropped or why.  Re-run stage 2a to record it.")
            print()
            continue
        summary = loaded["summary"]
        summary = {**summary, "prompts": loaded["prompts"]}
        prompts, outcomes, rows = outcome_matrix(summary)
        print(
            f"  {summary.get('total')} track(s) from {len(prompts)} prompt(s); "
            f"{len(summary.get('prompts_with_no_track') or [])} returned no track"
        )
        print()
        print(_table(["prompt", *outcomes], rows))
        no_track = summary.get("prompts_with_no_track") or []
        if no_track:
            print()
            print(f"  returned no track at all: {', '.join(map(str, no_track))}")
        print()
        for outcome in outcomes:
            if outcome in OUTCOME_MEANING:
                print(f"  {outcome:22s} {OUTCOME_MEANING[outcome]}")
        print()
        reports[run_dir.name] = summary

    if args.json_out is not None:
        args.json_out.expanduser().resolve().write_text(
            json.dumps(reports, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"json_out={args.json_out.expanduser().resolve()}")


if __name__ == "__main__":
    main()
