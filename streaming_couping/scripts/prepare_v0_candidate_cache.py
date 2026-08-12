#!/usr/bin/env python3
"""Find and link a provenance-compatible V0 cache before rebuilding it."""

from __future__ import annotations

import argparse
from pathlib import Path

from streaming_couping.src.learned_pose.cache import (
    _cache_complete,
    cache_path,
)
from streaming_couping.src.learned_pose.config import load_learned_pose_config
from streaming_couping.src.v0_track_ba import validate_track_cache
from streaming_couping.src.learned_pose.cache import load_feature_cache


def main() -> None:
    args = _parse_args()
    config = load_learned_pose_config(args.config)
    clip = next(item for item in config.clips if item.name == args.clip)
    target = cache_path(config, clip)
    if _compatible(target, config=config, clip=clip):
        print(f"V0 candidate cache ready={target}")
        return

    candidates = _compatible_candidates(
        args.search_root,
        target=target,
        config=config,
        clip=clip,
    )
    if not candidates:
        # Do not leave a stale link pointing at a deleted historical output;
        # the subsequent cache builder must be able to create a real file at
        # the stable V0-owned target.
        if target.is_symlink() or target.exists():
            target.unlink()
        print(f"V0 compatible cache not found; target={target}")
        raise SystemExit(2)

    source = candidates[0]
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() or target.exists():
        target.unlink()
    target.symlink_to(source.resolve())
    print(f"V0 candidate cache linked source={source} target={target}")


def _compatible_candidates(
    root: Path,
    *,
    target: Path,
    config,
    clip,
) -> list[Path]:
    if not root.exists():
        return []
    candidates = []
    for path in root.rglob(f"{clip.name}.pt"):
        if path.absolute() == target.absolute() or path.is_dir():
            continue
        if _compatible(path, config=config, clip=clip):
            candidates.append(path)
    return sorted(
        candidates,
        key=lambda path: (-path.stat().st_mtime_ns, str(path)),
    )


def _compatible(path: Path, *, config, clip) -> bool:
    if not _cache_complete(
        path,
        config=config,
        clip=clip,
        require_identity=config.fusion.strict_identity_gate,
    ):
        return False
    try:
        validate_track_cache(load_feature_cache(path))
    except (OSError, RuntimeError, TypeError, ValueError):
        return False
    return True


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="streaming_couping/configs/v0_baseline.yaml",
    )
    parser.add_argument("--clip", required=True)
    parser.add_argument("--search-root", type=Path, default=Path("outputs"))
    return parser.parse_args()


if __name__ == "__main__":
    main()
