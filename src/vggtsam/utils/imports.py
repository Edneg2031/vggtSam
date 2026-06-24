"""Helpers for optional external repositories."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional


def maybe_add_repo_to_path(repo_path: Optional[str | Path]) -> Optional[Path]:
    """Add an optional external repository to `sys.path`.

    Missing paths are ignored so scripts can still work when a dependency is
    installed in the environment instead of linked under `externals/`.
    """
    if repo_path is None:
        return None
    path = Path(repo_path).expanduser()
    if not path.exists():
        return None
    path = path.resolve()
    candidates = [path]
    src_path = path / "src"
    if src_path.is_dir():
        candidates.insert(0, src_path)
    for candidate in candidates:
        path_str = str(candidate)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
    return path
