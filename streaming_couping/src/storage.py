"""Shared path expansion for large datasets and experiment artifacts."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


DEFAULT_STORAGE_ROOT = Path("/data184/open_source/vggtSam")


def storage_root() -> Path:
    """Return the mounted storage root used by server-side experiments."""

    raw = os.environ.get("VGGT_SAM_STORAGE_ROOT", str(DEFAULT_STORAGE_ROOT))
    return Path(os.path.expandvars(raw)).expanduser().resolve()


def expand_storage_path(
    value: Any,
    *,
    base: Path | None = None,
    prefer_cwd: bool = True,
) -> Path:
    """Expand the storage placeholder and resolve a configured path."""

    if value is None or str(value).strip() == "":
        raise ValueError("A required path is missing from the configuration.")
    raw = str(value).strip()
    root = str(storage_root())
    raw = raw.replace("${VGGT_SAM_STORAGE_ROOT}", root)
    raw = raw.replace("$VGGT_SAM_STORAGE_ROOT", root)
    path = Path(os.path.expandvars(raw)).expanduser()
    if path.is_absolute():
        return path.resolve()
    if base is None:
        return (Path.cwd() / path).resolve()
    cwd_path = (Path.cwd() / path).resolve()
    base_path = (Path(base) / path).resolve()
    if prefer_cwd and (cwd_path.exists() or not base_path.exists()):
        return cwd_path
    return base_path
