"""Portable cache helpers for the isolated HorizonStream process."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from PIL import Image


HORIZONSTREAM_CACHE_SCHEMA = "horizonstream_semantic_geometry"
HORIZONSTREAM_CACHE_VERSION = 1


def load_horizonstream_cache(path: str | Path) -> dict[str, Any]:
    cache_path = Path(path).expanduser().resolve()
    if not cache_path.is_file():
        raise FileNotFoundError(
            f"HorizonStream geometry cache does not exist: {cache_path}"
        )
    try:
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(cache_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(
            "HorizonStream geometry cache must contain a dictionary, got "
            f"{type(payload)!r}."
        )
    validate_horizonstream_cache(payload)
    return payload


def validate_horizonstream_cache(
    payload: Mapping[str, Any],
    *,
    expected_image_paths: Sequence[str | Path] | None = None,
) -> None:
    if payload.get("schema") != HORIZONSTREAM_CACHE_SCHEMA:
        raise ValueError(
            "Unsupported geometry cache schema: "
            f"{payload.get('schema')!r}; expected {HORIZONSTREAM_CACHE_SCHEMA!r}."
        )
    if int(payload.get("schema_version", -1)) != HORIZONSTREAM_CACHE_VERSION:
        raise ValueError(
            "Unsupported HorizonStream cache version: "
            f"{payload.get('schema_version')!r}."
        )
    if str(payload.get("backend")) != "horizonstream":
        raise ValueError("Geometry cache backend must be 'horizonstream'.")

    depth = _tensor(payload, "depth").detach().cpu()
    confidence = _tensor(payload, "confidence").detach().cpu()
    world_to_camera = _tensor(payload, "world_to_camera").detach().cpu()
    intrinsics = _tensor(payload, "intrinsics").detach().cpu()
    processed_rgb = _tensor(payload, "processed_rgb").detach().cpu()
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if confidence.ndim == 4 and confidence.shape[-1] == 1:
        confidence = confidence[..., 0]
    if depth.ndim != 3:
        raise ValueError("HorizonStream depth must have shape [S,H,W].")
    frame_count, height, width = (int(value) for value in depth.shape)
    if tuple(confidence.shape) != (frame_count, height, width):
        raise ValueError("HorizonStream confidence is not aligned to depth.")
    if tuple(world_to_camera.shape) != (frame_count, 3, 4):
        raise ValueError("HorizonStream world_to_camera must have shape [S,3,4].")
    if tuple(intrinsics.shape) != (frame_count, 3, 3):
        raise ValueError("HorizonStream intrinsics must have shape [S,3,3].")
    if tuple(processed_rgb.shape) != (frame_count, height, width, 3):
        raise ValueError("HorizonStream processed_rgb must have shape [S,H,W,3].")
    if processed_rgb.dtype != torch.uint8:
        raise ValueError("HorizonStream processed_rgb must use uint8 storage.")
    if tuple(int(value) for value in payload.get("processed_size", ())) != (
        height,
        width,
    ):
        raise ValueError("HorizonStream processed_size does not match depth.")

    image_paths = tuple(str(value) for value in payload.get("image_paths", ()))
    source_sizes = tuple(payload.get("source_sizes", ()))
    if len(image_paths) != frame_count:
        raise ValueError("HorizonStream image_paths do not match the frame count.")
    if len(source_sizes) != frame_count:
        raise ValueError("HorizonStream source_sizes do not match the frame count.")
    try:
        normalized_source_sizes = tuple(
            tuple(int(value) for value in size) for size in source_sizes
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "HorizonStream source_sizes must contain (height, width)."
        ) from exc
    if any(
        len(size) != 2 or size[0] <= 0 or size[1] <= 0
        for size in normalized_source_sizes
    ):
        raise ValueError(
            "HorizonStream source_sizes must contain positive (height, width)."
        )
    frame_ids = tuple(
        int(value) for value in payload.get("frame_ids", range(frame_count))
    )
    if len(frame_ids) != frame_count:
        raise ValueError("HorizonStream frame_ids do not match the frame count.")
    if frame_ids != tuple(range(frame_count)):
        raise ValueError("HorizonStream frame_ids must be zero-based and contiguous.")
    if str(payload.get("scale_type")) != "metric":
        raise ValueError("HorizonStream cache must declare scale_type='metric'.")

    if not bool(torch.isfinite(confidence).all()):
        raise ValueError("HorizonStream confidence contains non-finite values.")
    if float(confidence.min()) < 0.0 or float(confidence.max()) > 1.0:
        raise ValueError("HorizonStream confidence must be normalized to [0,1].")
    valid_depth = torch.isfinite(depth) & (depth > 0.0)
    if not bool(valid_depth.flatten(1).any(dim=1).all()):
        raise ValueError("Every HorizonStream frame must contain positive finite depth.")
    if not bool(torch.isfinite(world_to_camera).all()):
        raise ValueError("HorizonStream world_to_camera contains non-finite values.")
    if not bool(torch.isfinite(intrinsics).all()):
        raise ValueError("HorizonStream intrinsics contains non-finite values.")
    if bool((intrinsics[:, 0, 0].abs() <= 1e-8).any()) or bool(
        (intrinsics[:, 1, 1].abs() <= 1e-8).any()
    ):
        raise ValueError("HorizonStream intrinsics contain zero focal lengths.")

    if expected_image_paths is not None:
        expected = tuple(_canonical_path(path) for path in expected_image_paths)
        cached = tuple(_canonical_path(path) for path in image_paths)
        if expected != cached:
            mismatch = next(
                (
                    index
                    for index, (left, right) in enumerate(zip(expected, cached))
                    if left != right
                ),
                min(len(expected), len(cached)),
            )
            raise ValueError(
                "HorizonStream cache RGB order differs from the selected input "
                f"at position {mismatch}: expected_count={len(expected)} "
                f"cached_count={len(cached)}. Regenerate the geometry cache."
            )


def normalize_horizonstream_confidence(
    confidence: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize upstream expp1 confidence independently in each frame."""

    value = torch.nan_to_num(
        torch.as_tensor(confidence).detach().float().cpu(),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    if value.ndim == 4 and value.shape[-1] == 1:
        value = value[..., 0]
    if value.ndim != 3:
        raise ValueError("HorizonStream confidence must have shape [S,H,W].")
    flat = value.flatten(1)
    low = torch.quantile(flat, 0.05, dim=1, keepdim=True)
    high = torch.quantile(flat, 0.95, dim=1, keepdim=True)
    normalized = ((flat - low) / (high - low).clamp_min(1e-6)).clamp(0.0, 1.0)
    return normalized.reshape_as(value), low[:, 0], high[:, 0]


def materialize_horizonstream_rgb(
    payload: Mapping[str, Any],
    directory: str | Path,
) -> tuple[Path, ...]:
    """Write model-aligned RGB frames for the separate SAM process."""

    validate_horizonstream_cache(payload)
    output_dir = Path(directory).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir.chmod(0o755)
    rgb = torch.as_tensor(payload["processed_rgb"]).detach().cpu().numpy()
    paths: list[Path] = []
    for frame_id, frame in enumerate(rgb):
        path = output_dir / f"{frame_id:06d}.jpg"
        Image.fromarray(frame, mode="RGB").save(
            path,
            format="JPEG",
            quality=95,
            subsampling=0,
        )
        path.chmod(0o644)
        paths.append(path)
    return tuple(paths)


def horizonstream_cache_matches(
    payload: Mapping[str, Any],
    *,
    image_paths: Sequence[str | Path],
    checkpoint: str | Path,
    settings: Mapping[str, Any],
) -> bool:
    try:
        validate_horizonstream_cache(payload, expected_image_paths=image_paths)
    except (KeyError, OSError, TypeError, ValueError):
        return False
    request = payload.get("request", {})
    if not isinstance(request, Mapping):
        return False
    if _canonical_path(request.get("checkpoint", "")) != _canonical_path(checkpoint):
        return False
    cached_settings = request.get("settings", {})
    if not isinstance(cached_settings, Mapping):
        return False
    if not all(cached_settings.get(key) == value for key, value in settings.items()):
        return False
    return request.get("image_signatures") == image_file_signatures(image_paths)


def image_file_signatures(
    image_paths: Sequence[str | Path],
) -> list[dict[str, int]]:
    """Return cheap content-change indicators for an already frozen image list."""

    output = []
    for path in image_paths:
        stat = Path(path).expanduser().resolve().stat()
        output.append(
            {
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )
    return output


def _tensor(payload: Mapping[str, Any], name: str) -> torch.Tensor:
    if name not in payload:
        raise KeyError(f"HorizonStream cache is missing {name!r}.")
    return torch.as_tensor(payload[name])


def _canonical_path(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve())
