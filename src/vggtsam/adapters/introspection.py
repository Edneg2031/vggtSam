"""Small utilities to summarize unknown model outputs."""

from __future__ import annotations

from typing import Any, Dict, List


def summarize_object(obj: Any, *, max_depth: int = 4) -> Dict[str, Any]:
    return _summarize(obj, name="root", depth=0, max_depth=max_depth)


def tensor_candidates(obj: Any, *, max_depth: int = 5) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    _collect_tensors(obj, "root", 0, max_depth, out)
    return out


def _summarize(obj: Any, *, name: str, depth: int, max_depth: int) -> Dict[str, Any]:
    if depth > max_depth:
        return {"name": name, "type": type(obj).__name__, "truncated": True}

    shape = getattr(obj, "shape", None)
    dtype = getattr(obj, "dtype", None)
    device = getattr(obj, "device", None)
    if shape is not None and dtype is not None:
        return {
            "name": name,
            "type": type(obj).__name__,
            "shape": [int(v) for v in shape],
            "dtype": str(dtype),
            "device": str(device) if device is not None else None,
        }

    if isinstance(obj, dict):
        return {
            "name": name,
            "type": "dict",
            "keys": list(obj.keys()),
            "children": [
                _summarize(value, name=str(key), depth=depth + 1, max_depth=max_depth)
                for key, value in obj.items()
            ],
        }

    if isinstance(obj, (list, tuple)):
        return {
            "name": name,
            "type": type(obj).__name__,
            "len": len(obj),
            "children": [
                _summarize(value, name=f"{name}[{idx}]", depth=depth + 1, max_depth=max_depth)
                for idx, value in enumerate(obj[:8])
            ],
        }

    return {"name": name, "type": type(obj).__name__, "repr": _short_repr(obj)}


def _collect_tensors(
    obj: Any,
    path: str,
    depth: int,
    max_depth: int,
    out: List[Dict[str, Any]],
) -> None:
    if depth > max_depth:
        return

    shape = getattr(obj, "shape", None)
    dtype = getattr(obj, "dtype", None)
    if shape is not None and dtype is not None:
        out.append(
            {
                "path": path,
                "type": type(obj).__name__,
                "shape": [int(v) for v in shape],
                "dtype": str(dtype),
                "rank": len(shape),
            }
        )
        return

    if isinstance(obj, dict):
        for key, value in obj.items():
            _collect_tensors(value, f"{path}.{key}", depth + 1, max_depth, out)
    elif isinstance(obj, (list, tuple)):
        for idx, value in enumerate(obj):
            _collect_tensors(value, f"{path}[{idx}]", depth + 1, max_depth, out)


def _short_repr(obj: Any, limit: int = 120) -> str:
    text = repr(obj)
    if len(text) > limit:
        text = text[: limit - 3] + "..."
    return text
