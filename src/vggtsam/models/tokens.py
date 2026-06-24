"""Typed token containers for foundation-model adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import torch


@dataclass
class GeometryTokens:
    """Geometry-side tokens produced by VGGT or StreamVGGT."""

    tokens: torch.Tensor
    camera_tokens: Optional[torch.Tensor] = None
    pointmap: Optional[torch.Tensor] = None
    depth: Optional[torch.Tensor] = None
    spatial_shape: Optional[Tuple[int, int]] = None
    kv_cache: Optional[Any] = None
    aux: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SemanticTokens:
    """Semantic/object tokens produced by SAM3 or a mask-pooling adapter."""

    tokens: torch.Tensor
    masks: Optional[torch.Tensor] = None
    object_ids: Optional[torch.Tensor] = None
    spatial_shape: Optional[Tuple[int, int]] = None
    aux: Dict[str, Any] = field(default_factory=dict)
