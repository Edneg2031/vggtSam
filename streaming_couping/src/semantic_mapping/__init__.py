"""Backend-neutral semantic-instance mapping pipeline.

The package deliberately keeps model-specific code at the adapter boundary.
The mapper consumes canonical geometry and segmentation observations, so a
future geometry model can replace StreamVGGT without changing map fusion.
"""

from .contracts import (
    GeometryFrame,
    GeometryProvider,
    ObjectObservation,
    SegmentationFrame,
    SegmentationProvider,
    StreamingGeometryProvider,
)
from .adapters import (
    SAM31SegmentationAdapter,
    StreamVGGTGeometryAdapter,
    V0CacheGeometryAdapter,
    V0CacheSegmentationAdapter,
)
from .export import export_semantic_map
from .mapping import (
    MapUpdateStats,
    ObjectTrackMap,
    SemanticMapBuilder,
    SemanticMapConfig,
    SemanticMapResult,
)
from .pipeline import SemanticMapPipeline

__all__ = [
    "GeometryFrame",
    "GeometryProvider",
    "SAM31SegmentationAdapter",
    "StreamVGGTGeometryAdapter",
    "V0CacheGeometryAdapter",
    "V0CacheSegmentationAdapter",
    "MapUpdateStats",
    "ObjectObservation",
    "ObjectTrackMap",
    "SegmentationFrame",
    "SegmentationProvider",
    "SemanticMapBuilder",
    "SemanticMapConfig",
    "SemanticMapPipeline",
    "SemanticMapResult",
    "StreamingGeometryProvider",
    "export_semantic_map",
]
