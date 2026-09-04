"""Backend-neutral semantic-instance mapping pipeline.

The package deliberately keeps model-specific code at the adapter boundary.
The mapper consumes canonical geometry and segmentation observations, so a
future geometry model can replace StreamVGGT without changing map fusion.
"""

from .contracts import (
    GeometryFrame,
    GeometryAwareSegmentationProvider,
    GeometryProvider,
    ObjectObservation,
    SegmentationFrame,
    SegmentationProvider,
    StreamingGeometryProvider,
)
from .adapters import (
    GeometryAwareSAM31SegmentationAdapter,
    HorizonStreamGeometryCacheAdapter,
    SAM31SegmentationAdapter,
    StreamVGGTGeometryAdapter,
    V0CacheGeometryAdapter,
    V0CacheSegmentationAdapter,
)
from .export import export_semantic_map
from .mapping import (
    MapWriteGateConfig,
    MapUpdateStats,
    ObjectTrackMap,
    SemanticMapBuilder,
    SemanticMapConfig,
    SemanticMapResult,
)
from .pipeline import SemanticMapPipeline
from .pipeline import SemanticMapPoseRefinementRun
from .object_pose_refinement import (
    DinoV3PatchFeatureMatcher,
    FeatureMatch,
    ObjectCorrespondence,
    ObjectFeatureMatcher,
    ObjectFramePair,
    ObjectObservationRecord,
    ObjectPoseEdge,
    ObjectPoseRefinementConfig,
    ObjectPoseRefiner,
    PoseRefinementResult,
    RGBPatchFeatureMatcher,
    RigidRegistration,
    apply_refined_camera_poses,
    create_object_feature_matcher,
    estimate_rigid_transform_ransac,
    weighted_rigid_transform,
    write_pose_refinement_debug,
)
from .object_pose_loss_refinement import (
    ObjectCloudObservation,
    ObjectLossEdge,
    ObjectPoseLossRefinementConfig,
    ObjectPoseLossRefiner,
)
from .evaluation import (
    ExportedMapMetricConfig,
    SimilarityAlignment,
    evaluate_exported_semantic_map,
    evaluate_pointmap_alignment,
    extract_exported_objects,
    fit_reference_alignment,
)
from .temporal_consensus import (
    TemporalConsensusConfig,
    TemporalConsensusDecision,
    TemporalConsensusMemory,
)
from .instance_point_consistency import (
    InstancePointConsistencyConfig,
    InstancePointConsistencyDecision,
    InstancePointConsistencyMemory,
)

__all__ = [
    "GeometryFrame",
    "GeometryAwareSegmentationProvider",
    "GeometryProvider",
    "GeometryAwareSAM31SegmentationAdapter",
    "HorizonStreamGeometryCacheAdapter",
    "SAM31SegmentationAdapter",
    "StreamVGGTGeometryAdapter",
    "V0CacheGeometryAdapter",
    "V0CacheSegmentationAdapter",
    "MapUpdateStats",
    "MapWriteGateConfig",
    "ObjectObservation",
    "ObjectTrackMap",
    "SegmentationFrame",
    "SegmentationProvider",
    "SemanticMapBuilder",
    "SemanticMapConfig",
    "SemanticMapPipeline",
    "SemanticMapPoseRefinementRun",
    "SemanticMapResult",
    "StreamingGeometryProvider",
    "ExportedMapMetricConfig",
    "SimilarityAlignment",
    "evaluate_exported_semantic_map",
    "evaluate_pointmap_alignment",
    "extract_exported_objects",
    "fit_reference_alignment",
    "TemporalConsensusConfig",
    "TemporalConsensusDecision",
    "TemporalConsensusMemory",
    "InstancePointConsistencyConfig",
    "InstancePointConsistencyDecision",
    "InstancePointConsistencyMemory",
    "export_semantic_map",
    "DinoV3PatchFeatureMatcher",
    "FeatureMatch",
    "ObjectCorrespondence",
    "ObjectFeatureMatcher",
    "ObjectFramePair",
    "ObjectObservationRecord",
    "ObjectPoseEdge",
    "ObjectPoseRefinementConfig",
    "ObjectPoseRefiner",
    "PoseRefinementResult",
    "RGBPatchFeatureMatcher",
    "RigidRegistration",
    "apply_refined_camera_poses",
    "create_object_feature_matcher",
    "estimate_rigid_transform_ransac",
    "weighted_rigid_transform",
    "write_pose_refinement_debug",
    "ObjectCloudObservation",
    "ObjectLossEdge",
    "ObjectPoseLossRefinementConfig",
    "ObjectPoseLossRefiner",
]
