"""Causal dynamic-instance guidance for StreamVGGT camera pose."""

from typing import TYPE_CHECKING

from .config import LearnedPoseConfig, load_learned_pose_config

if TYPE_CHECKING:
    from .dynamic_instance_baseline import (
        BaselineModelConfig,
        CameraPoseBaseline,
        DynamicInstanceGeometryRefiner,
    )

__all__ = [
    "BaselineModelConfig",
    "CameraPoseBaseline",
    "DynamicInstanceGeometryRefiner",
    "LearnedPoseConfig",
    "load_learned_pose_config",
]


def __getattr__(name: str):
    """Keep YAML/config inspection independent of the PyTorch runtime."""

    if name in {
        "BaselineModelConfig",
        "CameraPoseBaseline",
        "DynamicInstanceGeometryRefiner",
    }:
        from . import dynamic_instance_baseline

        return getattr(dynamic_instance_baseline, name)
    raise AttributeError(name)
