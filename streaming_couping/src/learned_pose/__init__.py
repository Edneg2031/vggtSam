"""Learned persistent-instance guidance for StreamVGGT camera pose."""

from typing import TYPE_CHECKING

from .config import LearnedPoseConfig, load_learned_pose_config

if TYPE_CHECKING:
    from .model import InstancePoseAdapter

__all__ = [
    "InstancePoseAdapter",
    "LearnedPoseConfig",
    "load_learned_pose_config",
]


def __getattr__(name: str):
    """Keep configuration inspection independent of the PyTorch runtime."""

    if name == "InstancePoseAdapter":
        from .model import InstancePoseAdapter

        return InstancePoseAdapter
    raise AttributeError(name)
