"""VGGT loading and output inspection helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from vggtsam.utils.imports import maybe_add_repo_to_path


def preprocess_vggt_images(
    images,
    *,
    patch_multiple: int = 14,
    value_scale: float = 1.0,
):
    """Convert image tensors to the padded format used by VGGT tests.

    Args:
        images: `[B, 3, H, W]` RGB tensor. Values may be 0-255 or 0-1.
        value_scale: Keep `1.0` to match the user's previous smoke test.
    """
    if images.ndim != 4:
        raise ValueError(f"Expected [B, 3, H, W], got {tuple(images.shape)}")

    import torch.nn.functional as F

    images = images.float()
    if value_scale != 1.0:
        images = images / value_scale

    _, _, height, width = images.shape
    new_h = ((height + patch_multiple - 1) // patch_multiple) * patch_multiple
    new_w = ((width + patch_multiple - 1) // patch_multiple) * patch_multiple
    if new_h == height and new_w == width:
        return images
    return F.interpolate(images, size=(new_h, new_w), mode="bilinear", align_corners=False)


def load_vggt_model(
    *,
    repo_path: Optional[str | Path],
    checkpoint_path: str | Path,
    device: str,
    strict: bool = False,
) -> Any:
    import torch

    maybe_add_repo_to_path(repo_path)
    from vggt.models.vggt import VGGT

    model = VGGT()
    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=strict)
    return model.to(device).eval()


def load_streamvggt_model(
    *,
    repo_path: Optional[str | Path],
    checkpoint_path: str | Path,
    device: str,
    strict: bool = True,
) -> Any:
    import torch

    maybe_add_repo_to_path(repo_path)
    from streamvggt.models.streamvggt import StreamVGGT

    model = StreamVGGT()
    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=strict)
    return model.to(device).eval()


def run_vggt_forward(
    model,
    images,
    *,
    patch_multiple: int = 14,
    value_scale: float = 1.0,
) -> Any:
    import torch

    images = preprocess_vggt_images(
        images, patch_multiple=patch_multiple, value_scale=value_scale
    )
    with torch.no_grad():
        return model(images)


def run_streamvggt_inference(
    model,
    frame_paths,
    *,
    device: str,
) -> Any:
    import torch

    from streamvggt.utils.load_fn import load_and_preprocess_images

    image_paths = [str(path) for path in frame_paths]
    images = load_and_preprocess_images(image_paths).to(device)
    frames = [{"img": images[idx].unsqueeze(0)} for idx in range(images.shape[0])]

    with torch.no_grad():
        if str(device).startswith("cuda"):
            dtype = (
                torch.bfloat16
                if torch.cuda.get_device_capability()[0] >= 8
                else torch.float16
            )
            with torch.cuda.amp.autocast(dtype=dtype):
                return model.inference(frames)
        return model.inference(frames)
