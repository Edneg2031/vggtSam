"""SAM3 intermediate feature adapter.

This adapter intentionally bypasses SAM3 final mask filtering. It reads the
image/text backbone outputs that exist before the detector/tracker decides
whether an object is present.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from PIL import Image

from vggtsam.models.tokens import SemanticTokens
from vggtsam.utils.imports import maybe_add_repo_to_path


@dataclass
class SAM3IntermediateOutput:
    semantic: SemanticTokens
    backbone_out: Dict[str, Any] = field(default_factory=dict)
    text_out: Dict[str, Any] = field(default_factory=dict)


def load_sam3_image_model(
    *,
    repo_path: Optional[str | Path],
    checkpoint_path: str | Path,
    device: str,
    enable_segmentation: bool = False,
    enable_inst_interactivity: bool = False,
):
    repo = maybe_add_repo_to_path(repo_path)
    if repo_path is not None:
        expected = Path(repo_path).expanduser()
        if repo is None:
            raise RuntimeError(
                f"SAM3 repo path does not exist: {expected}\n"
                "Run `git submodule update --init --recursive`, or pass the correct repo path."
            )
        if not ((repo / "sam3").is_dir() or (repo / "src" / "sam3").is_dir()):
            raise RuntimeError(
                f"SAM3 repo at {repo} does not look initialized; missing package `sam3`."
            )
    try:
        from sam3.model_builder import build_sam3_image_model
    except ModuleNotFoundError as exc:
        if exc.name == "sam3":
            raise RuntimeError(
                "Could not import `sam3`. Run `git submodule update --init --recursive` "
                "or pass `--sam3-repo` to a SAM3 repo."
            ) from exc
        raise

    build_device = "cuda" if str(device).startswith("cuda") else "cpu"
    model = build_sam3_image_model(
        checkpoint_path=str(checkpoint_path),
        device=build_device,
        eval_mode=True,
        load_from_HF=False,
        enable_segmentation=enable_segmentation,
        enable_inst_interactivity=enable_inst_interactivity,
        compile=False,
    )
    return model.to(device).eval()


class SAM3IntermediateAdapter:
    """Extract 72x72-ish SAM3 latent tokens instead of final masks."""

    def __init__(
        self,
        model,
        *,
        device: str,
        resolution: int = 1008,
        source: str = "detector_fpn2",
        text_conditioning: str = "concat",
        token_grid: Tuple[int, int] = (72, 72),
    ) -> None:
        self.model = model.eval()
        self.device = device
        self.resolution = int(resolution)
        self.source = source
        self.text_conditioning = text_conditioning
        self.token_grid = token_grid
        self.transform = self._build_transform()

    @torch.no_grad()
    def extract_from_paths(
        self,
        image_paths: Sequence[str | Path],
        *,
        prompt: str = "object",
    ) -> SAM3IntermediateOutput:
        images = self._load_images(image_paths)
        return self.extract(images, prompt=prompt)

    @torch.no_grad()
    def extract(
        self,
        images: torch.Tensor,
        *,
        prompt: str = "object",
    ) -> SAM3IntermediateOutput:
        if images.ndim != 4:
            raise ValueError(f"Expected images [T, 3, H, W], got {tuple(images.shape)}")

        with sam3_autocast(self.device):
            backbone_out = self.model.backbone.forward_image(images.to(self.device))
            text_out = self.model.backbone.forward_text([prompt], device=self.device)

        spatial = select_sam3_spatial_feature(backbone_out, source=self.source)
        spatial = ensure_bchw_tensor(spatial).float()
        spatial = resize_feature_map(spatial, self.token_grid)

        tokens = spatial.permute(0, 2, 3, 1).reshape(
            1, spatial.shape[0] * self.token_grid[0] * self.token_grid[1], spatial.shape[1]
        )

        language = pool_language_features(text_out)
        aux: Dict[str, Any] = {
            "source": self.source,
            "text_conditioning": self.text_conditioning,
            "spatial_shape": self.token_grid,
            "raw_spatial_shape": tuple(int(v) for v in spatial.shape[-2:]),
        }
        if language is not None:
            language = language.to(tokens.device, dtype=tokens.dtype)
            aux["language_shape"] = tuple(int(v) for v in language.shape)
            if self.text_conditioning == "concat":
                language_tokens = language[:, None, :].expand(tokens.shape[0], tokens.shape[1], -1)
                tokens = torch.cat([tokens, language_tokens], dim=-1)
            elif self.text_conditioning == "add":
                if language.shape[-1] != tokens.shape[-1]:
                    raise ValueError(
                        "SAM3 text_conditioning='add' requires language dim to match spatial dim, "
                        f"got {language.shape[-1]} and {tokens.shape[-1]}"
                    )
                tokens = tokens + language[:, None, :]
            elif self.text_conditioning == "none":
                pass
            else:
                raise ValueError(f"Unknown text_conditioning: {self.text_conditioning}")

        return SAM3IntermediateOutput(
            semantic=SemanticTokens(
                tokens=tokens,
                spatial_shape=self.token_grid,
                aux=aux,
            ),
            backbone_out=backbone_out,
            text_out=text_out,
        )

    def _load_images(self, image_paths: Sequence[str | Path]) -> torch.Tensor:
        if not image_paths:
            raise ValueError("At least one image path is required")
        tensors = []
        for image_path in image_paths:
            image = Image.open(image_path).convert("RGB")
            tensors.append(self.transform(image))
        return torch.stack(tensors, dim=0).to(self.device)

    def _build_transform(self):
        from torchvision.transforms import v2

        return v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.uint8, scale=True),
                v2.Resize(size=(self.resolution, self.resolution)),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )


def select_sam3_spatial_feature(backbone_out: Dict[str, Any], *, source: str) -> Any:
    source = source.strip().lower()
    if source in {"detector_fpn2", "sam3_fpn2"}:
        return backbone_out["backbone_fpn"][-1]
    if source in {"detector_fpn1", "sam3_fpn1"}:
        return backbone_out["backbone_fpn"][-2]
    if source in {"detector_fpn0", "sam3_fpn0"}:
        return backbone_out["backbone_fpn"][-3]
    if source == "vision_features":
        return backbone_out["vision_features"]
    if source in {
        "tracker_fpn0",
        "tracker_fpn1",
        "tracker_fpn2",
        "sam2_fpn0",
        "sam2_fpn1",
        "sam2_fpn2",
        "tracker_vision_features",
        "sam2_vision_features",
    }:
        sam2 = backbone_out.get("sam2_backbone_out")
        if sam2 is None:
            raise KeyError(
                "SAM3 backbone_out does not contain sam2_backbone_out. "
                "Build the image model with enable_inst_interactivity=True."
            )
        if source in {"tracker_fpn2", "sam2_fpn2"}:
            return sam2["backbone_fpn"][-1]
        if source in {"tracker_fpn1", "sam2_fpn1"}:
            return sam2["backbone_fpn"][-2]
        if source in {"tracker_fpn0", "sam2_fpn0"}:
            return sam2["backbone_fpn"][-3]
        return sam2["vision_features"]
    raise KeyError(f"Unknown SAM3 feature source: {source}")


def ensure_bchw_tensor(value: Any) -> torch.Tensor:
    tensor = getattr(value, "tensors", value)
    if not torch.is_tensor(tensor):
        raise TypeError(f"Expected tensor-like SAM3 feature, got {type(value).__name__}")
    if tensor.ndim != 4:
        raise ValueError(f"Expected SAM3 feature [B, C, H, W], got {tuple(tensor.shape)}")
    return tensor


def resize_feature_map(feature: torch.Tensor, token_grid: Tuple[int, int]) -> torch.Tensor:
    if tuple(feature.shape[-2:]) == tuple(token_grid):
        return feature
    return F.interpolate(feature, size=token_grid, mode="bilinear", align_corners=False)


def pool_language_features(text_out: Dict[str, Any]) -> Optional[torch.Tensor]:
    features = text_out.get("language_features")
    if features is None or not torch.is_tensor(features):
        return None
    features = features.float()
    # SAM3 VE text features are usually [text_len, B_prompt, C].
    if features.ndim == 3:
        if features.shape[1] == 1:
            return features.mean(dim=0)
        if features.shape[0] == 1:
            return features.mean(dim=1)
        return features.mean(dim=0, keepdim=False)[:1]
    if features.ndim == 2:
        return features.mean(dim=0, keepdim=True)
    raise ValueError(f"Unsupported language feature shape: {tuple(features.shape)}")


def sam3_autocast(device: str):
    if str(device).startswith("cuda"):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()
