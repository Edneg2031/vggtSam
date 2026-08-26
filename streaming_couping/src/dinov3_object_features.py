"""DINOv3 dense features and causal object-level feature memory.

The module is deliberately independent of SAM3 and StreamVGGT model weights.
It consumes frozen RGB frames and frozen SAM stream masks, producing one
feature per mask slot for single-frame, persistent-EMA, and shuffled-ID
controls.  The latter is important: an apparent gain from aggregation is not
evidence for persistent identity unless it beats the shuffled control.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.nn import functional as F


DEFAULT_DINOV3_ROOT = Path(
    "/home/bod/86Nas/95_data_bak/FoundationModels/dinov3"
)
DEFAULT_DINOV3_VARIANTS = (
    "dinov3-vitl16",
    "dinov3-vith16",
    "dinov3-vit7b16",
)
DEFAULT_DENSE_LAYER_FRACTIONS = (0.25, 0.50, 0.75, 1.0)


@dataclass(frozen=True)
class DinoV3FeatureConfig:
    """Local DINOv3 inference settings."""

    checkpoint: Path | None = None
    root: Path = DEFAULT_DINOV3_ROOT
    preferred_variant: str = "dinov3-vitl16"
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    ema_beta: float = 0.90
    projection_dim: int = 256
    mask_threshold: float = 0.5
    normalize_features: bool = True


def resolve_dinov3_checkpoint(
    checkpoint: str | Path | None = None,
    *,
    root: str | Path = DEFAULT_DINOV3_ROOT,
    preferred_variant: str = "dinov3-vitl16",
) -> Path:
    """Resolve an existing local Hugging Face DINOv3 directory.

    The preferred order is ViT-L/16, ViT-H/16, then the supplied ViT-7B/16
    checkpoint.  This keeps the first experiment tractable while supporting
    the exact 7B directory supplied by the user without any code change.
    """

    explicit = Path(checkpoint).expanduser() if checkpoint else None
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    root_path = Path(root).expanduser()
    variants = [str(preferred_variant)] + [
        value
        for value in DEFAULT_DINOV3_VARIANTS
        if value != str(preferred_variant)
    ]
    candidates.extend(root_path / value for value in variants)
    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if _is_huggingface_checkpoint(candidate):
            return candidate
    attempted = "\n".join(f"  - {value}" for value in candidates)
    raise FileNotFoundError(
        "No local DINOv3 Hugging Face checkpoint was found.\n" + attempted
    )


def _is_huggingface_checkpoint(path: Path) -> bool:
    if not path.is_dir() or not (path / "config.json").is_file():
        return False
    return any(
        value.is_file()
        for value in path.glob("*.safetensors")
    ) or (path / "model.safetensors.index.json").is_file()


def dtype_from_name(name: str) -> torch.dtype:
    value = str(name).strip().lower()
    if value in {"float32", "fp32", "32"}:
        return torch.float32
    if value in {"float16", "fp16", "16"}:
        return torch.float16
    if value in {"bfloat16", "bf16", "bf"}:
        return torch.bfloat16
    raise ValueError(f"Unsupported DINOv3 dtype {name!r}.")


class DinoV3DenseEncoder:
    """Offline local DINOv3 encoder returning dense patch features."""

    def __init__(self, config: DinoV3FeatureConfig):
        self.config = config
        self.checkpoint = resolve_dinov3_checkpoint(
            config.checkpoint,
            root=config.root,
            preferred_variant=config.preferred_variant,
        )
        try:
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as exc:  # pragma: no cover - server environment path
            raise RuntimeError(
                "DINOv3 requires the transformers package in the server environment."
            ) from exc

        self.device = torch.device(config.device)
        self.dtype = dtype_from_name(config.dtype)
        load_dtype = self.dtype if self.device.type == "cuda" else torch.float32
        self.processor = AutoImageProcessor.from_pretrained(
            str(self.checkpoint),
            local_files_only=True,
        )
        self.model = AutoModel.from_pretrained(
            str(self.checkpoint),
            local_files_only=True,
            torch_dtype=load_dtype,
        ).eval().to(self.device)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        model_config = getattr(self.model, "config", None)
        patch_size = getattr(model_config, "patch_size", 16)
        if isinstance(patch_size, Mapping):
            patch_size = patch_size.get("height", patch_size.get("width", 16))
        self.patch_size = int(patch_size)
        if self.patch_size <= 0:
            raise ValueError(f"Invalid DINOv3 patch size {self.patch_size}.")
        self.image_mean = torch.tensor(
            getattr(self.processor, "image_mean", (0.485, 0.456, 0.406)),
            dtype=torch.float32,
        ).view(1, 3, 1, 1)
        self.image_std = torch.tensor(
            getattr(self.processor, "image_std", (0.229, 0.224, 0.225)),
            dtype=torch.float32,
        ).view(1, 3, 1, 1)

    @property
    def feature_dim(self) -> int:
        value = getattr(self.model.config, "hidden_size", None)
        if value is None:
            value = getattr(self.model.config, "embed_dim", None)
        if value is None:
            raise RuntimeError("Cannot infer DINOv3 hidden size from config.")
        return int(value)

    @property
    def hidden_layer_count(self) -> int:
        value = getattr(self.model.config, "num_hidden_layers", None)
        if value is None:
            value = getattr(self.model.config, "num_layers", None)
        if value is None:
            raise RuntimeError("Cannot infer DINOv3 transformer depth from config.")
        return int(value)

    def representative_layer_ids(self, count: int = 4) -> tuple[int, ...]:
        """Return explicit 1-based transformer block IDs spread through depth."""

        count = min(max(1, int(count)), self.hidden_layer_count)
        fractions = DEFAULT_DENSE_LAYER_FRACTIONS
        if count != len(fractions):
            fractions = tuple(
                float(index + 1) / float(count) for index in range(count)
            )
        selected = sorted(
            {
                max(1, min(self.hidden_layer_count, round(self.hidden_layer_count * fraction)))
                for fraction in fractions
            }
        )
        # Rounding can collapse two fractions for unusually shallow models.
        for layer_id in range(1, self.hidden_layer_count + 1):
            if len(selected) >= count:
                break
            if layer_id not in selected:
                selected.append(layer_id)
                selected.sort()
        return tuple(selected)

    def _prepare_inputs(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, int, int, int, int]:
        values = _image_tensor(images)
        batch, _, height, width = values.shape
        target_height = max(
            self.patch_size,
            round(height / self.patch_size) * self.patch_size,
        )
        target_width = max(
            self.patch_size,
            round(width / self.patch_size) * self.patch_size,
        )
        if (target_height, target_width) != (height, width):
            values = F.interpolate(
                values,
                size=(target_height, target_width),
                mode="bilinear",
                align_corners=False,
            )
        mean = self.image_mean.to(values.device)
        std = self.image_std.to(values.device)
        values = (values - mean) / std
        return values.to(self.device), batch, target_height, target_width, (
            target_height // self.patch_size
        ) * (target_width // self.patch_size)

    def _base_metadata(
        self,
        *,
        target_height: int,
        target_width: int,
        patch_height: int,
        patch_width: int,
        layer_ids: Sequence[int] | None = None,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "input_size": [int(target_height), int(target_width)],
            "patch_size": int(self.patch_size),
            "patch_shape": [int(patch_height), int(patch_width)],
            "feature_dim": int(self.feature_dim),
            "checkpoint": str(self.checkpoint),
            "num_hidden_layers": int(self.hidden_layer_count),
        }
        if layer_ids is not None:
            metadata["layer_ids"] = [int(value) for value in layer_ids]
            metadata["layer_indexing"] = "1-based transformer block; hidden_states[0] is embeddings"
        return metadata

    def _tokens_to_dense(
        self,
        tokens: torch.Tensor,
        *,
        batch: int,
        patch_height: int,
        patch_width: int,
    ) -> torch.Tensor:
        patch_count = patch_height * patch_width
        if int(tokens.shape[1]) < patch_count:
            raise RuntimeError(
                "DINOv3 returned fewer tokens than the expected patch grid: "
                f"tokens={tuple(tokens.shape)}, expected_patches={patch_count}."
            )
        return tokens[:, -patch_count:, :].reshape(
            batch, patch_height, patch_width, int(tokens.shape[-1])
        ).float().cpu()

    @torch.inference_mode()
    def encode_dense(self, images: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        """Return ``[B, patch_h, patch_w, C]`` dense features.

        Images are resized without center cropping, preserving mask alignment
        with the original frame.  The dimensions are rounded to the patch
        size, and the same resize is applied to masks by masked_mean_pool.
        """

        values, batch, target_height, target_width, _ = self._prepare_inputs(images)
        with torch.autocast(
            device_type=self.device.type,
            dtype=self.dtype,
            enabled=self.device.type == "cuda" and self.dtype != torch.float32,
        ):
            output = self.model(pixel_values=values)
        tokens = getattr(output, "last_hidden_state", None)
        if tokens is None:
            tokens = output[0]
        patch_height = target_height // self.patch_size
        patch_width = target_width // self.patch_size
        dense = self._tokens_to_dense(
            tokens,
            batch=batch,
            patch_height=patch_height,
            patch_width=patch_width,
        )
        metadata = self._base_metadata(
            target_height=target_height,
            target_width=target_width,
            patch_height=patch_height,
            patch_width=patch_width,
        )
        return dense, metadata

    @torch.inference_mode()
    def encode_dense_layers(
        self,
        images: torch.Tensor,
        *,
        layer_ids: Sequence[int] | None = None,
    ) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
        """Return dense features from explicit intermediate transformer blocks."""

        values, batch, target_height, target_width, _ = self._prepare_inputs(images)
        requested = (
            self.representative_layer_ids()
            if layer_ids is None
            else tuple(int(value) for value in layer_ids)
        )
        if not requested or any(
            value < 1 or value > self.hidden_layer_count for value in requested
        ):
            raise ValueError(
                f"DINO layer IDs must be in [1,{self.hidden_layer_count}], got {requested}."
            )
        with torch.autocast(
            device_type=self.device.type,
            dtype=self.dtype,
            enabled=self.device.type == "cuda" and self.dtype != torch.float32,
        ):
            output = self.model(pixel_values=values, output_hidden_states=True)
        hidden_states = getattr(output, "hidden_states", None)
        if hidden_states is None:
            raise RuntimeError(
                "DINOv3 model did not return hidden_states; cannot run multi-layer "
                "geometry-prior diagnosis."
            )
        actual_depth = len(hidden_states) - 1
        if actual_depth < self.hidden_layer_count:
            raise RuntimeError(
                f"DINO hidden-state depth is {actual_depth}, config reports "
                f"{self.hidden_layer_count}."
            )
        patch_height = target_height // self.patch_size
        patch_width = target_width // self.patch_size
        dense = {
            int(layer_id): self._tokens_to_dense(
                hidden_states[int(layer_id)],
                batch=batch,
                patch_height=patch_height,
                patch_width=patch_width,
            )
            for layer_id in requested
        }
        metadata = self._base_metadata(
            target_height=target_height,
            target_width=target_width,
            patch_height=patch_height,
            patch_width=patch_width,
            layer_ids=requested,
        )
        return dense, metadata


def masked_mean_pool(
    dense_features: torch.Tensor,
    masks: torch.Tensor,
    *,
    normalize: bool = True,
    mask_threshold: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pool dense features inside each mask.

    Args:
        dense_features: ``[B, h, w, C]``.
        masks: ``[B, K, H, W]`` boolean or soft masks.
    Returns:
        ``features`` with shape ``[B, K, C]`` and ``valid`` with shape
        ``[B, K]``.
    """

    dense = torch.as_tensor(dense_features).float()
    mask_values = torch.as_tensor(masks).float()
    if dense.ndim != 4 or mask_values.ndim != 4:
        raise ValueError("DINO dense features and masks must both be 4-D.")
    if dense.shape[0] != mask_values.shape[0]:
        raise ValueError("DINO feature batch and mask batch disagree.")
    resized = F.interpolate(
        mask_values,
        size=tuple(int(value) for value in dense.shape[1:3]),
        mode="nearest",
    )
    binary = resized >= float(mask_threshold)
    weights = binary.float()
    numerator = torch.einsum("bkhw,bhwc->bkc", weights, dense)
    denominator = weights.sum(dim=(-1, -2))
    valid = denominator > 0.0
    pooled = numerator / denominator.clamp_min(1.0).unsqueeze(-1)
    if normalize:
        pooled = F.normalize(pooled, dim=-1)
    pooled = torch.where(valid.unsqueeze(-1), pooled, torch.zeros_like(pooled))
    return pooled, valid


def aggregate_persistent_features(
    single_features: torch.Tensor,
    valid: torch.Tensor,
    *,
    beta: float = 0.90,
    track_ids: torch.Tensor | None = None,
    normalize: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Causally aggregate per-frame object features with an EMA."""

    features = torch.as_tensor(single_features).float().cpu()
    observed = torch.as_tensor(valid).bool().cpu()
    if features.ndim != 3 or observed.shape != features.shape[:2]:
        raise ValueError("single_features must be [S,K,C] and valid must be [S,K].")
    if not 0.0 <= float(beta) < 1.0:
        raise ValueError("DINO EMA beta must be in [0, 1).")
    ids = None if track_ids is None else torch.as_tensor(track_ids).cpu()
    if ids is not None and ids.shape not in {features.shape[:2], features.shape[1:2]}:
        raise ValueError("track_ids must be [S,K] or [K].")
    sequence, slots, channels = map(int, features.shape)
    output = torch.zeros_like(features)
    output_valid = torch.zeros(sequence, slots, dtype=torch.bool)
    state = torch.zeros(slots, channels, dtype=torch.float32)
    seen = torch.zeros(slots, dtype=torch.bool)
    previous_ids: list[int | None] = [None] * slots
    for frame in range(sequence):
        for slot in range(slots):
            if not bool(observed[frame, slot]):
                if bool(seen[slot]):
                    output[frame, slot] = state[slot]
                    output_valid[frame, slot] = True
                continue
            current_id = _track_id(ids, frame, slot)
            if current_id is not None and previous_ids[slot] not in {None, current_id}:
                seen[slot] = False
            value = features[frame, slot]
            if not bool(seen[slot]):
                state[slot] = value
                seen[slot] = True
            else:
                state[slot] = float(beta) * state[slot] + (1.0 - float(beta)) * value
                if normalize:
                    state[slot] = F.normalize(state[slot].unsqueeze(0), dim=-1)[0]
            previous_ids[slot] = current_id
            output[frame, slot] = state[slot]
            output_valid[frame, slot] = True
    if normalize:
        output = F.normalize(output, dim=-1)
        output = torch.where(output_valid.unsqueeze(-1), output, torch.zeros_like(output))
    return output, output_valid


def shuffled_persistent_features(
    single_features: torch.Tensor,
    valid: torch.Tensor,
    *,
    beta: float = 0.90,
    seed: int = 2026,
    normalize: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a deterministic wrong-ID control with the same feature budget."""

    features = torch.as_tensor(single_features).float().cpu()
    observed = torch.as_tensor(valid).bool().cpu()
    if features.ndim != 3 or observed.shape != features.shape[:2]:
        raise ValueError("single_features must be [S,K,C] and valid must be [S,K].")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    permutation = torch.randperm(int(features.shape[1]), generator=generator)
    shuffled_features = features[:, permutation]
    shuffled_valid = observed[:, permutation]
    persistent, persistent_valid = aggregate_persistent_features(
        shuffled_features,
        shuffled_valid,
        beta=beta,
        normalize=normalize,
    )
    return persistent, persistent_valid, permutation


def _track_id(ids: torch.Tensor | None, frame: int, slot: int) -> int | None:
    if ids is None:
        return None
    if ids.ndim == 1:
        return int(ids[slot])
    value = int(ids[frame, slot])
    return None if value < 0 else value


def _image_tensor(images: torch.Tensor) -> torch.Tensor:
    values = torch.as_tensor(images).float()
    if values.ndim == 3:
        values = values.unsqueeze(0)
    if values.ndim != 4:
        raise ValueError("Images must have shape [B,3,H,W] or [B,H,W,3].")
    if values.shape[1] not in {1, 3} and values.shape[-1] in {1, 3}:
        values = values.permute(0, 3, 1, 2)
    if values.shape[1] == 1:
        values = values.repeat(1, 3, 1, 1)
    if values.shape[1] != 3:
        raise ValueError(f"Expected RGB images, got shape {tuple(values.shape)}.")
    if float(values.max()) > 1.5:
        values = values / 255.0
    return values.clamp(0.0, 1.0)


__all__ = [
    "DEFAULT_DINOV3_ROOT",
    "DEFAULT_DINOV3_VARIANTS",
    "DinoV3FeatureConfig",
    "DinoV3DenseEncoder",
    "aggregate_persistent_features",
    "dtype_from_name",
    "masked_mean_pool",
    "resolve_dinov3_checkpoint",
    "shuffled_persistent_features",
]
