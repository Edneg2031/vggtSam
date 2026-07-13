"""SAM3 video tracker adapter.

This adapter uses SAM3's original video predictor and memory propagation. It is
kept separate from the SAM3 intermediate-feature adapter because the first use
case is validation and visualization, not training loss.
"""

from __future__ import annotations

import contextlib
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from vggtsam.utils.imports import maybe_add_repo_to_path


@dataclass
class SAM3TrackOutput:
    masks: torch.Tensor
    selected_obj_id: Optional[int]
    prompt_frame_idx: int
    prompt_box_xywh: Optional[tuple[float, float, float, float]]
    scores: Optional[torch.Tensor] = None
    frame_objects: Dict[int, Dict[int, torch.Tensor]] = field(default_factory=dict)
    aux: Dict[str, Any] = field(default_factory=dict)


def load_sam3_video_predictor(
    *,
    repo_path: Optional[str | Path],
    checkpoint_path: str | Path,
    device: str,
    async_loading_frames: bool = False,
    quiet: bool = True,
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
    if not str(device).startswith("cuda"):
        raise RuntimeError("SAM3 video predictor requires a CUDA device.")

    with quiet_sam3_output(quiet):
        try:
            from sam3.model_builder import build_sam3_video_predictor
        except ModuleNotFoundError as exc:
            if exc.name == "sam3":
                raise RuntimeError(
                    "Could not import `sam3`. Run `git submodule update --init --recursive` "
                    "or pass `--sam3-repo` to a SAM3 repo."
                ) from exc
            raise

        gpu_id = parse_cuda_device_index(device)
        return build_sam3_video_predictor(
            checkpoint_path=str(checkpoint_path),
            gpus_to_use=[gpu_id],
            compile=False,
            async_loading_frames=async_loading_frames,
        )


class SAM3VideoTrackerAdapter:
    """Track a prompted object with SAM3's original video memory."""

    def __init__(
        self,
        predictor,
        *,
        output_prob_thresh: float = 0.5,
        prompt_with_box: bool = True,
    ) -> None:
        self.predictor = predictor
        self.output_prob_thresh = float(output_prob_thresh)
        self.prompt_with_box = bool(prompt_with_box)
        install_vggtsam_sam3_feature_hooks(self.predictor)

    @torch.no_grad()
    def track_from_paths(
        self,
        image_paths: Sequence[str | Path],
        *,
        prompt: str,
        output_size: tuple[int, int],
        prompt_frame_idx: int = 0,
        reference_mask: torch.Tensor | np.ndarray | None = None,
        tracker_fpn_residuals: Optional[Sequence[torch.Tensor]] = None,
        tracker_fpn_residual_scale: float = 1.0,
        quiet: bool = True,
    ) -> SAM3TrackOutput:
        if not image_paths:
            raise ValueError("At least one image path is required for SAM3 tracking.")
        prompt_frame_idx = int(prompt_frame_idx)
        if prompt_frame_idx < 0 or prompt_frame_idx >= len(image_paths):
            raise ValueError(
                f"prompt_frame_idx={prompt_frame_idx} is out of range for "
                f"{len(image_paths)} frames."
            )

        with tempfile.TemporaryDirectory(prefix="sam3_track_") as tmp:
            tmp_dir = Path(tmp)
            materialize_video_dir(image_paths, tmp_dir)
            with quiet_sam3_output(quiet):
                session = self.predictor.start_session(resource_path=str(tmp_dir))
                session_id = session["session_id"] if isinstance(session, dict) else session
                try:
                    residual_enabled = tracker_fpn_residuals is not None
                    if residual_enabled:
                        set_vggtsam_tracker_fpn_residuals(
                            self.predictor,
                            session_id=session_id,
                            residuals=tracker_fpn_residuals,
                            scale=tracker_fpn_residual_scale,
                        )
                    prompt_box = None
                    reference_mask_out = normalize_reference_mask(
                        reference_mask,
                        output_size=output_size,
                    )
                    if self.prompt_with_box and reference_mask_out is not None:
                        prompt_box = mask_to_normalized_box(
                            reference_mask_out,
                            image_path=Path(image_paths[prompt_frame_idx]),
                        )

                    add_kwargs: Dict[str, Any] = {
                        "session_id": session_id,
                        "frame_idx": prompt_frame_idx,
                        "text": prompt,
                        "output_prob_thresh": self.output_prob_thresh,
                    }
                    if prompt_box is not None:
                        add_kwargs["bounding_boxes"] = [prompt_box]
                        add_kwargs["bounding_box_labels"] = [1]
                    prompted = self.predictor.add_prompt(**add_kwargs)
                    propagated = list(
                        self.predictor.propagate_in_video(
                            session_id=session_id,
                            propagation_direction="both",
                            start_frame_idx=prompt_frame_idx,
                            max_frame_num_to_track=len(image_paths),
                            output_prob_thresh=self.output_prob_thresh,
                        )
                    )
                finally:
                    self.predictor.close_session(session_id)

        frame_objects = collect_frame_objects(
            [prompted, *propagated],
            output_size=output_size,
        )
        frame_scores = collect_frame_scores([prompted, *propagated])
        selected_obj_id = select_tracked_object_id(
            frame_objects,
            prompt_frame_idx=prompt_frame_idx,
            reference_mask=reference_mask_out,
        )
        masks = masks_for_selected_object(
            frame_objects,
            selected_obj_id=selected_obj_id,
            num_frames=len(image_paths),
            output_size=output_size,
        )
        scores = scores_for_selected_object(
            frame_scores,
            selected_obj_id=selected_obj_id,
            num_frames=len(image_paths),
        )
        return SAM3TrackOutput(
            masks=masks,
            selected_obj_id=selected_obj_id,
            prompt_frame_idx=prompt_frame_idx,
            prompt_box_xywh=tuple(prompt_box) if prompt_box is not None else None,
            scores=scores,
            frame_objects=frame_objects,
            aux={
                "prompt": prompt,
                "num_frames": len(image_paths),
                "vggtsam_tracker_fpn_residuals": bool(
                    tracker_fpn_residuals is not None
                ),
                "vggtsam_tracker_fpn_residual_scale": float(
                    tracker_fpn_residual_scale
                ),
                "frame_object_counts": {
                    int(frame_idx): len(objects)
                    for frame_idx, objects in frame_objects.items()
                },
                "score_semantics": (
                    "SAM3 out_probs for the selected ID; propagation keeps the "
                    "initial detector score, while a missing ID is assigned zero"
                ),
            },
        )


@contextlib.contextmanager
def quiet_sam3_output(enabled: bool = True):
    if not enabled:
        yield
        return

    previous_levels: Dict[str, int] = {}
    set_sam3_loggers(logging.WARNING, previous_levels)
    with open(os.devnull, "w", encoding="utf8") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            try:
                yield
            finally:
                restore_loggers(previous_levels)


def set_sam3_loggers(level: int, previous_levels: Dict[str, int]) -> None:
    names = {"", "sam3"}
    names.update(
        name
        for name in logging.Logger.manager.loggerDict.keys()
        if str(name).startswith("sam3")
    )
    for name in names:
        logger = logging.getLogger(name)
        previous_levels.setdefault(name, logger.level)
        logger.setLevel(level)


def restore_loggers(previous_levels: Dict[str, int]) -> None:
    for name, level in previous_levels.items():
        logging.getLogger(name).setLevel(level)


def set_vggtsam_tracker_fpn_residuals(
    predictor,
    *,
    session_id: str,
    residuals: Sequence[torch.Tensor],
    scale: float,
) -> None:
    """Attach per-frame FPN residuals to SAM3's local inference feature cache."""
    if not hasattr(predictor, "_get_session"):
        raise RuntimeError("SAM3 predictor does not expose _get_session for local hooks.")
    session = predictor._get_session(session_id)
    inference_state = session["state"]
    feature_cache = inference_state.setdefault("feature_cache", {})
    feature_cache["vggtsam_tracker_fpn_residuals"] = [
        residual.detach().cpu() if torch.is_tensor(residual) else residual
        for residual in residuals
    ]
    feature_cache["vggtsam_tracker_fpn_residual_scale"] = float(scale)


def install_vggtsam_sam3_feature_hooks(predictor) -> None:
    """Patch local SAM3 source objects so VGGT residuals enter full video flow."""
    model = getattr(predictor, "model", None)
    if model is None or getattr(model, "_vggtsam_feature_hooks_installed", False):
        return

    original_reset_state = model.reset_state

    def reset_state_with_vggtsam_cache(inference_state, *args, **kwargs):
        cache = inference_state.get("feature_cache", {})
        keep = {
            key: value
            for key, value in cache.items()
            if str(key).startswith("vggtsam_")
        }
        result = original_reset_state(inference_state, *args, **kwargs)
        inference_state.setdefault("feature_cache", {}).update(keep)
        return result

    original_run_backbone = model.run_backbone_and_detection

    def run_backbone_with_vggtsam_residuals(*args, **kwargs):
        frame_idx = kwargs.get("frame_idx", args[0] if len(args) > 0 else None)
        feature_cache = kwargs.get(
            "feature_cache",
            args[4] if len(args) > 4 else None,
        )
        result = original_run_backbone(*args, **kwargs)
        if frame_idx is not None and feature_cache is not None:
            apply_vggtsam_tracker_fpn_residuals(
                feature_cache,
                frame_idx=int(frame_idx),
            )
        return result

    model.reset_state = reset_state_with_vggtsam_cache
    model.run_backbone_and_detection = run_backbone_with_vggtsam_residuals
    model._vggtsam_feature_hooks_installed = True


def apply_vggtsam_tracker_fpn_residuals(
    feature_cache: Dict,
    *,
    frame_idx: int,
) -> None:
    """Inject cached residuals into SAM3 tracker FPNs after SAM3 builds them."""
    residuals = feature_cache.get("vggtsam_tracker_fpn_residuals")
    if residuals is None:
        return
    cached = feature_cache.get(int(frame_idx))
    if cached is None:
        return
    _, backbone_cache = cached
    scale = float(feature_cache.get("vggtsam_tracker_fpn_residual_scale", 1.0))
    for key in ("tracker_backbone_out", "sam2_backbone_out", "interactive"):
        backbone_out = backbone_cache.get(key)
        if backbone_out is None:
            continue
        fpn = backbone_out.get("backbone_fpn")
        if fpn is None:
            continue
        for level in range(len(fpn)):
            residual = select_vggtsam_fpn_residual(residuals, level, frame_idx)
            if residual is None:
                continue
            feature = fpn[level]
            if hasattr(feature, "tensors"):
                feature.tensors = add_vggtsam_residual(
                    feature.tensors,
                    residual,
                    scale,
                )
            else:
                fpn[level] = add_vggtsam_residual(feature, residual, scale)
        last = fpn[-1]
        backbone_out["vision_features"] = (
            last.tensors if hasattr(last, "tensors") else last
        )


def select_vggtsam_fpn_residual(
    residuals: Any,
    level: int,
    frame_idx: int,
) -> Optional[torch.Tensor]:
    if isinstance(residuals, dict):
        residual = residuals.get(level)
        if residual is None:
            residual = residuals.get(str(level))
        if residual is None:
            residual = residuals.get(f"fpn{level}")
    else:
        residual = residuals[level] if level < len(residuals) else None
    if residual is None:
        return None
    if residual.ndim == 4 and residual.shape[0] > frame_idx:
        residual = residual[frame_idx : frame_idx + 1]
    elif residual.ndim == 3:
        residual = residual.unsqueeze(0)
    return residual


def add_vggtsam_residual(
    feature: torch.Tensor,
    residual: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    residual = residual.to(device=feature.device, dtype=feature.dtype)
    if residual.shape[-2:] != feature.shape[-2:]:
        residual = F.interpolate(
            residual,
            size=feature.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    if residual.shape[0] == 1 and feature.shape[0] > 1:
        residual = residual.expand(feature.shape[0], -1, -1, -1)
    if residual.shape[:2] != feature.shape[:2]:
        raise RuntimeError(
            "vggtsam tracker FPN residual shape mismatch: "
            f"feature={tuple(feature.shape)} residual={tuple(residual.shape)}"
        )
    return feature + float(scale) * residual


def parse_cuda_device_index(device: str) -> int:
    if device == "cuda":
        return int(torch.cuda.current_device())
    if device.startswith("cuda:"):
        return int(device.split(":", 1)[1])
    raise ValueError(f"Expected CUDA device string, got {device!r}")


def materialize_video_dir(image_paths: Sequence[str | Path], output_dir: Path) -> None:
    for idx, image_path in enumerate(image_paths):
        src = Path(image_path).expanduser().resolve()
        dst = output_dir / f"{idx:05d}.jpg"
        try:
            os.symlink(src, dst)
        except OSError:
            shutil.copyfile(src, dst)


def normalize_reference_mask(
    reference_mask: torch.Tensor | np.ndarray | None,
    *,
    output_size: tuple[int, int],
) -> torch.Tensor | None:
    if reference_mask is None:
        return None
    mask = torch.as_tensor(reference_mask).detach().cpu()
    if mask.ndim != 2:
        raise ValueError(f"Expected reference mask [H, W], got {tuple(mask.shape)}")
    if tuple(mask.shape) != tuple(output_size):
        mask = resize_bool_mask(mask, output_size)
    return mask.bool()


def mask_to_normalized_box(
    mask: torch.Tensor,
    *,
    image_path: Path,
) -> list[float] | None:
    if not mask.any():
        return None
    width, height = Image.open(image_path).size
    full = resize_bool_mask(mask, (height, width))
    ys, xs = full.nonzero(as_tuple=True)
    if xs.numel() == 0:
        return None
    x0 = float(xs.min().item())
    x1 = float(xs.max().item() + 1)
    y0 = float(ys.min().item())
    y1 = float(ys.max().item() + 1)
    return [
        max(0.0, min(1.0, x0 / float(width))),
        max(0.0, min(1.0, y0 / float(height))),
        max(1.0 / float(width), min(1.0, (x1 - x0) / float(width))),
        max(1.0 / float(height), min(1.0, (y1 - y0) / float(height))),
    ]


def collect_frame_objects(
    results: Sequence[Dict[str, Any]],
    *,
    output_size: tuple[int, int],
) -> Dict[int, Dict[int, torch.Tensor]]:
    frame_objects: Dict[int, Dict[int, torch.Tensor]] = {}
    for result in results:
        if result is None:
            continue
        frame_idx = int(result.get("frame_index", -1))
        if frame_idx < 0:
            continue
        outputs = result.get("outputs", {}) or {}
        obj_ids = np.asarray(outputs.get("out_obj_ids", []), dtype=np.int64).reshape(-1)
        raw_masks = outputs.get("out_binary_masks", [])
        masks_np = np.asarray(raw_masks)
        if obj_ids.size == 0 or masks_np.size == 0:
            frame_objects.setdefault(frame_idx, {})
            continue
        if masks_np.ndim == 4 and masks_np.shape[1] == 1:
            masks_np = masks_np[:, 0]
        if masks_np.ndim == 2:
            masks_np = masks_np[None]
        objects = frame_objects.setdefault(frame_idx, {})
        for obj_id, mask_np in zip(obj_ids.tolist(), masks_np):
            mask = torch.from_numpy(np.asarray(mask_np).astype(bool))
            objects[int(obj_id)] = resize_bool_mask(mask, output_size)
    return frame_objects


def collect_frame_scores(
    results: Sequence[Dict[str, Any]],
) -> Dict[int, Dict[int, float]]:
    """Collect SAM3 detector/tracker probabilities without changing mask selection."""

    frame_scores: Dict[int, Dict[int, float]] = {}
    for result in results:
        if result is None:
            continue
        frame_idx = int(result.get("frame_index", -1))
        if frame_idx < 0:
            continue
        outputs = result.get("outputs", {}) or {}
        obj_ids = np.asarray(outputs.get("out_obj_ids", []), dtype=np.int64).reshape(-1)
        probabilities = np.asarray(outputs.get("out_probs", []), dtype=np.float32).reshape(-1)
        scores = frame_scores.setdefault(frame_idx, {})
        for obj_id, probability in zip(obj_ids.tolist(), probabilities.tolist()):
            scores[int(obj_id)] = float(probability)
    return frame_scores


def scores_for_selected_object(
    frame_scores: Dict[int, Dict[int, float]],
    *,
    selected_obj_id: Optional[int],
    num_frames: int,
) -> torch.Tensor:
    scores = torch.zeros(int(num_frames), dtype=torch.float32)
    if selected_obj_id is None:
        return scores
    for frame_idx in range(int(num_frames)):
        scores[frame_idx] = float(
            frame_scores.get(frame_idx, {}).get(int(selected_obj_id), 0.0)
        )
    return scores


def select_tracked_object_id(
    frame_objects: Dict[int, Dict[int, torch.Tensor]],
    *,
    prompt_frame_idx: int,
    reference_mask: torch.Tensor | None,
) -> Optional[int]:
    objects = frame_objects.get(int(prompt_frame_idx), {})
    if not objects:
        return None
    if reference_mask is not None and reference_mask.any():
        best_obj_id = None
        best_iou = -1.0
        for obj_id, mask in objects.items():
            iou = binary_iou(mask, reference_mask)
            if iou > best_iou:
                best_iou = iou
                best_obj_id = obj_id
        if best_obj_id is not None:
            return int(best_obj_id)
    return int(max(objects.items(), key=lambda item: int(item[1].sum().item()))[0])


def masks_for_selected_object(
    frame_objects: Dict[int, Dict[int, torch.Tensor]],
    *,
    selected_obj_id: Optional[int],
    num_frames: int,
    output_size: tuple[int, int],
) -> torch.Tensor:
    masks = torch.zeros(
        int(num_frames),
        int(output_size[0]),
        int(output_size[1]),
        dtype=torch.bool,
    )
    if selected_obj_id is None:
        return masks
    for frame_idx in range(num_frames):
        mask = frame_objects.get(frame_idx, {}).get(int(selected_obj_id))
        if mask is not None:
            masks[frame_idx] = mask.bool()
    return masks


def resize_bool_mask(mask: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
    if tuple(mask.shape[-2:]) == tuple(output_size):
        return mask.bool()
    resized = F.interpolate(
        mask.float()[None, None],
        size=output_size,
        mode="nearest",
    )
    return resized[0, 0].bool()


def binary_iou(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred = pred.bool()
    target = target.bool()
    union = (pred | target).sum().item()
    if union == 0:
        return 1.0
    return float((pred & target).sum().item() / union)
