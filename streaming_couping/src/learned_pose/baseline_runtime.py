"""Runtime helpers for the retained V0 tracking and pose baseline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import yaml


@dataclass(frozen=True)
class BaselineRunConfig:
    source_path: Path
    version: str
    output_dir: Path
    clip_name: str
    audit_device: str
    evaluation_frames: tuple[int, ...]
    selected_pose_branch: str
    qk_pose_output: Path
    allow_raw_pose_fallback: bool


def load_baseline_run_config(path: str | Path) -> BaselineRunConfig:
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf8") as handle:
        raw = yaml.safe_load(handle) or {}
    section = raw.get("baseline", {})
    frames = section.get("frames", {})
    pose = section.get("pose", {})
    qk = raw.get("qk_pose_retrieval", {})
    qk_output_dir = _path(
        qk.get(
            "output_dir",
            "outputs/streaming_couping_v0/clean_qk_pose_retrieval",
        )
    )
    config = BaselineRunConfig(
        source_path=source,
        version=str(section.get("version", "")).strip().lower(),
        output_dir=_path(section.get("output_dir", "outputs/streaming_couping_v0")),
        clip_name=str(section.get("clip_name", "")),
        audit_device=str(
            section.get(
                "audit_device",
                section.get("training_device", "cuda:0"),
            )
        ),
        evaluation_frames=_int_tuple(frames.get("evaluation", ())),
        selected_pose_branch=str(
            pose.get("selected_branch", "raw_streamvggt")
        ).strip().lower(),
        qk_pose_output=_path(
            pose.get(
                "qk_pose_output",
                qk_output_dir / "clean_qk_pose_output.pt",
            )
        ),
        allow_raw_pose_fallback=bool(
            pose.get("allow_raw_fallback", True)
        ),
    )
    _validate_config(config)
    return config


def decode_cached_poses(
    payload: dict,
    *,
    pose_decoder,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    baseline_encoding = payload.get("baseline_pose_encoding")
    target_encoding = payload.get("target_pose_encoding")
    if not torch.is_tensor(baseline_encoding):
        raise ValueError("Baseline cache lacks tensor field 'baseline_pose_encoding'.")
    if not torch.is_tensor(target_encoding):
        raise ValueError("Baseline cache lacks tensor field 'target_pose_encoding'.")
    image_size = tuple(int(value) for value in payload["image_size"])
    baseline, _ = pose_decoder(
        baseline_encoding.unsqueeze(0).to(device),
        image_size_hw=image_size,
    )
    target, _ = pose_decoder(
        target_encoding.unsqueeze(0).to(device),
        image_size_hw=image_size,
    )
    return baseline, target


def pose_metrics(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    reference_index: int,
    evaluation_indices: list[int] | None,
) -> dict[str, float]:
    indices = evaluation_indices_or_default(
        predicted.shape[1],
        reference_index=reference_index,
        evaluation_indices=evaluation_indices,
    )
    index = torch.tensor(indices, dtype=torch.long, device=predicted.device)
    predicted_eval = predicted.index_select(1, index)
    target_eval = target.index_select(1, index)
    relative = (
        predicted_eval[..., :3, :3]
        @ target_eval[..., :3, :3].transpose(-1, -2)
    )
    cosine = (
        torch.diagonal(relative, dim1=-2, dim2=-1).sum(dim=-1) - 1.0
    ) * 0.5
    rotation = torch.rad2deg(torch.acos(cosine.clamp(-1, 1))).mean()
    center = torch.linalg.vector_norm(
        camera_centers(predicted_eval) - camera_centers(target_eval),
        dim=-1,
    ).mean()
    return {
        "rotation_degrees": float(rotation.cpu()),
        "center_error_native": float(center.cpu()),
    }


def evaluation_indices_or_default(
    sequence_length: int,
    *,
    reference_index: int,
    evaluation_indices: list[int] | None,
) -> list[int]:
    indices = (
        [index for index in range(sequence_length) if index != int(reference_index)]
        if evaluation_indices is None
        else [int(value) for value in evaluation_indices]
    )
    if not indices or int(reference_index) in indices:
        raise ValueError("Evaluation indices are empty or include the reference.")
    if len(indices) != len(set(indices)):
        raise ValueError("Evaluation indices contain duplicates.")
    if any(index < 0 or index >= sequence_length for index in indices):
        raise ValueError("Evaluation index is outside the sequence.")
    return indices


def camera_centers(world_to_camera: torch.Tensor) -> torch.Tensor:
    rotation = world_to_camera[..., :3, :3]
    translation = world_to_camera[..., :3, 3]
    return -(rotation.transpose(-1, -2) @ translation[..., None]).squeeze(-1)


def tracking_audit(
    payload: dict,
    *,
    reference_index: int,
) -> dict[str, object]:
    """Validate causal persistent-track registry invariants from a V0 cache."""

    frames = tuple(int(value) for value in payload.get("frame_indices", ()))
    births = tuple(int(value) for value in payload.get("sam_birth_indices", ()))
    geometry_births = tuple(
        int(value) for value in payload.get("instance_birth_indices", ())
    )
    instance_ids = tuple(int(value) for value in payload.get("instance_ids", ()))
    sam_track_ids = tuple(
        int(value) for value in payload.get("sam_track_ids", ())
    )
    sam_track_prompts = tuple(
        str(value) for value in payload.get("sam_track_prompts", ())
    )
    registry_shapes_match = (
        len(instance_ids) == len(births)
        and len(geometry_births) == len(births)
        and len(sam_track_ids) == len(births)
        and len(sam_track_prompts) == len(births)
    )
    rows = sorted(
        (dict(row) for row in payload.get("dynamic_instance_diagnostics", ())),
        key=lambda row: int(row["sequence_index"]),
    )
    rows_aligned = len(rows) == len(frames) and all(
        int(row["sequence_index"]) == index
        and int(row["frame_index"]) == frames[index]
        for index, row in enumerate(rows)
    )
    expected_discovered = tuple(
        sum(birth >= 0 and birth <= index for birth in births)
        for index in range(len(frames))
    )
    reported_discovered = tuple(
        int(row.get("discovered_tracks", -1)) for row in rows
    )
    discovery_exact = rows_aligned and reported_discovered == expected_discovered
    discovery_monotonic = all(
        left <= right
        for left, right in zip(
            reported_discovered,
            reported_discovered[1:],
        )
    )
    mature_is_causal = rows_aligned and all(
        int(row.get("mature_tracks", -1))
        <= sum(birth >= 0 and birth < index for birth in geometry_births)
        for index, row in enumerate(rows)
    )
    valid_track_keys = tuple(
        (sam_track_prompts[index], track_id)
        for index, track_id in enumerate(sam_track_ids)
        if track_id >= 0
    )
    track_keys_unique = len(valid_track_keys) == len(set(valid_track_keys))
    discovered_track_count = sum(birth >= 0 for birth in births)
    track_count_matches_births = len(valid_track_keys) == discovered_track_count
    late_births = tuple(
        birth for birth in births if birth > int(reference_index)
    )
    future_birth_supported = bool(late_births)
    passed = all(
        (
            rows_aligned,
            registry_shapes_match,
            discovery_exact,
            discovery_monotonic,
            mature_is_causal,
            track_keys_unique,
            track_count_matches_births,
            future_birth_supported,
        )
    )
    return {
        "rows_aligned": int(rows_aligned),
        "registry_shapes_match": int(registry_shapes_match),
        "discovery_exact_from_birth_registry": int(discovery_exact),
        "discovery_monotonic": int(discovery_monotonic),
        "mature_tracks_require_prior_geometry_birth": int(mature_is_causal),
        "persistent_prompt_track_keys_unique": int(track_keys_unique),
        "track_count_matches_birth_registry": int(track_count_matches_births),
        "future_birth_supported": int(future_birth_supported),
        "late_birth_count": len(late_births),
        "late_birth_sequence_indices": late_births,
        "discovered_track_count": discovered_track_count,
        "permanent_slot_capacity": len(births),
        "tracking_audit_pass": int(passed),
    }


def _int_tuple(value) -> tuple[int, ...]:
    return tuple(int(item) for item in (value or ()))


def _path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def _validate_config(config: BaselineRunConfig) -> None:
    if config.version != "v0":
        raise ValueError("The retained baseline config must declare baseline.version=v0.")
    if not config.clip_name:
        raise ValueError("baseline.clip_name is required.")
    if len(config.evaluation_frames) != 12:
        raise ValueError("V0 requires twelve future evaluation frames.")
    if len(set(config.evaluation_frames)) != len(config.evaluation_frames):
        raise ValueError("V0 evaluation frames must be unique.")
    if tuple(sorted(config.evaluation_frames)) != config.evaluation_frames:
        raise ValueError("V0 evaluation frames must be strictly time ordered.")
    if config.selected_pose_branch not in {"raw_streamvggt", "retrieve_qk"}:
        raise ValueError(
            "baseline.pose.selected_branch must be raw_streamvggt or retrieve_qk."
        )
