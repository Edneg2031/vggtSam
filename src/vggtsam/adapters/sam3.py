"""SAM3 video predictor helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Optional

from vggtsam.utils.imports import maybe_add_repo_to_path


def load_sam3_video_predictor(
    *,
    repo_path: Optional[str | Path],
    checkpoint_path: str | Path,
    gpus_to_use: Optional[List[int]] = None,
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
                f"SAM3 repo at {repo} does not look initialized; missing package `sam3`.\n"
                "Run `git submodule update --init --recursive`, or pass the correct repo path."
            )
    try:
        from sam3.model_builder import build_sam3_video_predictor
    except ModuleNotFoundError as exc:
        if exc.name == "sam3":
            raise RuntimeError(
                "Could not import `sam3`. Run `git submodule update --init --recursive` "
                "or pass `--sam3-repo` to a SAM3 repo."
            ) from exc
        raise

    kwargs = {"checkpoint_path": str(checkpoint_path)}
    if gpus_to_use is not None:
        kwargs["gpus_to_use"] = gpus_to_use
    return build_sam3_video_predictor(**kwargs)


def prepare_video_frame_dir(
    frame_paths: Iterable[str | Path],
    output_dir: str | Path,
    *,
    extension: str = ".jpg",
) -> Path:
    """Create a SAM3-compatible numbered frame directory using symlinks."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, frame_path in enumerate(frame_paths):
        src = Path(frame_path).resolve()
        suffix = src.suffix or extension
        dst = output_dir / f"{idx:05d}{suffix}"
        if dst.exists() or dst.is_symlink():
            continue
        os.symlink(src, dst)
    return output_dir


def run_sam3_text_prompt(
    predictor,
    *,
    frame_dir: str | Path,
    prompt: str,
    frame_idx: int = 0,
):
    session = predictor.start_session(resource_path=str(frame_dir))
    session_id = session["session_id"] if isinstance(session, dict) else session
    predictor.add_prompt(session_id=session_id, frame_idx=frame_idx, text=prompt)
    return list(predictor.propagate_in_video(session_id=session_id))
