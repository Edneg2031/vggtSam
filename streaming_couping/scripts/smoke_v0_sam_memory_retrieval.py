#!/usr/bin/env python3
"""CPU smoke for native-QK retrieval and SAM persistent-instance controls."""

from __future__ import annotations

import torch

from streaming_couping.src.backbones.streamvggt_parallel import (
    _first_global_layer_instance_frame_scores,
    assert_frame_repository_cache_equivalence,
    assert_processed_key_cache_equivalence,
)
from streaming_couping.src.config import load_config
from streaming_couping.src.external_repos import maybe_add_repo_to_path
from streaming_couping.src.sam_memory_retrieval import (
    RetrievalPolicy,
    persistent_visibility,
    same_instance_history_frames,
    select_retrieval_history,
)
from streaming_couping.scripts.run_v0_sam_memory_retrieval import (
    _candidate_generation_payload,
)


def main() -> None:
    recovery = load_config(
        "streaming_couping/configs/recovery_dynamic_instance.yaml"
    )
    maybe_add_repo_to_path(recovery.streamvggt_repo)
    assert_processed_key_cache_equivalence()
    assert_frame_repository_cache_equivalence()
    _test_persistent_candidates()
    _test_fixed_budget_selection()
    _test_masked_instance_qk_identity_control()
    deployable = _candidate_generation_payload(
        {
            "stream_images": torch.zeros(1, 3, 4, 4),
            "tracking_masks_output": torch.zeros(1, 1, 4, 4),
            "tracking_masks_stream": torch.zeros(1, 1, 4, 4),
            "tracking_scores": torch.ones(1, 1),
            "sam_track_ids": [7],
            "sam_track_prompts": ["chair"],
            "frame_indices": [0],
            "patch_shape": [1, 1],
            "target_pose_encoding": torch.ones(1, 9),
        }
    )
    assert not any(name.startswith("target_") for name in deployable)
    print("V0 SAM-memory masked-QK/full-frame-KV retrieval smoke passed")


def _test_persistent_candidates() -> None:
    masks = torch.zeros(4, 3, 4, 4, dtype=torch.bool)
    masks[0, 0, 0, 0] = True
    masks[1, 1, 0, 0] = True
    masks[2, 0, 0, 0] = True
    masks[3, 0, 0, 0] = True
    scores = torch.ones(4, 3)
    visibility = persistent_visibility(
        masks,
        scores,
        track_ids=(10, 11, -1),
        minimum_score=0.5,
    )
    assert same_instance_history_frames(visibility, 3) == (0, 2)
    assert same_instance_history_frames(
        visibility,
        3,
        shuffled_identity=True,
    ) == (1,)
    scores[2, 0] = 0.1
    filtered = persistent_visibility(
        masks,
        scores,
        track_ids=(10, 11, -1),
        minimum_score=0.5,
    )
    assert same_instance_history_frames(filtered, 3) == (0,)


def _test_fixed_budget_selection() -> None:
    policy = RetrievalPolicy(
        total_frame_budget=5,
        anchor_frames=1,
        sam_frame_quota=2,
    )
    qk = torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    sam_region = torch.tensor([0.0, 0.9, 0.8, 0.1, 0.2, 0.3])
    shuffled_region = torch.tensor([0.0, 0.7, 0.1, 0.8, 0.2, 0.3])
    raw = select_retrieval_history(
        method="raw_full_history",
        frame_index=6,
        qk_scores=qk,
        sam_region_scores=sam_region,
        shuffled_region_scores=shuffled_region,
        sam_candidates=(1, 2),
        shuffled_candidates=(3, 4),
        policy=policy,
    )
    assert raw == (0, 1, 2, 3, 4, 5)
    retrieve = select_retrieval_history(
        method="retrieve_qk",
        frame_index=6,
        qk_scores=qk,
        sam_region_scores=sam_region,
        shuffled_region_scores=shuffled_region,
        sam_candidates=(1, 2),
        shuffled_candidates=(3, 4),
        policy=policy,
    )
    assert retrieve == (0, 2, 3, 4, 5)
    hybrid = select_retrieval_history(
        method="sam_hybrid_qk",
        frame_index=6,
        qk_scores=qk,
        sam_region_scores=sam_region,
        shuffled_region_scores=shuffled_region,
        sam_candidates=(1, 2),
        shuffled_candidates=(3, 4),
        policy=policy,
    )
    shuffled = select_retrieval_history(
        method="shuffled_instance_memory",
        frame_index=6,
        qk_scores=qk,
        sam_region_scores=sam_region,
        shuffled_region_scores=shuffled_region,
        sam_candidates=(1, 2),
        shuffled_candidates=(1, 3),
        policy=policy,
    )
    assert hybrid == (0, 1, 2, 4, 5)
    assert shuffled == (0, 1, 3, 4, 5)
    assert len(hybrid) == len(shuffled) == policy.total_frame_budget
    assert len(set(hybrid)) == len(hybrid)
    assert len(set(shuffled)) == len(shuffled)
    assert max(hybrid) < 6 and max(shuffled) < 6


def _test_masked_instance_qk_identity_control() -> None:
    class Identity(torch.nn.Module):
        def forward(self, value):
            return value

    class Attention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.num_heads = 1
            self.head_dim = 2
            self.qkv = torch.nn.Linear(2, 6, bias=False)
            self.q_norm = Identity()
            self.rope = None
            with torch.no_grad():
                self.qkv.weight.zero_()
                self.qkv.weight[0:2].copy_(torch.eye(2))

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.norm1 = Identity()
            self.attn = Attention()

    block = Block()
    current = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    value = torch.zeros(1, 1, 1, 2, 2)
    same_key = torch.tensor([[[[[1.0, 0.0], [0.0, 1.0]]]]])
    swapped_key = torch.tensor([[[[[0.0, 1.0], [1.0, 0.0]]]]])
    masks = torch.tensor(
        [
            [[[1.0, 0.0]], [[0.0, 1.0]]],
            [[[1.0, 0.0]], [[0.0, 1.0]]],
            [[[1.0, 0.0]], [[0.0, 1.0]]],
        ]
    )
    same, shuffled = _first_global_layer_instance_frame_scores(
        block,
        current,
        position=None,
        frame_key_values=((same_key, value), (swapped_key, value)),
        patch_start_idx=0,
        region_patch_masks=masks,
        frame_index=2,
    )
    torch.testing.assert_close(same, torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(shuffled, torch.tensor([0.0, 1.0]))


if __name__ == "__main__":
    main()
