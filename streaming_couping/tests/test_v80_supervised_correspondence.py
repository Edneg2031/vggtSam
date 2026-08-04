import torch

from streaming_couping.src.learned_pose.v80_supervised_correspondence import (
    V80MatchingConfig,
    causal_reference_targets,
    compute_v80_matching_loss,
    sample_gt_world_at_local_tokens,
)


def test_gt_world_sampling_uses_cached_normalized_uv_without_pose_transform():
    dense = torch.zeros(1, 2, 3, 4, 3)
    for frame in range(2):
        for y in range(3):
            for x in range(4):
                dense[0, frame, y, x] = torch.tensor([x, y, frame])
    local = torch.zeros(1, 2, 1, 2, 7)
    local[..., 3:5] = torch.tensor(
        [[[-1.0, -1.0], [1.0, 1.0]]]
    )
    valid = torch.ones(1, 2, 1, 2, dtype=torch.bool)
    sampled, sampled_valid = sample_gt_world_at_local_tokens(
        local_features=local,
        local_valid=valid,
        target_world_points=dense,
    )
    assert torch.equal(sampled[0, 0, 0, 0], torch.tensor([0.0, 0.0, 0.0]))
    assert torch.equal(sampled[0, 1, 0, 1], torch.tensor([3.0, 2.0, 1.0]))
    assert bool(sampled_valid.all())


def test_causal_target_memory_overwrites_with_invalid_gt_when_geometry_writes():
    values = torch.tensor(
        [[[[[1.0, 0.0, 0.0]]], [[[0.0, 0.0, 0.0]]], [[[3.0, 0.0, 0.0]]]]]
    )
    target_valid = torch.tensor([[[[1]], [[0]], [[1]]]], dtype=torch.bool)
    source_valid = torch.ones_like(target_valid)
    write = torch.ones(1, 3, 1, dtype=torch.bool)
    reference, reference_valid = causal_reference_targets(
        values,
        target_valid,
        source_valid=source_valid,
        memory_write=write,
    )
    assert torch.equal(reference[0, 1], values[0, 0])
    assert bool(reference_valid[0, 1].all())
    assert torch.equal(reference[0, 2], values[0, 1])
    assert not bool(reference_valid[0, 2].any())


def test_match_loss_supervises_only_mutual_in_radius_queries_and_backpropagates():
    probability_logits = torch.zeros(1, 2, 1, 3, 3, requires_grad=True)
    probability = probability_logits.softmax(dim=-1)
    valid = torch.ones(1, 2, 1, 3, dtype=torch.bool)
    valid[:, 0] = False  # no history before frame zero
    output = {
        "transport_probability": probability,
        "transport_query_valid": torch.ones_like(valid),
        "transport_key_valid": valid,
    }
    reference = torch.tensor(
        [[0.0, 0.0, 1.0], [0.2, 0.0, 1.0], [0.4, 0.0, 1.0]]
    )
    current = reference.clone()
    current[2] = torch.tensor([5.0, 5.0, 5.0])
    world = torch.stack([reference, current]).reshape(1, 2, 1, 3, 3)
    result = compute_v80_matching_loss(
        output,
        current_gt_world=world,
        current_gt_valid=torch.ones_like(valid),
        source_local_valid=torch.ones_like(valid),
        memory_write=torch.ones(1, 2, 1, dtype=torch.bool),
        config=V80MatchingConfig(max_distance=0.05, temperature=0.01),
        sequence_indices=[1],
    )
    assert int(result.supervised_queries) == 2
    assert int(result.candidate_queries) == 3
    assert float(result.loss) > 0.0
    result.loss.backward()
    assert probability_logits.grad is not None
    assert float(probability_logits.grad.abs().sum()) > 0.0


def test_no_match_returns_differentiable_zero():
    logits = torch.zeros(1, 2, 1, 2, 2, requires_grad=True)
    probability = logits.softmax(dim=-1)
    valid = torch.zeros(1, 2, 1, 2, dtype=torch.bool)
    output = {
        "transport_probability": probability,
        "transport_query_valid": valid,
        "transport_key_valid": valid,
    }
    world = torch.zeros(1, 2, 1, 2, 3)
    result = compute_v80_matching_loss(
        output,
        current_gt_world=world,
        current_gt_valid=valid,
        source_local_valid=valid,
        memory_write=torch.zeros(1, 2, 1, dtype=torch.bool),
        config=V80MatchingConfig(),
        sequence_indices=[1],
    )
    assert float(result.loss) == 0.0
    result.loss.backward()
    assert logits.grad is not None
    assert float(logits.grad.abs().sum()) == 0.0
