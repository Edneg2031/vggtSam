from long_sequence_baselines.run_streamvggt_3gpu import partition_layers


def test_24_layers_are_split_eight_per_gpu() -> None:
    assignment = partition_layers(24, 3)
    assert assignment == [0] * 8 + [1] * 8 + [2] * 8


def test_uneven_partition_is_contiguous_and_balanced() -> None:
    assignment = partition_layers(10, 3)
    assert assignment == [0] * 4 + [1] * 3 + [2] * 3
