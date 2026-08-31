"""Tests for the frozen V0 temporal-prompt matrix diagnostic."""

from streaming_couping.scripts.smoke_temporal_prompt_matrix import main


def test_temporal_prompt_matrix_smoke() -> None:
    main()
