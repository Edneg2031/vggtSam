"""Tests for the real-SAM temporal prompt A/B bookkeeping."""

from streaming_couping.scripts.smoke_temporal_prompt_sam_ab import main


def test_temporal_prompt_sam_ab_smoke() -> None:
    main()
