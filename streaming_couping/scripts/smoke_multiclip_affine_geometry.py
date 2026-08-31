#!/usr/bin/env python3
"""No-data smoke test for the multi-clip affine geometry diagnostic."""

from __future__ import annotations

from streaming_couping.scripts.analyze_multiclip_affine_geometry import (
    _reconstruct_depth_smoke,
)


def main() -> None:
    result = _reconstruct_depth_smoke()
    print(
        "multi-clip affine geometry smoke passed "
        f"applied_pixels={result['applied_pixels']}"
    )


if __name__ == "__main__":
    main()
