"""CPU-only shape smoke test for all feature mergers."""

from __future__ import annotations

import torch

from test_sam.fusion import FUSION_METHODS, SAM3GeometryFusion


def main() -> None:
    sam = torch.randn(1, 256, 72, 72)
    geometry = [torch.randn(1, 2048, 12, 12) for _ in range(4)]
    with torch.no_grad():
        for method in FUSION_METHODS:
            model = SAM3GeometryFusion(method=method).eval()
            levels = None if method == "sam_only" else geometry
            outputs = model(sam, levels)
            shapes = [tuple(output.shape) for output in outputs]
            expected = [(1, 32, 288, 288), (1, 64, 144, 144), (1, 256, 72, 72)]
            assert shapes == expected, (method, shapes)
            print(f"{method}: {shapes}")


if __name__ == "__main__":
    main()
