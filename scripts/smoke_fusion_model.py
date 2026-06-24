#!/usr/bin/env python3
"""Smoke-test the latent fusion module without external backbones."""

from __future__ import annotations

import argparse

import torch

from vggtsam.models import GeometryTokens, LatentGeometrySemanticFusion, SemanticTokens


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--geometry-tokens", type=int, default=196)
    parser.add_argument("--semantic-tokens", type=int, default=8)
    parser.add_argument("--geometry-dim", type=int, default=1024)
    parser.add_argument("--semantic-dim", type=int, default=256)
    parser.add_argument("--camera-dim", type=int, default=128)
    parser.add_argument("--d-fuse", type=int, default=512)
    parser.add_argument("--num-classes", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    model = LatentGeometrySemanticFusion(
        geometry_dim=args.geometry_dim,
        semantic_dim=args.semantic_dim,
        camera_dim=args.camera_dim,
        d_fuse=args.d_fuse,
        num_classes=args.num_classes,
    ).to(args.device)
    geometry = GeometryTokens(
        tokens=torch.randn(
            args.batch_size, args.geometry_tokens, args.geometry_dim, device=args.device
        ),
        camera_tokens=torch.randn(
            args.batch_size, 1, args.camera_dim, device=args.device
        ),
    )
    semantics = SemanticTokens(
        tokens=torch.randn(
            args.batch_size, args.semantic_tokens, args.semantic_dim, device=args.device
        )
    )

    with torch.no_grad():
        out = model(geometry, semantics, return_attention=True)
        corr = model.correspondence_logits(out.match_embeddings, out.match_embeddings)

    print("fused_tokens", tuple(out.fused_tokens.shape))
    print("pred_pointmap", tuple(out.pred_pointmap.shape))
    print("pred_logits", tuple(out.pred_logits.shape))
    print("match_embeddings", tuple(out.match_embeddings.shape))
    print("correspondence_logits", tuple(corr.shape))
    if out.attention_weights is not None:
        print("attention_weights", tuple(out.attention_weights.shape))


if __name__ == "__main__":
    main()
