#!/usr/bin/env python3
"""Compatibility entry point; this baseline contains no trainable bridge."""

import warnings

from streaming_couping.src.pipeline import main


if __name__ == "__main__":
    warnings.warn(
        "train_bridge.py was a nonfunctional generated skeleton. Running the "
        "frozen explicit-bridge evaluation instead; prefer run_bridge.py.",
        stacklevel=1,
    )
    main()

