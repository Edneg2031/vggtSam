#!/usr/bin/env python3
"""Prepare 2D semantic and instance labels for ScanNet++ pinhole frames."""

from __future__ import annotations

from vggtsam.data.scannetpp.preprocess import main


if __name__ == "__main__":
    main()
