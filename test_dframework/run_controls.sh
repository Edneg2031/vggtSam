#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src:. python -m test_dframework.run_stage_b \
  --config test_dframework/config.yaml \
  --geometry-modes zero aligned shuffled \
  "$@"

