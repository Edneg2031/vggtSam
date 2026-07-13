#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-test_sam/config.yaml}"
ITERATIONS="${ITERATIONS:-700}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/test_sam_layer17_controls}"
FUSION_LR="${FUSION_LR:-0.001}"
RESIDUAL_INIT_STD="${RESIDUAL_INIT_STD:-0.01}"
GEOMETRY_LAYER_INDEX="${GEOMETRY_LAYER_INDEX:-17}"

export PYTHONPATH="src:.:${PYTHONPATH:-}"
mkdir -p "${OUTPUT_ROOT}"

run_control() {
  local name="$1"
  shift
  local output_dir="${OUTPUT_ROOT}/${name}"
  mkdir -p "${output_dir}"
  echo
  echo "================================================================"
  echo "Layer-${GEOMETRY_LAYER_INDEX} control=${name}, iterations=${ITERATIONS}"
  echo "Output: ${output_dir}"
  echo "================================================================"
  python -m test_sam.train_fusion_ablation \
    --config "${CONFIG}" \
    --fusion-method cross_attention \
    --geometry-layer-index "${GEOMETRY_LAYER_INDEX}" \
    --residual-init-std "${RESIDUAL_INIT_STD}" \
    --lr "${FUSION_LR}" \
    --no-train-tracker \
    --iterations "${ITERATIONS}" \
    --output-dir "${output_dir}" \
    "$@" 2>&1 | tee "${output_dir}/run.log"
}

# Same fusion graph and parameter count, but no geometry content.
run_control "01_zero_geometry" --zero-geometry

# Correct frame-aligned StreamVGGT layer-17 features.
run_control "02_aligned_geometry" --no-compare-direct

# Same feature distribution with deliberately incorrect frame correspondence.
run_control "03_shuffled_geometry" --shuffle-geometry --no-compare-direct

python -m test_sam.summarize_ablations \
  "${OUTPUT_ROOT}" \
  --output "${OUTPUT_ROOT}/ablation_summary.csv"

echo
echo "Layer-${GEOMETRY_LAYER_INDEX} controls completed."
echo "Summary: ${OUTPUT_ROOT}/ablation_summary.csv"
