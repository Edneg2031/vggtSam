#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-test_sam/config.yaml}"
ITERATIONS="${ITERATIONS:-700}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/test_sam_ablation}"
FULL_ABLATIONS="${FULL_ABLATIONS:-0}"

export PYTHONPATH="src:.:${PYTHONPATH:-}"
mkdir -p "${OUTPUT_ROOT}"

run_ablation() {
  local name="$1"
  local method="$2"
  shift 2

  local output_dir="${OUTPUT_ROOT}/${name}"
  mkdir -p "${output_dir}"
  echo
  echo "================================================================"
  echo "Running ${name}: method=${method}, iterations=${ITERATIONS}"
  echo "Output: ${output_dir}"
  echo "================================================================"

  python -m test_sam.train_fusion_ablation \
    --config "${CONFIG}" \
    --fusion-method "${method}" \
    --iterations "${ITERATIONS}" \
    --output-dir "${output_dir}" \
    "$@" 2>&1 | tee "${output_dir}/run.log"
}

echo "Checking fusion output shapes..."
python test_sam/test_fusion_shapes.py

# Original SAM3 is evaluated in this first run. It is deterministic for the
# fixed sequence, so later runs skip loading a duplicate comparison model.
run_ablation "01_sam_only" "sam_only"

# Same graph and trainable capacity as 03, but geometry values are all zero.
run_ablation \
  "02_cross_attention_zero_geometry" \
  "cross_attention" \
  --zero-geometry \
  --no-compare-direct

run_ablation \
  "03_cross_attention" \
  "cross_attention" \
  --no-compare-direct

run_ablation \
  "04_cross_attention_shuffled_geometry" \
  "cross_attention" \
  --shuffle-geometry \
  --no-compare-direct

if [[ "${FULL_ABLATIONS}" == "1" ]]; then
  run_ablation \
    "05_multilevel_cross_attention" \
    "multilevel_cross_attention" \
    --no-compare-direct
  run_ablation "06_add" "add" --no-compare-direct
  run_ablation "07_concat_conv" "concat_conv" --no-compare-direct
  run_ablation \
    "08_gated_cross_attention" \
    "gated_cross_attention" \
    --no-compare-direct
  run_ablation "09_film" "film" --no-compare-direct
fi

python -m test_sam.summarize_ablations \
  "${OUTPUT_ROOT}" \
  --output "${OUTPUT_ROOT}/ablation_summary.csv"

echo
echo "All ablations completed."
echo "Summary: ${OUTPUT_ROOT}/ablation_summary.csv"
