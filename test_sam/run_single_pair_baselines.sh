#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-test_sam/debug_single_pair.yaml}"
ITERATIONS="${ITERATIONS:-1000}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/debug_single_pair_baselines}"
STRICT_IOU_GATE="${STRICT_IOU_GATE:-0}"
IOU_TARGET="${IOU_TARGET:-0.95}"

export PYTHONPATH="src:.:${PYTHONPATH:-}"
mkdir -p "${OUTPUT_ROOT}"

run_mode() {
  local mode="$1"
  local output_dir="${OUTPUT_ROOT}/${mode}"
  echo
  echo "================================================================"
  echo "Single-pair mode=${mode}, iterations=${ITERATIONS}"
  echo "Output: ${output_dir}"
  echo "================================================================"
  python scripts/debug_single_pair_overfit.py \
    --config "${CONFIG}" \
    --mode "${mode}" \
    --iterations "${ITERATIONS}" \
    --output-dir "${output_dir}" 2>&1 | tee "${output_dir}.log"
}

check_train_iou() {
  local mode="$1"
  python - "${OUTPUT_ROOT}/${mode}/training_history.csv" "${mode}" "${IOU_TARGET}" <<'PY'
import csv
import sys

path, mode, target = sys.argv[1:]
target = float(target)
with open(path, newline="", encoding="utf8") as handle:
    rows = list(csv.DictReader(handle))
if not rows:
    raise SystemExit(f"{mode}: no training metrics were written")
final = rows[-1]
train_iou = float(final["train_iou"])
eval_iou = float(final["eval_iou"])
print(f"{mode}: final train_iou={train_iou:.4f}, eval_iou={eval_iou:.4f}")
if train_iou < target:
    raise SystemExit(2)
PY
}

run_mode sam_only
sam_ok=1
check_train_iou sam_only || sam_ok=0

run_mode constant_prompt
prompt_ok=1
check_train_iou constant_prompt || prompt_ok=0

if [[ "${STRICT_IOU_GATE}" == "1" && ("${sam_ok}" != "1" || "${prompt_ok}" != "1") ]]; then
  echo
  echo "STOP: strict diagnostic gate enabled and a baseline did not reach train IoU ${IOU_TARGET}."
  echo "Inspect parameter_audit.csv, tensor_audit.json, and module_diagnostics.csv."
  exit 2
fi

run_mode random_geometry
random_ok=1
check_train_iou random_geometry || random_ok=0

echo
echo "All baseline experiments completed (no default IoU gate)."
echo "Reached train IoU ${IOU_TARGET}: sam_only=${sam_ok}, constant_prompt=${prompt_ok}, random_geometry=${random_ok}."
