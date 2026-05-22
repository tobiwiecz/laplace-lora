#!/usr/bin/env bash
# End-to-end pipeline: VP calibration on openbookqa (~10 epochs) + comparison table.
# Usage: bash pipeline_openbookqa.sh

set -e
DIR="$(dirname "$0")"

TASKS=openbookqa \
SEEDS="${SEEDS:-21 42 87 13 100}" \
LOAD_STEPS="${LOAD_STEPS:-4000}" \
TOTAL_STEPS="${TOTAL_STEPS:-20}" \
EVAL_ON_TEST=1 \
    bash "$DIR/calibrate_vp.sh"

echo ""
echo "══════════════════════════════════════════════════════"
echo " Comparison table"
echo "══════════════════════════════════════════════════════"
source "$DIR/.venv/bin/activate"
python "$DIR/compare_openbookqa.py" \
    --hessian "${HESSIAN:-diag}" \
    --load_step "${LOAD_STEPS:-4000}"
