#!/usr/bin/env bash
#
# Selectively rsync the large Laplace / calibration .pt files to Ada,
# preserving the outputs_laplace/<task>/.../step_4000/ directory tree but
# WITHOUT shipping the rest of the codebase.
#
# Mirrors the rsyncada() helper in ~/.bashrc (host + home dir below).
# Run bare:  bash sync_checkpoints_to_ada.sh

set -euo pipefail

# ══════════════════════════════════════════════════════════════════════════════
# HYPERPARAMETERS — edit before running
# ══════════════════════════════════════════════════════════════════════════════

# Local source root (trailing slash = sync CONTENTS into DEST_DIR).
SRC_DIR="outputs_laplace/"

# Remote target (Ada). Tree under SRC_DIR is recreated here.
REMOTE="tw65vyka@130.83.183.133"
DEST_DIR="/home/tw65vyka/laplace-lora/outputs_laplace/"

# Only these filenames get transferred (the ~860 GB of big tensors).
# Add/remove names to change the selection.
BIG_FILES=(
    "calib_params_per_layer_logit.pt"   # ~24.6 GB each
    "calib_params_per_block_st.pt"      # ~24.6 GB each
    "laplace_H_kron_all.pt"             # ~8.1  GB each
)

# Safety toggles.
DRY_RUN=false          # true = show what WOULD transfer, move nothing. Flip to false to run for real.
DELETE_SOURCE=true   # true = delete each local file AFTER rsync verifies it landed (--remove-source-files).

# ══════════════════════════════════════════════════════════════════════════════

# Build rsync filter rules: descend all dirs, include only BIG_FILES, drop the rest.
filters=( --prune-empty-dirs --include='*/' )
for f in "${BIG_FILES[@]}"; do
    filters+=( --include="$f" )
done
filters+=( --exclude='*' )

# Base flags. No -z: .pt are dense binary, gzip wastes CPU at this scale.
# --mkpath: create missing parent dirs of DEST on the remote (rsync >=3.2.3).
opts=( -a --info=progress2 --partial --mkpath )

if [[ "$DRY_RUN" == "true" ]]; then
    opts+=( --dry-run )
    echo ">>> DRY RUN — no data will move. Set DRY_RUN=false to transfer."
fi

if [[ "$DELETE_SOURCE" == "true" ]]; then
    opts+=( --remove-source-files )
    echo ">>> DELETE_SOURCE=true — local files will be removed after verified transfer."
fi

echo ">>> ${SRC_DIR}  ->  ${REMOTE}:${DEST_DIR}"
echo ">>> selecting: ${BIG_FILES[*]}"
echo

rsync "${opts[@]}" "${filters[@]}" "$SRC_DIR" "${REMOTE}:${DEST_DIR}"

echo
echo ">>> done."
