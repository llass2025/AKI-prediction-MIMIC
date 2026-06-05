#!/usr/bin/env bash
# ============================================================
# Bootstrap CI Pipeline
# Computes 95% CIs for AUROC and AUPRC for all AKI models.
#
# Run from project root:
#   bash uncertainty_intervals/run_bootstrap_ci.sh
#
# Optional: faster run with fewer resamples for testing:
#   bash uncertainty_intervals/run_bootstrap_ci.sh --n-bootstrap 100
# ============================================================

set -euo pipefail
cd "$(dirname "$0")/.."

N_BOOTSTRAP=1000
for arg in "$@"; do
  case $arg in
    --n-bootstrap=*) N_BOOTSTRAP="${arg#--n-bootstrap=}" ;;
  esac
done

log() { echo; echo "==> $*"; }

log "Computing bootstrap CIs (n=${N_BOOTSTRAP} resamples)"
python3 -m uncertainty_intervals.bootstrap_ci \
  --artifacts-dir artifacts \
  --n-bootstrap   "${N_BOOTSTRAP}" \
  --seed          42 \
  --outdir        uncertainty_intervals/artifacts

log "Bootstrap CIs complete. Outputs in: uncertainty_intervals/artifacts/"
