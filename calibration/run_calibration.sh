#!/usr/bin/env bash
# ============================================================
# Calibration Assessment Pipeline
# Computes Brier scores, ECE, HL test, and calibration plots
# for all AKI prediction models.
#
# Run from project root:
#   bash calibration/run_calibration.sh
#
# Prerequisites:
#   Prediction CSVs must exist in artifacts/*/
#   (run run.sh first if they don't)
# ============================================================

set -euo pipefail
cd "$(dirname "$0")/.."

log() { echo; echo "==> $*"; }

log "Running calibration assessment for all models"
python3 -m calibration.calibration_utils \
  --artifacts-dir  artifacts \
  --features-parquet features/features_all.parquet \
  --outdir         calibration/artifacts \
  --n-bins         10

log "Calibration complete. Outputs in: calibration/artifacts/"
