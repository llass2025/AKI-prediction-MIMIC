#!/usr/bin/env bash
# ============================================================
# Age × Sex RCS Logistic Regression Pipeline
#
# Run from project root:
#   bash age_sex_rcs/run_age_sex_rcs.sh
# ============================================================

set -euo pipefail
cd "$(dirname "$0")/.."

log() { echo; echo "==> $*"; }

log "Fitting Elastic Net logistic regression with RCS age × sex interactions"
python3 -m age_sex_rcs.rcs_logreg \
  --features    features/features_all.parquet \
  --outdir      age_sex_rcs/artifacts \
  --seed        42 \
  --n-bootstrap 300 \
  --knot-pcts   5 35 65 95

log "Age × sex RCS complete. Outputs in: age_sex_rcs/artifacts/"
