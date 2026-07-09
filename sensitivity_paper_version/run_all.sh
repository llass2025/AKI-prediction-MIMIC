#!/usr/bin/env bash
# ============================================================
# sensitivity_paper_version/run_all.sh
#
# Re-runs all four sensitivity analyses using the aki-analysis
# data and models (previous round), saving outputs here rather
# than overwriting the primary sensitivity folders.
#
# Run from project root:
#   bash sensitivity_paper_version/run_all.sh
# ============================================================
set -euo pipefail
cd "$(dirname "$0")/.."

AKI_ANALYSIS="$HOME/coding/aki-analysis"
OUTBASE="sensitivity_paper_version"

log() { echo; echo "==> $*"; }

log "1/4  Age × sex RCS"
bash sensitivity_paper_version/age_sex_rcs/run.sh

log "2/4  CKD definition (lab-based eGFR)"
bash sensitivity_paper_version/ckd_definition/run.sh

log "3/4  AKI timing (KDIGO onset)"
bash sensitivity_paper_version/aki_timing/run.sh

log "4/4  Autocar exclusion (retrain all 6 models)"
bash sensitivity_paper_version/autocar/run.sh

log "All sensitivity_paper_version analyses complete."
