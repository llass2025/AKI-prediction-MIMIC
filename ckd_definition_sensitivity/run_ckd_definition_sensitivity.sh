#!/usr/bin/env bash
# ============================================================
# CKD Definition Sensitivity Pipeline
# Replaces ICD-code-based CKD with lab-based eGFR definition
# (2021 CKD-EPI, sustained eGFR < 60 on two measurements ≥90d apart).
#
# Run from project root:
#   bash ckd_definition_sensitivity/run_ckd_definition_sensitivity.sh
#
# Prerequisites:
#   1. data.pkl must exist
#   2. ckd_survival/incident_ckd_survival.csv must exist
#   3. aki_timing_sensitivity/ feature parquets must exist
#      (dem_cmb, labs, medsprocs) — reused from AKI timing sensitivity
#
# All outputs written to ckd_definition_sensitivity/
# ============================================================

set -euo pipefail
cd "$(dirname "$0")/.."

OUTDIR="ckd_definition_sensitivity"
SURVIVAL_LABCKD="${OUTDIR}/incident_ckd_survival_labckd.csv"
DATA_PKL="data.pkl"

log() { echo; echo "==> $*"; }

# ── Step 0: Compute lab-CKD events ────────────────────────────────────────────
if [ ! -f "${SURVIVAL_LABCKD}" ]; then
  log "Computing lab-based CKD events (eGFR < 60, 2 measurements ≥90d apart)"
  python3 -m ckd_definition_sensitivity.lab_ckd_definition \
    --data-pkl "${DATA_PKL}" \
    --survival-csv ckd_survival/incident_ckd_survival.csv \
    --target target.parquet \
    --outdir "${OUTDIR}"
else
  log "Lab-CKD survival CSV already exists — skipping (${SURVIVAL_LABCKD})"
fi

# ── Step 1: Merge features with lab-CKD survival outcomes ─────────────────────
log "Merging features -> ${OUTDIR}/features_ckd.parquet"
python3 -m ckd_survival.src.features.merge_ckd_features \
  --targets    "${SURVIVAL_LABCKD}" \
  --dem-cmb    "aki_timing_sensitivity/features_dem_cmb_ckd.parquet" \
  --labs       "aki_timing_sensitivity/features_labs_preckd.parquet" \
  --medsprocs  "aki_timing_sensitivity/features_medsprocs_preckd.parquet" \
  --outdir     "${OUTDIR}"

# ── Step 2: Train survival model with lab-CKD outcome ─────────────────────────
mkdir -p "${OUTDIR}/artifacts"

log "Training XGBoost CoxPH — time to lab-CKD"
python3 -m ckd_survival.src.train_xgboost_time_to_ckd \
  --input      "${OUTDIR}/features_ckd.parquet" \
  --event-col  "is_ckd" \
  --out-prefix "${OUTDIR}/artifacts/xgb_ckd_labckd"

log "CKD definition sensitivity pipeline complete. Outputs in: ${OUTDIR}/"
