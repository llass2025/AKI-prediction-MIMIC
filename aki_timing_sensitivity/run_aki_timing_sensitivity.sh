#!/usr/bin/env bash
# ============================================================
# AKI Timing Sensitivity Pipeline
# Runs the full CKD/death survival analysis anchored to
# KDIGO creatinine onset time instead of admission time.
#
# Run from project root:
#   bash aki_timing_sensitivity/run_aki_timing_sensitivity.sh
#
# Prerequisites:
#   1. data.pkl must exist
#   2. ckd_survival/incident_ckd_survival.csv must exist
#   3. ckd_survival/incident_ckd_patient.csv must exist
#
# All outputs written to aki_timing_sensitivity/ — does NOT
# touch ckd_survival/ primary results.
# ============================================================

set -euo pipefail
cd "$(dirname "$0")/.."

OUTDIR="aki_timing_sensitivity"
SURVIVAL_KDIGO="${OUTDIR}/incident_ckd_survival_kdigo.csv"
PATIENT_CSV="ckd_survival/incident_ckd_patient.csv"
DATA_PKL="data.pkl"
AKI_FEAT="features/features_all_imputed.parquet"

log() { echo; echo "==> $*"; }

# ── Step 0: KDIGO onset timing ────────────────────────────────────────────────
if [ ! -f "${SURVIVAL_KDIGO}" ]; then
  log "Computing KDIGO onset times -> ${SURVIVAL_KDIGO}"
  python3 -m aki_timing_sensitivity.kdigo_onset_timing \
    --data-pkl "${DATA_PKL}" \
    --target target.parquet \
    --survival-csv ckd_survival/incident_ckd_survival.csv \
    --outdir "${OUTDIR}"
else
  log "KDIGO onset times already exist — skipping (${SURVIVAL_KDIGO})"
fi

# ── Step 1: Build CKD features (KDIGO-anchored) ───────────────────────────────
log "Building demographics + comorbidity features"
python3 -m ckd_survival.src.features.demographics_comorbidity_ckd \
  --aki-feat "${AKI_FEAT}" \
  --outdir "${OUTDIR}"

log "Building pre-CKD lab features"
python3 -m ckd_survival.src.features.labs_preckd \
  --data-pkl "${DATA_PKL}" \
  --target "${PATIENT_CSV}" \
  --outdir "${OUTDIR}"

log "Building medication/procedure history features"
python3 -m ckd_survival.src.features.meds_procedures_history \
  --data-pkl "${DATA_PKL}" \
  --target "${PATIENT_CSV}" \
  --outdir "${OUTDIR}"

log "Merging all CKD features -> ${OUTDIR}/features_ckd.parquet"
python3 -m ckd_survival.src.features.merge_ckd_features \
  --targets "${SURVIVAL_KDIGO}" \
  --dem-cmb "${OUTDIR}/features_dem_cmb_ckd.parquet" \
  --labs    "${OUTDIR}/features_labs_preckd.parquet" \
  --medsprocs "${OUTDIR}/features_medsprocs_preckd.parquet" \
  --outdir  "${OUTDIR}"

# ── Step 2: Train survival models ─────────────────────────────────────────────
mkdir -p "${OUTDIR}/artifacts"

log "Training XGBoost CoxPH — time to CKD (KDIGO-anchored)"
python3 -m ckd_survival.src.train_xgboost_time_to_ckd \
  --input "${OUTDIR}/features_ckd.parquet" \
  --out-prefix "${OUTDIR}/artifacts/xgb_ckd_kdigo"

log "Training XGBoost CoxPH — time to death (KDIGO-anchored)"
python3 -m ckd_survival.src.train_xgboost_time_to_death \
  --input "${OUTDIR}/features_ckd.parquet" \
  --out-prefix "${OUTDIR}/artifacts/xgb_death_kdigo"

log "AKI timing sensitivity pipeline complete. Outputs in: ${OUTDIR}/"
