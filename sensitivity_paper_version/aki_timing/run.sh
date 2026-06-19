#!/usr/bin/env bash
# AKI timing sensitivity — KDIGO onset, using aki-analysis data
set -euo pipefail
cd "$(dirname "$0")/../.."

AKI_ANALYSIS="$HOME/coding/aki-analysis"
OUTDIR="sensitivity_paper_version/aki_timing"
SURVIVAL_KDIGO="${OUTDIR}/incident_ckd_survival_kdigo.csv"
PATIENT_CSV="${AKI_ANALYSIS}/ckd_survival/incident_ckd_patient.csv"

log() { echo; echo "==> $*"; }

# Step 0: KDIGO onset timing
if [ ! -f "${SURVIVAL_KDIGO}" ]; then
  log "Computing KDIGO onset times"
  python3 -m aki_timing_sensitivity.kdigo_onset_timing \
    --data-pkl     "${AKI_ANALYSIS}/data.pkl" \
    --target       "${AKI_ANALYSIS}/target.parquet" \
    --survival-csv "${AKI_ANALYSIS}/ckd_survival/incident_ckd_survival.csv" \
    --outdir       "${OUTDIR}"
else
  log "KDIGO survival CSV exists — skipping"
fi

# Step 1: build CKD features (reuse aki-analysis CKD features)
log "Building demographics + comorbidity features"
python3 -m ckd_survival.src.features.demographics_comorbidity_ckd \
  --aki-feat "${AKI_ANALYSIS}/features/features_all_imputed.parquet" \
  --outdir   "${OUTDIR}"

log "Building pre-CKD lab features"
python3 -m ckd_survival.src.features.labs_preckd \
  --data-pkl "${AKI_ANALYSIS}/data.pkl" \
  --target   "${PATIENT_CSV}" \
  --outdir   "${OUTDIR}"

log "Building medication/procedure history features"
python3 -m ckd_survival.src.features.meds_procedures_history \
  --data-pkl "${AKI_ANALYSIS}/data.pkl" \
  --target   "${PATIENT_CSV}" \
  --outdir   "${OUTDIR}"

log "Merging all CKD features"
python3 -m ckd_survival.src.features.merge_ckd_features \
  --targets    "${SURVIVAL_KDIGO}" \
  --dem-cmb    "${OUTDIR}/features_dem_cmb_ckd.parquet" \
  --labs       "${OUTDIR}/features_labs_preckd.parquet" \
  --medsprocs  "${OUTDIR}/features_medsprocs_preckd.parquet" \
  --outdir     "${OUTDIR}"

# Step 2: train survival models
log "Training XGBoost CoxPH — time to CKD (KDIGO-anchored)"
python3 -m ckd_survival.src.train_xgboost_time_to_ckd \
  --input      "${OUTDIR}/features_ckd.parquet" \
  --out-prefix "${OUTDIR}/artifacts/xgb_ckd_kdigo"

log "Training XGBoost CoxPH — time to death (KDIGO-anchored)"
python3 -m ckd_survival.src.train_xgboost_time_to_death \
  --input      "${OUTDIR}/features_ckd.parquet" \
  --out-prefix "${OUTDIR}/artifacts/xgb_death_kdigo"
