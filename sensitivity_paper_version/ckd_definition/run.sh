#!/usr/bin/env bash
# CKD definition sensitivity — lab-based eGFR, using aki-analysis data
set -euo pipefail
cd "$(dirname "$0")/../.."

AKI_ANALYSIS="$HOME/coding/aki-analysis"
OUTDIR="sensitivity_paper_version/ckd_definition"
SURVIVAL_LABCKD="${OUTDIR}/incident_ckd_survival_labckd.csv"

log() { echo; echo "==> $*"; }

# Step 0: compute lab-CKD events
if [ ! -f "${SURVIVAL_LABCKD}" ]; then
  log "Computing lab-based CKD events"
  python3 -m ckd_definition_sensitivity.lab_ckd_definition \
    --data-pkl      "${AKI_ANALYSIS}/data.pkl" \
    --survival-csv  "${AKI_ANALYSIS}/ckd_survival/incident_ckd_survival.csv" \
    --target        "${AKI_ANALYSIS}/target.parquet" \
    --outdir        "${OUTDIR}"
else
  log "Lab-CKD survival CSV exists — skipping"
fi

# Step 1: merge features (reuse aki-analysis CKD features)
log "Merging features"
python3 -m ckd_survival.src.features.merge_ckd_features \
  --targets    "${SURVIVAL_LABCKD}" \
  --dem-cmb    "${AKI_ANALYSIS}/ckd_survival/features_dem_cmb_ckd.parquet" \
  --labs       "${AKI_ANALYSIS}/ckd_survival/features_labs_preckd.parquet" \
  --medsprocs  "${AKI_ANALYSIS}/ckd_survival/features_medsprocs_preckd.parquet" \
  --outdir     "${OUTDIR}"

# Step 2: train survival model
log "Training XGBoost CoxPH — time to lab-CKD"
python3 -m ckd_survival.src.train_xgboost_time_to_ckd \
  --input      "${OUTDIR}/features_ckd.parquet" \
  --event-col  "is_ckd" \
  --out-prefix "${OUTDIR}/artifacts/xgb_ckd_labckd"
