#!/usr/bin/env bash
# Autocar sensitivity — retrain all 6 AKI prediction models excluding
# auto accident/trauma admissions, using aki-analysis data.
set -euo pipefail
cd "$(dirname "$0")/../.."

AKI_ANALYSIS="$HOME/coding/aki-analysis"
OUTDIR="sensitivity_paper_version/autocar"
FEAT_FILTERED="${OUTDIR}/features_all_imputed_noautocar.parquet"

log() { echo; echo "==> $*"; }

# Step 0: filter out autocar admissions
if [ ! -f "${FEAT_FILTERED}" ]; then
  log "Filtering autocar admissions from features"
  conda run -n genai python3 sensitivity_paper_version/autocar/prepare_features.py \
    --features "${AKI_ANALYSIS}/features/features_all_imputed.parquet" \
    --target   "${AKI_ANALYSIS}/target.parquet" \
    --out      "${FEAT_FILTERED}"
else
  log "Filtered features exist — skipping"
fi

SEED=20250831

# Step 1: retrain all 6 models
log "Training Logistic Regression (noautocar)"
conda run -n genai python3 -m src.train_logreg \
  --input      "${FEAT_FILTERED}" \
  --out-prefix "${OUTDIR}/artifacts/logreg_noautocar"

log "Training XGBoost (noautocar)"
conda run -n genai python3 -m src.train_xgboost \
  --input      "${FEAT_FILTERED}" \
  --out-prefix "${OUTDIR}/artifacts/xgb_noautocar"

log "Training Random Forest (noautocar)"
conda run -n genai python3 -m src.train_rf \
  --input      "${FEAT_FILTERED}" \
  --out-prefix "${OUTDIR}/artifacts/rf_noautocar"

log "Training DNN (noautocar)"
conda run -n genai python3 -m src.train_dnn \
  --input      "${FEAT_FILTERED}" \
  --out-prefix "${OUTDIR}/artifacts/dnn_noautocar"

log "Training DCN (noautocar)"
conda run -n genai python3 -m src.train_dcn \
  --input      "${FEAT_FILTERED}" \
  --out-prefix "${OUTDIR}/artifacts/dcn_noautocar"

log "Training Self-Attention DNN (noautocar)"
conda run -n genai python3 -m src.train_selfattn \
  --input      "${FEAT_FILTERED}" \
  --out-prefix "${OUTDIR}/artifacts/selfatten_noautocar"

log "Autocar sensitivity complete. Outputs in: ${OUTDIR}/artifacts/"
