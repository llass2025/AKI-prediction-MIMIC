#!/usr/bin/env bash
# Run the AKI prediction pipeline with additional exclusions for gap ICD codes:
#   Z940 (kidney transplant), N13 (obstructive uropathy), N28 (other renal disorders)
# Builds on target_gap_addressed.parquet (derived from target_no_cmb_ckd.parquet).
# Outputs to features_gap_addressed/ and artifacts_gap_addressed/ — does NOT overwrite existing runs.
#
# Run from project root:
#   bash run_gap_addressed.sh [--skip-features] [--skip-imputation] [--skip-training] [--device=cpu|cuda|mps]

set -euo pipefail
cd "$(dirname "$0")"

SKIP_FEATURES=0; SKIP_IMPUTATION=0; SKIP_TRAINING=0; DEVICE=""

for arg in "$@"; do
  case $arg in
    --skip-features)   SKIP_FEATURES=1 ;;
    --skip-imputation) SKIP_IMPUTATION=1 ;;
    --skip-training)   SKIP_TRAINING=1 ;;
    --device=*)        DEVICE="${arg#--device=}" ;;
    *) echo "Unknown flag: $arg"; exit 1 ;;
  esac
done

if [ -z "$DEVICE" ]; then
  if python3 -c "import torch; assert torch.backends.mps.is_available()" 2>/dev/null; then
    DEVICE="mps"
  elif python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    DEVICE="cuda"
  else
    DEVICE="cpu"
  fi
fi

FEAT_DIR="features_gap_addressed"
ART_DIR="artifacts_gap_addressed"
TARGET="target_gap_addressed.parquet"

log() { echo; echo "==> $*"; }

if [ ! -f "$TARGET" ]; then
  echo "ERROR: $TARGET not found. Run the gap exclusion step first."
  exit 1
fi

# ── Step 1: Feature engineering ───────────────────────────────────────────────
if [ "$SKIP_FEATURES" -eq 0 ]; then
  log "Engineering features -> $FEAT_DIR/"
  mkdir -p "$FEAT_DIR"

  python3 -m src.features.demographics \
    --data-pkl data.pkl --target "$TARGET" --outdir "$FEAT_DIR"

  python3 -m src.features.insurance \
    --data-pkl data.pkl --target "$TARGET" --outdir "$FEAT_DIR"

  python3 -m src.features.comorbidities \
    --data-pkl data.pkl --target "$TARGET" --outdir "$FEAT_DIR"

  python3 -m src.features.meds_procedures_history \
    --data-pkl data.pkl --target "$TARGET" --outdir "$FEAT_DIR"

  python3 -m src.features.vitals_preicu_48h \
    --data-pkl data.pkl --target "$TARGET" --outdir "$FEAT_DIR"

  python3 -m src.features.labs_preicu_7d \
    --data-pkl data.pkl --target "$TARGET" --outdir "$FEAT_DIR"

  python3 -m src.features.fluids_preicu_48h \
    --data-pkl data.pkl --target "$TARGET" --outdir "$FEAT_DIR"

  log "Merging features -> $FEAT_DIR/features_all.parquet"
  python3 -m src.build_features \
    --target "$TARGET" --features-dir "$FEAT_DIR" --outdir "$FEAT_DIR"
else
  log "Skipping feature engineering"
fi

# ── Step 2: Imputation ────────────────────────────────────────────────────────
if [ "$SKIP_IMPUTATION" -eq 0 ]; then
  log "Imputing -> $FEAT_DIR/features_all_imputed.parquet"
  python3 -m src.imputation train \
    "$FEAT_DIR/features_all.parquet" \
    "$FEAT_DIR/features_all_imputed.parquet"
else
  log "Skipping imputation"
fi

# ── Step 3: Train models ──────────────────────────────────────────────────────
if [ "$SKIP_TRAINING" -eq 0 ]; then
  mkdir -p "$ART_DIR/xgb_aki" "$ART_DIR/rf_aki" "$ART_DIR/logreg_aki" \
           "$ART_DIR/dnn_aki" "$ART_DIR/selfatten_aki" "$ART_DIR/dcn_aki"

  log "Training XGBoost"
  python3 -m src.train_xgboost \
    --input "$FEAT_DIR/features_all.parquet" \
    --out-prefix "$ART_DIR/xgb_aki/xgb"

  log "Training Random Forest"
  python3 -m src.train_rf \
    --input "$FEAT_DIR/features_all.parquet" \
    --impute \
    --out-prefix "$ART_DIR/rf_aki/rf"

  log "Training Logistic Regression"
  python3 -m src.train_logreg \
    --input "$FEAT_DIR/features_all_imputed.parquet" \
    --out-prefix "$ART_DIR/logreg_aki/logreg"

  log "Training DNN (device=$DEVICE)"
  python3 -m src.train_dnn \
    --input "$FEAT_DIR/features_all_imputed.parquet" \
    --device "$DEVICE" \
    --out-prefix "$ART_DIR/dnn_aki/dnn"

  log "Training Self-Attention (device=$DEVICE)"
  python3 -m src.train_selfattn \
    --input "$FEAT_DIR/features_all_imputed.parquet" \
    --device "$DEVICE" \
    --out-prefix "$ART_DIR/selfatten_aki/selfatten"

  log "Training DCN (device=$DEVICE)"
  python3 -m src.train_dcn \
    --input "$FEAT_DIR/features_all_imputed.parquet" \
    --device "$DEVICE" \
    --out-prefix "$ART_DIR/dcn_aki/dcn"
else
  log "Skipping model training"
fi

log "Pipeline complete. Outputs in $FEAT_DIR/ and $ART_DIR/"
