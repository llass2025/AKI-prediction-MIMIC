#!/usr/bin/env bash
# Standalone end-to-end pipeline with cmb_ckd and renal_impaired excluded UPSTREAM
# for both the AKI prediction pipeline and the CKD survival pipeline.
#
# Exclusion happens before feature engineering — not at modeling time.
#
# Outputs:
#   src_with_exclusions/features/        AKI features (clean cohort)
#   src_with_exclusions/artifacts/       AKI model artifacts
#   src_with_exclusions/ckd_survival/    CKD survival features + artifacts
#
# Run from project root:
#   bash src_with_exclusions/run_all.sh [--skip-aki-features] [--skip-aki-training]
#                                       [--skip-ckd-targets] [--skip-ckd-features]
#                                       [--skip-ckd-training] [--device=cpu|mps|cuda]

set -euo pipefail
cd "$(dirname "$0")/.."

SKIP_AKI_FEATURES=0; SKIP_AKI_TRAINING=0
SKIP_CKD_TARGETS=0; SKIP_CKD_FEATURES=0; SKIP_CKD_TRAINING=0
DEVICE=""

for arg in "$@"; do
  case $arg in
    --skip-aki-features)  SKIP_AKI_FEATURES=1 ;;
    --skip-aki-training)  SKIP_AKI_TRAINING=1 ;;
    --skip-ckd-targets)   SKIP_CKD_TARGETS=1 ;;
    --skip-ckd-features)  SKIP_CKD_FEATURES=1 ;;
    --skip-ckd-training)  SKIP_CKD_TRAINING=1 ;;
    --device=*)           DEVICE="${arg#--device=}" ;;
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

OUTDIR="src_with_exclusions"
FEAT_DIR="$OUTDIR/features"
ART_DIR="$OUTDIR/artifacts"
CKD_DIR="$OUTDIR/ckd_survival"
TARGET="$OUTDIR/target.parquet"

log() { echo; echo "==> $*"; }

mkdir -p "$FEAT_DIR" "$CKD_DIR"

# ══════════════════════════════════════════════════════════════════════════════
# PART 1 — AKI PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

# ── Step 1a: Build AKI targets ────────────────────────────────────────────────
log "Building AKI target admissions"
python3 -m src_with_exclusions.src.compute_target_admissions \
  --data-pkl data.pkl \
  --keep_autocar \
  --outdir "$OUTDIR"

log "Excluding cmb_ckd and renal_impaired patients from AKI target (upstream)"
python3 src_with_exclusions/filter_aki_target.py \
  --target "$OUTDIR/target_admissions.parquet" \
  --features features/features_all.parquet \
  --out "$TARGET"

# ── Step 1b: AKI feature engineering ─────────────────────────────────────────
if [ "$SKIP_AKI_FEATURES" -eq 0 ]; then
  log "Engineering AKI features -> $FEAT_DIR/"

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

  log "Merging AKI features -> $FEAT_DIR/features_all.parquet"
  python3 -m src.build_features \
    --target "$TARGET" --features-dir "$FEAT_DIR" --outdir "$FEAT_DIR"

  log "Imputing -> $FEAT_DIR/features_all_imputed.parquet"
  python3 -m src.imputation train \
    "$FEAT_DIR/features_all.parquet" \
    "$FEAT_DIR/features_all_imputed.parquet"
else
  log "Skipping AKI feature engineering"
fi

# ── Step 1c: Train AKI models ─────────────────────────────────────────────────
if [ "$SKIP_AKI_TRAINING" -eq 0 ]; then
  mkdir -p "$ART_DIR/xgb_aki" "$ART_DIR/rf_aki" "$ART_DIR/logreg_aki" \
           "$ART_DIR/dnn_aki" "$ART_DIR/selfatten_aki" "$ART_DIR/dcn_aki"

  log "Training XGBoost"
  python3 -m src_with_exclusions.src.train_xgboost \
    --input "$FEAT_DIR/features_all.parquet" \
    --out-prefix "$ART_DIR/xgb_aki/xgb"

  log "Training Random Forest"
  python3 -m src_with_exclusions.src.train_rf \
    --input "$FEAT_DIR/features_all.parquet" \
    --impute \
    --out-prefix "$ART_DIR/rf_aki/rf"

  log "Training Logistic Regression"
  python3 -m src_with_exclusions.src.train_logreg \
    --input "$FEAT_DIR/features_all_imputed.parquet" \
    --out-prefix "$ART_DIR/logreg_aki/logreg"

  log "Training DNN (device=$DEVICE)"
  python3 -m src_with_exclusions.src.train_dnn \
    --input "$FEAT_DIR/features_all_imputed.parquet" \
    --device "$DEVICE" \
    --out-prefix "$ART_DIR/dnn_aki/dnn"

  log "Training Self-Attention (device=$DEVICE)"
  python3 -m src_with_exclusions.src.train_selfattn \
    --input "$FEAT_DIR/features_all_imputed.parquet" \
    --device "$DEVICE" \
    --out-prefix "$ART_DIR/selfatten_aki/selfatten"

  log "Training DCN (device=$DEVICE)"
  python3 -m src_with_exclusions.src.train_dcn \
    --input "$FEAT_DIR/features_all_imputed.parquet" \
    --device "$DEVICE" \
    --out-prefix "$ART_DIR/dcn_aki/dcn"
else
  log "Skipping AKI model training"
fi

# ══════════════════════════════════════════════════════════════════════════════
# PART 2 — CKD SURVIVAL PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

# ── Step 2a: Build CKD targets ────────────────────────────────────────────────
if [ "$SKIP_CKD_TARGETS" -eq 0 ]; then
  log "Building CKD target admissions -> $CKD_DIR/"
  python3 -m src_with_exclusions.ckd_survival.src.compute_target_admissions \
    --data-pkl data.pkl \
    --keep_autocar \
    --aki-target "$TARGET" \
    --outdir "$CKD_DIR"

  log "Building patient-level CKD survival targets -> $CKD_DIR/"
  python3 -m src_with_exclusions.ckd_survival.src.incident_ckd_target \
    --adm "$CKD_DIR/incident_ckd_admission.csv" \
    --outdir "$CKD_DIR"
else
  log "Skipping CKD target construction"
fi

# ── Step 2b: CKD feature engineering ─────────────────────────────────────────
if [ "$SKIP_CKD_FEATURES" -eq 0 ]; then
  log "Building CKD features -> $CKD_DIR/"

  python3 -m ckd_survival.src.features.demographics_comorbidity_ckd \
    --aki-feat "$FEAT_DIR/features_all_imputed.parquet" \
    --outdir "$CKD_DIR"

  python3 -m ckd_survival.src.features.labs_preckd \
    --data-pkl data.pkl \
    --target "$CKD_DIR/incident_ckd_patient.csv" \
    --outdir "$CKD_DIR"

  python3 -m ckd_survival.src.features.meds_procedures_history \
    --data-pkl data.pkl \
    --target "$CKD_DIR/incident_ckd_patient.csv" \
    --outdir "$CKD_DIR"

  log "Merging CKD features -> $CKD_DIR/features_ckd.parquet"
  python3 -m ckd_survival.src.features.merge_ckd_features \
    --targets "$CKD_DIR/incident_ckd_patient.csv" \
    --dem-cmb "$CKD_DIR/features_dem_cmb_ckd.parquet" \
    --labs "$CKD_DIR/features_labs_preckd.parquet" \
    --medsprocs "$CKD_DIR/features_medsprocs_preckd.parquet" \
    --outdir "$CKD_DIR"
else
  log "Skipping CKD feature engineering"
fi

# ── Step 2c: Train CKD models ─────────────────────────────────────────────────
if [ "$SKIP_CKD_TRAINING" -eq 0 ]; then
  mkdir -p "$CKD_DIR/artifacts"

  log "Training XGBoost time-to-CKD"
  python3 -m ckd_survival.src.train_xgboost_time_to_ckd \
    --input "$CKD_DIR/features_ckd.parquet" \
    --out-prefix "$CKD_DIR/artifacts/xgb_ckd"

  log "Training XGBoost time-to-death"
  python3 -m ckd_survival.src.train_xgboost_time_to_death \
    --input "$CKD_DIR/features_ckd.parquet" \
    --out-prefix "$CKD_DIR/artifacts/xgb_death"

  log "Training DeepSurv time-to-CKD"
  python3 -m ckd_survival.src.train_deepsurv_time_to_ckd \
    --input "$CKD_DIR/features_ckd.parquet" \
    --out-prefix "$CKD_DIR/artifacts/deepsurv_ckd"

  log "Training DeepSurv time-to-death"
  python3 -m ckd_survival.src.train_deepsurv_time_to_death \
    --input "$CKD_DIR/features_ckd.parquet" \
    --out-prefix "$CKD_DIR/artifacts/deepsurv_death"
else
  log "Skipping CKD model training"
fi

# ══════════════════════════════════════════════════════════════════════════════
# PART 3 — LAB-BASED CKD SENSITIVITY
# ══════════════════════════════════════════════════════════════════════════════

log "Running lab-based CKD definition (sensitivity analysis)"
python3 ckd_definition_sensitivity/lab_ckd_definition.py \
  --data-pkl data.pkl \
  --survival-csv "$CKD_DIR/incident_ckd_survival.csv" \
  --target "$TARGET" \
  --outdir "$CKD_DIR"

log "Pipeline complete. Outputs:"
log "  AKI features:    $FEAT_DIR/"
log "  AKI artifacts:   $ART_DIR/"
log "  CKD survival:    $CKD_DIR/"
log "  Lab CKD:         $CKD_DIR/incident_ckd_survival_labckd.csv"
