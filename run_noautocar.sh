#!/usr/bin/env bash
# Run the full AKI/CKD prediction pipeline
# Run from project root: bash run.sh
# Optional flags:
#   --skip-install     skip pip install
#   --skip-compile     skip compile_data (use existing data.pkl)
#   --skip-targets     skip target construction
#   --skip-features    skip feature engineering
#   --skip-imputation  skip imputation
#   --skip-training    skip model training
#   --device=cpu|cuda  device for deep models (default: auto-detect)

set -euo pipefail
cd "$(dirname "$0")"

SKIP_INSTALL=0; SKIP_COMPILE=0; SKIP_TARGETS=0
SKIP_FEATURES=0; SKIP_IMPUTATION=0; SKIP_TRAINING=0; DEVICE=""

for arg in "$@"; do
  case $arg in
    --skip-install)    SKIP_INSTALL=1 ;;
    --skip-compile)    SKIP_COMPILE=1 ;;
    --skip-targets)    SKIP_TARGETS=1 ;;
    --skip-features)   SKIP_FEATURES=1 ;;
    --skip-imputation) SKIP_IMPUTATION=1 ;;
    --skip-training)   SKIP_TRAINING=1 ;;
    --device=*)        DEVICE="${arg#--device=}" ;;
    *) echo "Unknown flag: $arg"; exit 1 ;;
  esac
done

if [ -z "$DEVICE" ]; then
  if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    DEVICE="cuda"
  else
    DEVICE="cpu"
  fi
fi

log() { echo; echo "==> $*"; }

# ── Step 0: Install dependencies ──────────────────────────────────────────────
if [ "$SKIP_INSTALL" -eq 0 ]; then
  log "Installing dependencies"
  pip install -q pandas numpy scikit-learn xgboost torch joblib pyarrow matplotlib captum pycox
fi

# ── Step 1: Compile data ──────────────────────────────────────────────────────
if [ "$SKIP_COMPILE" -eq 0 ]; then
  if [ -f data.pkl ]; then
    log "data.pkl already exists — skipping compile"
  else
    log "Compiling MIMIC data -> data.pkl"
    python3 -m src.compile_data --output_pickle data.pkl
  fi
else
  log "Skipping compile_data"
fi

# ── Step 2: Build targets ─────────────────────────────────────────────────────
if [ "$SKIP_TARGETS" -eq 0 ]; then
  log "Computing target admissions -> target_admissions.parquet"
  python3 -m src.compute_target_admissions --data-pkl data.pkl --outdir .

  log "Building incident AKI target -> incident_aki_target.parquet"
  python3 -m src.incident_aki_target --input target_admissions.parquet --outdir .

  log "Deduplicating to patient level -> target.parquet"
  python3 -m src.dedup_patient_level --input incident_aki_target.parquet --outdir .
else
  log "Skipping target construction"
fi

# ── Step 3: Feature engineering ───────────────────────────────────────────────
if [ "$SKIP_FEATURES" -eq 0 ]; then
  log "Engineering features"

  python3 -m src.features.demographics \
    --data-pkl data.pkl --target target.parquet --outdir .

  python3 -m src.features.insurance \
    --data-pkl data.pkl --target target.parquet --outdir .

  python3 -m src.features.comorbidities \
    --data-pkl data.pkl --target target.parquet --outdir .

  python3 -m src.features.meds_procedures_history \
    --data-pkl data.pkl --target target.parquet --outdir .

  python3 -m src.features.vitals_preicu_48h \
    --data-pkl data.pkl --target target.parquet --outdir .

  python3 -m src.features.labs_preicu_7d \
    --data-pkl data.pkl --target target.parquet --outdir .

  python3 -m src.features.fluids_preicu_48h \
    --data-pkl data.pkl --target target.parquet --outdir .

  log "Merging all features -> features/features_all.parquet"
  mkdir -p features
  python3 -m src.build_features \
    --target target.parquet --features-dir . --outdir ./features
else
  log "Skipping feature engineering"
fi

# ── Step 4: Imputation ────────────────────────────────────────────────────────
if [ "$SKIP_IMPUTATION" -eq 0 ]; then
  log "Imputing -> features/features_all_imputed.parquet"
  python3 -m src.imputation train \
    features/features_all.parquet \
    features/features_all_imputed.parquet
else
  log "Skipping imputation"
fi

# ── Step 5: Train models ──────────────────────────────────────────────────────
if [ "$SKIP_TRAINING" -eq 0 ]; then
  mkdir -p artifacts/xgb_aki artifacts/rf_aki artifacts/logreg_aki \
           artifacts/dnn_aki artifacts/selfatten_aki artifacts/dcn_aki

  log "Training XGBoost"
  python3 -m src.train_xgboost \
    --input features/features_all.parquet \
    --out-prefix artifacts/xgb_aki/xgb

  log "Training Random Forest"
  python3 -m src.train_rf \
    --input features/features_all.parquet \
    --impute \
    --out-prefix artifacts/rf_aki/rf

  log "Training Logistic Regression"
  python3 -m src.train_logreg \
    --input features/features_all_imputed.parquet \
    --out-prefix artifacts/logreg_aki/logreg

  log "Training DNN (device=$DEVICE)"
  python3 -m src.train_dnn \
    --input features/features_all_imputed.parquet \
    --device "$DEVICE" \
    --out-prefix artifacts/dnn_aki/dnn

  log "Training Self-Attention (device=$DEVICE)"
  python3 -m src.train_selfattn \
    --input features/features_all_imputed.parquet \
    --device "$DEVICE" \
    --out-prefix artifacts/selfatten_aki/selfatten

  log "Training DCN (device=$DEVICE)"
  python3 -m src.train_dcn \
    --input features/features_all_imputed.parquet \
    --device "$DEVICE" \
    --out-prefix artifacts/dcn_aki/dcn
else
  log "Skipping model training"
fi

log "Pipeline complete."
