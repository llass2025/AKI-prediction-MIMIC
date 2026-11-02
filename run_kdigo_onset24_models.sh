#!/usr/bin/env bash
# Rerun all models with the KDIGO in-hospital-onset label (positive = KDIGO with first qualifying creatinine >= 24 h
# after admission; negative = KDIGO-negative; early-onset KDIGO-positive patients are left out).
# Inputs are written by:  python -m src.concordance.model_inputs   (features_kdigo_in_hospital_onset/)
# Same commands and defaults as step 5 of run.sh; only --input and --out-prefix differ. Outputs go to artifacts_kdigo_onset24/.
# Usage: bash run_kdigo_onset24_models.sh [--device=cpu|cuda|mps]

set -euo pipefail
cd "$(dirname "$0")"

IN="${FEATURES_DIR:-features_kdigo_in_hospital_onset}"
OUT="${ARTIFACTS_DIR:-artifacts_kdigo_onset24}"
DEVICE=""
for arg in "$@"; do
  case $arg in
    --device=*) DEVICE="${arg#--device=}" ;;
    *) echo "Unknown flag: $arg"; exit 1 ;;
  esac
done
if [ -z "$DEVICE" ]; then
  if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then DEVICE="cuda"; else DEVICE="cpu"; fi
fi
log() { echo; echo "==> $*"; }

for f in "$IN/features_all.parquet" "$IN/features_all_imputed.parquet"; do
  [ -f "$f" ] || { echo "Missing $f. Run: python -m src.concordance.model_inputs"; exit 1; }
done
mkdir -p "$OUT"/{xgb_aki,rf_aki,logreg_aki,elasticnet_aki,dnn_aki,selfatten_aki,dcn_aki}

log "XGBoost";            python3 -m src.train_xgboost --input "$IN/features_all.parquet" --out-prefix "$OUT/xgb_aki/xgb"
log "Random Forest";      python3 -m src.train_rf --input "$IN/features_all.parquet" --impute --out-prefix "$OUT/rf_aki/rf"
log "Logistic Regression"; python3 -m src.train_logreg --input "$IN/features_all_imputed.parquet" --out-prefix "$OUT/logreg_aki/logreg"
log "Elastic Net";        python3 -m src.train_elasticnet --input "$IN/features_all_imputed.parquet" --n-alphas 11 --n-lambdas 20 --cv-folds 10 --seed 20250422 --out-prefix "$OUT/elasticnet_aki/elasticnet"
log "DNN ($DEVICE)";      python3 -m src.train_dnn --input "$IN/features_all_imputed.parquet" --device "$DEVICE" --out-prefix "$OUT/dnn_aki/dnn"
log "Self-Attention ($DEVICE)"; python3 -m src.train_selfattn --input "$IN/features_all_imputed.parquet" --device "$DEVICE" --out-prefix "$OUT/selfatten_aki/selfatten"
log "DCN ($DEVICE)";      python3 -m src.train_dcn --input "$IN/features_all_imputed.parquet" --device "$DEVICE" --out-prefix "$OUT/dcn_aki/dcn"
log "Done. Artifacts in $OUT"
