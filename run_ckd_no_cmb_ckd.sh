#!/usr/bin/env bash
# Run the CKD survival pipeline excluding cmb_ckd==1 and renal_impaired_at_adm==1 patients upstream.
# Outputs to ckd_survival_no_cmb_ckd/ — does NOT overwrite existing runs.
#
# Run from project root:
#   bash run_ckd_no_cmb_ckd.sh [--skip-targets] [--skip-features] [--skip-training] [--device=cpu|mps|cuda]

set -euo pipefail
cd "$(dirname "$0")"

SKIP_TARGETS=0; SKIP_FEATURES=0; SKIP_TRAINING=0; DEVICE="cpu"

for arg in "$@"; do
  case $arg in
    --skip-targets)  SKIP_TARGETS=1 ;;
    --skip-features) SKIP_FEATURES=1 ;;
    --skip-training) SKIP_TRAINING=1 ;;
    --device=*)      DEVICE="${arg#--device=}" ;;
    *) echo "Unknown flag: $arg"; exit 1 ;;
  esac
done

OUTDIR="ckd_survival_no_cmb_ckd"
AKI_FEAT="features_no_cmb_ckd/features_all_imputed.parquet"

log() { echo; echo "==> $*"; }

mkdir -p "$OUTDIR"

# ── Step 1: Build CKD targets ─────────────────────────────────────────────────
if [ "$SKIP_TARGETS" -eq 0 ]; then
  log "Computing CKD target admissions -> $OUTDIR/"
  python3 -m ckd_survival.src.compute_target_admissions \
    --data-pkl data.pkl \
    --keep_autocar \
    --outdir "$OUTDIR"

  log "Filtering out cmb_ckd and renal_impaired patients"
  python3 - << EOF
import pandas as pd

# get excluded subject_ids from AKI features
feat = pd.read_parquet("features/features_all.parquet",
    columns=["subject_id","cmb_ckd","renal_impaired_at_adm"], engine="fastparquet")
excl_sids = set(feat.loc[(feat["cmb_ckd"]==1) | (feat["renal_impaired_at_adm"]==1), "subject_id"])
print(f"Excluding {len(excl_sids)} subjects with cmb_ckd or renal_impaired")

adm = pd.read_csv("$OUTDIR/incident_ckd_admission.csv")
print(f"incident_ckd_admission before: {adm['subject_id'].nunique()} patients, {len(adm)} rows")
adm_filtered = adm[~adm["subject_id"].isin(excl_sids)]
print(f"incident_ckd_admission after:  {adm_filtered['subject_id'].nunique()} patients, {len(adm_filtered)} rows")
adm_filtered.to_csv("$OUTDIR/incident_ckd_admission.csv", index=False)
print("Saved filtered incident_ckd_admission.csv")
EOF

  log "Building patient-level CKD survival targets -> $OUTDIR/"
  python3 -m ckd_survival.src.incident_ckd_target \
    --adm "$OUTDIR/incident_ckd_admission.csv" \
    --outdir "$OUTDIR"
else
  log "Skipping target construction"
fi

# ── Step 2: Feature engineering ───────────────────────────────────────────────
if [ "$SKIP_FEATURES" -eq 0 ]; then
  log "Building CKD features -> $OUTDIR/"

  python3 -m ckd_survival.src.features.demographics_comorbidity_ckd \
    --aki-feat "$AKI_FEAT" \
    --outdir "$OUTDIR"

  python3 -m ckd_survival.src.features.labs_preckd \
    --data-pkl data.pkl \
    --target "$OUTDIR/incident_ckd_patient.csv" \
    --outdir "$OUTDIR"

  python3 -m ckd_survival.src.features.meds_procedures_history \
    --data-pkl data.pkl \
    --target "$OUTDIR/incident_ckd_patient.csv" \
    --outdir "$OUTDIR"

  log "Merging CKD features -> $OUTDIR/features_ckd.parquet"
  python3 -m ckd_survival.src.features.merge_ckd_features \
    --targets "$OUTDIR/incident_ckd_patient.csv" \
    --dem-cmb "$OUTDIR/features_dem_cmb_ckd.parquet" \
    --labs "$OUTDIR/features_labs_preckd.parquet" \
    --medsprocs "$OUTDIR/features_medsprocs_preckd.parquet" \
    --outdir "$OUTDIR"
else
  log "Skipping feature engineering"
fi

# ── Step 3: Train CKD models ──────────────────────────────────────────────────
if [ "$SKIP_TRAINING" -eq 0 ]; then
  log "Training XGBoost time-to-CKD"
  python3 -m ckd_survival.src.train_xgboost_time_to_ckd \
    --input "$OUTDIR/features_ckd.parquet" \
    --out-prefix "$OUTDIR/artifacts/xgb_ckd"

  log "Training XGBoost time-to-death"
  python3 -m ckd_survival.src.train_xgboost_time_to_death \
    --input "$OUTDIR/features_ckd.parquet" \
    --out-prefix "$OUTDIR/artifacts/xgb_death"

  log "Training DeepSurv time-to-CKD"
  python3 -m ckd_survival.src.train_deepsurv_time_to_ckd \
    --input "$OUTDIR/features_ckd.parquet" \
    --out-prefix "$OUTDIR/artifacts/deepsurv_ckd"

  log "Training DeepSurv time-to-death"
  python3 -m ckd_survival.src.train_deepsurv_time_to_death \
    --input "$OUTDIR/features_ckd.parquet" \
    --out-prefix "$OUTDIR/artifacts/deepsurv_death"
else
  log "Skipping model training"
fi

log "CKD pipeline complete. Outputs in $OUTDIR/"
