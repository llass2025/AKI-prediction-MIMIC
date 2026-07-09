#!/usr/bin/env bash
# Age x Sex RCS — using aki-analysis features
set -euo pipefail
cd "$(dirname "$0")/../.."

AKI_ANALYSIS="$HOME/coding/aki-analysis"

python3 -m age_sex_rcs.rcs_logreg \
  --features    "${AKI_ANALYSIS}/features/features_all.parquet" \
  --outdir      "sensitivity_paper_version/age_sex_rcs/artifacts" \
  --seed        42 \
  --n-bootstrap 300 \
  --knot-pcts   5 35 65 95
