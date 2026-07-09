#!/usr/bin/env bash
# Run Elastic Net logistic regression for incident AKI
# From project root: bash run_elasticnet.sh

set -euo pipefail
cd "$(dirname "$0")"

echo "==> Elastic Net logistic regression with sex interactions"
python -m src.train_elasticnet \
    --input      features/features_all_imputed.parquet \
    --out-prefix artifacts/elasticnet_aki/elasticnet \
    --seed       20250422 \
    --n-alphas   51 \
    --n-lambdas  50 \
    --cv-folds   10

echo "==> Done. Artifacts in artifacts/elasticnet_aki/"
