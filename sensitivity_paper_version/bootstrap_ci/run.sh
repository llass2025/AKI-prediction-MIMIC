#!/usr/bin/env bash
# Bootstrap 95% CIs for AKI prediction models using aki-analysis artifacts.
# Results saved to sensitivity_paper_version/bootstrap_ci/artifacts/
# Run from project root: bash sensitivity_paper_version/bootstrap_ci/run.sh

set -euo pipefail
cd "$(dirname "$0")/../.."

AKI_ANALYSIS=~/coding/aki-analysis/artifacts
OUTDIR=sensitivity_paper_version/bootstrap_ci/artifacts
N_BOOTSTRAP=1000

/Users/lb353/anaconda3/envs/aki-modeling/bin/python3 sensitivity_paper_version/bootstrap_ci/bootstrap_ci_paper.py \
  --aki-analysis-dir "$AKI_ANALYSIS" \
  --n-bootstrap "$N_BOOTSTRAP" \
  --seed 42 \
  --outdir "$OUTDIR"
