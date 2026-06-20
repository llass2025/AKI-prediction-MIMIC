#!/usr/bin/env python3
"""
Bootstrap 95% CIs for elastic net logistic regression coefficients (RCS model).
Refits the same model (saga, l1_ratio=0.5, C=1.0) on 1000 bootstrap resamples
of the training set. Saves per-feature point estimate + CI to artifacts/.

Run from project root:
  python3 sensitivity_paper_version/bootstrap_ci/bootstrap_elasticnet_coef.py \
    --features /path/to/features_all.parquet \
    --n-bootstrap 1000 \
    --outdir sensitivity_paper_version/bootstrap_ci/artifacts
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from age_sex_rcs.rcs_logreg import add_rcs_features, load_and_split

EXCLUDE_COLS = {
    'version', 'subject_id', 'hadm_id', 'admittime',
    'target_meta__age', 'gender', 'hospital_expire_flag', 'death_flag',
    'time_to_death_from_adm_hr', 'time_to_death_from_disch_hr', 'cmb_ckd',
    'is_aki', 'is_ckd', 'ckd_only_flag', 'ckd_admission_flag', 'is_autocar',
    'kdigo-aki', 'renal_impaired_at_adm', 'aki_history_flag', 'onset_aki_flag',
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', default='/Users/lb353/coding/aki-analysis/features/features_all.parquet')
    ap.add_argument('--n-bootstrap', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--outdir', default='sensitivity_paper_version/bootstrap_ci/artifacts')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print('Loading features ...')
    df, _, feat_cols, y, groups, train_idx, val_idx, test_idx = load_and_split(args.features, seed=args.seed)

    knot_pcts = [5, 35, 65, 95]
    train_ages = df.iloc[train_idx]['age'].values
    knots = np.percentile(train_ages, knot_pcts)
    print(f'Knots: {knots.round(1)}')

    df_rcs, knots, basis_cols, interaction_cols = add_rcs_features(
        df, age_col='age', sex_col='dem_sex_M', knots=knots
    )
    new_cols = basis_cols + interaction_cols
    feat_cols_rcs = feat_cols + [c for c in new_cols if c not in feat_cols]

    X_all = df_rcs[feat_cols_rcs].values
    y_all = y
    g_all = groups

    # Fit original model on full training set
    print('Fitting original model ...')
    imputer = SimpleImputer(strategy='median')
    X_train_orig = imputer.fit_transform(X_all[train_idx])
    scaler = StandardScaler()
    X_train_orig = scaler.fit_transform(X_train_orig)
    y_train = y_all[train_idx]

    clf_orig = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, C=1.0, max_iter=10000, tol=1e-3, random_state=args.seed)
    clf_orig.fit(X_train_orig, y_train)

    try:
        feat_names = list(imputer.get_feature_names_out(feat_cols_rcs))
    except Exception:
        feat_names = feat_cols_rcs

    point_coefs = clf_orig.coef_[0]
    print(f'  Original: {len(point_coefs)} coefficients, {(point_coefs != 0).sum()} nonzero')

    # Bootstrap
    subsample_frac = 0.30
    subsample_n = int(len(train_idx) * subsample_frac)
    print(f'Bootstrapping {args.n_bootstrap} resamples (subsample {subsample_frac:.0%} = {subsample_n:,} rows each) ...')
    boot_coefs = np.zeros((args.n_bootstrap, len(point_coefs)))
    n_train = len(train_idx)

    for b in range(args.n_bootstrap):
        if (b + 1) % 100 == 0:
            print(f'  {b+1}/{args.n_bootstrap}', flush=True)

        # Subsample without replacement (m-out-of-n bootstrap)
        idx = rng.choice(n_train, size=subsample_n, replace=False)
        boot_abs_idx = train_idx[idx]

        X_b = X_all[boot_abs_idx]
        y_b = y_all[boot_abs_idx]

        if len(np.unique(y_b)) < 2:
            boot_coefs[b] = point_coefs
            continue

        imp_b = SimpleImputer(strategy='median')
        X_b = imp_b.fit_transform(X_b)
        sc_b = StandardScaler()
        X_b = sc_b.fit_transform(X_b)

        clf_b = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, C=1.0, max_iter=5000, tol=1e-3, random_state=int(rng.integers(0, 1e6)))
        try:
            clf_b.fit(X_b, y_b)
            boot_coefs[b] = clf_b.coef_[0]
        except Exception:
            boot_coefs[b] = point_coefs

    ci_lo = np.percentile(boot_coefs, 2.5, axis=0)
    ci_hi = np.percentile(boot_coefs, 97.5, axis=0)

    results = pd.DataFrame({
        'feature': feat_names,
        'coefficient': point_coefs,
        'ci_lo': ci_lo,
        'ci_hi': ci_hi,
        'ci_str': [f'{v:.3f} ({lo:.3f}–{hi:.3f})' for v, lo, hi in zip(point_coefs, ci_lo, ci_hi)],
        'crosses_zero': ((ci_lo < 0) & (ci_hi > 0)),
        'n_bootstrap': args.n_bootstrap,
    }).sort_values('coefficient', key=abs, ascending=False)

    out = os.path.join(args.outdir, 'elasticnet_coef_bootstrap_ci.csv')
    results.to_csv(out, index=False)
    print(f'\n✓ Saved {out}')

    print('\nTop 20 features with CIs:')
    top = results.head(20)[['feature', 'coefficient', 'ci_lo', 'ci_hi', 'crosses_zero']]
    print(top.to_string(index=False))

    rcs_rows = results[results['feature'].str.contains('rcs|dem_sex_M_x|dem_sex_F', na=False)]
    print('\nRCS / sex interaction coefficients:')
    print(rcs_rows[['feature', 'coefficient', 'ci_lo', 'ci_hi', 'crosses_zero']].to_string(index=False))

    print('\nDone.')


if __name__ == '__main__':
    main()
