#!/usr/bin/env python3
"""
Likelihood ratio test for the sex x age interaction terms in the RCS elastic net model.

Compares:
  Full model:    all features + age_rcs1 + age_rcs2 + dem_sex_M_x_age_rcs1 + dem_sex_M_x_age_rcs2
  Reduced model: same but without the two interaction terms

LR = 2 * (LL_full - LL_reduced) ~ chi-squared(df=2) by Wilks' theorem.

Run from project root:
  python3 age_sex_rcs/lrt_sex_age_interaction.py
"""

import sys, os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from age_sex_rcs.rcs_logreg import add_rcs_features, load_and_split

FEATURES_PATH = "/Users/lb353/coding/aki-analysis/features/features_all.parquet"
SEED = 42


def main():
    print("Loading data...")
    df, _, feat_cols, y, groups, train_idx, val_idx, test_idx = load_and_split(
        FEATURES_PATH, seed=SEED)

    knots = np.percentile(df.iloc[train_idx]["age"].values, [5, 35, 65, 95])
    print(f"Knots: {knots.round(1)}")

    df_rcs, knots, basis_cols, interaction_cols = add_rcs_features(
        df, age_col="age", sex_col="dem_sex_M", knots=knots)

    feat_cols_full  = feat_cols + [c for c in basis_cols + interaction_cols if c not in feat_cols]
    feat_cols_noint = feat_cols + [c for c in basis_cols if c not in feat_cols]

    y_train = y[train_idx]

    def fit(feat_cols_subset):
        X = df_rcs[feat_cols_subset].values
        imp = SimpleImputer(strategy="median")
        sc  = StandardScaler()
        X_tr = sc.fit_transform(imp.fit_transform(X[train_idx]))
        clf = LogisticRegression(penalty="elasticnet", solver="saga",
                                 l1_ratio=0.5, C=1.0, max_iter=10000,
                                 tol=1e-3, random_state=SEED)
        clf.fit(X_tr, y_train)
        ll = -log_loss(y_train, clf.predict_proba(X_tr), normalize=False)
        return clf, ll

    print("\nFitting full model (with sex×age interaction)...")
    clf_full, ll_full = fit(feat_cols_full)
    print(f"  LL_full = {ll_full:.4f}")

    print("Fitting reduced model (no interaction terms)...")
    clf_red, ll_red = fit(feat_cols_noint)
    print(f"  LL_reduced = {ll_red:.4f}")

    lr_stat = 2 * (ll_full - ll_red)
    df_diff = len(interaction_cols)
    p_val   = stats.chi2.sf(lr_stat, df=df_diff)

    print(f"\n=== Likelihood Ratio Test ===")
    print(f"Interaction terms tested: {interaction_cols}")
    print(f"LR statistic: {lr_stat:.4f}  (df={df_diff})")
    print(f"p-value:      {p_val:.4e}")

    if p_val < 0.001:
        print("=> Interaction is statistically significant (p < 0.001)")
    elif p_val < 0.05:
        print("=> Interaction is statistically significant (p < 0.05)")
    else:
        print("=> Interaction is NOT statistically significant")


if __name__ == "__main__":
    main()
