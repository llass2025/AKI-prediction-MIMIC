#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
rcs_logreg.py

Re-fit the Elastic Net logistic regression model replacing the linear age
term with restricted cubic splines (RCS, 4 knots) and adding age×sex
interaction terms. Produces:

  - AUROC/AUPRC comparison vs primary linear logistic
  - Age-risk curves by sex with 95% bootstrap CI
  - Spline coefficients table
  - Predictions CSV

Inputs:
  --features   features/features_all.parquet  (same as primary model)
  --outdir     age_sex_rcs/artifacts/

Knot placement: 5th, 35th, 65th, 95th percentiles of age in training set.
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


# ---------------------------------------------------------------------------
# RCS basis construction
# ---------------------------------------------------------------------------

def rcs_basis(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """
    Restricted cubic spline basis for vector x given knot locations.
    With k knots, returns k-2 additional basis columns (beyond x itself).
    Uses Harrell's parameterization.

    Returns array of shape (n, k-2).
    """
    k = len(knots)
    n = len(x)
    cols = []
    tk = knots[-1]  # last knot
    tk1 = knots[-2]  # second-to-last knot
    denom = (tk - knots[0]) ** 2

    for j in range(k - 2):
        tj  = knots[j]
        tj1 = knots[j + 1]

        def _term(t):
            return np.maximum(x - t, 0) ** 3

        col = (
            _term(tj)
            - _term(tk1) * (tk - tj) / (tk - tk1)
            + _term(tk)  * (tk1 - tj) / (tk - tk1)
        ) / denom
        cols.append(col)

    return np.column_stack(cols) if cols else np.zeros((n, 0))


def add_rcs_features(
    df: pd.DataFrame,
    age_col: str = "age",
    sex_col: str = "dem_sex_M",
    knot_percentiles: list = None,
    knots: np.ndarray = None,
) -> tuple:
    """
    Add RCS basis columns and age×sex interaction terms to df.
    Returns (df_expanded, knot_locations, basis_col_names, interaction_col_names).
    """
    if knot_percentiles is None:
        knot_percentiles = [5, 35, 65, 95]

    age = df[age_col].values.astype(float)

    if knots is None:
        knots = np.percentile(age, knot_percentiles)

    basis = rcs_basis(age, knots)
    k = len(knots)
    basis_cols = [f"age_rcs{i+1}" for i in range(k - 2)]

    df = df.copy()
    for i, col in enumerate(basis_cols):
        df[col] = basis[:, i]

    # Interaction terms: sex × each spline basis
    interaction_cols = []
    if sex_col in df.columns:
        for col in basis_cols:
            iname = f"{sex_col}_x_{col}"
            df[iname] = df[sex_col] * df[col]
            interaction_cols.append(iname)

    return df, knots, basis_cols, interaction_cols


# ---------------------------------------------------------------------------
# Data loading helpers (mirrors train_logreg.py)
# ---------------------------------------------------------------------------

EXCLUDE_COLS = {
    'version', 'subject_id', 'hadm_id', 'admittime',
    'target_meta__age', 'gender', 'hospital_expire_flag', 'death_flag',
    'time_to_death_from_adm_hr', 'time_to_death_from_disch_hr', 'cmb_ckd',
    'is_aki', 'is_ckd', 'ckd_only_flag', 'ckd_admission_flag', 'is_autocar',
    'kdigo-aki', 'renal_impaired_at_adm', 'aki_history_flag', 'onset_aki_flag',
}


def load_and_split(features_path: str, seed: int = 42):
    df_full = pd.read_parquet(features_path)
    df_full = df_full[~((df_full['cmb_ckd'] == 1) | (df_full['renal_impaired_at_adm'] == 1))]
    df_full['incident_aki_label'] = (df_full['incident_aki_label'].astype(float) > 0).astype(int)

    # MIMIC-IV for train/val/test splits
    df = df_full[df_full['version'] == 'mimic4'].copy()
    df = df[~df['incident_aki_label'].isna()].copy()

    # MIMIC-III for external validation
    df_m3 = df_full[df_full['version'] == 'mimic3'].copy()
    df_m3 = df_m3[~df_m3['incident_aki_label'].isna()].copy()

    # feature cols
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feat_cols = [c for c in num_cols if c not in EXCLUDE_COLS
                 and not c.startswith('emb_')
                 and c != 'incident_aki_label']

    y = df['incident_aki_label'].values
    groups = df['subject_id'].values

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    trainval_idx, test_idx = next(gss.split(df, y, groups=groups))
    rel_val = 0.1 / 0.9
    gss2 = GroupShuffleSplit(n_splits=1, test_size=rel_val, random_state=seed + 1)
    tr_idx_rel, val_idx_rel = next(gss2.split(
        trainval_idx, y[trainval_idx], groups=groups[trainval_idx]
    ))
    train_idx = trainval_idx[tr_idx_rel]
    val_idx   = trainval_idx[val_idx_rel]

    return df, df_m3, feat_cols, y, groups, train_idx, val_idx, test_idx


# ---------------------------------------------------------------------------
# Age-risk curves with bootstrap CI
# ---------------------------------------------------------------------------

def compute_age_curves(
    clf,
    imputer,
    scaler: StandardScaler,
    feat_cols: list,
    fitted_feat_names: list,
    knots: np.ndarray,
    df_train: pd.DataFrame,
    n_bootstrap: int = 500,
    seed: int = 42,
    age_range: tuple = (18, 90),
):
    """
    Compute predicted AKI probability vs age curves for male and female,
    holding all other features at training-set medians.
    Returns curves dict with ages, mean predictions, and bootstrap CIs.

    feat_cols: full RCS feature list (pre-imputation)
    fitted_feat_names: feature names after imputer (may be shorter if all-NaN cols dropped)
    """
    ages = np.linspace(age_range[0], age_range[1], 100)
    rng  = np.random.default_rng(seed)

    curves = {}
    for sex_label, sex_val in [("Female", 0), ("Male", 1)]:
        preds = np.zeros((len(ages), n_bootstrap))

        for b in range(n_bootstrap):
            # Resample training set for CI
            idx = rng.integers(0, len(df_train), size=len(df_train))
            boot_med = df_train.iloc[idx][feat_cols].median()

            row_list = []
            for age in ages:
                row = boot_med.copy()
                row["age"] = age
                row["dem_sex_M"] = sex_val
                row["dem_sex_F"] = 1 - sex_val
                row_list.append(row)

            X_grid = pd.DataFrame(row_list)

            # Add RCS and interaction
            X_grid, _, _, _ = add_rcs_features(
                X_grid, age_col="age", sex_col="dem_sex_M", knots=knots
            )

            # Align to feat_cols (pre-imputation)
            for c in feat_cols:
                if c not in X_grid.columns:
                    X_grid[c] = 0.0
            # Run through the same imputer → scaler pipeline
            X_arr = imputer.transform(X_grid[feat_cols].values)
            # imputer may output fewer columns (all-NaN cols); align to fitted_feat_names
            if X_arr.shape[1] != len(fitted_feat_names):
                # fall back: select by position up to scaler's expected width
                X_arr = X_arr[:, :len(fitted_feat_names)]
            X_arr = scaler.transform(X_arr)
            preds[:, b] = clf.predict_proba(X_arr)[:, 1]

        curves[sex_label] = {
            "ages":    ages,
            "mean":    preds.mean(axis=1),
            "ci_lo":   np.percentile(preds, 2.5, axis=1),
            "ci_hi":   np.percentile(preds, 97.5, axis=1),
        }

    return curves


def plot_age_curves(curves: dict, out_path: str, auroc_rcs: float, auroc_lin: float):
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"Female": "#C00000", "Male": "#2E74B5"}

    for sex_label, d in curves.items():
        c = colors[sex_label]
        ax.plot(d["ages"], d["mean"], color=c, lw=2.5, label=sex_label)
        ax.fill_between(d["ages"], d["ci_lo"], d["ci_hi"], color=c, alpha=0.15)

    ax.set_xlabel("Age (years)", fontsize=12)
    ax.set_ylabel("Predicted AKI probability", fontsize=12)
    ax.set_title("Predicted AKI Risk by Age and Sex\n(Elastic Net + RCS, medians for other features)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=11)
    ax.text(0.02, 0.97,
            f"AUROC — RCS: {auroc_rcs:.3f}  |  Linear: {auroc_lin:.3f}",
            transform=ax.transAxes, va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    ax.set_xlim(18, 90)
    ax.set_ylim(0, None)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"✓ Saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="RCS age×sex logistic regression for AKI.")
    ap.add_argument("--features",    default="features/features_all.parquet")
    ap.add_argument("--outdir",      default="age_sex_rcs/artifacts")
    ap.add_argument("--seed",        type=int, default=42)
    ap.add_argument("--n-bootstrap", type=int, default=300,
                    help="Bootstrap resamples for age-risk curve CIs (default: 300)")
    ap.add_argument("--knot-pcts",   nargs=4, type=float,
                    default=[5, 35, 65, 95],
                    help="Knot percentiles for RCS (default: 5 35 65 95)")
    ap.add_argument("--skip-curves", action="store_true",
                    help="Skip age-risk curve bootstrap (faster, for debugging)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # ── Load and split ────────────────────────────────────────────────────
    print("Loading features ...")
    df, df_m3, feat_cols, y, groups, train_idx, val_idx, test_idx = \
        load_and_split(args.features, seed=args.seed)

    df_train = df.iloc[train_idx].copy()

    # ── Add RCS features ──────────────────────────────────────────────────
    print("Adding RCS basis and age×sex interactions ...")
    knot_pcts = args.knot_pcts
    train_ages = df_train["age"].values

    knots = np.percentile(train_ages, knot_pcts)
    print(f"  Knot locations (age): {knots.round(1)}")

    df_rcs, knots, basis_cols, interaction_cols = add_rcs_features(
        df, age_col="age", sex_col="dem_sex_M", knots=knots
    )

    new_cols = basis_cols + interaction_cols
    feat_cols_rcs = feat_cols + [c for c in new_cols if c not in feat_cols]

    # ── Impute ────────────────────────────────────────────────────────────
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(df_rcs.iloc[train_idx][feat_cols_rcs])
    X_val   = imputer.transform(df_rcs.iloc[val_idx][feat_cols_rcs])
    X_test  = imputer.transform(df_rcs.iloc[test_idx][feat_cols_rcs])

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    y_train = y[train_idx]
    y_val   = y[val_idx]
    y_test  = y[test_idx]

    # ── Fit RCS logistic ──────────────────────────────────────────────────
    print("Fitting Elastic Net logistic regression with RCS ...")
    # Use new sklearn 1.8+ API: set l1_ratio directly, omit penalty
    clf = LogisticRegression(
        solver="saga",
        l1_ratio=0.5, C=1.0,
        max_iter=10000, random_state=args.seed,
    )
    clf.fit(X_train, y_train)

    # ── Evaluate ──────────────────────────────────────────────────────────
    results = {}
    for name, X, yt in [("train", X_train, y_train),
                         ("val",   X_val,   y_val),
                         ("test",  X_test,  y_test)]:
        probs = clf.predict_proba(X)[:, 1]
        auroc = roc_auc_score(yt, probs) if len(np.unique(yt)) > 1 else np.nan
        auprc = average_precision_score(yt, probs) if len(np.unique(yt)) > 1 else np.nan
        results[name] = {"auroc": round(float(auroc), 4), "auprc": round(float(auprc), 4), "n": int(len(yt))}
        print(f"  {name:6s}  AUROC={auroc:.4f}  AUPRC={auprc:.4f}")

    # MIMIC-III external validation
    if len(df_m3):
        df_m3_rcs, _, _, _ = add_rcs_features(
            df_m3, age_col="age", sex_col="dem_sex_M", knots=knots
        )
        for c in feat_cols_rcs:
            if c not in df_m3_rcs.columns:
                df_m3_rcs[c] = 0.0
        X_m3 = scaler.transform(imputer.transform(df_m3_rcs[feat_cols_rcs]))
        y_m3 = df_m3["incident_aki_label"].astype(int).values
        probs_m3 = clf.predict_proba(X_m3)[:, 1]
        auroc_m3 = roc_auc_score(y_m3, probs_m3)
        auprc_m3 = average_precision_score(y_m3, probs_m3)
        results["mimic3_full"] = {"auroc": round(float(auroc_m3), 4),
                                   "auprc": round(float(auprc_m3), 4),
                                   "n": len(df_m3)}
        print(f"  {'mimic3':6s}  AUROC={auroc_m3:.4f}  AUPRC={auprc_m3:.4f}")

    # ── Save metrics ──────────────────────────────────────────────────────
    with open(os.path.join(args.outdir, "rcs_metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    # ── Coefficients ──────────────────────────────────────────────────────
    # Use imputer's output feature names (all-NaN cols may be skipped/zeroed)
    try:
        fitted_feat_names = list(imputer.get_feature_names_out(feat_cols_rcs))
    except Exception:
        fitted_feat_names = feat_cols_rcs
    coef_df = pd.DataFrame({
        "feature": fitted_feat_names,
        "coefficient": clf.coef_[0]
    }).sort_values("coefficient", key=abs, ascending=False)
    coef_df.to_csv(os.path.join(args.outdir, "rcs_coefficients.csv"), index=False)
    print("\nTop RCS/interaction coefficients:")
    rcs_coefs = coef_df[coef_df["feature"].str.contains("rcs|dem_sex_M_x")]
    print(rcs_coefs.to_string(index=False))

    # ── Age-risk curves ───────────────────────────────────────────────────
    if not args.skip_curves:
        print(f"\nComputing age-risk curves (n_bootstrap={args.n_bootstrap}) ...")
        df_train_unscaled = df_rcs.iloc[train_idx].copy()
        try:
            curves = compute_age_curves(
                clf=clf,
                imputer=imputer,
                scaler=scaler,
                feat_cols=feat_cols_rcs,
                fitted_feat_names=fitted_feat_names,
                knots=knots,
                df_train=df_train_unscaled,
                n_bootstrap=args.n_bootstrap,
                seed=args.seed,
            )
            auroc_lin = 0.870  # from primary run (keepautocar)
            plot_age_curves(
                curves=curves,
                out_path=os.path.join(args.outdir, "rcs_age_sex_curves.png"),
                auroc_rcs=results["test"]["auroc"],
                auroc_lin=auroc_lin,
            )
        except Exception as e:
            print(f"⚠️  Age-risk curves failed ({e}) — skipping. Run with --skip-curves to suppress.")
    else:
        print("\nSkipping age-risk curves (--skip-curves).")

    # ── Predictions CSV ───────────────────────────────────────────────────
    test_df = df.iloc[test_idx][["subject_id", "hadm_id", "incident_aki_label"]].copy()
    test_probs = clf.predict_proba(X_test)[:, 1]
    test_df["prob"] = test_probs
    test_df["split"] = "test"
    test_df.to_csv(os.path.join(args.outdir, "rcs_predictions.csv"), index=False)

    print(f"\n✓ All outputs written to {args.outdir}/")
    print("\nSummary:")
    for split, r in results.items():
        print(f"  {split:12s}  AUROC={r['auroc']:.4f}  AUPRC={r['auprc']:.4f}  n={r['n']:,}")


if __name__ == "__main__":
    main()
