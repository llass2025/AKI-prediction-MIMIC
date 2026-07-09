#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_elasticnet.py

Elastic Net logistic regression for incident AKI with sex-predictor interactions.
Closely follows the R glmnet analysis described in the paper:

Algorithm:
  1. Build feature matrix:
       - Continuous predictors (labs, vitals, fluid, age, comorbidities, hx) — standardized
       - Continuous × male interactions — standardized (male: impute with male mean; female: set to 0)
       - Categorical dummies K-1 (insurance, race) — unscaled
       - Categorical × male interactions — unscaled
       - Sex dummies (dem_sex_F, dem_sex_M) — unscaled
  2. 10-fold CV over alpha (l1_ratio: 51 values 0→1) and lambda (C grid)
     optimising AUC — mirrors R cva.glmnet.
  3. Fit final elastic net with best (alpha, lambda) → extract nonzero coefficients.
  4. Refit plain LogisticRegression on selected features only.
  5. Evaluate on MIMIC-IV test set and MIMIC-III external validation.

Inputs:
  features/features_all_imputed.parquet

Outputs:
  artifacts/elasticnet_aki/elasticnet.{joblib,features.txt,coefficients.csv,
                                        predictions.csv,metrics.json,summary.txt}
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.model_selection import GroupShuffleSplit, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, classification_report
import joblib

warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# Feature column definitions
# ---------------------------------------------------------------------------

NON_FEATURE_COLS = {
    "version", "subject_id", "hadm_id", "admittime", "target_meta__age",
    "gender", "hospital_expire_flag", "death_flag",
    "time_to_death_from_adm_hr", "time_to_death_from_disch_hr",
    "cmb_ckd", "is_aki", "is_ckd", "ckd_only_flag", "ckd_admission_flag",
    "is_autocar", "kdigo-aki", "renal_impaired_at_adm",
    "aki_history_flag", "onset_aki_flag", "incident_aki_label",
}

NULL_COLS = [
    "vital48h_minute_ventilation_min", "vital48h_minute_ventilation_max",
    "vital48h_minute_ventilation_mean", "vital48h_minute_ventilation_std",
    "vital48h_minute_ventilation_last", "vital48h_minute_ventilation_count",
]

RRT_COLS_PATTERN = "rrt"

# Categorical groups — drop one per group (K-1 constraint)
# Reference categories (dropped):
INS_REF  = "ins_UNKNOWN"       # drop UNKNOWN as reference
RACE_REF = "dem_race_UNK"      # drop UNKNOWN/UNK as reference
# Sex: keep both dem_sex_F and dem_sex_M (consistent with R code keeping all GENDER dummies)
# dem_sex_M is used as the interaction indicator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_feature_groups(df: pd.DataFrame):
    """Return (continuous_cols, categorical_cols) based on column naming."""
    all_cols = set(df.columns) - NON_FEATURE_COLS - set(NULL_COLS)
    all_cols = {c for c in all_cols if RRT_COLS_PATTERN not in c}
    all_cols = {c for c in all_cols if not c.endswith("_missing")}
    all_cols = {c for c in all_cols if not c.startswith("emb_")}

    # Categorical groups (already one-hot encoded in data)
    ins_cols  = sorted(c for c in all_cols if c.startswith("ins_") and c != INS_REF)
    race_cols = sorted(c for c in all_cols if c.startswith("dem_race_") and c != RACE_REF)
    sex_cols  = sorted(c for c in all_cols if c.startswith("dem_sex_"))

    categorical_cols = set(ins_cols + race_cols + sex_cols)

    # Everything else is continuous
    continuous_cols = sorted(all_cols - categorical_cols)

    return continuous_cols, ins_cols, race_cols, sex_cols


def build_design_matrix(df: pd.DataFrame,
                        continuous_cols: list,
                        ins_cols: list,
                        race_cols: list,
                        sex_cols: list,
                        cont_scaler=None,
                        cont_int_scaler=None,
                        male_col_means: pd.Series = None,
                        fit: bool = True):
    """
    Build the design matrix following the paper's R code:
      - X_cont:  continuous features, scaled
      - X_int:   continuous × male interactions, scaled
                 (impute male-mean for males; 0 for females pre-scale)
      - X_cat:   categorical dummies K-1 (ins + race + sex), unscaled
      - X_cat_int: categorical × male interactions, unscaled
    Returns (X_full, cont_scaler, cont_int_scaler, male_col_means, feature_names)
    """
    male = df["dem_sex_M"].values  # 1 = male, 0 = female

    # ── Continuous ─────────────────────────────────────────────────────────
    X_cont = df[continuous_cols].copy()

    # ── Continuous × male interaction ──────────────────────────────────────
    X_int = X_cont.copy()
    for col in continuous_cols:
        X_int[col] = X_cont[col] * male
    X_int = X_int.astype(float)

    # Imputation for interaction terms:
    # males: mean of male values; females: 0 (since male=0 → interaction=0)
    if fit:
        male_idx = (male == 1)
        male_col_means = X_cont[male_idx].mean()

    for col in continuous_cols:
        # for males, fill remaining NAs (should be 0 after imputation upstream)
        male_mask = (male == 1)
        female_mask = (male == 0)
        # females: interaction is always 0
        X_int.loc[female_mask, col] = 0.0
        # males: fill any NaN with male mean (data already imputed upstream, just in case)
        if male_col_means is not None:
            X_int.loc[male_mask & X_int[col].isna(), col] = male_col_means[col]

    # Rename interaction columns
    X_int.columns = [f"{c}_M" for c in continuous_cols]

    # ── Scale continuous ───────────────────────────────────────────────────
    if fit:
        cont_scaler = StandardScaler()
        X_cont_s = cont_scaler.fit_transform(X_cont.fillna(0.0))
        cont_int_scaler = StandardScaler()
        X_int_s = cont_int_scaler.fit_transform(X_int.fillna(0.0))
    else:
        X_cont_s = cont_scaler.transform(X_cont.fillna(0.0))
        X_int_s  = cont_int_scaler.transform(X_int.fillna(0.0))

    # ── Categorical (unscaled) ─────────────────────────────────────────────
    cat_cols = ins_cols + race_cols + sex_cols
    X_cat = df[cat_cols].fillna(0.0).values

    # ── Categorical × male interactions (unscaled) ─────────────────────────
    # Exclude dem_sex_M × dem_sex_M (trivial) and dem_sex_F × dem_sex_M
    cat_int_cols = ins_cols + race_cols  # sex dummies not interacted with sex
    X_cat_int = (df[cat_int_cols].fillna(0.0).values * male[:, None])
    cat_int_names = [f"{c}_M" for c in cat_int_cols]

    # ── Assemble ───────────────────────────────────────────────────────────
    cont_names     = continuous_cols
    int_names      = [f"{c}_M" for c in continuous_cols]
    cat_names      = cat_cols
    all_feat_names = cont_names + int_names + cat_names + cat_int_names

    X_full = np.hstack([X_cont_s, X_int_s, X_cat, X_cat_int])

    return X_full, cont_scaler, cont_int_scaler, male_col_means, all_feat_names


def evaluate_split(y_true, y_score, thr=0.5):
    y_true  = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    auroc = float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else float("nan")
    auprc = float(average_precision_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else float("nan")
    y_pred = (y_score >= thr).astype(int)
    cm  = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0,0]), int(cm[0,1]), int(cm[1,0]), int(cm[1,1])
    report = classification_report(y_true, y_pred, digits=4, zero_division=0, output_dict=True)
    return {
        "n": int(len(y_true)),
        "prevalence": float(y_true.mean()),
        "auroc": auroc, "auprc": auprc,
        "threshold": float(thr),
        "sensitivity": tp / (tp + fn + 1e-12),
        "specificity": tn / (tn + fp + 1e-12),
        "precision":   tp / (tp + fp + 1e-12),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "classification_report": report,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Elastic Net logistic regression for incident AKI.")
    ap.add_argument("--input",      default="features/features_all_imputed.parquet")
    ap.add_argument("--out-prefix", default="artifacts/elasticnet_aki/elasticnet")
    ap.add_argument("--seed",       type=int, default=20250422)
    ap.add_argument("--n-alphas",   type=int, default=51,
                    help="Number of alpha (l1_ratio) values in grid 0→1 (default: 51)")
    ap.add_argument("--n-lambdas",  type=int, default=50,
                    help="Number of lambda (C) values in log-grid (default: 50)")
    ap.add_argument("--cv-folds",   type=int, default=10)
    ap.add_argument("--threshold",  type=float, default=0.5)
    args = ap.parse_args()

    np.random.seed(args.seed)
    out = Path(args.out_prefix)
    out.parent.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────
    print("Loading features ...")
    df = pd.read_parquet(args.input)
    df = df[~((df["cmb_ckd"] == 1) | (df["renal_impaired_at_adm"] == 1))]
    df["incident_aki_label"] = (df["incident_aki_label"].astype(float) > 0).astype(int)

    df4 = df[df["version"] == "mimic4"].copy().reset_index(drop=True)
    df3 = df[df["version"] == "mimic3"].copy().reset_index(drop=True)

    print(f"MIMIC-IV: {len(df4):,} patients | MIMIC-III: {len(df3):,} patients")

    # ── Feature groups ────────────────────────────────────────────────────
    continuous_cols, ins_cols, race_cols, sex_cols = get_feature_groups(df4)
    print(f"Continuous features: {len(continuous_cols)}")
    print(f"Categorical (ins K-1): {len(ins_cols)} | race (K-1): {len(race_cols)} | sex: {len(sex_cols)}")
    print(f"Total features before interactions: {len(continuous_cols)+len(ins_cols)+len(race_cols)+len(sex_cols)}")

    # ── Train/val/test split (GroupShuffleSplit on subject_id, 60/20/20) ──
    y4      = df4["incident_aki_label"].values
    groups4 = df4["subject_id"].values

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed)
    trainval_idx, test_idx = next(gss.split(df4, y4, groups=groups4))

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=args.seed + 1)
    tr_rel, val_rel = next(gss2.split(trainval_idx, y4[trainval_idx],
                                       groups=groups4[trainval_idx]))
    train_idx = trainval_idx[tr_rel]
    val_idx   = trainval_idx[val_rel]

    df_train = df4.iloc[train_idx].reset_index(drop=True)
    df_val   = df4.iloc[val_idx].reset_index(drop=True)
    df_test  = df4.iloc[test_idx].reset_index(drop=True)

    print(f"Train: {len(df_train):,} | Val: {len(df_val):,} | Test: {len(df_test):,}")

    y_train = df_train["incident_aki_label"].values
    y_val   = df_val["incident_aki_label"].values
    y_test  = df_test["incident_aki_label"].values

    # ── Build design matrices ─────────────────────────────────────────────
    print("Building design matrices (continuous + interactions + categoricals) ...")
    X_train, cont_sc, int_sc, male_means, feat_names = build_design_matrix(
        df_train, continuous_cols, ins_cols, race_cols, sex_cols, fit=True
    )
    X_val, _, _, _, _ = build_design_matrix(
        df_val, continuous_cols, ins_cols, race_cols, sex_cols,
        cont_scaler=cont_sc, cont_int_scaler=int_sc, male_col_means=male_means, fit=False
    )
    X_test, _, _, _, _ = build_design_matrix(
        df_test, continuous_cols, ins_cols, race_cols, sex_cols,
        cont_scaler=cont_sc, cont_int_scaler=int_sc, male_col_means=male_means, fit=False
    )

    print(f"Design matrix shape: {X_train.shape} (features including interactions)")

    # ── Step 2: 10-fold CV over alpha × lambda grid ───────────────────────
    # Mirrors R: cva.glmnet with alpha=seq(0,1,len=51), nfolds=10, type.measure="auc"
    print(f"\nStep 2: 10-fold CV over {args.n_alphas} alpha values × {args.n_lambdas} lambda values ...")
    alpha_grid   = np.linspace(0, 1, args.n_alphas)
    # C = 1/lambda; sklearn uses log-spaced Cs
    C_grid       = np.logspace(-4, 2, args.n_lambdas)

    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)

    clf_cv = LogisticRegressionCV(
        solver="saga",
        l1_ratios=list(alpha_grid),
        Cs=C_grid,
        cv=cv,
        scoring="roc_auc",
        max_iter=3000,
        random_state=args.seed,
        n_jobs=-1,
    )
    clf_cv.fit(X_train, y_train)

    best_alpha = float(clf_cv.l1_ratio_[0])
    best_C     = float(clf_cv.C_[0])
    val_proba  = clf_cv.predict_proba(X_val)[:, 1]
    best_auc   = float(roc_auc_score(y_val, val_proba)) if len(np.unique(y_val)) > 1 else -np.inf

    print(f"\nBest: alpha={best_alpha:.4f}  C={best_C:.5f}  val_AUC={best_auc:.4f}")

    # ── Step 3: Fit final elastic net → nonzero coefficients ─────────────
    print("\nStep 3: Fit final elastic net with best (alpha, lambda) ...")
    clf_en = LogisticRegression(
        solver="saga",
        l1_ratio=float(best_alpha),
        C=float(best_C),
        max_iter=5000,
        random_state=args.seed,
    )
    # Combine train+val for final fitting (mirrors R which fits on full data)
    X_trainval = np.vstack([X_train, X_val])
    y_trainval = np.concatenate([y_train, y_val])
    clf_en.fit(X_trainval, y_trainval)

    coef_vals   = clf_en.coef_[0]
    nonzero_idx = np.where(np.abs(coef_vals) > 0)[0]
    selected_features = [feat_names[i] for i in nonzero_idx]
    print(f"Nonzero coefficients: {len(nonzero_idx)} / {len(feat_names)}")
    print(f"Selected features (top 20 by |coef|):")
    top = sorted(zip(selected_features, coef_vals[nonzero_idx]), key=lambda x: abs(x[1]), reverse=True)
    for f, c in top[:20]:
        print(f"  {f:50s}  {c:+.4f}")

    # Save all coefficients
    coef_df = pd.DataFrame({
        "feature": feat_names,
        "coefficient_elasticnet": coef_vals,
        "selected": (np.abs(coef_vals) > 0).astype(int),
    }).sort_values("coefficient_elasticnet", key=abs, ascending=False)

    # ── Step 4: Refit plain logistic on selected features ────────────────
    print("\nStep 4: Refit plain logistic regression on selected features ...")
    if len(nonzero_idx) == 0:
        print("WARNING: No features selected — using all features")
        nonzero_idx = np.arange(len(feat_names))

    X_train_sel    = X_trainval[:, nonzero_idx]
    X_test_sel     = X_test[:, nonzero_idx]

    clf_final = LogisticRegression(
        C=np.inf,
        solver="lbfgs",
        max_iter=5000,
        random_state=args.seed,
    )
    clf_final.fit(X_train_sel, y_trainval)

    # ── Evaluate ──────────────────────────────────────────────────────────
    train_proba = clf_final.predict_proba(X_train_sel)[:, 1]
    test_proba  = clf_final.predict_proba(X_test_sel)[:, 1]

    train_metrics = evaluate_split(y_trainval, train_proba, args.threshold)
    test_metrics  = evaluate_split(y_test,  test_proba,  args.threshold)

    print(f"\nTrain AUROC={train_metrics['auroc']:.4f}  AUPRC={train_metrics['auprc']:.4f}")
    print(f"Test  AUROC={test_metrics['auroc']:.4f}  AUPRC={test_metrics['auprc']:.4f}")

    results = {
        "model": "ElasticNet → plain logistic (selected features)",
        "best_alpha_l1_ratio": float(best_alpha),
        "best_C": float(best_C),
        "best_lambda": float(1 / best_C),
        "cv_folds": args.cv_folds,
        "n_features_total": len(feat_names),
        "n_features_selected": int(len(nonzero_idx)),
        "train": train_metrics,
        "test": test_metrics,
    }

    # ── MIMIC-III external validation ─────────────────────────────────────
    if len(df3):
        print("\nExternal validation on MIMIC-III ...")
        df3_clean = df3.copy().reset_index(drop=True)
        df3_clean["incident_aki_label"] = (df3_clean["incident_aki_label"].astype(float) > 0).astype(int)
        y3 = df3_clean["incident_aki_label"].values

        X3, _, _, _, _ = build_design_matrix(
            df3_clean, continuous_cols, ins_cols, race_cols, sex_cols,
            cont_scaler=cont_sc, cont_int_scaler=int_sc, male_col_means=male_means, fit=False
        )
        X3_sel   = X3[:, nonzero_idx]
        prob3    = clf_final.predict_proba(X3_sel)[:, 1]
        m3_metrics = evaluate_split(y3, prob3, args.threshold)
        results["mimic3_full"] = m3_metrics
        print(f"MIMIC-III AUROC={m3_metrics['auroc']:.4f}  AUPRC={m3_metrics['auprc']:.4f}")

    # ── Predictions CSV ───────────────────────────────────────────────────
    preds = []
    for split, df_s, probs, y_s in [
        ("trainval", df4.iloc[np.concatenate([train_idx, val_idx])], train_proba, y_trainval),
        ("test",     df_test, test_proba, y_test),
    ]:
        p = df_s[["subject_id", "hadm_id"]].copy()
        p["incident_aki_label"] = y_s
        p["prob"] = probs
        p["split"] = split
        preds.append(p)

    if len(df3):
        p3 = df3_clean[["subject_id", "hadm_id"]].copy()
        p3["incident_aki_label"] = y3
        p3["prob"] = prob3
        p3["split"] = "mimic3_full"
        preds.append(p3)

    preds_df = pd.concat(preds, ignore_index=True)

    # ── Save artifacts ────────────────────────────────────────────────────
    joblib.dump({
        "clf_elasticnet": clf_en,
        "clf_final": clf_final,
        "cont_scaler": cont_sc,
        "cont_int_scaler": int_sc,
        "male_col_means": male_means,
        "feat_names": feat_names,
        "selected_features": selected_features,
        "nonzero_idx": nonzero_idx,
        "continuous_cols": continuous_cols,
        "ins_cols": ins_cols,
        "race_cols": race_cols,
        "sex_cols": sex_cols,
        "best_alpha": float(best_alpha),
        "best_C": float(best_C),
    }, f"{out}.joblib")

    Path(f"{out}.features.txt").write_text("\n".join(feat_names))
    Path(f"{out}.selected_features.txt").write_text("\n".join(selected_features))
    coef_df.to_csv(f"{out}.coefficients.csv", index=False)
    preds_df.to_csv(f"{out}.predictions.csv", index=False)
    Path(f"{out}.metrics.json").write_text(json.dumps(results, indent=2))

    # Summary
    lines = [
        "# Elastic Net Logistic Regression — AKI Prediction",
        f"- Alpha (l1_ratio): {best_alpha:.4f}  Lambda: {1/best_C:.5f}  C: {best_C:.5f}",
        f"- CV: {args.cv_folds}-fold, metric: AUC",
        f"- Total features (incl. interactions): {len(feat_names)}",
        f"- Selected features (nonzero): {len(nonzero_idx)}",
        "",
        "## TRAIN+VAL",
        f"  AUROC={train_metrics['auroc']:.4f}  AUPRC={train_metrics['auprc']:.4f}  N={train_metrics['n']:,}",
        "",
        "## TEST",
        f"  AUROC={test_metrics['auroc']:.4f}  AUPRC={test_metrics['auprc']:.4f}  N={test_metrics['n']:,}",
    ]
    if "mimic3_full" in results:
        m = results["mimic3_full"]
        lines += ["", "## MIMIC-III (external)",
                  f"  AUROC={m['auroc']:.4f}  AUPRC={m['auprc']:.4f}  N={m['n']:,}"]

    lines += ["", "## TOP 30 SELECTED FEATURES (by |coef| in plain logistic)"]
    fin_coef = pd.Series(clf_final.coef_[0], index=selected_features).sort_values(key=abs, ascending=False)
    for f, c in fin_coef.head(30).items():
        lines.append(f"  {f:55s}  {c:+.4f}")

    Path(f"{out}.summary.txt").write_text("\n".join(lines))

    print(f"\n✓ All artifacts written to {out}.*")


if __name__ == "__main__":
    main()
