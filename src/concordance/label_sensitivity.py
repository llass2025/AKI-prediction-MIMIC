"""
Does the choice of AKI label (ICD vs KDIGO variants) change the model or the sex findings? (opt-in)

Cohort: the flowchart cohort (paper-run features_all, cmb_ckd and prior renal impairment excluded), joined to kdigo_final_label.parquet.
Label variants (rows with an undefined label are left out of that variant):

  ICD incident                       the pipeline label (reference)
  KDIGO final                        B3 baseline + 24 h pre-admission window, any onset
  KDIGO in-hospital onset            positive = KDIGO with first qualifying creatinine >= 24 h after admission;
                                     negative = KDIGO-negative; KDIGO-positive with earlier onset is left out
  ICD and KDIGO agree                positive = both; negative = neither (discordant patients left out)
  ICD, arrival-onset removed         positive = ICD-coded and not (KDIGO-positive with onset < 24 h); negative = ICD-negative;
                                     ICD-coded patients whose creatinine shows AKI on arrival are left out

For each variant: positive rate by sex, crude and age-adjusted sex odds ratios, an XGBoost model (MIMIC-IV 80/20 patient split,
MIMIC-III external) with AUROC / AUPRC overall and by sex, feature-importance agreement with the ICD model, and agreement of the
predicted risks. Outputs are suppressed aggregates; nothing row-level is written.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .stats import Suppressor

FEATURES_DIR = os.environ.get("AKI_FEATURES_DIR", "features")     # folder holding features_all.parquet (set AKI_FEATURES_DIR for another run)

LABELS = ("ICD incident", "KDIGO final", "KDIGO in-hospital onset", "ICD and KDIGO agree", "ICD, arrival-onset removed")
EXCLUDE = {"version", "subject_id", "hadm_id", "admittime", "target_meta__age", "gender", "hospital_expire_flag", "death_flag",
           "time_to_death_from_adm_hr", "time_to_death_from_disch_hr", "cmb_ckd", "is_aki", "is_ckd", "ckd_only_flag",
           "ckd_admission_flag", "is_autocar", "kdigo-aki", "renal_impaired_at_adm", "aki_history_flag", "onset_aki_flag",
           "incident_aki_label"}


def make_labels(d: pd.DataFrame) -> dict:
    """d needs icd_incident_label, kdigo_final, kdigo_incident_24h. Returns {name: float Series, NaN = left out}."""
    icd, k, inc = d["icd_incident_label"].to_numpy() == 1, d["kdigo_final"].to_numpy(bool), d["kdigo_incident_24h"].to_numpy(bool)
    out = {
        LABELS[0]: np.where(icd, 1.0, 0.0),
        LABELS[1]: np.where(k, 1.0, 0.0),
        LABELS[2]: np.where(inc, 1.0, np.where(~k, 0.0, np.nan)),
        LABELS[3]: np.where(icd & k, 1.0, np.where(~icd & ~k, 0.0, np.nan)),
        LABELS[4]: np.where(icd & k & ~inc, np.nan, np.where(icd, 1.0, 0.0)),
    }
    return {n: pd.Series(v, index=d.index) for n, v in out.items()}


def logit_or(y: np.ndarray, X: np.ndarray, j: int):
    """Odds ratio and Wald 95% CI for column j of X (intercept added); unpenalized logistic regression."""
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(penalty=None, max_iter=500, solver="lbfgs").fit(X, y)
    Z = np.hstack([np.ones((len(X), 1)), X])
    beta = np.r_[m.intercept_, m.coef_.ravel()]
    p = 1 / (1 + np.exp(-Z @ beta))
    cov = np.linalg.inv((Z * (p * (1 - p))[:, None]).T @ Z)
    b, se = beta[j + 1], np.sqrt(cov[j + 1, j + 1])
    return float(np.exp(b)), float(np.exp(b - 1.96 * se)), float(np.exp(b + 1.96 * se))


def auroc_ci(y, s, n_boot, seed):
    from sklearn.metrics import roc_auc_score
    y, s = np.asarray(y), np.asarray(s)
    est = roc_auc_score(y, s)
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            bs.append(roc_auc_score(y[i], s[i]))
    lo, hi = np.percentile(bs, [2.5, 97.5]) if bs else (np.nan, np.nan)
    return float(est), float(lo), float(hi)


def split_subjects(subjects: np.ndarray, seed: int, test_frac: float = 0.2) -> np.ndarray:
    """Boolean test mask over `subjects` (unique ids), fixed by seed so every label variant uses the same patients."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(subjects))
    test = np.zeros(len(subjects), bool)
    test[order[: int(round(test_frac * len(subjects)))]] = True
    return test


def load_cohort(a) -> pd.DataFrame:
    labels = pd.read_parquet(a.label_file)
    feat = pd.read_parquet(a.features_all)
    feat = feat[(feat["cmb_ckd"] != 1) & (feat["renal_impaired_at_adm"] != 1)]
    d = feat.merge(labels[["version", "hadm_id", "icd_incident_label", "kdigo_final", "kdigo_incident_24h", "onset_hours"]],
                   on=["version", "hadm_id"], how="inner", validate="one_to_one")
    assert len(d) == len(labels), f"feature table covers {len(d)} of {len(labels)} label rows"
    assert (d["incident_aki_label"].astype(int) == d["icd_incident_label"]).all(), "ICD labels differ between the two files"
    return d.reset_index(drop=True)


def run(a) -> None:
    from scipy.stats import spearmanr
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier

    log = (lambda m: None) if a.quiet else (lambda m: print(m, flush=True))
    sup = Suppressor(a.min_cell)
    d = load_cohort(a)
    labs = make_labels(d)
    sex_m = d["dem_sex_M"].to_numpy() == 1
    sex_f = d["dem_sex_F"].to_numpy() == 1
    feature_cols = [c for c in d.columns if c not in EXCLUDE and not c.startswith("emb_")
                    and c not in ("icd_incident_label", "kdigo_final", "kdigo_incident_24h", "onset_hours")
                    and pd.api.types.is_numeric_dtype(d[c])]
    is4 = (d["version"] == "mimic4").to_numpy()
    uniq = d.loc[is4, "subject_id"].to_numpy()
    assert len(set(uniq)) == len(uniq), "a MIMIC-IV patient appears twice"
    test4 = np.zeros(len(d), bool)
    test4[np.flatnonzero(is4)[split_subjects(uniq, a.seed)]] = True
    train4 = is4 & ~test4
    null_rate = d.loc[train4, feature_cols].isna().mean()
    feature_cols = [c for c in feature_cols if null_rate[c] <= 0.9]
    log(f"cohort {len(d)}  features {len(feature_cols)}  train {int(train4.sum())}  test {int(test4.sum())}  external {int((~is4).sum())}")

    counts, ors, perf, imps, adj = [], [], [], [], []
    scores, importance = {}, {}
    age = d["age"].to_numpy(float)
    for name in LABELS:
        y = labs[name].to_numpy()
        ok = ~np.isnan(y)
        for ver, vm in (("mimic4", is4), ("mimic3", ~is4)):
            for sx, m in (("all", np.ones(len(d), bool)), ("F", sex_f), ("M", sex_m)):
                s = ok & vm & m
                n, k = int(s.sum()), int(np.nansum(y[s]))
                counts.append(dict(label=name, version=ver, sex=sx, n=sup.count(n), n_pos=sup.count(k),
                                   pct_pos=sup.pct(k, n, k, n - k)))
            s = ok & vm & (sex_m | sex_f)
            male = sex_m[s].astype(float)
            c = logit_or(y[s], male[:, None], 0)
            ad = logit_or(y[s], np.column_stack([male, (age[s] - age[s].mean()) / 10.0]), 0)
            for kind, v in (("crude", c), ("age-adjusted", ad)):
                ors.append(dict(label=name, version=ver, model=kind, male_vs_female_or=f"{v[0]:.3f}", ci_low=f"{v[1]:.3f}", ci_high=f"{v[2]:.3f}"))
        # XGBoost on the MIMIC-IV training patients
        tr = ok & train4
        clf = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, eval_metric="logloss", random_state=a.seed, n_jobs=-1)
        clf.fit(d.loc[tr, feature_cols], y[tr].astype(int))
        scores[name] = clf.predict_proba(d[feature_cols])[:, 1]
        importance[name] = pd.Series(clf.feature_importances_, index=feature_cols)
        # ridge logistic with every feature: adjusted male-vs-female odds ratio (point estimate)
        med = d.loc[tr, feature_cols].median()
        Xtr = d.loc[tr, feature_cols].fillna(med)
        sc = StandardScaler().fit(Xtr)
        lr = LogisticRegression(C=0.05, max_iter=300).fit(sc.transform(Xtr), y[tr].astype(int))
        co = pd.Series(lr.coef_.ravel() / sc.scale_, index=feature_cols)
        adj.append(dict(label=name, adjusted_male_vs_female_or=f"{np.exp(co['dem_sex_M'] - co['dem_sex_F']):.3f}"))
        for split, sm in (("MIMIC-IV test", test4), ("MIMIC-III external", ~is4)):
            for sx, m in (("all", np.ones(len(d), bool)), ("F", sex_f), ("M", sex_m)):
                s = ok & sm & m
                n, k = int(s.sum()), int(y[s].sum())
                if k < a.min_cell or n - k < a.min_cell:
                    perf.append(dict(label=name, split=split, sex=sx, n=sup.count(n), n_pos=sup.count(k), auroc="suppressed",
                                     auroc_ci_low="suppressed", auroc_ci_high="suppressed", auprc="suppressed"))
                    continue
                est, lo, hi = auroc_ci(y[s], scores[name][s], a.n_boot, a.seed)
                perf.append(dict(label=name, split=split, sex=sx, n=sup.count(n), n_pos=sup.count(k), auroc=f"{est:.4f}",
                                 auroc_ci_low=f"{lo:.4f}", auroc_ci_high=f"{hi:.4f}",
                                 auprc=f"{average_precision_score(y[s], scores[name][s]):.4f}"))
        log(f"done {name}")

    ref = LABELS[0]
    top = lambda s, k=20: set(s.sort_values(ascending=False).head(k).index)  # noqa: E731
    shared = test4.copy()
    for name in LABELS:
        shared &= ~np.isnan(labs[name].to_numpy())
    for name in LABELS:
        rho = spearmanr(importance[name], importance[ref]).correlation
        rank = importance[name].rank(ascending=False)
        imps.append(dict(label=name, spearman_importance_vs_icd=f"{rho:.3f}", top20_overlap_with_icd=len(top(importance[name]) & top(importance[ref])),
                         sex_F_rank=int(rank["dem_sex_F"]), sex_M_rank=int(rank["dem_sex_M"]), age_rank=int(rank["age"]),
                         top5=", ".join(importance[name].sort_values(ascending=False).head(5).index),
                         spearman_risk_vs_icd_shared_test=f"{spearmanr(scores[name][shared], scores[ref][shared]).correlation:.3f}",
                         n_shared_test=sup.count(int(shared.sum()))))
    os.makedirs(a.outdir, exist_ok=True)
    out = Path(a.outdir)
    tables = {"s1_label_counts": counts, "s2_sex_odds_ratios": ors, "s3_model_performance": perf,
              "s4_importance_and_risk_agreement": imps, "s5_adjusted_sex_or_ridge": adj}
    for k, rows in tables.items():
        pd.DataFrame(rows).to_csv(out / f"{k}.csv", index=False, lineterminator="\n")
    log("wrote " + ", ".join(tables))


def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog="python -m src.concordance.label_sensitivity",
                                 description="Sensitivity of the model and sex findings to the AKI label.")
    ap.add_argument("--features-all", default=os.path.join(FEATURES_DIR, "features_all.parquet"))
    ap.add_argument("--label-file", default="kdigo_final_label.parquet")
    ap.add_argument("--outdir", default="reports/label_sensitivity")
    ap.add_argument("--min-cell", type=int, default=10)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20250831)
    ap.add_argument("--quiet", action="store_true")
    return ap.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
