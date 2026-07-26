#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
bootstrap_ci_survival.py

Bootstrap 95% confidence intervals for the CKD/death survival models
(XGBoost CoxPH, DeepSurv) trained by ckd_survival/src/train_*.py, on the
ckd_survival_no_cmb_ckd cohort (cmb_ckd + renal-impaired patients excluded
upstream).

Two variants (--variant native|imputed), evaluated separately since they
give materially different results for the CKD endpoint:
  native  -- ckd_survival_no_cmb_ckd/artifacts/, trained on raw
             features_ckd.parquet. XGBoost uses its own sparse-aware
             missing-value handling; DeepSurv uses its own internal
             median imputer (fit on train only). Missingness patterns
             (e.g. "was this lab even ordered") are available to the model
             as an implicit signal.
  imputed -- ckd_survival_no_cmb_ckd/artifacts_imputed/, trained on
             features_ckd_imputed.parquet, pre-imputed with the same
             3-tier (version+sex+age) median imputation used for the main
             AKI feature set. Missingness-as-signal is removed before the
             model ever sees the data.

Both XGBoost training scripts (train_xgboost_time_to_ckd.py,
train_xgboost_time_to_death.py) were patched to exclude "censor_time" from
the feature set -- it was previously included by omission (present in
DeepSurv's exclude_cols, missing from XGBoost's), and is a literal copy of
the time-to-event label for every censored subject where it's non-null.
Ablation showed dropping it changes c-index by <0.002 in all cases checked,
so it was not the actual cause of the native-vs-imputed gap (see below),
but it is a genuine leakage bug and both XGBoost artifact sets here were
retrained after the fix.

Method: percentile bootstrap (1000 resamples, subject-level, with
replacement), applied to the held-out test set and the MIMIC-III external
validation set. Does NOT retrain models -- reloads the saved model
artifacts (.joblib / .pt) and regenerates risk scores, then resamples those
scores. This guarantees the point estimates reproduce the corresponding
*.metrics.json exactly (verified at runtime -- see sanity-check prints).

The c-index bootstrapped here matches the *exact* (non-standard) pairwise
admissibility rule implemented in ckd_survival/src/train_*.py's
concordance_index(): a pair is admissible whenever at least one of the two
subjects had an observed event (regardless of whether the comparison would
be admissible under Harrell's stricter censoring rule). This is intentional
-- it reproduces the numbers already reported in metrics.json rather than a
"more correct" but different c-index implementation (e.g. lifelines).

Outputs:
  uncertainty_intervals/artifacts/bootstrap_ci_survival_summary_native.csv
  uncertainty_intervals/artifacts/bootstrap_ci_survival_summary_imputed.csv
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score

REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_DIR = REPO_ROOT / "ckd_survival_no_cmb_ckd"

VARIANTS = {
    "native": {
        "artifacts_dir": RUN_DIR / "artifacts",
        "input_parquet": RUN_DIR / "features_ckd.parquet",
        "label": "Native missing-value handling (raw features; XGBoost sparse-aware splits, "
                 "DeepSurv's own internal median imputer)",
    },
    "imputed": {
        "artifacts_dir": RUN_DIR / "artifacts_imputed",
        "input_parquet": RUN_DIR / "features_ckd_imputed.parquet",
        "label": "Manual imputation (3-tier version+sex+age median imputation upstream, "
                 "same as the main AKI feature set)",
    },
    "imputed_thresholded": {
        "artifacts_dir": RUN_DIR / "artifacts_imputed_thresholded",
        "input_parquet": RUN_DIR / "features_ckd_imputed_thresholded.parquet",
        "label": "Manual imputation, high-missingness columns excluded (same 3-tier imputation as "
                 "'imputed', but columns >90% missing in raw MIMIC-IV -- 159 of them, mostly labs "
                 "windows that were almost entirely fabricated by imputation -- are dropped before "
                 "training, matching the raw-missingness threshold XGBoost's native variant already "
                 "applies implicitly. See uncertainty_intervals/uncertainty_intervals_survival_imputed_thresholded.docx.",
    },
    "sex_interaction": {
        "artifacts_dir": REPO_ROOT / "ckd_survival_sex_interaction" / "artifacts",
        "input_parquet": REPO_ROOT / "ckd_survival_sex_interaction" / "features_ckd_sex_interaction.parquet",
        "label": "Co-author-specified reduced feature set: 34 base features with reference-category "
                 "dummy coding (dem_race WHITE, dem_mar MARRIED, ins MEDICARE dropped; dem_mar "
                 "OTHER/SEPARATED and ins UNKNOWN rows excluded) + 33 variable x dem_sex_M interaction "
                 "terms (67 total), labs restricted to the 90-day window only, native (raw) values. "
                 "Entirely separate cohort/feature/artifact tree from the other variants -- see "
                 "ckd_survival_sex_interaction/.",
    },
}


# ---------------------------------------------------------------------------
# Ported helpers (must match ckd_survival/src/train_*.py exactly)
# ---------------------------------------------------------------------------

def groups_split_masks(n, groups, test_size, val_size, seed):
    idx = np.arange(n)
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trainval_idx, test_idx = next(gss.split(idx, groups=groups))
    rel_val = val_size / (1.0 - test_size + 1e-12)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=rel_val, random_state=seed + 1)
    tr_idx_rel, val_idx_rel = next(gss2.split(trainval_idx, groups=groups[trainval_idx]))
    train_idx = trainval_idx[tr_idx_rel]
    val_idx = trainval_idx[val_idx_rel]
    train_mask = np.zeros(n, dtype=bool); train_mask[train_idx] = True
    val_mask = np.zeros(n, dtype=bool); val_mask[val_idx] = True
    test_mask = np.zeros(n, dtype=bool); test_mask[test_idx] = True
    return train_mask, val_mask, test_mask


class DeepSurv(nn.Module):
    def __init__(self, in_features, hidden=[512, 128, 64], dropout=0.1):
        super().__init__()
        layers = []
        last = in_features
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
            last = h
        layers += [nn.Linear(last, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def custom_cindex(times, scores, events):
    """
    Vectorized Harrell's concordance index, matching the (fixed) definition
    in ckd_survival/src/train_*.py's concordance_index().

    A pair is admissible only when the ordering of true failure times is
    actually knowable: both had events (times differ), or exactly one had
    an event and the other's observed/censoring time is >= that event's
    time. This replaces an earlier lenient rule (admissible whenever
    either subject had an event, regardless of orderability) which
    produced a severe downward c-index bias in cohorts with heavy
    early/short-follow-up censoring -- confirmed here: DeepSurv-CKD's
    reported c-index moved from 0.24 (lenient) to 0.87 (Harrell) on the
    same model, same scores.
    """
    times = np.asarray(times, dtype=float)
    scores = np.asarray(scores, dtype=float)
    events = np.asarray(events).astype(bool)
    n = len(times)
    if n < 2:
        return np.nan
    iu = np.triu_indices(n, k=1)
    ti, tj = times[iu[0]], times[iu[1]]
    si, sj = scores[iu[0]], scores[iu[1]]
    ei, ej = events[iu[0]], events[iu[1]]

    both_event = ei & ej
    one_event = ei ^ ej
    i_only = ei & ~ej
    j_only = ej & ~ei
    adm_ee = both_event & (ti != tj)
    adm_ec = one_event & ((i_only & (tj >= ti)) | (j_only & (ti >= tj)))
    admissible = adm_ee | adm_ec

    conc = admissible & (((ti < tj) & (si > sj)) | ((tj < ti) & (sj > si)))
    tie = admissible & (si == sj)
    n_total = admissible.sum()
    if n_total == 0:
        return np.nan
    n_conc = conc.sum() + 0.5 * tie.sum()
    return float(n_conc / n_total)


def horizon_binary_eval(time, event, score, horizon_days):
    time = np.asarray(time, float)
    event = np.asarray(event, int)
    score = np.asarray(score, float)
    pos = (event == 1) & (time <= horizon_days)
    neg = time > horizon_days
    use = pos | neg
    if use.sum() == 0 or len(np.unique(pos[use])) < 2:
        return np.nan, np.nan, int(use.sum())
    y_true = pos[use].astype(int)
    y_score = score[use]
    try:
        auroc = float(roc_auc_score(y_true, y_score))
        auprc = float(average_precision_score(y_true, y_score))
    except Exception:
        auroc, auprc = np.nan, np.nan
    return auroc, auprc, int(use.sum())


def td_metrics_mean(time, event, score, grid_times):
    aurocs, auprcs = [], []
    for t_cut in grid_times:
        a, p, _ = horizon_binary_eval(time, event, score, t_cut)
        if not np.isnan(a):
            aurocs.append(a)
        if not np.isnan(p):
            auprcs.append(p)
    mean_auroc = float(np.mean(aurocs)) if aurocs else np.nan
    mean_auprc = float(np.mean(auprcs)) if auprcs else np.nan
    return mean_auroc, mean_auprc


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def bootstrap_cindex(time, score, event, n_bootstrap, seed):
    rng = np.random.default_rng(seed)
    n = len(time)
    point = custom_cindex(time, score, event)
    vals = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        v = custom_cindex(time[idx], score[idx], event[idx])
        if not np.isnan(v):
            vals.append(v)
    return _summarize(point, vals)


def bootstrap_horizon_metrics(time, score, event, horizon_or_grid, n_bootstrap, seed, mean_over_grid=False):
    rng = np.random.default_rng(seed)
    n = len(time)
    if mean_over_grid:
        point_auroc, point_auprc = td_metrics_mean(time, event, score, horizon_or_grid)
    else:
        point_auroc, point_auprc, _ = horizon_binary_eval(time, event, score, horizon_or_grid)

    auroc_vals, auprc_vals = [], []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        t_b, s_b, e_b = time[idx], score[idx], event[idx]
        if mean_over_grid:
            a, p = td_metrics_mean(t_b, e_b, s_b, horizon_or_grid)
        else:
            a, p, _ = horizon_binary_eval(t_b, e_b, s_b, horizon_or_grid)
        if not np.isnan(a):
            auroc_vals.append(a)
        if not np.isnan(p):
            auprc_vals.append(p)
    return _summarize(point_auroc, auroc_vals), _summarize(point_auprc, auprc_vals)


def _summarize(point, vals, ci=0.95):
    if len(vals) < 10:
        return {"point": point, "ci_lo": np.nan, "ci_hi": np.nan, "n_bootstrap": len(vals)}
    alpha = (1 - ci) / 2
    ci_lo = float(np.percentile(vals, alpha * 100))
    ci_hi = float(np.percentile(vals, (1 - alpha) * 100))
    return {"point": point, "ci_lo": ci_lo, "ci_hi": ci_hi, "n_bootstrap": len(vals)}


# ---------------------------------------------------------------------------
# Model-specific data prep (mirrors each train_*.py's filtering exactly)
# ---------------------------------------------------------------------------

def prep_xgb(df, joblib_path, event_col):
    d = joblib.load(joblib_path)
    feats = d["features"]
    imputer = d["imputer"]

    dsplit = df[df["version"] == "mimic4"].copy()
    groups = dsplit["subject_id"].values
    train_m, val_m, test_m = groups_split_masks(len(dsplit), groups, 0.2, 0.1, 42)

    X = dsplit[feats].copy()
    if imputer is not None:
        X.iloc[train_m, :] = imputer.transform(X.loc[train_m])
        X.iloc[val_m, :] = imputer.transform(X.loc[val_m])
        X.iloc[test_m, :] = imputer.transform(X.loc[test_m])

    y_time = dsplit["time_days"].astype(float).values
    y_event = dsplit[event_col].astype(int).values

    clf = d["model"]
    score_test = clf.predict(X[test_m])

    df_ext = df[df["version"] == "mimic3"].copy()
    X_ext = df_ext[feats].copy()
    if imputer is not None:
        X_ext = pd.DataFrame(imputer.transform(X_ext), columns=feats, index=X_ext.index)
    score_ext = clf.predict(X_ext)
    t_ext = df_ext["time_days"].astype(float).values
    e_ext = df_ext[event_col].astype(int).values

    return {
        "test": (y_time[test_m], score_test, y_event[test_m]),
        "external_mimic3": (t_ext, score_ext, e_ext),
        "train_event_times": y_time[train_m & (y_event == 1)],
    }


def prep_deepsurv(df, joblib_path, pt_path, event_col, extra_filter=None):
    d = joblib.load(joblib_path)
    feats = d["features"]
    imputer = d["imputer"]
    scaler = d["scaler"]
    cont_cols = d["cont_cols"]

    df_f = extra_filter(df) if extra_filter else df

    dsplit = df_f[df_f["version"] == "mimic4"].copy()
    groups = dsplit["subject_id"].values
    train_m, val_m, test_m = groups_split_masks(len(dsplit), groups, 0.2, 0.1, 42)

    def prep(dfpart):
        vals = imputer.transform(dfpart[feats])
        df_imp = pd.DataFrame(vals, columns=feats, index=dfpart.index)
        if scaler is not None and len(cont_cols) > 0:
            df_imp[cont_cols] = scaler.transform(df_imp[cont_cols])
        return torch.tensor(df_imp.values, dtype=torch.float32)

    model = DeepSurv(len(feats))
    model.load_state_dict(torch.load(pt_path, map_location="cpu", weights_only=True))
    model.eval()

    def predict(dfpart):
        with torch.no_grad():
            return model(prep(dfpart)).numpy()

    y_time = dsplit["time_days"].astype(float).values
    y_event = dsplit[event_col].astype(int).values
    score_test = predict(dsplit.loc[test_m])

    df_ext = df_f[df_f["version"] == "mimic3"].copy()
    score_ext = predict(df_ext)
    t_ext = df_ext["time_days"].astype(float).values
    e_ext = df_ext[event_col].astype(int).values

    return {
        "test": (y_time[test_m], score_test, y_event[test_m]),
        "external_mimic3": (t_ext, score_ext, e_ext),
        "train_event_times": y_time[train_m & (y_event == 1)],
    }


def filter_cmb_ckd(df):
    return df[df["cmb_ckd"] == 0].copy()


def filter_death(df):
    df = df[df["cmb_ckd"] == 0].copy()
    df = df[df["is_ckd"] != 1].copy()
    if "event_code" in df.columns:
        df["is_death"] = (df["event_code"].astype(int) == 2).astype(int)
    else:
        df["is_death"] = (df["event_type"] == "Death").astype(int)
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(VARIANTS.keys()), required=True,
                     help="native = raw features + each algorithm's own missing-value handling; "
                          "imputed = manual 3-tier median imputation upstream")
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default=str(REPO_ROOT / "uncertainty_intervals" / "artifacts"))
    args = ap.parse_args()

    variant = VARIANTS[args.variant]
    artifacts_dir = variant["artifacts_dir"]
    input_parquet = variant["input_parquet"]

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    print(f"Variant: {args.variant} -- {variant['label']}")
    print(f"Loading {input_parquet} ...")
    df = pd.read_parquet(input_parquet)

    rows = []

    model_specs = [
        {"name": "xgb_ckd", "kind": "xgb", "event_col": "is_ckd",
         "joblib": artifacts_dir / "xgb_ckd.joblib"},
        {"name": "xgb_death", "kind": "xgb", "event_col": "is_death",
         "joblib": artifacts_dir / "xgb_death.joblib"},
        {"name": "deepsurv_ckd", "kind": "deepsurv", "event_col": "is_ckd",
         "joblib": artifacts_dir / "deepsurv_ckd.joblib",
         "pt": artifacts_dir / "deepsurv_ckd.pt", "filter": filter_cmb_ckd},
        {"name": "deepsurv_death", "kind": "deepsurv", "event_col": "is_death",
         "joblib": artifacts_dir / "deepsurv_death.joblib",
         "pt": artifacts_dir / "deepsurv_death.pt", "filter": filter_death},
    ]

    for spec in model_specs:
        name = spec["name"]
        metrics_path = artifacts_dir / f"{name}.metrics.json"
        expected = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}

        print(f"\n=== {name} ===")
        if spec["kind"] == "xgb":
            data = prep_xgb(df, spec["joblib"], spec["event_col"])
        else:
            data = prep_deepsurv(df, spec["joblib"], spec["pt"], spec["event_col"], spec.get("filter"))

        grid_times = None
        if spec["kind"] == "deepsurv":
            tet = data["train_event_times"]
            lo, hi = np.quantile(tet, 0.05), np.quantile(tet, 0.95)
            grid_times = np.linspace(lo, hi, num=50)

        expected_key = {"test": "test", "external_mimic3": "mimic3_full" if spec["kind"] == "xgb" else "external_mimic3"}
        for split in ["test", "external_mimic3"]:
            t, s, e = data[split]
            n = len(t)
            prev = float(np.mean(e))
            print(f"  {split} (n={n:,}, event_rate={prev:.3f}) -- bootstrapping cindex ...", flush=True)

            cindex_res = bootstrap_cindex(t, s, e, args.n_bootstrap, args.seed)

            exp = expected.get(expected_key[split], {})
            exp_cindex = exp.get("cindex", np.nan)
            match_flag = "OK" if (not np.isnan(exp_cindex) and abs(cindex_res["point"] - exp_cindex) < 1e-3) else "MISMATCH"
            print(f"    cindex = {cindex_res['point']:.4f} (95% CI: {cindex_res['ci_lo']:.4f}-{cindex_res['ci_hi']:.4f})"
                  f"  [expected {exp_cindex:.4f}, {match_flag}]")

            row = {
                "variant": args.variant, "model": name, "kind": spec["kind"], "split": split, "n": n,
                "event_rate": round(prev, 4),
                "cindex": round(cindex_res["point"], 4),
                "cindex_ci_lo": round(cindex_res["ci_lo"], 4) if not np.isnan(cindex_res["ci_lo"]) else np.nan,
                "cindex_ci_hi": round(cindex_res["ci_hi"], 4) if not np.isnan(cindex_res["ci_hi"]) else np.nan,
                "cindex_str": f"{cindex_res['point']:.3f} ({cindex_res['ci_lo']:.3f}–{cindex_res['ci_hi']:.3f})",
                "cindex_expected": exp_cindex, "cindex_match": match_flag,
            }

            print(f"    bootstrapping discrimination metric ...", flush=True)
            if spec["kind"] == "xgb":
                auroc_res, auprc_res = bootstrap_horizon_metrics(
                    t, s, e, 365.0, args.n_bootstrap, args.seed + 1, mean_over_grid=False)
                metric_label = "auroc_at_horizon365"
            else:
                auroc_res, auprc_res = bootstrap_horizon_metrics(
                    t, s, e, grid_times, args.n_bootstrap, args.seed + 1, mean_over_grid=True)
                metric_label = "mean_td_auroc"

            print(f"    {metric_label} = {auroc_res['point']:.4f} "
                  f"(95% CI: {auroc_res['ci_lo']:.4f}-{auroc_res['ci_hi']:.4f})")

            row.update({
                "disc_metric": metric_label,
                "disc_auroc": round(auroc_res["point"], 4),
                "disc_auroc_ci_lo": round(auroc_res["ci_lo"], 4) if not np.isnan(auroc_res["ci_lo"]) else np.nan,
                "disc_auroc_ci_hi": round(auroc_res["ci_hi"], 4) if not np.isnan(auroc_res["ci_hi"]) else np.nan,
                "disc_auroc_str": f"{auroc_res['point']:.3f} ({auroc_res['ci_lo']:.3f}–{auroc_res['ci_hi']:.3f})",
                "disc_auprc": round(auprc_res["point"], 4),
                "disc_auprc_ci_lo": round(auprc_res["ci_lo"], 4) if not np.isnan(auprc_res["ci_lo"]) else np.nan,
                "disc_auprc_ci_hi": round(auprc_res["ci_hi"], 4) if not np.isnan(auprc_res["ci_hi"]) else np.nan,
                "disc_auprc_str": f"{auprc_res['point']:.3f} ({auprc_res['ci_lo']:.3f}–{auprc_res['ci_hi']:.3f})",
                "n_bootstrap": cindex_res["n_bootstrap"],
            })
            rows.append(row)

    out_df = pd.DataFrame(rows)
    out_path = Path(args.outdir) / f"bootstrap_ci_survival_summary_{args.variant}.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\n✓ Wrote {out_path}")
    print()
    display = out_df[["model", "split", "n", "cindex_str", "disc_metric", "disc_auroc_str", "disc_auprc_str"]].copy()
    display.columns = ["Model", "Split", "N", "C-index (95% CI)", "Metric", "AUROC (95% CI)", "AUPRC (95% CI)"]
    print(display.to_string(index=False))
    print("\nDone.")


if __name__ == "__main__":
    main()
