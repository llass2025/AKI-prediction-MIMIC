#!/usr/bin/env python3
"""
Bootstrap 95% CIs for AUROC, AUPRC, and threshold-based metrics
(precision, sensitivity/recall, specificity, accuracy, F1) using
aki-analysis prediction CSVs.

Threshold: Youden's index (maximises sensitivity + specificity) derived
from the training split of each model, then applied to test / MIMIC-III.

Saves to sensitivity_paper_version/bootstrap_ci/artifacts/ — does not
touch the primary uncertainty_intervals/artifacts/ results.
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             roc_curve, precision_score, recall_score,
                             f1_score, accuracy_score)


def youden_threshold(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    idx = np.argmax(tpr - fpr)
    return float(thresholds[idx])


def threshold_metrics(y_true, y_score, threshold):
    y_pred = (y_score >= threshold).astype(int)
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    return {
        "precision":   float(precision_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": specificity,
        "accuracy":    float(accuracy_score(y_true, y_pred)),
        "f1":          float(f1_score(y_true, y_pred, zero_division=0)),
    }


def bootstrap_metric(y_true, y_score, metric_fn, n_bootstrap=1000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    point = float(metric_fn(y_true, y_score))
    boot_vals = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        try:
            boot_vals.append(float(metric_fn(yt, ys)))
        except Exception:
            continue
    if len(boot_vals) < 10:
        return {"point": point, "ci_lo": np.nan, "ci_hi": np.nan, "n_bootstrap": len(boot_vals)}
    return {"point": point,
            "ci_lo": float(np.percentile(boot_vals, 2.5)),
            "ci_hi": float(np.percentile(boot_vals, 97.5)),
            "n_bootstrap": len(boot_vals)}


def bootstrap_threshold_metrics(y_true, y_score, threshold, n_bootstrap=1000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    point = threshold_metrics(y_true, y_score, threshold)
    boots = {k: [] for k in point}
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        m = threshold_metrics(yt, ys, threshold)
        for k in boots:
            boots[k].append(m[k])
    results = {}
    for k, vals in boots.items():
        if len(vals) < 10:
            results[k] = {"point": point[k], "ci_lo": np.nan, "ci_hi": np.nan}
        else:
            results[k] = {"point": point[k],
                          "ci_lo": float(np.percentile(vals, 2.5)),
                          "ci_hi": float(np.percentile(vals, 97.5))}
    return results


def fmt(d):
    return f"{d['point']:.3f} ({d['ci_lo']:.3f}–{d['ci_hi']:.3f})"


def evaluate(model_name, pred_csv, n_bootstrap, seed):
    df = pd.read_csv(pred_csv)
    if "incident_aki_label" not in df.columns and "true" in df.columns:
        df = df.rename(columns={"true": "incident_aki_label"})
    df["incident_aki_label"] = pd.to_numeric(df["incident_aki_label"], errors="coerce")
    df["prob"] = pd.to_numeric(df["prob"], errors="coerce")
    df = df.dropna(subset=["incident_aki_label", "prob"])
    df["incident_aki_label"] = df["incident_aki_label"].astype(int)

    # Derive threshold from training split
    train = df[df["split"] == "train"]
    if len(train) > 0 and len(train["incident_aki_label"].unique()) == 2:
        threshold = youden_threshold(train["incident_aki_label"].values, train["prob"].values)
    else:
        threshold = 0.5
    print(f"  Youden threshold: {threshold:.3f}")

    results = []
    for split in ["test", "mimic3_full"]:
        sub = df[df["split"] == split]
        if len(sub) < 50 or len(sub["incident_aki_label"].unique()) < 2:
            print(f"  {split}: insufficient data — skipping")
            continue
        y_true, y_score = sub["incident_aki_label"].values, sub["prob"].values
        print(f"  {split} (n={len(sub):,}, prev={y_true.mean():.3f}) ...", flush=True)

        ar = bootstrap_metric(y_true, y_score, roc_auc_score,            n_bootstrap, seed)
        pr = bootstrap_metric(y_true, y_score, average_precision_score,  n_bootstrap, seed + 1)
        tm = bootstrap_threshold_metrics(y_true, y_score, threshold,     n_bootstrap, seed + 2)

        row = {
            "model": model_name, "split": split, "n": len(sub),
            "prevalence": round(float(y_true.mean()), 4),
            "threshold": round(threshold, 4),
            "auroc":        round(ar["point"], 4),
            "auroc_ci_lo":  round(ar["ci_lo"], 4), "auroc_ci_hi": round(ar["ci_hi"], 4),
            "auroc_str":    fmt(ar),
            "auprc":        round(pr["point"], 4),
            "auprc_ci_lo":  round(pr["ci_lo"], 4), "auprc_ci_hi": round(pr["ci_hi"], 4),
            "auprc_str":    fmt(pr),
            "n_bootstrap":  ar["n_bootstrap"],
        }
        for metric in ["precision", "sensitivity", "specificity", "accuracy", "f1"]:
            d = tm[metric]
            row[metric]          = round(d["point"], 4)
            row[f"{metric}_ci_lo"] = round(d["ci_lo"], 4)
            row[f"{metric}_ci_hi"] = round(d["ci_hi"], 4)
            row[f"{metric}_str"]   = fmt(d)

        results.append(row)

        print(f"    AUROC       = {fmt(ar)}")
        print(f"    AUPRC       = {fmt(pr)}")
        print(f"    Sensitivity = {fmt(tm['sensitivity'])}")
        print(f"    Specificity = {fmt(tm['specificity'])}")
        print(f"    Precision   = {fmt(tm['precision'])}")
        print(f"    Accuracy    = {fmt(tm['accuracy'])}")
        print(f"    F1          = {fmt(tm['f1'])}")

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aki-analysis-dir", required=True)
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default="sensitivity_paper_version/bootstrap_ci/artifacts")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    base = os.path.expanduser(args.aki_analysis_dir)

    models = {
        "logreg":    os.path.join(base, "logreg_model",                "logreg.predictions.csv"),
        "rf":        os.path.join(base, "rf_model",                    "rf.predictions.csv"),
        "xgb":       os.path.join(base, "xgboost_model",               "xgb.predictions.csv"),
        "dnn":       os.path.join(base, "dnn_model",                   "dnn.predictions.csv"),
        "selfatten": os.path.join(base, "multiheaded_selfatten_model",  "selfatten.predictions.csv"),
        "dcn":       os.path.join(base, "dcn_model",                   "dcn.predictions.csv"),
    }

    all_results = []
    for model_name, pred_csv in models.items():
        if not os.path.exists(pred_csv):
            print(f"⚠️  {model_name}: not found ({pred_csv}) — skipping")
            continue
        print(f"\nBootstrapping {model_name} ...")
        all_results.extend(evaluate(model_name, pred_csv, args.n_bootstrap, args.seed))

    if all_results:
        df = pd.DataFrame(all_results)
        out = os.path.join(args.outdir, "bootstrap_ci_summary.csv")
        df.to_csv(out, index=False)
        print(f"\n✓ Saved {out}\n")
        cols = ["model", "split", "n", "auroc_str", "auprc_str",
                "sensitivity_str", "specificity_str", "precision_str", "accuracy_str", "f1_str"]
        display = df[cols].copy()
        display.columns = ["Model", "Split", "N", "AUROC", "AUPRC",
                           "Sensitivity", "Specificity", "Precision", "Accuracy", "F1"]
        print(display.to_string(index=False))
    else:
        print("No results — check prediction CSVs.")
    print("\nDone.")


if __name__ == "__main__":
    main()
