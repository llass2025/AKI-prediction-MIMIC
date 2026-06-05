#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
bootstrap_ci.py

Compute bootstrap 95% confidence intervals for AUROC and AUPRC
for all AKI prediction models.

Method: percentile bootstrap (1000 resamples, with replacement).
Applies to test set and MIMIC-III external validation.

Inputs:
  --artifacts-dir   root artifacts directory (contains model subdirs)
  --n-bootstrap     number of bootstrap resamples (default: 1000)
  --seed            random seed (default: 42)
  --outdir          output directory (default: uncertainty_intervals/artifacts)

Outputs:
  bootstrap_ci_summary.csv   point estimate + 95% CI for AUROC and AUPRC
                              per model per split
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def bootstrap_metric(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn,
    n_bootstrap: int = 1000,
    seed: int = 42,
    ci: float = 0.95,
) -> dict:
    """
    Percentile bootstrap CI for a scalar metric.
    Returns point estimate, lower and upper CI bounds.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    point = float(metric_fn(y_true, y_score))

    boot_vals = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        ys = y_score[idx]
        if len(np.unique(yt)) < 2:
            continue  # skip degenerate resample
        try:
            boot_vals.append(float(metric_fn(yt, ys)))
        except Exception:
            continue

    if len(boot_vals) < 10:
        return {"point": point, "ci_lo": np.nan, "ci_hi": np.nan,
                "n_bootstrap": len(boot_vals)}

    alpha = (1 - ci) / 2
    ci_lo = float(np.percentile(boot_vals, alpha * 100))
    ci_hi = float(np.percentile(boot_vals, (1 - alpha) * 100))
    return {"point": point, "ci_lo": ci_lo, "ci_hi": ci_hi,
            "n_bootstrap": len(boot_vals)}


def bootstrap_auroc(y_true, y_score, n_bootstrap=1000, seed=42):
    return bootstrap_metric(y_true, y_score, roc_auc_score, n_bootstrap, seed)


def bootstrap_auprc(y_true, y_score, n_bootstrap=1000, seed=42):
    return bootstrap_metric(y_true, y_score, average_precision_score, n_bootstrap, seed)


# ---------------------------------------------------------------------------
# Per-model evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model_name: str,
    pred_csv: str,
    n_bootstrap: int,
    seed: int,
) -> list:
    df = pd.read_csv(pred_csv)

    # Normalize label column
    if "incident_aki_label" not in df.columns and "true" in df.columns:
        df = df.rename(columns={"true": "incident_aki_label"})

    df["incident_aki_label"] = pd.to_numeric(df["incident_aki_label"], errors="coerce")
    df["prob"] = pd.to_numeric(df["prob"], errors="coerce")
    df = df.dropna(subset=["incident_aki_label", "prob"])
    df["incident_aki_label"] = df["incident_aki_label"].astype(int)

    results = []
    for split in ["test", "mimic3_full"]:
        sub = df[df["split"] == split].copy()
        if len(sub) < 50 or len(sub["incident_aki_label"].unique()) < 2:
            print(f"  {split}: insufficient data — skipping")
            continue

        y_true  = sub["incident_aki_label"].values
        y_score = sub["prob"].values
        n       = len(sub)
        prev    = float(y_true.mean())

        print(f"  {split} (n={n:,}, prev={prev:.3f}) — bootstrapping ...", flush=True)

        auroc_res = bootstrap_auroc(y_true, y_score, n_bootstrap, seed)
        auprc_res = bootstrap_auprc(y_true, y_score, n_bootstrap, seed + 1)

        results.append({
            "model":        model_name,
            "split":        split,
            "n":            n,
            "prevalence":   round(prev, 4),
            # AUROC
            "auroc":        round(auroc_res["point"], 4),
            "auroc_ci_lo":  round(auroc_res["ci_lo"], 4),
            "auroc_ci_hi":  round(auroc_res["ci_hi"], 4),
            "auroc_str":    f"{auroc_res['point']:.3f} ({auroc_res['ci_lo']:.3f}–{auroc_res['ci_hi']:.3f})",
            # AUPRC
            "auprc":        round(auprc_res["point"], 4),
            "auprc_ci_lo":  round(auprc_res["ci_lo"], 4),
            "auprc_ci_hi":  round(auprc_res["ci_hi"], 4),
            "auprc_str":    f"{auprc_res['point']:.3f} ({auprc_res['ci_lo']:.3f}–{auprc_res['ci_hi']:.3f})",
            "n_bootstrap":  auroc_res["n_bootstrap"],
        })

        print(f"    AUROC = {auroc_res['point']:.3f} "
              f"(95% CI: {auroc_res['ci_lo']:.3f}–{auroc_res['ci_hi']:.3f})")
        print(f"    AUPRC = {auprc_res['point']:.3f} "
              f"(95% CI: {auprc_res['ci_lo']:.3f}–{auprc_res['ci_hi']:.3f})")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Bootstrap 95% CIs for AUROC and AUPRC."
    )
    ap.add_argument("--artifacts-dir", default="artifacts")
    ap.add_argument("--n-bootstrap",   type=int, default=1000)
    ap.add_argument("--seed",          type=int, default=42)
    ap.add_argument("--outdir",        default="uncertainty_intervals/artifacts")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    models = {
        "logreg":    os.path.join(args.artifacts_dir, "logreg_aki",    "logreg.predictions.csv"),
        "rf":        os.path.join(args.artifacts_dir, "rf_aki",        "rf.predictions.csv"),
        "xgb":       os.path.join(args.artifacts_dir, "xgb_aki",       "xgb.predictions.csv"),
        "dnn":       os.path.join(args.artifacts_dir, "dnn_aki",       "dnn.predictions.csv"),
        "selfatten": os.path.join(args.artifacts_dir, "selfatten_aki", "selfatten.predictions.csv"),
        "dcn":       os.path.join(args.artifacts_dir, "dcn_aki",       "dcn.predictions.csv"),
    }

    all_results = []
    for model_name, pred_csv in models.items():
        if not os.path.exists(pred_csv):
            print(f"⚠️  {model_name}: not found ({pred_csv}) — skipping")
            continue
        print(f"\nBootstrapping {model_name} (n={args.n_bootstrap}) ...")
        results = evaluate_model(model_name, pred_csv, args.n_bootstrap, args.seed)
        all_results.extend(results)

    if all_results:
        df = pd.DataFrame(all_results)
        out = os.path.join(args.outdir, "bootstrap_ci_summary.csv")
        df.to_csv(out, index=False)
        print(f"\n✓ Wrote {out}")
        print()
        # Pretty print
        display = df[["model", "split", "n", "auroc_str", "auprc_str"]].copy()
        display.columns = ["Model", "Split", "N", "AUROC (95% CI)", "AUPRC (95% CI)"]
        print(display.to_string(index=False))
    else:
        print("No results — check prediction CSVs exist.")

    print("\nDone.")


if __name__ == "__main__":
    main()
