#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
calibration_utils.py

Compute and plot model calibration for all AKI prediction models.

Metrics:
  - Brier score & Brier Skill Score (vs. no-skill baseline)
  - Expected Calibration Error (ECE, 10 equal-width bins)
  - Hosmer-Lemeshow goodness-of-fit (10 bins)
  - Calibration plots (reliability diagrams) — overall and sex-stratified

Inputs:
  Reads *.predictions.csv from each model's artifacts directory.
  Expected columns: subject_id, hadm_id, incident_aki_label, split, prob
  Optional: dem_sex_F column for sex-stratified calibration (joined from features)

Outputs (all in --outdir):
  calibration_summary.csv          — all metrics across models and splits
  cal_<model>_<split>.png          — calibration plot per model per split
  cal_<model>_sex_test.png         — sex-stratified calibration plot (test set)
"""

import os
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(np.mean((y_prob - y_true) ** 2))


def brier_skill_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    prev = y_true.mean()
    brier_ref = prev * (1 - prev)  # no-skill: always predict prevalence
    bs = brier_score(y_true, y_prob)
    return float(1.0 - bs / brier_ref) if brier_ref > 0 else np.nan


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() == 0:
            continue
        frac = mask.sum() / n
        obs = y_true[mask].mean()
        pred = y_prob[mask].mean()
        ece += frac * abs(obs - pred)
    return float(ece)


def hosmer_lemeshow(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> dict:
    """Hosmer-Lemeshow goodness-of-fit test (decile-of-risk)."""
    order = np.argsort(y_prob)
    y_true_s = y_true[order]
    y_prob_s = y_prob[order]
    groups = np.array_split(np.arange(len(y_true_s)), n_bins)
    hl_stat = 0.0
    for g in groups:
        obs_pos  = y_true_s[g].sum()
        exp_pos  = y_prob_s[g].sum()
        obs_neg  = len(g) - obs_pos
        exp_neg  = len(g) - exp_pos
        if exp_pos > 0:
            hl_stat += (obs_pos - exp_pos) ** 2 / exp_pos
        if exp_neg > 0:
            hl_stat += (obs_neg - exp_neg) ** 2 / exp_neg
    df = n_bins - 2
    p_val = 1 - stats.chi2.cdf(hl_stat, df)
    return {"hl_stat": float(hl_stat), "hl_df": df, "hl_pvalue": float(p_val)}


def calibration_curve(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> pd.DataFrame:
    bins = np.linspace(0, 1, n_bins + 1)
    rows = []
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() == 0:
            continue
        rows.append({
            "bin_low": bins[i],
            "bin_high": bins[i + 1],
            "n": int(mask.sum()),
            "mean_pred": float(y_prob[mask].mean()),
            "obs_rate": float(y_true[mask].mean()),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_calibration(
    curve_df: pd.DataFrame,
    title: str,
    out_path: str,
    brier: float,
    ece: float,
    hl_p: float,
    prevalence: float,
):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    ax.plot(
        curve_df["mean_pred"], curve_df["obs_rate"],
        "o-", color="#2E74B5", lw=2, ms=6, label="Model"
    )
    ax.axhline(prevalence, color="gray", lw=1, ls=":", label=f"Prevalence ({prevalence:.2f})")
    ax.set_xlabel("Mean predicted probability", fontsize=11)
    ax.set_ylabel("Observed event rate", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.text(
        0.98, 0.02,
        f"Brier={brier:.3f}  ECE={ece:.3f}\nHL p={hl_p:.3f}",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
    )
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_calibration_bysex(
    df: pd.DataFrame, title: str, out_path: str
):
    """Sex-stratified calibration plot."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    colors = {"Female": "#C00000", "Male": "#2E74B5"}

    for ax, (sex_label, sex_col) in zip(axes, [("Female", "F"), ("Male", "M")]):
        mask = df["sex"] == sex_col
        sub = df[mask]
        if len(sub) < 20:
            ax.set_title(f"{sex_label} (insufficient data)")
            continue

        y_true = sub["incident_aki_label"].values.astype(int)
        y_prob = sub["prob"].values

        curve = calibration_curve(y_true, y_prob)
        bs  = brier_score(y_true, y_prob)
        ece = expected_calibration_error(y_true, y_prob)
        hl  = hosmer_lemeshow(y_true, y_prob)
        prev = y_true.mean()

        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.plot(curve["mean_pred"], curve["obs_rate"], "o-",
                color=colors[sex_label], lw=2, ms=6)
        ax.axhline(prev, color="gray", lw=1, ls=":")
        ax.set_xlabel("Mean predicted probability", fontsize=10)
        ax.set_title(sex_label, fontsize=11, fontweight="bold")
        ax.text(0.98, 0.02,
                f"Brier={bs:.3f}  ECE={ece:.3f}\nHL p={hl['hl_pvalue']:.3f}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    axes[0].set_ylabel("Observed event rate", fontsize=10)
    fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-model calibration
# ---------------------------------------------------------------------------

def evaluate_model_calibration(
    model_name: str,
    pred_csv: str,
    features_parquet: str,
    outdir: str,
    n_bins: int = 10,
) -> list:
    """
    Load predictions CSV, compute calibration metrics for test and mimic3_full splits.
    Returns list of result dicts.
    """
    df = pd.read_csv(pred_csv)
    # Normalize label column — deep models use 'true', others use 'incident_aki_label'
    if "incident_aki_label" not in df.columns and "true" in df.columns:
        df = df.rename(columns={"true": "incident_aki_label"})
    df["incident_aki_label"] = pd.to_numeric(df["incident_aki_label"], errors="coerce")
    df = df.dropna(subset=["incident_aki_label", "prob"])
    df["incident_aki_label"] = df["incident_aki_label"].astype(int)

    # Try to join sex column from features
    has_sex = False
    if features_parquet and os.path.exists(features_parquet):
        try:
            feats = pd.read_parquet(features_parquet, columns=["subject_id", "hadm_id", "dem_sex_F"])
            join_cols = [c for c in ["subject_id", "hadm_id"] if c in df.columns]
            if join_cols:
                df = df.merge(feats, on=join_cols, how="left")
                df["sex"] = np.where(df["dem_sex_F"] == 1, "F", "M")
                has_sex = True
        except Exception:
            pass

    results = []
    for split in ["test", "mimic3_full"]:
        sub = df[df["split"] == split].copy()
        if len(sub) < 20:
            continue

        y_true = sub["incident_aki_label"].values
        y_prob = sub["prob"].values
        prev   = float(y_true.mean())

        bs   = brier_score(y_true, y_prob)
        bss  = brier_skill_score(y_true, y_prob)
        ece  = expected_calibration_error(y_true, y_prob, n_bins)
        hl   = hosmer_lemeshow(y_true, y_prob, n_bins)
        curve = calibration_curve(y_true, y_prob, n_bins)

        results.append({
            "model": model_name,
            "split": split,
            "n": len(sub),
            "prevalence": round(prev, 4),
            "brier_score": round(bs, 4),
            "brier_skill_score": round(bss, 4),
            "ece": round(ece, 4),
            "hl_stat": round(hl["hl_stat"], 3),
            "hl_pvalue": round(hl["hl_pvalue"], 4),
        })

        plot_calibration(
            curve_df=curve,
            title=f"{model_name} — {split}",
            out_path=os.path.join(outdir, f"cal_{model_name}_{split}.png"),
            brier=bs, ece=ece, hl_p=hl["hl_pvalue"], prevalence=prev,
        )

        # Sex-stratified (test set only)
        if split == "test" and has_sex:
            plot_calibration_bysex(
                df=sub,
                title=f"{model_name} — {split} (by sex)",
                out_path=os.path.join(outdir, f"cal_{model_name}_sex_{split}.png"),
            )

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Calibration assessment for AKI prediction models.")
    ap.add_argument("--artifacts-dir", default="artifacts",
                    help="Root artifacts directory containing model subdirs.")
    ap.add_argument("--features-parquet", default="features/features_all.parquet",
                    help="Features parquet for joining sex column.")
    ap.add_argument("--outdir", default="calibration/artifacts")
    ap.add_argument("--n-bins", type=int, default=10)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Model prediction CSV paths
    models = {
        "logreg":   os.path.join(args.artifacts_dir, "logreg_aki",    "logreg.predictions.csv"),
        "rf":       os.path.join(args.artifacts_dir, "rf_aki",        "rf.predictions.csv"),
        "xgb":      os.path.join(args.artifacts_dir, "xgb_aki",       "xgb.predictions.csv"),
        "dnn":      os.path.join(args.artifacts_dir, "dnn_aki",       "dnn.predictions.csv"),
        "selfatten":os.path.join(args.artifacts_dir, "selfatten_aki", "selfatten.predictions.csv"),
        "dcn":      os.path.join(args.artifacts_dir, "dcn_aki",       "dcn.predictions.csv"),
    }

    all_results = []
    for model_name, pred_csv in models.items():
        if not os.path.exists(pred_csv):
            print(f"⚠️  {model_name}: predictions CSV not found ({pred_csv}) — skipping")
            continue
        print(f"Evaluating {model_name} ...")
        results = evaluate_model_calibration(
            model_name=model_name,
            pred_csv=pred_csv,
            features_parquet=args.features_parquet,
            outdir=args.outdir,
            n_bins=args.n_bins,
        )
        all_results.extend(results)
        for r in results:
            print(f"  {r['split']:12s}  Brier={r['brier_score']:.4f}  "
                  f"BSS={r['brier_skill_score']:.4f}  ECE={r['ece']:.4f}  "
                  f"HL-p={r['hl_pvalue']:.4f}")

    if all_results:
        summary_df = pd.DataFrame(all_results)
        out_csv = os.path.join(args.outdir, "calibration_summary.csv")
        summary_df.to_csv(out_csv, index=False)
        print(f"\n✓ Wrote {out_csv}")
        print()
        print(summary_df.to_string(index=False))
    else:
        print("No results computed — check that prediction CSVs exist.")

    print("\nDone.")


if __name__ == "__main__":
    main()
