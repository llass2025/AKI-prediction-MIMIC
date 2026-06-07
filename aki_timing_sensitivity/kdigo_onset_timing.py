#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
kdigo_onset_timing.py

Compute the first in-admission creatinine timestamp that meets KDIGO AKI
criteria for each AKI admission, and produce a KDIGO-anchored survival
dataset for the post-AKI CKD/death sensitivity analysis.

KDIGO criteria (creatinine-based):
  (A) Absolute rise >= 0.3 mg/dL within any 48h window, OR
  (B) Rise to >= 1.5x patient baseline within 7 days of first in-admission value.

Baseline creatinine:
  Median of creatinine values in the 7 days prior to admittime
  (consistent with the primary analysis lookback window).
  If no prior labs available, the minimum in-admission value is used as fallback.

Scope:
  Applies ONLY to the post-AKI survival models (time-to-CKD, time-to-death).
  The AKI prediction model remains admission-anchored by design — see
  aki_timing_sensitivity.docx for full rationale.

Inputs:
  --data-pkl       data.pkl (contains labevents, labitems, admissions)
  --target         target.parquet (patient-level AKI cohort)
  --survival-csv   ckd_survival/incident_ckd_survival.csv (admission-anchored)
  --outdir         aki_timing_sensitivity/

Outputs:
  kdigo_onset_times.parquet        per-admission KDIGO onset times
  kdigo_onset_summary.csv          summary stats (coverage, median hours to onset)
  incident_ckd_survival_kdigo.csv  survival dataset re-anchored to KDIGO onset
"""

import os
import argparse
import pickle
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Reuse creatinine helpers from pre_aki_lab_flags
# ---------------------------------------------------------------------------

def to_datetime(df: pd.DataFrame, cols) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")


def select_creatinine_itemids(labitems: pd.DataFrame) -> set:
    li = labitems.copy()
    li.columns = [c.lower() for c in li.columns]
    mask = (
        li["label"].astype(str).str.contains(r"\bcreatinine\b", case=False, na=False)
        & ~li["label"].astype(str).str.contains(r"urine|ur\b", case=False, na=False)
    )
    return set(li.loc[mask, "itemid"].astype(int).unique().tolist())


def prep_creatinine_labs(labevents: pd.DataFrame, itemids: set) -> pd.DataFrame:
    labs = labevents.copy()
    labs.columns = [c.lower() for c in labs.columns]
    to_datetime(labs, ["charttime"])
    labs["valuenum"] = pd.to_numeric(labs.get("valuenum", np.nan), errors="coerce")
    labs = labs[labs["itemid"].isin(itemids)]
    labs = labs[labs["valuenum"].notna() & (labs["valuenum"] > 0) & labs["charttime"].notna()]
    keep = [c for c in ["version", "subject_id", "hadm_id", "charttime", "valuenum"] if c in labs.columns]
    return labs[keep].sort_values(["subject_id", "charttime"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Baseline creatinine: median in 7 days prior to admittime
# ---------------------------------------------------------------------------

def compute_baselines(admissions: pd.DataFrame, labs: pd.DataFrame) -> pd.DataFrame:
    """
    For each (subject_id, hadm_id), compute median creatinine in [admit-7d, admit).
    Falls back to NaN if no prior labs available.
    """
    adm = admissions[["version", "subject_id", "hadm_id", "admittime"]].copy()
    to_datetime(adm, ["admittime"])

    # merge all labs for the subject
    merged = adm.merge(
        labs[["subject_id", "charttime", "valuenum"]],
        on="subject_id", how="left"
    )
    window_start = merged["admittime"] - pd.Timedelta(days=7)
    in_window = (merged["charttime"] >= window_start) & (merged["charttime"] < merged["admittime"])
    prior = merged[in_window].copy()

    baseline = (
        prior.groupby(["subject_id", "hadm_id"])["valuenum"]
        .median()
        .rename("baseline_creat")
        .reset_index()
    )
    adm = adm.merge(baseline, on=["subject_id", "hadm_id"], how="left")
    return adm  # columns: version, subject_id, hadm_id, admittime, baseline_creat


# ---------------------------------------------------------------------------
# KDIGO onset: first timestamp meeting criteria within the admission
# ---------------------------------------------------------------------------

def kdigo_onset_for_admission(
    in_adm_labs: pd.DataFrame,
    baseline: float,
) -> pd.Timestamp | None:
    """
    Given creatinine values within one admission (sorted by charttime),
    return the first timestamp at which KDIGO criteria are met, or None.

    Criteria:
      (A) abs rise >= 0.3 mg/dL within any rolling 48h window, OR
      (B) value >= 1.5 * baseline (using pre-admission median; fallback = min in-admission)
    """
    g = in_adm_labs.sort_values("charttime").copy()
    if g.empty:
        return None

    s = g.set_index("charttime")["valuenum"].dropna()
    if s.empty:
        return None

    # Baseline fallback: minimum in-admission value
    if pd.isna(baseline):
        baseline = float(s.min())

    # Criterion A: rolling 48h absolute rise
    min_48h = s.rolling("48h").min()
    crit_a = (s - min_48h) >= 0.3

    # Criterion B: >= 1.5x baseline
    with np.errstate(divide="ignore", invalid="ignore"):
        crit_b = s >= (1.5 * baseline)

    met = crit_a | crit_b
    met_times = met[met].index

    return met_times.min() if len(met_times) > 0 else None


def compute_kdigo_onset_times(
    aki_admissions: pd.DataFrame,
    labs: pd.DataFrame,
    baselines: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each AKI admission, find the first creatinine timestamp meeting KDIGO criteria.

    Returns DataFrame with columns:
        version, subject_id, hadm_id, admittime, dischtime,
        kdigo_onset_time, hours_to_onset, kdigo_found
    """
    adm = aki_admissions.copy()
    to_datetime(adm, ["admittime", "dischtime"])

    # attach baselines
    adm = adm.merge(
        baselines[["subject_id", "hadm_id", "baseline_creat"]],
        on=["subject_id", "hadm_id"], how="left"
    )

    # merge labs into admissions — keep only in-admission labs
    merged = adm.merge(
        labs[["subject_id", "hadm_id", "charttime", "valuenum"]],
        on=["subject_id", "hadm_id"], how="left"
    )
    in_adm = (
        merged["charttime"].notna()
        & (merged["charttime"] >= merged["admittime"])
        & (
            merged["dischtime"].isna()
            | (merged["charttime"] <= merged["dischtime"])
        )
    )
    merged = merged[in_adm].copy()

    results = []
    for (sid, hadm), grp in merged.groupby(["subject_id", "hadm_id"], sort=False):
        row = adm[(adm["subject_id"] == sid) & (adm["hadm_id"] == hadm)].iloc[0]
        baseline = float(row.get("baseline_creat", np.nan))

        onset = kdigo_onset_for_admission(grp[["charttime", "valuenum"]], baseline)

        hours = (
            (onset - row["admittime"]).total_seconds() / 3600.0
            if onset is not None and pd.notna(row["admittime"])
            else np.nan
        )
        results.append({
            "version":          row["version"],
            "subject_id":       int(sid),
            "hadm_id":          int(hadm),
            "admittime":        row["admittime"],
            "dischtime":        row.get("dischtime", pd.NaT),
            "baseline_creat":   baseline,
            "kdigo_onset_time": onset,
            "hours_to_onset":   hours,
            "kdigo_found":      int(onset is not None),
        })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Re-anchor survival dataset to KDIGO onset time
# ---------------------------------------------------------------------------

def reanchor_survival(
    survival_csv: str,
    onset_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Load the admission-anchored survival CSV and shift time_days by the
    hours between admittime and kdigo_onset_time.

    Rows where kdigo_found == 0 are retained with a flag (kdigo_found=0)
    so downstream analysts can decide how to handle them.
    """
    surv = pd.read_csv(survival_csv)

    # merge onset info
    surv = surv.merge(
        onset_df[["subject_id", "hadm_id", "hours_to_onset", "kdigo_found"]],
        on=["subject_id", "hadm_id"] if "hadm_id" in surv.columns else ["subject_id"],
        how="left",
    )

    # shift time_days: subtract hours_to_onset/24 (KDIGO onset is later than admission)
    shift_days = surv["hours_to_onset"].fillna(0) / 24.0
    surv["time_days_kdigo"] = (surv["time_days"] - shift_days).clip(lower=0)
    surv["time_days_original"] = surv["time_days"]
    surv["time_days"] = surv["time_days_kdigo"]
    surv["kdigo_found"] = surv["kdigo_found"].fillna(0).astype(int)

    return surv


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Compute KDIGO onset times and re-anchor survival dataset."
    )
    ap.add_argument("--data-pkl",      default="data.pkl")
    ap.add_argument("--target",        default="target.parquet",
                    help="Patient-level AKI target (keepautocar or noautocar).")
    ap.add_argument("--survival-csv",  default="ckd_survival/incident_ckd_survival.csv",
                    help="Admission-anchored survival CSV from incident_ckd_target.py.")
    ap.add_argument("--outdir",        default="aki_timing_sensitivity/")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────
    print("Loading data.pkl ...")
    with open(args.data_pkl, "rb") as f:
        data = pickle.load(f)

    labevents  = data["labevents"].copy()
    labitems   = data["labitems"].copy()
    admissions = data["admissions"].copy()
    admissions.columns = [c.lower() for c in admissions.columns]
    to_datetime(admissions, ["admittime", "dischtime"])

    target = pd.read_parquet(args.target)
    aki_hadm = set(
        target.loc[target["is_aki"] == 1, "hadm_id"].astype(int).unique()
    )
    print(f"AKI admissions to process: {len(aki_hadm):,}")

    # ── Creatinine labs ────────────────────────────────────────────────────
    print("Selecting creatinine labs ...")
    itemids = select_creatinine_itemids(labitems)
    labs = prep_creatinine_labs(labevents, itemids)
    print(f"  Creatinine lab events: {len(labs):,}")

    # ── Baselines (7d prior) ───────────────────────────────────────────────
    print("Computing pre-admission baselines ...")
    aki_adm = admissions[admissions["hadm_id"].isin(aki_hadm)].copy()
    # attach version from target only if not already present
    if "version" not in aki_adm.columns:
        ver_map = target[["hadm_id", "version"]].drop_duplicates()
        aki_adm = aki_adm.merge(ver_map, on="hadm_id", how="left")
    baselines = compute_baselines(aki_adm, labs)

    # ── KDIGO onset ────────────────────────────────────────────────────────
    print("Computing KDIGO onset times (this may take a few minutes) ...")
    onset_df = compute_kdigo_onset_times(aki_adm, labs, baselines)

    n_found  = int(onset_df["kdigo_found"].sum())
    n_total  = len(onset_df)
    coverage = n_found / n_total * 100 if n_total > 0 else 0
    median_h = onset_df.loc[onset_df["kdigo_found"] == 1, "hours_to_onset"].median()
    print(f"  KDIGO onset found:   {n_found:,} / {n_total:,}  ({coverage:.1f}%)")
    print(f"  Median hours to onset (when found): {median_h:.1f}h")

    # ── Save onset times ───────────────────────────────────────────────────
    out_onset = os.path.join(args.outdir, "kdigo_onset_times.parquet")
    onset_df.to_parquet(out_onset, index=False)
    print(f"  Wrote {out_onset}")

    # Summary CSV
    summary = onset_df.groupby("version").agg(
        n_aki_admissions   = ("hadm_id", "count"),
        n_kdigo_found      = ("kdigo_found", "sum"),
        median_hours_onset = ("hours_to_onset", "median"),
        pct_within_24h     = ("hours_to_onset", lambda x: (x <= 24).mean() * 100),
        pct_within_48h     = ("hours_to_onset", lambda x: (x <= 48).mean() * 100),
    ).reset_index()
    out_summary = os.path.join(args.outdir, "kdigo_onset_summary.csv")
    summary.to_csv(out_summary, index=False)
    print(f"  Wrote {out_summary}")
    print()
    print(summary.to_string(index=False))

    # ── Re-anchor survival dataset ─────────────────────────────────────────
    if os.path.exists(args.survival_csv):
        print(f"\nRe-anchoring {args.survival_csv} to KDIGO onset times ...")
        surv_kdigo = reanchor_survival(args.survival_csv, onset_df)
        out_surv = os.path.join(args.outdir, "incident_ckd_survival_kdigo.csv")
        surv_kdigo.to_csv(out_surv, index=False)
        print(f"  Wrote {out_surv}  ({len(surv_kdigo):,} rows)")

        # quick comparison
        orig_med  = surv_kdigo["time_days_original"].median()
        kdigo_med = surv_kdigo["time_days_kdigo"].median()
        print(f"  Median time_days — original: {orig_med:.1f}d  |  KDIGO-anchored: {kdigo_med:.1f}d")
    else:
        print(f"\nWarning: {args.survival_csv} not found — skipping survival re-anchor.")
        print("Run ckd_survival/src/incident_ckd_target.py first.")

    print("\nDone.")


if __name__ == "__main__":
    main()
