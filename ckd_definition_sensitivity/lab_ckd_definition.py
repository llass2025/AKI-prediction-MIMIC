#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
lab_ckd_definition.py

Compute lab-based incident CKD after AKI using the 2021 race-free CKD-EPI
equation, as a sensitivity analysis against the ICD-code-based primary definition.

Lab-CKD criteria (mirrors KDIGO CKD diagnostic criteria):
  - At least two post-AKI serum creatinine measurements available
  - Both measurements yield eGFR < 60 mL/min/1.73m²
  - The two qualifying measurements are ≥90 days apart
  - Both measurements occur ≥90 days after the index AKI admission
  - No pre-existing lab-CKD (eGFR < 60 on two measurements ≥90d apart before AKI)

eGFR Formula:
  2021 CKD-EPI creatinine equation (race-free):
    eGFR = 142 × min(Scr/κ, 1)^α × max(Scr/κ, 1)^(−1.200) × 0.9938^Age × [1.012 if female]
    κ = 0.7 (female), 0.9 (male)
    α = −0.241 (female), −0.302 (male)

Inputs:
  --data-pkl        data.pkl
  --survival-csv    ckd_survival/incident_ckd_survival.csv (ICD-based, for comparison)
  --target          target.parquet (for demographics: age, sex)
  --outdir          ckd_definition_sensitivity/

Outputs:
  lab_ckd_events.parquet         per-patient: lab_ckd_flag, time_to_lab_ckd_days,
                                  n_post_aki_creat, has_sufficient_labs
  lab_ckd_summary.csv            concordance ICD vs lab-CKD by version and sex
  incident_ckd_survival_labckd.csv  survival dataset with lab-CKD as outcome
"""

import os
import argparse
import pickle
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# CKD-EPI 2021 (race-free)
# ---------------------------------------------------------------------------

def ckd_epi_2021(scr: float, age: float, is_female: bool) -> float:
    """
    2021 CKD-EPI creatinine equation (race-free).
    Returns eGFR in mL/min/1.73m².
    """
    if pd.isna(scr) or pd.isna(age) or scr <= 0 or age <= 0:
        return np.nan
    kappa = 0.7 if is_female else 0.9
    alpha = -0.241 if is_female else -0.302
    sex_factor = 1.012 if is_female else 1.0
    ratio = scr / kappa
    egfr = (142
            * (min(ratio, 1.0) ** alpha)
            * (max(ratio, 1.0) ** -1.200)
            * (0.9938 ** age)
            * sex_factor)
    return float(egfr)


def compute_egfr_series(labs: pd.DataFrame, age: float, is_female: bool) -> pd.DataFrame:
    """Apply CKD-EPI to a creatinine time series for one patient."""
    labs = labs.copy()
    labs["egfr"] = labs["valuenum"].apply(
        lambda scr: ckd_epi_2021(scr, age, is_female)
    )
    return labs


# ---------------------------------------------------------------------------
# Creatinine helpers (reuse pattern from pre_aki_lab_flags)
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
    keep = [c for c in ["subject_id", "hadm_id", "charttime", "valuenum"] if c in labs.columns]
    return labs[keep].sort_values(["subject_id", "charttime"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Pre-existing lab-CKD check
# ---------------------------------------------------------------------------

def has_pre_aki_lab_ckd(pre_labs: pd.DataFrame, age: float, is_female: bool,
                         min_gap_days: int = 90) -> bool:
    """
    Returns True if the patient had eGFR < 60 on two measurements ≥90 days
    apart BEFORE the AKI admission (pre-existing lab-CKD).
    """
    if pre_labs.empty or len(pre_labs) < 2:
        return False
    pre_labs = pre_labs.copy()
    pre_labs["egfr"] = pre_labs["valuenum"].apply(
        lambda s: ckd_epi_2021(s, age, is_female)
    )
    low = pre_labs[pre_labs["egfr"] < 60].sort_values("charttime")
    if len(low) < 2:
        return False
    for i in range(len(low)):
        for j in range(i + 1, len(low)):
            gap = (low.iloc[j]["charttime"] - low.iloc[i]["charttime"]).days
            if gap >= min_gap_days:
                return True
    return False


# ---------------------------------------------------------------------------
# Main lab-CKD detection per patient
# ---------------------------------------------------------------------------

def detect_lab_ckd(
    subject_id: int,
    index_admit: pd.Timestamp,
    age_at_index: float,
    is_female: bool,
    all_labs: pd.DataFrame,
    min_gap_days: int = 90,
    post_aki_lag_days: int = 90,
) -> dict:
    """
    For one AKI patient, detect lab-based incident CKD.

    Returns a dict with:
      lab_ckd_flag, time_to_lab_ckd_days,
      n_post_aki_creat, has_sufficient_labs,
      pre_existing_lab_ckd
    """
    result = {
        "subject_id": subject_id,
        "lab_ckd_flag": 0,
        "time_to_lab_ckd_days": np.nan,
        "n_post_aki_creat": 0,
        "has_sufficient_labs": 0,
        "pre_existing_lab_ckd": 0,
    }

    pt_labs = all_labs[all_labs["subject_id"] == subject_id].copy()
    if pt_labs.empty:
        return result

    # Pre-AKI labs
    pre = pt_labs[pt_labs["charttime"] < index_admit]
    if has_pre_aki_lab_ckd(pre, age_at_index, is_female, min_gap_days):
        result["pre_existing_lab_ckd"] = 1
        return result  # exclude pre-existing lab-CKD

    # Post-AKI labs: must be ≥ post_aki_lag_days after index
    cutoff = index_admit + pd.Timedelta(days=post_aki_lag_days)
    post = pt_labs[pt_labs["charttime"] >= cutoff].copy()
    result["n_post_aki_creat"] = len(post)

    if len(post) < 2:
        return result  # insufficient data

    result["has_sufficient_labs"] = 1

    # Compute eGFR and find first qualifying pair
    post = post.sort_values("charttime").copy()
    post["age_at_meas"] = age_at_index + (
        (post["charttime"] - index_admit).dt.days / 365.25
    )
    post["egfr"] = post.apply(
        lambda r: ckd_epi_2021(r["valuenum"], r["age_at_meas"], is_female), axis=1
    )

    low = post[post["egfr"] < 60].reset_index(drop=True)
    if len(low) < 2:
        return result

    # Find first pair ≥ min_gap_days apart
    for i in range(len(low)):
        for j in range(i + 1, len(low)):
            gap = (low.iloc[j]["charttime"] - low.iloc[i]["charttime"]).days
            if gap >= min_gap_days:
                result["lab_ckd_flag"] = 1
                result["time_to_lab_ckd_days"] = (
                    low.iloc[i]["charttime"] - index_admit
                ).days
                return result

    return result


# ---------------------------------------------------------------------------
# Build KDIGO-style survival CSV with lab-CKD outcome
# ---------------------------------------------------------------------------

def build_lab_ckd_survival(
    icd_survival: pd.DataFrame,
    lab_ckd_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Replace CKD/PostCKD event_type with lab-CKD determination.
    Patients with lab_ckd_flag=1 get event_type='LabCKD' and time_days=time_to_lab_ckd_days.
    Patients originally CKD/PostCKD but lab_ckd_flag=0 become Censor or Death depending on
    their original death status.
    Patients with pre_existing_lab_ckd=1 are flagged and excluded.
    """
    surv = icd_survival.merge(
        lab_ckd_df[["subject_id", "lab_ckd_flag", "time_to_lab_ckd_days",
                     "has_sufficient_labs", "pre_existing_lab_ckd"]],
        on="subject_id", how="left"
    )

    surv["lab_ckd_flag"] = surv["lab_ckd_flag"].fillna(0).astype(int)
    surv["pre_existing_lab_ckd"] = surv["pre_existing_lab_ckd"].fillna(0).astype(int)
    surv["has_sufficient_labs"] = surv["has_sufficient_labs"].fillna(0).astype(int)

    # Reassign event_type and time_days based on lab-CKD
    surv["event_type_labckd"] = surv["event_type"].copy()
    surv["time_days_labckd"] = surv["time_days"].copy()

    # Lab-CKD positive
    mask_labckd = surv["lab_ckd_flag"] == 1
    surv.loc[mask_labckd, "event_type_labckd"] = "LabCKD"
    surv.loc[mask_labckd, "time_days_labckd"] = surv.loc[mask_labckd, "time_to_lab_ckd_days"]

    # ICD-CKD but no lab-CKD → demote to Censor (if alive) or keep Death
    mask_icd_only = (
        surv["event_type"].isin(["CKD", "PostCKD"])
        & (surv["lab_ckd_flag"] == 0)
    )
    surv.loc[mask_icd_only, "event_type_labckd"] = "Censor"

    # Use lab-CKD outcome as primary
    surv["time_days"] = surv["time_days_labckd"]
    surv["event_type"] = surv["event_type_labckd"]
    surv["is_ckd"] = (surv["event_type"] == "LabCKD").astype(int)
    surv["is_postckd"] = 0
    surv["is_censored"] = (surv["event_type"] == "Censor").astype(int)
    surv["is_death"] = (surv["event_type"] == "Death").astype(int)

    return surv


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Compute lab-based CKD events using 2021 CKD-EPI eGFR."
    )
    ap.add_argument("--data-pkl",     default="data.pkl")
    ap.add_argument("--survival-csv", default="ckd_survival/incident_ckd_survival.csv")
    ap.add_argument("--target",       default="target.parquet")
    ap.add_argument("--outdir",       default="ckd_definition_sensitivity/")
    ap.add_argument("--min-gap-days", type=int, default=90,
                    help="Min days between two eGFR<60 measurements to confirm CKD (default: 90)")
    ap.add_argument("--post-aki-lag", type=int, default=90,
                    help="Min days post-AKI before CKD measurements count (default: 90)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────────
    print("Loading data.pkl ...")
    with open(args.data_pkl, "rb") as f:
        data = pickle.load(f)

    labevents = data["labevents"].copy()
    labitems  = data["labitems"].copy()

    print("Loading survival CSV and target ...")
    surv   = pd.read_csv(args.survival_csv)
    target = pd.read_parquet(args.target)

    # Demographics: age and sex per subject from first AKI admission
    demo = (
        target[target["is_aki"] == 1]
        .sort_values("admittime")
        .groupby("subject_id")
        .first()
        .reset_index()
    )[["subject_id", "age", "gender", "admittime"]]
    demo["is_female"] = demo["gender"].str.upper().isin(["F", "FEMALE"])
    to_datetime(demo, ["admittime"])

    # ── Creatinine labs ───────────────────────────────────────────────────
    print("Selecting creatinine labs ...")
    itemids = select_creatinine_itemids(labitems)
    labs = prep_creatinine_labs(labevents, itemids)
    print(f"  Creatinine events: {len(labs):,}")

    # ── Detect lab-CKD per patient ─────────────────────────────────────────
    patients = surv["subject_id"].unique()
    print(f"Processing {len(patients):,} patients ...")

    results = []
    for i, sid in enumerate(patients):
        if i % 5000 == 0:
            print(f"  {i:,} / {len(patients):,} ...", flush=True)

        row = demo[demo["subject_id"] == sid]
        if row.empty:
            results.append({
                "subject_id": sid, "lab_ckd_flag": 0,
                "time_to_lab_ckd_days": np.nan, "n_post_aki_creat": 0,
                "has_sufficient_labs": 0, "pre_existing_lab_ckd": 0,
            })
            continue

        row = row.iloc[0]
        res = detect_lab_ckd(
            subject_id    = int(sid),
            index_admit   = row["admittime"],
            age_at_index  = float(row["age"]),
            is_female     = bool(row["is_female"]),
            all_labs      = labs,
            min_gap_days  = args.min_gap_days,
            post_aki_lag_days = args.post_aki_lag,
        )
        results.append(res)

    lab_ckd_df = pd.DataFrame(results)

    # ── Save lab_ckd_events ───────────────────────────────────────────────
    # Attach version
    ver_map = demo[["subject_id"]].merge(
        target[["subject_id", "version"]].drop_duplicates(), on="subject_id", how="left"
    )
    lab_ckd_df = lab_ckd_df.merge(ver_map, on="subject_id", how="left")

    out_events = os.path.join(args.outdir, "lab_ckd_events.parquet")
    lab_ckd_df.to_parquet(out_events, index=False)
    print(f"\n✓ Wrote {out_events}")

    # ── Summary: concordance with ICD-CKD ─────────────────────────────────
    merge_cols = ["subject_id", "lab_ckd_flag", "time_to_lab_ckd_days",
                  "has_sufficient_labs", "pre_existing_lab_ckd", "version"]
    merge_cols = [c for c in merge_cols if c in lab_ckd_df.columns]
    merged = surv.merge(lab_ckd_df[merge_cols], on="subject_id", how="left",
                        suffixes=("", "_lab"))
    # resolve version column
    if "version" not in merged.columns and "version_lab" in merged.columns:
        merged = merged.rename(columns={"version_lab": "version"})
    elif "version_lab" in merged.columns:
        merged = merged.drop(columns=["version_lab"])
    merged["icd_ckd"] = merged["event_type"].isin(["CKD", "PostCKD"]).astype(int)

    print("\n=== ICD-CKD vs Lab-CKD Concordance ===")
    for ver in ["mimic3", "mimic4"]:
        sub = merged[merged["version"] == ver]
        if sub.empty:
            continue
        n = len(sub)
        icd_pos  = int(sub["icd_ckd"].sum())
        lab_pos  = int(sub["lab_ckd_flag"].sum())
        both     = int(((sub["icd_ckd"] == 1) & (sub["lab_ckd_flag"] == 1)).sum())
        icd_only = int(((sub["icd_ckd"] == 1) & (sub["lab_ckd_flag"] == 0)).sum())
        lab_only = int(((sub["icd_ckd"] == 0) & (sub["lab_ckd_flag"] == 1)).sum())
        neither  = int(((sub["icd_ckd"] == 0) & (sub["lab_ckd_flag"] == 0)).sum())
        insuf    = int((sub["has_sufficient_labs"] == 0).sum())
        pre_ckd  = int((sub["pre_existing_lab_ckd"] == 1).sum())
        print(f"\n{ver.upper()} (n={n:,}):")
        print(f"  ICD-CKD positive:     {icd_pos:,} ({icd_pos/n*100:.1f}%)")
        print(f"  Lab-CKD positive:     {lab_pos:,} ({lab_pos/n*100:.1f}%)")
        print(f"  Both positive:        {both:,}")
        print(f"  ICD-only:             {icd_only:,}")
        print(f"  Lab-only:             {lab_only:,}")
        print(f"  Neither:              {neither:,}")
        print(f"  Insuf. lab data:      {insuf:,}")
        print(f"  Pre-existing lab-CKD: {pre_ckd:,}")

    # Summary CSV
    summary_rows = []
    for ver in ["mimic3", "mimic4"]:
        sub = merged[merged["version"] == ver]
        if sub.empty:
            continue
        for sex in ["F", "M"]:
            s = sub[sub["gender"].str.upper().str.startswith(sex)]
            summary_rows.append({
                "version": ver, "sex": sex, "n": len(s),
                "icd_ckd_n": int(s["icd_ckd"].sum()),
                "lab_ckd_n": int(s["lab_ckd_flag"].sum()),
                "both_n":    int(((s["icd_ckd"]==1) & (s["lab_ckd_flag"]==1)).sum()),
                "icd_only_n": int(((s["icd_ckd"]==1) & (s["lab_ckd_flag"]==0)).sum()),
                "lab_only_n": int(((s["icd_ckd"]==0) & (s["lab_ckd_flag"]==1)).sum()),
                "insuf_labs_n": int((s["has_sufficient_labs"]==0).sum()),
            })
    summary_df = pd.DataFrame(summary_rows)
    out_summary = os.path.join(args.outdir, "lab_ckd_summary.csv")
    summary_df.to_csv(out_summary, index=False)
    print(f"\n✓ Wrote {out_summary}")

    # ── Build lab-CKD survival dataset ────────────────────────────────────
    print("\nBuilding lab-CKD survival dataset ...")
    surv_labckd = build_lab_ckd_survival(surv, lab_ckd_df)

    # Drop lab-specific columns that would leak into features
    leak_cols = ["lab_ckd_flag", "time_to_lab_ckd_days",
                 "has_sufficient_labs", "pre_existing_lab_ckd",
                 "event_type_labckd", "time_days_labckd"]
    surv_labckd = surv_labckd.drop(columns=[c for c in leak_cols if c in surv_labckd.columns])

    out_surv = os.path.join(args.outdir, "incident_ckd_survival_labckd.csv")
    surv_labckd.to_csv(out_surv, index=False)
    print(f"✓ Wrote {out_surv}  ({len(surv_labckd):,} rows)")
    print("\nEvent type distribution (lab-CKD):")
    print(surv_labckd["event_type"].value_counts())

    print("\nDone.")


if __name__ == "__main__":
    main()
