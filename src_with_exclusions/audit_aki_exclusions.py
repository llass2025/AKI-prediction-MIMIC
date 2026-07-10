#!/usr/bin/env python3
"""
Audit AKI admission exclusions step by step.

For each exclusion filter applied in the src_with_exclusions AKI pipeline,
records every (hadm_id, subject_id) that was removed and why.

Output CSV columns:
  step       - step number (int)
  filter     - human-readable reason for removal
  hadm_id    - admission ID removed
  subject_id - patient ID

Usage:
  python src_with_exclusions/audit_aki_exclusions.py \
    --data-pkl data.pkl \
    --features features/features_all.parquet \
    --out src_with_exclusions/aki_exclusion_audit.csv
"""

import argparse
import pandas as pd
import numpy as np

AKI_PREFIXES = ["584", "3995", "5498", "N17"]
CKD_PREFIXES = ["5851","5852","5853","5854","5855","5859","N181","N182","N183","N184","N185","N189"]

def startswith_any(code, prefixes):
    return any(str(code).startswith(p) for p in prefixes)

def get_age_row_ts(row):
    if row["version"] == "mimic3":
        admittime = row["admittime"].to_pydatetime()
        dob = row["dob"].to_pydatetime()
        age = (admittime - dob).days / 365
        return 90 if age > 150 else age
    elif row["version"] == "mimic4":
        return row["anchor_age"]
    return np.nan

def extract_kidney_flags(df_diagnosis):
    df = df_diagnosis.copy()
    df["icd_code"] = df["icd_code"].astype(str)
    df["is_aki"] = df["icd_code"].apply(lambda x: int(startswith_any(x, AKI_PREFIXES)))
    df["is_ckd"] = df["icd_code"].apply(lambda x: int(startswith_any(x, CKD_PREFIXES)))
    if "seq_num" not in df.columns:
        df["seq_num"] = np.nan
    df["seq_num"] = pd.to_numeric(df["seq_num"], errors="coerce")
    return df.groupby(["subject_id", "hadm_id", "seq_num"])[["is_aki", "is_ckd"]].max().reset_index()


def record_removed(step: int, filter_name: str, removed: pd.DataFrame) -> pd.DataFrame:
    """Return a tidy DataFrame of removed (hadm_id, subject_id) rows."""
    cols = [c for c in ["hadm_id", "subject_id"] if c in removed.columns]
    out = removed[cols].drop_duplicates().copy()
    out.insert(0, "step", step)
    out.insert(1, "filter", filter_name)
    return out


def main():
    ap = argparse.ArgumentParser(description="Audit AKI admission exclusions")
    ap.add_argument("--data-pkl",  default="data.pkl",                        help="Path to data.pkl")
    ap.add_argument("--features",  default="features/features_all.parquet",   help="Path to features_all.parquet")
    ap.add_argument("--out",       default="src_with_exclusions/aki_exclusion_audit.csv", help="Output CSV path")
    args = ap.parse_args()

    print("Loading data.pkl ...")
    with open(args.data_pkl, "rb") as f:
        data_dict = pd.read_pickle(f)

    df_patients   = data_dict["patients"].copy()
    df_admissions = data_dict["admissions"].copy()
    df_diagnosis  = data_dict["diagnosis"].copy()

    records = []
    step = 0

    # ── Step 0: full admission universe ──────────────────────────────────────────
    df_adm_pat = pd.merge(
        df_admissions,
        df_patients[["subject_id", "version", "gender", "dob", "dod", "anchor_age"]],
        on=["subject_id", "version"],
        how="left",
        validate="many_to_one",
    )
    for c in ("admittime", "dischtime", "dob", "dod"):
        if c in df_adm_pat.columns:
            df_adm_pat[c] = pd.to_datetime(df_adm_pat[c], errors="coerce")

    df_adm_pat["age"] = df_adm_pat[
        ["version", "admittime", "dob", "anchor_age"]
    ].apply(get_age_row_ts, axis=1)

    universe = df_adm_pat[["hadm_id", "subject_id"]].drop_duplicates()
    print(f"Total admissions: {len(universe):,}  |  subjects: {universe['subject_id'].nunique():,}")

    # ── Step 1: age < 18 ─────────────────────────────────────────────────────────
    step += 1
    removed_age = df_adm_pat[df_adm_pat["age"] < 18.0]
    records.append(record_removed(step, "age < 18", removed_age))
    df_adults = df_adm_pat[df_adm_pat["age"] >= 18.0].copy()
    print(f"Step {step} (age < 18):          removed {len(records[-1]):,} admissions, {records[-1]['subject_id'].nunique():,} subjects")

    # ── Step 2: primary AKI (presenting dx, seq_num == 1) ────────────────────────
    step += 1
    df_kdx = extract_kidney_flags(df_diagnosis)
    primary_aki_hadms = (
        df_kdx[(df_kdx["is_aki"] == 1) & (df_kdx["seq_num"] == 1.0)][["subject_id", "hadm_id"]]
        .drop_duplicates()
    )
    removed_primary = df_adults.merge(primary_aki_hadms, on=["subject_id", "hadm_id"], how="inner")
    records.append(record_removed(step, "primary AKI (presenting dx, seq_num=1)", removed_primary))
    adults_no_primary = df_adults[
        ~df_adults.set_index(["subject_id", "hadm_id"]).index.isin(
            primary_aki_hadms.set_index(["subject_id", "hadm_id"]).index
        )
    ]
    print(f"Step {step} (primary AKI):       removed {len(records[-1]):,} admissions, {records[-1]['subject_id'].nunique():,} subjects")

    # ── Step 3: no kidney ICD flag (dropped by inner merge with hadm_flags) ──────
    step += 1
    not_primary_mask = ~df_kdx.set_index(["subject_id", "hadm_id"]).index.isin(
        primary_aki_hadms.set_index(["subject_id", "hadm_id"]).index
    )
    hadm_flags = (
        df_kdx[not_primary_mask]
        .groupby(["subject_id", "hadm_id"])[["is_aki", "is_ckd"]]
        .max()
        .reset_index()
    )
    flagged_hadms = set(zip(hadm_flags["subject_id"], hadm_flags["hadm_id"]))
    removed_no_flag = adults_no_primary[
        ~adults_no_primary.apply(lambda r: (r["subject_id"], r["hadm_id"]) in flagged_hadms, axis=1)
    ]
    records.append(record_removed(step, "no AKI/CKD ICD flag (not in hadm_flags)", removed_no_flag))
    adults_flagged = adults_no_primary.merge(hadm_flags, on=["subject_id", "hadm_id"], how="inner")
    print(f"Step {step} (no kidney flag):    removed {len(records[-1]):,} admissions, {records[-1]['subject_id'].nunique():,} subjects")

    # ── Step 4: cmb_ckd == 1 (from features parquet) ────────────────────────────
    step += 1
    feat = pd.read_parquet(
        args.features,
        columns=["subject_id", "hadm_id", "cmb_ckd", "renal_impaired_at_adm"],
        engine="pyarrow",
    )
    cmb_ckd_sids = set(feat.loc[feat["cmb_ckd"] == 1, "subject_id"])
    removed_cmb = adults_flagged[adults_flagged["subject_id"].isin(cmb_ckd_sids)]
    records.append(record_removed(step, "cmb_ckd == 1 (concurrent AKI+CKD)", removed_cmb))
    after_cmb = adults_flagged[~adults_flagged["subject_id"].isin(cmb_ckd_sids)]
    print(f"Step {step} (cmb_ckd):           removed {len(records[-1]):,} admissions, {records[-1]['subject_id'].nunique():,} subjects")

    # ── Step 5: renal_impaired_at_adm == 1 (from features parquet) ──────────────
    step += 1
    renal_imp_sids = set(feat.loc[feat["renal_impaired_at_adm"] == 1, "subject_id"])
    removed_renal = after_cmb[after_cmb["subject_id"].isin(renal_imp_sids)]
    records.append(record_removed(step, "renal_impaired_at_adm == 1 (prior CKD history)", removed_renal))
    final_cohort = after_cmb[~after_cmb["subject_id"].isin(renal_imp_sids)]
    print(f"Step {step} (renal_impaired):    removed {len(records[-1]):,} admissions, {records[-1]['subject_id'].nunique():,} subjects")

    # ── Step 6: deduplication — first AKI, last non-AKI per subject ─────────────
    step += 1
    final_cohort = final_cohort.copy()
    final_cohort["admittime"] = pd.to_datetime(final_cohort["admittime"], errors="coerce")

    def pick_kept(g):
        aki = g[g["is_aki"] == 1].sort_values("admittime")
        if not aki.empty:
            return aki.iloc[:1]   # subject had AKI: keep first AKI admission only
        return g.sort_values("admittime").iloc[-1:]  # no AKI: keep last admission

    kept = (
        final_cohort.groupby("subject_id", group_keys=False)
        .apply(pick_kept)
        .reset_index(drop=True)
    )
    removed_dedup = final_cohort[~final_cohort.index.isin(kept.index)]
    records.append(record_removed(step, "dedup: not first AKI or last non-AKI per subject", removed_dedup))
    print(f"Step {step} (dedup):             removed {len(records[-1]):,} admissions, {records[-1]['subject_id'].nunique():,} subjects")

    modeling_cohort = kept
    print(f"\nModeling cohort: {len(modeling_cohort):,} admissions, {modeling_cohort['subject_id'].nunique():,} subjects")
    print(f"  AKI (is_aki==1): {(modeling_cohort['is_aki']==1).sum():,}")
    print(f"  No AKI (is_aki==0): {(modeling_cohort['is_aki']==0).sum():,}")

    # ── Summary ───────────────────────────────────────────────────────────────────
    print(f"\nPre-dedup cohort: {len(final_cohort):,} admissions, {final_cohort['subject_id'].nunique():,} subjects")

    # ── Write output ──────────────────────────────────────────────────────────────
    audit = pd.concat(records, ignore_index=True)
    audit.to_csv(args.out, index=False)
    print(f"\nAudit CSV written -> {args.out}")
    print(f"Total rows: {len(audit):,}  (one row per removed admission per step)")

    # ── Per-step summary ──────────────────────────────────────────────────────────
    summary = (
        audit.groupby(["step", "filter"])
        .agg(admissions_removed=("hadm_id", "nunique"), subjects_removed=("subject_id", "nunique"))
        .reset_index()
    )
    print("\n" + summary.to_string(index=False))


if __name__ == "__main__":
    main()
