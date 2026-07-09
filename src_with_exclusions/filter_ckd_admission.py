#!/usr/bin/env python3
"""
Upstream exclusion step for the CKD survival pipeline.

Removes patients with cmb_ckd==1 (concurrent AKI+CKD at index admission) or
renal_impaired_at_adm==1 (prior CKD history) from incident_ckd_admission.csv
BEFORE incident_ckd_target.py collapses it to patient-level survival data.
These flags are read from the reference features parquet (built once from the
full cohort) because they are not available at target-computation time.

Inputs:
  --adm      incident_ckd_admission.csv produced by ckd_survival compute_target_admissions.py
  --features features/features_all.parquet (reference; must contain cmb_ckd and
             renal_impaired_at_adm columns)
  --out      output path for the filtered CSV (default: overwrites --adm)

Outputs:
  Filtered CSV at --out (patients with cmb_ckd==1 or renal_impaired_at_adm==1 removed)
"""

import argparse
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description="Exclude cmb_ckd/renal_impaired patients from CKD admission CSV")
    ap.add_argument("--adm",      required=True, help="Path to incident_ckd_admission.csv")
    ap.add_argument("--features", required=True, help="Path to features_all.parquet (reference cohort)")
    ap.add_argument("--out",      default=None,  help="Output path (default: overwrite --adm)")
    args = ap.parse_args()

    out_path = args.out or args.adm

    adm = pd.read_csv(args.adm)
    print(f"CKD admission before exclusion: {adm['subject_id'].nunique():,} patients, {len(adm):,} rows")

    feat = pd.read_parquet(
        args.features,
        columns=["subject_id", "cmb_ckd", "renal_impaired_at_adm"],
        engine="fastparquet",
    )
    excl_cmb_ckd        = set(feat.loc[feat["cmb_ckd"] == 1,               "subject_id"])
    excl_renal_impaired = set(feat.loc[feat["renal_impaired_at_adm"] == 1,  "subject_id"])
    excl_sids = excl_cmb_ckd | excl_renal_impaired
    print(f"  cmb_ckd==1:              {len(excl_cmb_ckd):,} patients")
    print(f"  renal_impaired_at_adm==1:{len(excl_renal_impaired):,} patients")
    print(f"  union (excluded):        {len(excl_sids):,} patients")

    filtered = adm[~adm["subject_id"].isin(excl_sids)]
    print(f"CKD admission after exclusion:  {filtered['subject_id'].nunique():,} patients, {len(filtered):,} rows")

    filtered.to_csv(out_path, index=False)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
