#!/usr/bin/env python3
"""
Upstream exclusion step for the AKI prediction pipeline.

Removes patients with cmb_ckd==1 (concurrent AKI+CKD at index admission) or
renal_impaired_at_adm==1 (prior CKD history) from target_admissions.parquet
BEFORE feature engineering. These flags are read from the reference features
parquet (built once from the full cohort) because they are not available at
target-computation time.

Inputs:
  --target   target_admissions.parquet produced by compute_target_admissions.py
  --features features/features_all.parquet (reference; must contain cmb_ckd and
             renal_impaired_at_adm columns)
  --out      output path for the filtered target parquet (default: overwrites --target)

Outputs:
  Filtered parquet at --out (patients with cmb_ckd==1 or renal_impaired_at_adm==1 removed)
"""

import argparse
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description="Exclude cmb_ckd/renal_impaired patients from AKI target parquet")
    ap.add_argument("--target",   required=True, help="Path to target_admissions.parquet")
    ap.add_argument("--features", required=True, help="Path to features_all.parquet (reference cohort)")
    ap.add_argument("--out",      default=None,  help="Output path (default: overwrite --target)")
    args = ap.parse_args()

    out_path = args.out or args.target

    target = pd.read_parquet(args.target, engine="fastparquet")
    print(f"AKI target before exclusion: {target['subject_id'].nunique():,} patients, {len(target):,} rows")

    feat = pd.read_parquet(
        args.features,
        columns=["subject_id", "cmb_ckd", "renal_impaired_at_adm"],
        engine="fastparquet",
    )
    excl_cmb_ckd       = set(feat.loc[feat["cmb_ckd"] == 1,              "subject_id"])
    excl_renal_impaired = set(feat.loc[feat["renal_impaired_at_adm"] == 1, "subject_id"])
    excl_sids = excl_cmb_ckd | excl_renal_impaired
    print(f"  cmb_ckd==1:              {len(excl_cmb_ckd):,} patients")
    print(f"  renal_impaired_at_adm==1:{len(excl_renal_impaired):,} patients")
    print(f"  union (excluded):        {len(excl_sids):,} patients")

    filtered = target[~target["subject_id"].isin(excl_sids)]
    print(f"AKI target after exclusion:  {filtered['subject_id'].nunique():,} patients, {len(filtered):,} rows")

    filtered.to_parquet(out_path, index=False, engine="fastparquet")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
