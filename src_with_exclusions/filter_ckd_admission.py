#!/usr/bin/env python3
"""
Upstream exclusion step for the CKD survival pipeline.

Filters incident_ckd_admission.csv to only retain patients who are in the
already-filtered AKI target cohort (target.parquet). Since cmb_ckd and
renal_impaired_at_adm patients were removed from target.parquet by
filter_aki_target.py, this ensures the CKD survival pipeline starts from
exactly the same clean incident AKI cohort — consistent with how the
flowchart counts are derived.

Inputs:
  --adm     incident_ckd_admission.csv produced by ckd_survival compute_target_admissions.py
  --target  target.parquet (filtered AKI cohort from filter_aki_target.py)
  --out     output path for the filtered CSV (default: overwrites --adm)

Outputs:
  Filtered CSV at --out containing only admissions for patients in the AKI cohort
"""

import argparse
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description="Filter CKD admission CSV to the incident AKI cohort")
    ap.add_argument("--adm",    required=True, help="Path to incident_ckd_admission.csv")
    ap.add_argument("--target", required=True, help="Path to filtered target.parquet (AKI cohort)")
    ap.add_argument("--out",    default=None,  help="Output path (default: overwrite --adm)")
    args = ap.parse_args()

    out_path = args.out or args.adm

    adm = pd.read_csv(args.adm)
    print(f"CKD admission before filter: {adm['subject_id'].nunique():,} patients, {len(adm):,} rows")

    target = pd.read_parquet(args.target, engine="fastparquet")
    aki_sids = set(target["subject_id"].unique())
    print(f"Incident AKI cohort (filtered target): {len(aki_sids):,} patients")

    filtered = adm[adm["subject_id"].isin(aki_sids)]
    n_removed = adm["subject_id"].nunique() - filtered["subject_id"].nunique()
    print(f"Removed {n_removed:,} patients not in AKI cohort (cmb_ckd / renal_impaired / non-AKI)")
    print(f"CKD admission after filter:  {filtered['subject_id'].nunique():,} patients, {len(filtered):,} rows")

    filtered.to_csv(out_path, index=False)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
