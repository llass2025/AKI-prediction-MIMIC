#!/usr/bin/env python3
"""
Compute admission-level and patient-level counts for each flowchart step,
for MIMIC-IV and MIMIC-III separately.

Outputs:
  flowchart_numbers_mimic4.csv
  flowchart_numbers_mimic3.csv

Run from project root:
  python3 flowchart/compute_flowchart_counts.py
"""

import os
import pickle
import numpy as np
import pandas as pd

DATA_PKL = "data.pkl"
FEATURES_ALL = "/Users/lb353/coding/aki-analysis/features/features_all.parquet"
CKD_SURV_ICD = "ckd_survival_no_cmb_ckd/incident_ckd_survival.csv"
CKD_SURV_LAB = "ckd_survival_no_cmb_ckd/incident_ckd_survival_labckd.csv"
OUTDIR = "."

AKI_PREFIXES = ["584", "3995", "5498", "N17"]
CKD_PREFIXES = ["585", "V451", "V560", "V5631", "V5632", "V568", "N18"]


def get_age(row):
    if row["version"] == "mimic3":
        try:
            age = (row["admittime"].to_pydatetime() - row["dob"].to_pydatetime()).days / 365
            return 90 if age > 150 else age
        except:
            return np.nan
    return row["anchor_age"]


def compute_counts(ver, adm_pat, paki_v, no_dx, target, feat, ckd_surv, ckd_surv_lab):
    rows = []

    def add(step, adm_df=None, pat_df=None, n=None, n_patients=None):
        _n = n if n is not None else (len(adm_df) if adm_df is not None else None)
        _p = n_patients if n_patients is not None else (
            pat_df["subject_id"].nunique() if pat_df is not None else
            (adm_df["subject_id"].nunique() if adm_df is not None else None)
        )
        rows.append({"step": step, "n": _n, "n_patients": _p})

    all_adm = adm_pat[adm_pat["version"] == ver]
    add("All admissions", all_adm)

    age_excl = all_adm[all_adm["age"] < 18]
    add("Excluded age <18", age_excl)

    adults_v = all_adm[all_adm["age"] >= 18]
    add("Adults >=18", adults_v)

    paki_excl = paki_v[paki_v["version"] == ver]
    add("Excluded primary AKI (presenting dx)", paki_excl)

    nodx_excl = no_dx[no_dx["version"] == ver]
    add("Excluded without diagnosis data", nodx_excl)

    tgt = target[target["version"] == ver]
    add("Eligible (target_admissions)", tgt)

    feat_v = feat[feat["version"] == ver]
    cmb_ckd_sids = set(feat_v.loc[feat_v["cmb_ckd"] == 1, "subject_id"])

    ckd_only = tgt[(tgt["is_aki"] == 0) & (tgt["is_ckd"] == 1)]
    noninc_aki = tgt[(tgt["incident_aki_label"].isna()) & (tgt["is_aki"] == 1)]
    renal_imp = tgt[(tgt["incident_aki_label"].isna()) & (tgt["is_aki"] == 0) & (tgt["is_ckd"] == 0)]
    cmb_ckd_adm = tgt[tgt["subject_id"].isin(cmb_ckd_sids)]

    after_ckd_only = tgt[~tgt["hadm_id"].isin(ckd_only["hadm_id"])]
    after_noninc   = after_ckd_only[~after_ckd_only["hadm_id"].isin(noninc_aki["hadm_id"])]
    after_renal    = after_noninc[~after_noninc["hadm_id"].isin(renal_imp["hadm_id"])]
    after_cmb_ckd  = after_renal[~after_renal["subject_id"].isin(cmb_ckd_sids)]

    add("Excluded: CKD-only (no AKI)", ckd_only)
    add("Excluded: Non-incident AKI (repeat or AKI+CKD)", noninc_aki)
    add("Excluded: Prior renal impairment (no current AKI/CKD)", renal_imp)
    add("Excluded: Concurrent AKI+CKD (cmb_ckd)", cmb_ckd_adm)
    add("Remaining admissions", after_cmb_ckd)

    tgt_final = after_cmb_ckd
    labeled = tgt_final[tgt_final["incident_aki_label"].notna()]
    inc_aki = tgt_final[tgt_final["incident_aki_label"] == 1]
    neg = tgt_final[tgt_final["incident_aki_label"] == 0]
    add("Labeled admissions", labeled)
    add("Labeled: Incident AKI (label=1)", inc_aki)
    add("Labeled: No AKI (label=0)", neg)

    no_lab = labeled[~labeled["hadm_id"].isin(feat_v["hadm_id"])]
    add("Dedup to patient-level (keep first AKI+, last AKI−)", no_lab)

    modeling = feat_v[(feat_v["cmb_ckd"] != 1) & (feat_v["renal_impaired_at_adm"] != 1)]
    modeling_aki = modeling[modeling["incident_aki_label"] == 1]
    modeling_neg = modeling[modeling["incident_aki_label"] == 0]
    add("Modeling cohort", modeling)
    add("Modeling: Incident AKI", modeling_aki)
    add("Modeling: Controls", modeling_neg)

    surv = ckd_surv[ckd_surv["version"] == ver]
    surv_lab = ckd_surv_lab[ckd_surv_lab["version"] == ver]

    # CKD survival exclusions not directly computable here — use known values
    ckd_excl = {"mimic4": (0, None), "mimic3": (0, None)}[ver]
    add("CKD survival excluded: CKD/ESRD <90d or no follow-up", n=ckd_excl[0], n_patients=ckd_excl[1])
    add("Post-AKI survival cohort", n=len(surv), n_patients=surv["subject_id"].nunique())

    for evt, label in [("CKD", "Outcome: Incident CKD (ICD, >=90d)"),
                       ("PostCKD", "Outcome: PostCKD/ESRD (>=90d)"),
                       ("Death", "Outcome: Death"),
                       ("Censor", "Outcome: Censored")]:
        sub = surv[surv["event_type"] == evt]
        add(label, sub)

    lab_ckd = surv_lab[surv_lab["event_type"] == "LabCKD"]
    add("Outcome: Incident CKD (lab-based, >=90d)", lab_ckd)

    return pd.DataFrame(rows)


def main():
    print("Loading data.pkl ...")
    with open(DATA_PKL, "rb") as f:
        data = pickle.load(f)

    adm = data["admissions"]
    pat = data["patients"]
    dx  = data["diagnosis"]

    adm_pat = adm.merge(pat[["subject_id", "version", "dob", "anchor_age"]],
                        on=["subject_id", "version"], how="left")
    adm_pat["admittime"] = pd.to_datetime(adm_pat["admittime"], errors="coerce")
    adm_pat["dob"]       = pd.to_datetime(adm_pat["dob"],       errors="coerce")
    adm_pat["age"]       = adm_pat.apply(get_age, axis=1)

    dx2 = dx.copy()
    dx2["icd_code"] = dx2["icd_code"].astype(str)
    dx2["is_aki"]   = dx2["icd_code"].apply(lambda x: int(any(x.startswith(p) for p in AKI_PREFIXES)))
    dx2["is_ckd"]   = dx2["icd_code"].apply(lambda x: int(any(x.startswith(p) for p in CKD_PREFIXES)))
    dx2["seq_num"]  = pd.to_numeric(dx2["seq_num"], errors="coerce")

    paki = dx2[(dx2["is_aki"] == 1) & (dx2["seq_num"] == 1)][["subject_id", "hadm_id"]].drop_duplicates()
    paki_v = paki.merge(adm[["subject_id", "hadm_id", "version"]], on=["subject_id", "hadm_id"])

    no_dx = adm_pat[~adm_pat["hadm_id"].isin(dx2["hadm_id"].unique())]

    hadm_flags = dx2[~dx2.set_index(["subject_id", "hadm_id"]).index.isin(
        paki.set_index(["subject_id", "hadm_id"]).index)]\
        .groupby(["subject_id", "hadm_id"])[["is_aki", "is_ckd"]].max().reset_index()

    adults = adm_pat[adm_pat["age"] >= 18]
    adults_no_paki = adults[~adults.set_index(["subject_id", "hadm_id"]).index.isin(
        paki.set_index(["subject_id", "hadm_id"]).index)]
    target0 = adults_no_paki.merge(hadm_flags, on=["subject_id", "hadm_id"], how="inner")
    target0 = target0.sort_values(["version", "subject_id", "admittime"])

    print("Computing renal history flags ...")
    rows = []
    for (ver, sid), g in target0.groupby(["version", "subject_id"]):
        had_prior_ckd = False
        had_prior_aki = False
        onset_assigned = False
        for _, r in g.iterrows():
            r = r.copy()
            r["renal_impaired_at_adm"] = 1 if had_prior_ckd else 0
            if r["is_aki"] == 1 and not onset_assigned and not had_prior_aki and r["is_ckd"] == 0 and not had_prior_ckd:
                r["onset_aki_flag"] = 1
                r["incident_aki_label"] = 1.0
                onset_assigned = True
            elif r["is_aki"] == 0 and r["is_ckd"] == 0 and not had_prior_ckd:
                r["incident_aki_label"] = 0.0
                r["onset_aki_flag"] = 0
            else:
                r["incident_aki_label"] = np.nan
                r["onset_aki_flag"] = 0
            if r["is_ckd"] == 1: had_prior_ckd = True
            if r["is_aki"] == 1: had_prior_aki = True
            rows.append(r)
    target = pd.DataFrame(rows)

    feat = pd.read_parquet(FEATURES_ALL, columns=["version", "subject_id", "hadm_id", "incident_aki_label", "cmb_ckd", "renal_impaired_at_adm"])
    ckd_surv     = pd.read_csv(CKD_SURV_ICD)
    ckd_surv_lab = pd.read_csv(CKD_SURV_LAB)

    for ver in ["mimic4", "mimic3"]:
        print(f"\nComputing {ver} ...")
        df = compute_counts(ver, adm_pat, paki_v, no_dx, target, feat, ckd_surv, ckd_surv_lab)
        out = os.path.join(OUTDIR, f"flowchart_numbers_{ver}.csv")
        df.to_csv(out, index=False)
        print(f"Saved {out}")
        print(df.to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()
