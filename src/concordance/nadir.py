"""
Patient-level overlap of ICD-coded incident AKI and nadir-baseline KDIGO (opt-in).

Unit of analysis: the flowchart cohort, one admission per patient (first incident-AKI admission, else last AKI-free admission;
paper-run features_all with the cmb_ckd / renal-impairment exclusions). ICD side: incident_aki_label. KDIGO side, per variant:
creatinine (and dialysis) staging over the stay, optionally with the 24 h before admission, against one of several baselines:

  B2     the patient's own lowest creatinine in the previous 7 days
  B3     creatinine for eGFR 75 from age and sex (CKD-EPI 2021)
  Nadir  the lower of B3 and the patient's own lowest creatinine in the window, in any order (minimum-inpatient-creatinine
         surrogate, combined with B3)

Outputs are suppressed aggregates (counts < --min-cell print as '<N'); nothing row-level is written.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from . import data, mimic3
from .config import Config
from .stats import Suppressor, bootstrap_kappa_ci, kappa, wilson

FEATURES_DIR = os.environ.get("AKI_FEATURES_DIR", "features")     # folder holding features_all.parquet (set AKI_FEATURES_DIR for another run)

PRE_HOURS = 24
INCIDENT_GUARD_H = 24.0
PRIMARY = "Nadir, +24 h before admission"
# (label, pre-admission hours of the flag frame, flag column, minimum onset hours after admission or None)
VARIANTS = (
    ("B2: own lowest in prior 7 d", 0, "kdigo_b2", None),
    ("B3: eGFR 75 estimate", 0, "kdigo_b3", None),
    ("B3, +24 h before admission", PRE_HOURS, "kdigo_b3", None),
    ("Nadir", 0, "kdigo_b3n", None),
    (PRIMARY, PRE_HOURS, "kdigo_b3n", None),
    ("Nadir, +24 h, onset >= 24 h after admission", PRE_HOURS, "kdigo_b3n", INCIDENT_GUARD_H),
    ("B3, +24 h, onset >= 24 h after admission", PRE_HOURS, "kdigo_b3", INCIDENT_GUARD_H),
)
B3_PRIMARY = "B3, +24 h before admission"
SUFFIX = {"kdigo_b2": "B2", "kdigo_b3": "B3", "kdigo_b3n": "B3N"}
TIMING = [("before admission (ED value within 24 h)", -np.inf, 0.0), ("0 to <24 h", 0.0, 24.0),
          ("24 to <48 h", 24.0, 48.0), ("48 h or later", 48.0, np.inf)]
ICD_ONLY_REASONS = ("no creatinine in the window", "one creatinine only",
                    ">=2 creatinines, near the threshold (peak >= 80% of it, or a 0.2-0.3 mg/dL rise)",
                    ">=2 creatinines, clearly below the threshold")

COLS = {
    "n1_overlap_by_variant": ["version", "variant", "stratum", "n", "both", "icd_only", "kdigo_only", "neither",
                              "pct_icd_pos_also_kdigo", "ci_low_1", "ci_high_1", "pct_kdigo_pos_also_icd", "ci_low_2", "ci_high_2",
                              "pct_icd_neg_also_kdigo_neg", "accuracy", "jaccard", "kappa", "kappa_ci_low", "kappa_ci_high"],
    "n2_onset_timing": ["version", "variant", "group", "timing", "n", "pct_of_group"],
    "n3_stage": ["version", "stage", "n", "n_icd_pos", "pct_icd_pos"],
    "n4_icd_only_reasons": ["version", "reason", "n", "pct_of_icd_only"],
}


def kdigo_vector(frames: dict, hours: int, col: str, guard, hadm: pd.Series) -> np.ndarray:
    f = frames[hours].reindex(hadm.to_numpy())
    k = f[col].fillna(False).astype(bool)
    if guard is not None:
        k = k & (f["onset_h_" + SUFFIX[col]] >= guard).fillna(False)
    return k.to_numpy()


def overlap_row(sup: Suppressor, cfg, version: str, variant: str, stratum: str, icd, kd) -> dict:
    icd, kd = np.asarray(icd, bool), np.asarray(kd, bool)
    b, io, ko, ne = int((icd & kd).sum()), int((icd & ~kd).sum()), int((~icd & kd).sum()), int((~icd & ~kd).sum())
    cells, n = (b, io, ko, ne), len(icd)

    def prop(k, d):
        if d <= 0:
            return ("NA", "NA", "NA")
        lo, hi = wilson(k, d)
        return (sup.pct(k, d, *cells), sup.value(100 * lo, 2, *cells), sup.value(100 * hi, 2, *cells))
    p1, l1, h1 = prop(b, b + io)
    p2, l2, h2 = prop(b, b + ko)
    p3 = prop(ne, ne + ko)[0]
    if sup.ok(*cells):
        klo, khi = bootstrap_kappa_ci(cells, cfg.n_boot, cfg.seed, f"{version}|{variant}|{stratum}")
        kap, kl, kh = sup.value(kappa(*cells), 4, *cells), sup.value(klo, 4, *cells), sup.value(khi, 4, *cells)
    else:
        kap = kl = kh = "suppressed"
    return dict(version=version, variant=variant, stratum=stratum, n=sup.count(n), both=sup.count(b), icd_only=sup.count(io),
                kdigo_only=sup.count(ko), neither=sup.count(ne), pct_icd_pos_also_kdigo=p1, ci_low_1=l1, ci_high_1=h1,
                pct_kdigo_pos_also_icd=p2, ci_low_2=l2, ci_high_2=h2, pct_icd_neg_also_kdigo_neg=p3,
                accuracy=sup.pct(b + ne, n, *cells), jaccard=sup.value(b / (b + io + ko) if b + io + ko else float("nan"), 4, *cells),
                kappa=kap, kappa_ci_low=kl, kappa_ci_high=kh)


def icd_only_reason(f: pd.DataFrame, nadir: bool = True) -> pd.Series:
    """Why an ICD-positive patient is KDIGO-negative. f: n_scr, scr_max, scr_min, rise48, b3_base; nadir picks the baseline."""
    base = (np.minimum(f["b3_base"].to_numpy(), f["scr_min"].fillna(np.inf).to_numpy()) if nadir
            else f["b3_base"].to_numpy())
    ratio = f["scr_max"].to_numpy() / (1.5 * base)                       # peak as a fraction of the stage-1 ratio trigger
    near = (ratio >= 0.8) | (f["rise48"].to_numpy() >= 0.2)
    n = f["n_scr"].to_numpy()
    return pd.Series(np.select([n == 0, n == 1, near], ICD_ONLY_REASONS[:3], default=ICD_ONLY_REASONS[3]), index=f.index)


def tables(version: str, frames: dict, cohort: pd.DataFrame, cfg, sup: Suppressor, primary: str = PRIMARY) -> dict:
    """cohort: index = patient, columns hadm_id, icd (bool), female (bool). `primary` is the variant the detail tables describe."""
    spec = {lab: (h, col, g) for lab, h, col, g in VARIANTS}
    p_hours, p_col, _ = spec[primary]
    p_sfx = SUFFIX[p_col]
    out = {k: [] for k in COLS}
    icd, hadm = cohort["icd"].to_numpy(), cohort["hadm_id"]
    kd = {lab: kdigo_vector(frames, h, col, g, hadm) for lab, h, col, g in VARIANTS}
    for lab, *_ in VARIANTS:
        out["n1_overlap_by_variant"].append(overlap_row(sup, cfg, version, lab, "all", icd, kd[lab]))
    fem = cohort["female"].to_numpy()
    for lab in (B3_PRIMARY, PRIMARY, "Nadir, +24 h, onset >= 24 h after admission", "B3, +24 h, onset >= 24 h after admission"):
        for name, m in (("sex=F", fem), ("sex=M", ~fem)):
            out["n1_overlap_by_variant"].append(overlap_row(sup, cfg, version, lab, name, icd[m], kd[lab][m]))

    # onset timing of the first qualifying creatinine, for the patients who are KDIGO-positive
    for lab, h, col, g in VARIANTS:
        if lab not in (primary, "B3: eGFR 75 estimate"):
            continue
        onset = frames[h].reindex(hadm.to_numpy())["onset_h_" + SUFFIX[col]].to_numpy()
        for group, m in (("both ICD and KDIGO", icd & kd[lab]), ("KDIGO only", ~icd & kd[lab])):
            tot = int(m.sum())
            cats = [(t, int((m & (onset >= lo) & (onset < hi)).sum())) for t, lo, hi in TIMING]
            cats.append(("no creatinine time (dialysis only)", int((m & np.isnan(onset)).sum())))
            assert sum(c for _, c in cats) == tot, "timing categories do not add up"
            for t, c in cats:
                out["n2_onset_timing"].append(dict(version=version, variant=lab, group=group, timing=t, n=sup.count(c),
                                                   pct_of_group=sup.pct(c, tot, c, tot - c)))

    # stage (primary variant, creatinine and dialysis) and the share with an ICD code
    f24 = frames[p_hours].reindex(hadm.to_numpy())
    stage = np.maximum(f24["scr_" + p_sfx].fillna(0).to_numpy(), 3 * f24["rrt"].fillna(False).to_numpy().astype(int))
    for s in range(4):
        m = stage == s
        k = int((icd & m).sum())
        out["n3_stage"].append(dict(version=version, stage=s, n=sup.count(int(m.sum())), n_icd_pos=sup.count(k),
                                    pct_icd_pos=sup.pct(k, int(m.sum()), k, int(m.sum()) - k)))

    # why ICD-positive patients are KDIGO-negative
    io = icd & ~kd[primary]
    reasons = icd_only_reason(f24[io], nadir=(p_sfx == "B3N"))
    for r in ICD_ONLY_REASONS:
        c = int((reasons == r).sum())
        out["n4_icd_only_reasons"].append(dict(version=version, reason=r, n=sup.count(c),
                                               pct_of_icd_only=sup.pct(c, int(io.sum()), c, int(io.sum()) - c)))
    return out


def load(cfg: Config, a, log) -> dict:
    feat = pd.read_parquet(a.features_all, columns=["version", "subject_id", "hadm_id", "incident_aki_label", "cmb_ckd",
                                                    "renal_impaired_at_adm"])
    mod = feat[(feat["cmb_ckd"] != 1) & (feat["renal_impaired_at_adm"] != 1)]
    res = {}
    log("MIMIC-IV: creatinine staging ...")
    tab = data.load_tables(cfg)
    adm4, _, _ = data.build_admission_table(tab, cfg)
    cols = ["hadm_id", "subject_id", "admittime", "dischtime", "age", "female"]
    rrt, esrd = set(adm4.index[adm4["rrt_HOSP"]]), set(adm4.index[adm4["ESRD"]])
    res["mimic4"] = ({h: mimic3.hospital_flags(adm4[cols], tab.labs_raw, rrt, esrd, pre_hours=h) for h in (0, PRE_HOURS)},
                     adm4["female"])
    log("MIMIC-III: creatinine staging ...")
    frames3, female3 = {}, None
    for h in (0, PRE_HOURS):
        flags, adm3, *_ = mimic3.mimic3_flags(a.mimic3_dir, cfg, return_inputs=True, pre_hours=h)
        frames3[h] = flags
        female3 = adm3.drop_duplicates("hadm_id").set_index("hadm_id")["female"]
    res["mimic3"] = (frames3, female3)
    for ver in res:
        frames, female = res[ver]
        m = mod[mod["version"] == ver]
        assert m["hadm_id"].is_unique, "patient-level cohort has a repeated admission"
        cohort = pd.DataFrame({"hadm_id": m["hadm_id"].to_numpy(), "icd": (m["incident_aki_label"] == 1).to_numpy(),
                               "female": female.reindex(m["hadm_id"]).fillna(False).astype(bool).to_numpy()},
                              index=m["subject_id"].to_numpy())
        res[ver] = (frames, cohort)
    return res


def md(df: pd.DataFrame) -> str:
    rows = ["| " + " | ".join(df.columns) + " |", "|" + "|".join("---" for _ in df.columns) + "|"]
    rows += ["| " + " | ".join(r) + " |" for r in df.astype(str).values.tolist()]
    return "\n".join(rows)


def run(a) -> None:
    cfg = Config(mimic_dir=a.mimic4_dir, cache_dir=a.cache_dir, min_cell=a.min_cell, n_boot=a.n_boot, seed=a.seed,
                 use_uo=False, use_rrt=True, quiet=a.quiet)
    log = (lambda m: None) if a.quiet else (lambda m: print(m, flush=True))
    sup = Suppressor(a.min_cell)
    per_version = load(cfg, a, log)
    allt = {k: [] for k in COLS}
    for ver, (frames, cohort) in per_version.items():
        for k, rows in tables(ver, frames, cohort, cfg, sup, a.primary).items():
            allt[k] += rows
    os.makedirs(a.outdir, exist_ok=True)
    out = Path(a.outdir)
    frames_out = {k: pd.DataFrame(v, columns=COLS[k]) for k, v in allt.items()}
    for k, df in frames_out.items():
        df.to_csv(out / f"{k}.csv", index=False, lineterminator="\n")
    text = f"# Patient-level overlap tables (primary variant: {a.primary})\n\n" + "\n\n".join(f"## {k}\n\n{md(df)}" for k, df in frames_out.items()) + "\n"
    (out / f"{a.tag}_tables.md").write_text(text)
    log(text)


def parse_args(argv=None):
    d = Config()
    ap = argparse.ArgumentParser(prog="python -m src.concordance.nadir",
                                 description="Patient-level overlap of ICD incident AKI and nadir-baseline KDIGO.")
    ap.add_argument("--mimic4-dir", default=d.mimic_dir)
    ap.add_argument("--mimic3-dir", default="../data/mimic-iii-clinical-database-1.4")
    ap.add_argument("--features-all", default=os.path.join(FEATURES_DIR, "features_all.parquet"))
    ap.add_argument("--primary", default=PRIMARY, choices=[v[0] for v in VARIANTS],
                    help="Variant the onset-timing, stage and discordance tables describe.")
    ap.add_argument("--tag", default="nadir", help="Prefix of the markdown tables file.")
    ap.add_argument("--outdir", default="reports/nadir_baseline")
    ap.add_argument("--cache-dir", default=d.cache_dir)
    ap.add_argument("--min-cell", type=int, default=d.min_cell)
    ap.add_argument("--n-boot", type=int, default=d.n_boot)
    ap.add_argument("--seed", type=int, default=d.seed)
    ap.add_argument("--quiet", action="store_true")
    return ap.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
