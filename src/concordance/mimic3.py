"""
Hospital-wide KDIGO creatinine flags (B2 and B3 baselines) from plain frames, plus a MIMIC-III v1.4 reader.

`hospital_flags` takes version-neutral frames, so the same code serves MIMIC-IV (where it is checked against the
full admission table in data.py) and MIMIC-III. It mirrors the W_HOSP creatinine path of data.build_admission_table:
labs within [admittime, dischtime] of their own admission, the same 48 h / 7 d rolling minima, the same staging, and
dialysis (procedure code, non-ESRD) as stage 3. Urine output and ICU-only criteria are not part of it.

MIMIC-III specifics: blood creatinine is itemid 50912 only (the other 'creatinine' items in D_LABITEMS are urine,
ascites, pleural, joint and body fluid), ICD-9 only, ages as the pipeline computes them ((admit - DOB)/365, 90 when
the shifted DOB gives > 150).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from . import criteria, icd, kdigo
from .data import _CHUNK, _cached, _secs

MIMIC3_CREATININE_ITEMIDS = (50912,)


def hospital_flags(adm: pd.DataFrame, labs: pd.DataFrame, rrt_hadm: set, esrd_hadm: set, return_window: bool = False,
                   pre_hours: float = 0.0):
    """
    adm:  hadm_id, subject_id, admittime, dischtime, age, female
    labs: subject_id, hadm_id (NaN allowed), charttime, valuenum   (creatinine rows only)
    Returns a frame indexed by hadm_id: n_scr, scr_B2, scr_B3 (creatinine stage 0-3), rrt (non-ESRD dialysis),
    kdigo_b2, kdigo_b3 (stage >= 1 including dialysis); scr_B3N / kdigo_b3n use the lower of the B3 estimate and the
    patient's lowest creatinine in the window as baseline.
    pre_hours > 0 also admits the patient's creatinine from the `pre_hours` before admittime (ED values), whatever their
    hadm_id; the default 0 is the window of data.build_admission_table.
    """
    adm = adm.sort_values("hadm_id").reset_index(drop=True)
    adm["admit_s"], adm["disch_s"] = _secs(adm["admittime"]), _secs(adm["dischtime"])
    idx = adm["hadm_id"].to_numpy(dtype="int64")

    l = labs[labs["valuenum"].notna() & (labs["valuenum"] > 0) & labs["charttime"].notna()]
    l = l.sort_values(["subject_id", "charttime", "valuenum"], kind="mergesort").reset_index(drop=True)
    l["t"] = _secs(l["charttime"])
    l["lab_row"] = np.arange(len(l))
    subj, uniq = pd.factorize(l["subject_id"])
    l["prior7d_min"] = kdigo.grouped(kdigo.prior_min, subj, l["t"], l["valuenum"], kdigo.BASE_WINDOW_S)

    inh = l[l["hadm_id"].notna()].copy()
    inh["hadm_id"] = inh["hadm_id"].astype("int64")
    m = inh.merge(adm[["hadm_id", "admit_s", "disch_s"]], on="hadm_id", how="inner")
    m = m[(m["t"] >= m["admit_s"]) & (m["t"] <= m["disch_s"])]
    cols = ["hadm_id", "t", "valuenum", "prior7d_min", "lab_row"]
    m = m[cols]
    if pre_hours > 0:
        # labs are sorted by (subject, time), so (subject code, time) is a sorted key: find each admission's pre-admission slice
        stride = np.int64(1) << 36
        key = subj.astype("int64") * stride + l["t"].to_numpy()
        code = uniq.get_indexer(adm["subject_id"])
        base = np.where(code >= 0, code, 0).astype("int64") * stride + adm["admit_s"].to_numpy()
        lo = np.searchsorted(key, base - int(pre_hours * 3600), side="left")
        hi = np.searchsorted(key, base, side="left")
        cnt = np.where(code >= 0, hi - lo, 0)
        rep = np.repeat(np.arange(len(adm)), cnt)
        rows = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt) + np.repeat(lo, cnt)
        pre = pd.DataFrame({"hadm_id": idx[rep], "t": l["t"].to_numpy()[rows], "valuenum": l["valuenum"].to_numpy()[rows],
                            "prior7d_min": l["prior7d_min"].to_numpy()[rows], "lab_row": rows})
        m = pd.concat([m, pre], ignore_index=True).drop_duplicates(["hadm_id", "lab_row"])
    win = kdigo.make_window(m[["hadm_id", "t", "valuenum", "prior7d_min"]])

    b3 = pd.Series(kdigo.backcalc_scr(adm["age"].to_numpy(), adm["female"].to_numpy()), index=idx)
    out = pd.DataFrame(index=pd.Index(idx, name="hadm_id"))
    out["n_scr"] = pd.Series(win.counts, index=win.hadm).reindex(idx).fillna(0).astype(int)
    b3_lab = win.per_lab(win.aligned(b3))
    # nadir variant: the lower of the eGFR-75 estimate and the patient's lowest creatinine in the window (any order, so a
    # creatinine that falls after arrival counts against that lowest value)
    nadir = np.repeat(np.minimum.reduceat(win.v, win.starts), win.counts) if win.n_groups else np.zeros(0)
    admit = pd.Series(adm["admit_s"].to_numpy(), index=idx)
    for name, per_lab in (("scr_B2", win.prior7d), ("scr_B3", b3_lab), ("scr_B3N", np.minimum(b3_lab, nadir))):
        res = (kdigo.stage_window(win, per_lab) if win.n_groups
               else pd.DataFrame({"max_stage": [], "first_qual_t": []}))
        out[name] = res["max_stage"].reindex(idx).fillna(0).astype(int)
        ft = res["first_qual_t"].reindex(idx)        # seconds; -1 or NaN = never qualified
        out["onset_h_" + name[4:]] = np.where(ft >= 0, (ft - admit) / 3600.0, np.nan)   # hours after admittime (negative = before)
    # creatinine shape in the window, for the discordance breakdown
    if win.n_groups:
        shape = pd.DataFrame({"scr_max": np.maximum.reduceat(win.v, win.starts), "scr_min": np.minimum.reduceat(win.v, win.starts),
                              "rise48": np.maximum.reduceat(win.v - win.min48, win.starts)}, index=win.hadm).reindex(idx)
    else:
        shape = pd.DataFrame(np.nan, index=idx, columns=["scr_max", "scr_min", "rise48"])
    out[["scr_max", "scr_min", "rise48"]] = shape.to_numpy()
    out["b3_base"] = b3.to_numpy()
    out["rrt"] = [h in rrt_hadm and h not in esrd_hadm for h in idx]
    for b in ("B2", "B3", "B3N"):
        out[f"kdigo_{b.lower()}"] = (out[f"scr_{b}"] >= 1) | out["rrt"]
    return (out, win) if return_window else out


# ---------------------------------------------------------------------------
# MIMIC-III v1.4 reader
# ---------------------------------------------------------------------------

def _csv(root: Path, name: str) -> Path:
    for ext in (".csv.gz", ".csv"):
        p = root / f"{name}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"MIMIC-III file not found: {root / name}.csv(.gz) (set --mimic3-dir)")


def mimic3_flags(root: str, cfg, return_inputs: bool = False, pre_hours: float = 0.0):
    """Hospital-wide B2/B3 flags for every MIMIC-III admission with an age >= 0 (the caller selects the cohort).
    With return_inputs, also returns (adm, labs, esrd, window) for diagnostics."""
    root = Path(root)
    p_adm, p_pat, p_lab = _csv(root, "ADMISSIONS"), _csv(root, "PATIENTS"), _csv(root, "LABEVENTS")
    p_proc, p_dx = _csv(root, "PROCEDURES_ICD"), _csv(root, "DIAGNOSES_ICD")

    adm = pd.read_csv(p_adm, usecols=["SUBJECT_ID", "HADM_ID", "ADMITTIME", "DISCHTIME"],
                      parse_dates=["ADMITTIME", "DISCHTIME"]).rename(columns=str.lower)
    pat = pd.read_csv(p_pat, usecols=["SUBJECT_ID", "GENDER", "DOB"], parse_dates=["DOB"]).rename(columns=str.lower)
    adm = adm.merge(pat, on="subject_id", how="inner", validate="many_to_one")
    adm = adm[adm["dischtime"].notna()].copy()
    # seconds resolution: shifted DOBs of patients > 89 y are ~300 y before admission, which overflows nanoseconds
    age = (adm["admittime"].dt.as_unit("s") - adm["dob"].dt.as_unit("s")).dt.days / 365.0   # as get_age() in compute_flowchart_counts.py
    adm["age"] = np.where(age > 150, 90.0, age)
    adm["female"] = adm["gender"].astype(str).str.upper().eq("F")
    adm = adm[adm["age"].notna()]

    ids = set(MIMIC3_CREATININE_ITEMIDS)

    def build():
        parts = []
        for c in pd.read_csv(p_lab, usecols=["SUBJECT_ID", "HADM_ID", "ITEMID", "CHARTTIME", "VALUENUM"], chunksize=_CHUNK):
            c = c.rename(columns=str.lower)
            parts.append(c[c["itemid"].isin(ids)])
        labs_ = pd.concat(parts, ignore_index=True)
        labs_["charttime"] = pd.to_datetime(labs_["charttime"], errors="coerce")
        return labs_
    labs = _cached(cfg, f"creatinine_mimic3_{'-'.join(map(str, sorted(ids)))}", [p_lab], build)

    proc = pd.read_csv(p_proc, usecols=["HADM_ID", "ICD9_CODE"], dtype={"ICD9_CODE": str}).rename(
        columns={"HADM_ID": "hadm_id", "ICD9_CODE": "icd_code"}).dropna()
    proc["icd_version"] = 9
    dx = pd.read_csv(p_dx, usecols=["HADM_ID", "ICD9_CODE"], dtype={"ICD9_CODE": str}).rename(
        columns={"HADM_ID": "hadm_id", "ICD9_CODE": "icd_code"}).dropna()
    dx["icd_version"] = 9
    esrd = set(icd.flag_admissions(dx).query("ESRD").index.astype("int64"))
    rrt = criteria.rrt_icd_hadm(proc)
    if return_inputs:
        flags, win = hospital_flags(adm, labs, rrt, esrd, return_window=True, pre_hours=pre_hours)
        return flags, adm, labs, esrd, win
    return hospital_flags(adm, labs, rrt, esrd, pre_hours=pre_hours)
