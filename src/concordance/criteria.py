"""
Urine-output and dialysis (RRT) KDIGO criteria, to sit beside the creatinine criterion.

Urine output (ICU only; outputevents is not charted outside the ICU):
  hourly volume per ICU stay (hours counted from intime), summed over trailing windows.
    stage 1: < 0.5 mL/kg/h over 6 h      stage 2: < 0.5 mL/kg/h over 12 h
    stage 3: < 0.3 mL/kg/h over 24 h, or no urine over 12 h
  A window counts only if at least COVERAGE of its hours have a charted urine value, so
  gaps in charting are never read as oliguria. Weight is the stay's median inputevents
  patientweight; stays without a plausible weight are not assessable.
RRT: dialysis started in an ICU stay (procedureevents) or coded (procedures_icd, whole
  hospitalization only). It is stage 3 by itself. Admissions with an ESRD code are
  excluded, because maintenance dialysis is not AKI.
"""

import numpy as np
import pandas as pd

URINE_ITEMS = (226557, 226558, 226559, 226560, 226561, 226563, 226564, 226565, 226567, 226584, 227489)
IRRIGANT_IN = 227488                      # GU irrigant volume in: subtracted from urine out
RRT_PROC_ITEMS = (225441, 225802, 225803, 225805, 225809, 225955)
RRT_ICD9 = ("3995", "5498")               # hemodialysis, peritoneal dialysis
RRT_ICD10 = ("5A1D", "3E1M39Z")           # urinary filtration (PCS), peritoneal dialysate irrigation
UO_RULES = ((6, 0.5, 1), (12, 0.5, 2), (24, 0.3, 3))      # (hours, mL/kg/h threshold, stage)
ANURIA_HOURS = 12
COVERAGE = 5.0 / 6.0
MIN_WEIGHT, MAX_WEIGHT = 20.0, 400.0


def stage_one_stay(h: np.ndarray, v: np.ndarray, weight: float) -> int:
    """UO stage for one stay from hour bins h (ints >= 0) and their summed volumes v (mL)."""
    n_hours = int(h.max()) + 1
    vol = np.zeros(n_hours)
    charted = np.zeros(n_hours)
    np.add.at(vol, h, v)
    charted[h] = 1.0
    cv = np.concatenate([[0.0], np.cumsum(vol)])
    cc = np.concatenate([[0.0], np.cumsum(charted)])
    stage = 0
    for hours, thr, st in UO_RULES:
        if n_hours < hours:
            continue
        ws = cv[hours:] - cv[:-hours]
        ok = (cc[hours:] - cc[:-hours]) >= np.ceil(hours * COVERAGE)
        if (ok & (ws / (weight * hours) < thr)).any():
            stage = max(stage, st)
        if hours == ANURIA_HOURS and (ok & (ws <= 0)).any():
            stage = 3
    return stage


def uo_stage_by_stay(events: pd.DataFrame, weights: pd.Series) -> pd.DataFrame:
    """
    events: stay_id, hour (int, from intime), vol (mL, irrigant-in already negative).
    Returns a frame indexed by stay_id: uo_stage (0-3) and assessable (valid weight and urine charted).
    """
    if events.empty:
        return pd.DataFrame({"uo_stage": pd.Series(dtype="int8"), "assessable": pd.Series(dtype=bool)},
                            index=pd.Index([], name="stay_id"))
    agg = events.groupby(["stay_id", "hour"], sort=True)["vol"].sum().reset_index()
    stays, starts = np.unique(agg["stay_id"].to_numpy(), return_index=True)
    ends = np.append(starts[1:], len(agg))
    hrs, vols = agg["hour"].to_numpy(dtype=np.int64), agg["vol"].to_numpy(dtype=float)
    w = weights.reindex(stays).to_numpy(dtype=float)
    stage = np.zeros(len(stays), dtype=np.int8)
    ok = (w >= MIN_WEIGHT) & (w <= MAX_WEIGHT)
    for i in np.flatnonzero(ok):
        stage[i] = stage_one_stay(hrs[starts[i]:ends[i]], vols[starts[i]:ends[i]], w[i])
    return pd.DataFrame({"uo_stage": stage, "assessable": ok}, index=pd.Index(stays, name="stay_id"))


def rrt_icd_hadm(proc_icd: pd.DataFrame) -> set:
    """hadm_ids with a dialysis procedure code (no time stamp, so whole-hospitalization only)."""
    code = proc_icd["icd_code"].astype(str).str.strip().str.upper()
    ver = proc_icd["icd_version"].astype(int)
    hit = ((ver == 9) & code.str.startswith(RRT_ICD9)) | ((ver == 10) & code.str.startswith(RRT_ICD10))
    return set(proc_icd.loc[hit, "hadm_id"].astype("int64"))
