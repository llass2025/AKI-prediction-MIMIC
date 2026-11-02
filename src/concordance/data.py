"""
Load the raw MIMIC-IV tables the diagnostics need, and build the admission-level
analysis table (ICD flags, KDIGO stage per window / baseline / criterion).

Row-level frames live only in memory (and in --cache-dir, gitignored pickles of
extracted rows); nothing row-level reaches the reports.

Admission-table columns used downstream:
  scr_HOSP_<b>, scr_ICU_<b>   creatinine-only stage (0-3) for baseline b
  uo_ICU, uo_assessable       urine-output stage in the ICU window
  rrt_HOSP, rrt_ICU           dialysis started (non-ESRD)
  scr_primary_<w>             creatinine-only stage for the primary baseline
  stage_HOSP, stage_ICU       combined stage = max(SCr[primary], UO, 3 if RRT)
"""

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from . import criteria, icd, kdigo
from .config import BASELINES, Config

_CHUNK = 5_000_000


@dataclass
class Tables:
    admissions: pd.DataFrame
    patients: pd.DataFrame
    dx: pd.DataFrame
    icustays: pd.DataFrame
    labitems: pd.DataFrame
    labs_raw: pd.DataFrame                       # creatinine rows (all hadm_id incl. null)
    urine: Optional[pd.DataFrame] = None         # stay_id, charttime, itemid, value
    weights: Optional[pd.Series] = None          # stay_id -> kg
    rrt_proc: Optional[pd.DataFrame] = None      # stay_id, starttime
    proc_icd: Optional[pd.DataFrame] = None      # hadm_id, icd_code, icd_version


@dataclass
class Meta:
    n_adults: int = 0
    n_icu_cohort: int = 0
    n_multi_stay: int = 0
    n_no_scr_hosp: int = 0
    n_no_scr_icu: int = 0
    item_rows: Dict[int, int] = field(default_factory=dict)
    item_labels: Dict[int, str] = field(default_factory=dict)
    existing_regex_itemids: int = 0
    n_pipeline: int = 0
    n_modeling: int = 0
    n_modeling_pos: int = 0
    n_stays: int = 0
    n_stays_weight: int = 0
    n_stays_urine: int = 0
    n_uo_assessable: int = 0
    n_rrt_icu: int = 0
    n_rrt_hosp: int = 0
    n_rrt_esrd_excluded: int = 0
    b0_check: dict = field(default_factory=dict)


def _log(cfg: Config, msg: str) -> None:
    if not cfg.quiet:
        print(msg, flush=True)


# ---------------------------------------------------------------------------
# Loading (extracts are cached, keyed on the source files' size and mtime)
# ---------------------------------------------------------------------------

def _cached(cfg: Config, name: str, files, build):
    key = hashlib.md5("|".join(f"{p.name}:{p.stat().st_size}:{int(p.stat().st_mtime)}" for p in files).encode()
                      ).hexdigest()[:12]
    path = Path(cfg.cache_dir) / f"{name}_{key}.pkl"
    if path.exists() and not cfg.refresh_cache:
        _log(cfg, f"Using cached {name} extract")
        return pd.read_pickle(path)
    _log(cfg, f"Extracting {name} (one-off; cached afterwards) ...")
    obj = build()
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(obj, path)
    return obj


def _load_creatinine(mimic_dir: Path, cfg: Config) -> pd.DataFrame:
    src = mimic_dir / "hosp" / "labevents.csv.gz"
    ids = set(cfg.creatinine_itemids)

    def build():
        parts = [c[c["itemid"].isin(ids)] for c in pd.read_csv(
            src, usecols=["subject_id", "hadm_id", "itemid", "charttime", "valuenum"], chunksize=_CHUNK)]
        labs = pd.concat(parts, ignore_index=True)
        labs["charttime"] = pd.to_datetime(labs["charttime"], errors="coerce")
        return labs
    return _cached(cfg, f"creatinine_{'-'.join(map(str, sorted(ids)))}", [src], build)


def _load_icu_extras(d: Path, cfg: Config) -> dict:
    out = {}
    if cfg.use_uo:
        oe, ie = d / "icu/outputevents.csv.gz", d / "icu/inputevents.csv.gz"

        def build_urine():
            keep = set(criteria.URINE_ITEMS) | {criteria.IRRIGANT_IN}
            u = pd.read_csv(oe, usecols=["stay_id", "charttime", "itemid", "value"], parse_dates=["charttime"])
            return u[u["itemid"].isin(keep)].reset_index(drop=True)

        def build_weights():
            parts = []
            for c in pd.read_csv(ie, usecols=["stay_id", "patientweight"], chunksize=_CHUNK):
                parts.append(c.dropna().drop_duplicates())
            w = pd.concat(parts).drop_duplicates()
            return w.groupby("stay_id")["patientweight"].median()
        out["urine"] = _cached(cfg, "urine", [oe], build_urine)
        out["weights"] = _cached(cfg, "weights", [ie], build_weights)
    if cfg.use_rrt:
        pe, pi = d / "icu/procedureevents.csv.gz", d / "hosp/procedures_icd.csv.gz"

        def build_rrt():
            p = pd.read_csv(pe, usecols=["stay_id", "starttime", "itemid"], parse_dates=["starttime"])
            return p[p["itemid"].isin(criteria.RRT_PROC_ITEMS)].reset_index(drop=True)
        out["rrt_proc"] = _cached(cfg, "rrt_proc", [pe], build_rrt)
        out["proc_icd"] = pd.read_csv(pi, usecols=["hadm_id", "icd_code", "icd_version"], dtype={"icd_code": str})
    return out


def load_tables(cfg: Config) -> Tables:
    d = Path(cfg.mimic_dir)
    need = ["hosp/patients.csv.gz", "hosp/admissions.csv.gz", "hosp/diagnoses_icd.csv.gz",
            "hosp/d_labitems.csv.gz", "hosp/labevents.csv.gz", "icu/icustays.csv.gz"]
    if cfg.use_uo:
        need += ["icu/outputevents.csv.gz", "icu/inputevents.csv.gz"]
    if cfg.use_rrt:
        need += ["icu/procedureevents.csv.gz", "hosp/procedures_icd.csv.gz"]
    for rel in need:
        if not (d / rel).exists():
            raise FileNotFoundError(f"MIMIC-IV file not found: {d / rel} (set --mimic-dir, or --no-uo / --no-rrt)")
    return Tables(
        admissions=pd.read_csv(d / "hosp/admissions.csv.gz",
                               usecols=["subject_id", "hadm_id", "admittime", "dischtime"],
                               parse_dates=["admittime", "dischtime"]),
        patients=pd.read_csv(d / "hosp/patients.csv.gz",
                             usecols=["subject_id", "gender", "anchor_age", "anchor_year", "anchor_year_group"]),
        dx=pd.read_csv(d / "hosp/diagnoses_icd.csv.gz", dtype={"icd_code": str}),
        icustays=pd.read_csv(d / "icu/icustays.csv.gz",
                             usecols=["subject_id", "hadm_id", "stay_id", "intime", "outtime"],
                             parse_dates=["intime", "outtime"]),
        labitems=pd.read_csv(d / "hosp/d_labitems.csv.gz"),
        labs_raw=_load_creatinine(d, cfg),
        **_load_icu_extras(d, cfg),
    )


# ---------------------------------------------------------------------------
# Admission-level table
# ---------------------------------------------------------------------------

def _secs(s: pd.Series) -> np.ndarray:
    return s.astype("datetime64[ns]").astype("int64").to_numpy() // 1_000_000_000


def _col(res: pd.DataFrame, field_: str, index, fill, dtype):
    return res[field_].reindex(index, fill_value=fill).astype(dtype)


def build_admission_table(tab: Tables, cfg: Config):
    """Returns (adm, meta, windows) where adm is indexed by hadm_id (unnamed index)."""
    meta = Meta()
    onset = kdigo.load_existing_onset_module()

    # -- itemid bookkeeping (labels from d_labitems; compared with the pipeline's regex) --
    li = tab.labitems.set_index("itemid")
    for i in cfg.creatinine_itemids:
        if i not in li.index:
            raise ValueError(f"creatinine itemid {i} not in d_labitems")
        meta.item_labels[i] = f"{li.loc[i, 'label']} ({li.loc[i, 'fluid']}, {li.loc[i, 'category']})"
        meta.item_rows[i] = int((tab.labs_raw["itemid"] == i).sum())
    meta.existing_regex_itemids = len(onset.select_creatinine_itemids(tab.labitems))

    # -- adult admissions --
    adm = tab.admissions.merge(tab.patients, on="subject_id", how="inner", validate="many_to_one")
    adm["age"] = adm["anchor_age"] + adm["admittime"].dt.year - adm["anchor_year"]
    adm = adm[(adm["age"] >= 18) & adm["dischtime"].notna()].sort_values("hadm_id").reset_index(drop=True)
    adm["admit_s"] = _secs(adm["admittime"])
    adm["disch_s"] = _secs(adm["dischtime"])
    adm["female"] = adm["gender"].astype(str).str.upper().eq("F")
    meta.n_adults = len(adm)

    # -- ICU stays aggregated to the admission --
    icu = tab.icustays[tab.icustays["hadm_id"].isin(adm["hadm_id"])].copy()
    icu["in_s"] = _secs(icu["intime"])
    icu["out_s"] = _secs(icu["outtime"])
    per = icu.groupby("hadm_id").agg(n_icu_stays=("stay_id", "size"), first_intime_s=("in_s", "min"))
    adm = adm.merge(per, left_on="hadm_id", right_index=True, how="left")
    adm["n_icu_stays"] = adm["n_icu_stays"].fillna(0).astype(int)
    adm["in_icu_cohort"] = adm["n_icu_stays"] > 0
    meta.n_icu_cohort = int(adm["in_icu_cohort"].sum())
    meta.n_multi_stay = int((adm["n_icu_stays"] > 1).sum())

    # -- ICD flags --
    flags = icd.flag_admissions(tab.dx)
    adm = adm.merge(flags, left_on="hadm_id", right_index=True, how="left")
    for name in icd.ICD_SETS:
        adm[name] = adm[name].fillna(False).astype(bool)
    adm["icd_version_group"] = adm["icd_version_group"].fillna("no_codes")
    adm["ckd_status"] = icd.ckd_status(adm["CKD"], adm["ESRD"]).to_numpy()

    # -- creatinine rows: reuse the existing prep, then add seconds + subject-level 7d prior min --
    labs = onset.prep_creatinine_labs(tab.labs_raw, set(cfg.creatinine_itemids))
    labs = labs.sort_values(["subject_id", "charttime", "valuenum"], kind="mergesort").reset_index(drop=True)
    labs["t"] = _secs(labs["charttime"])
    labs["lab_row"] = np.arange(len(labs))
    subj_code = pd.factorize(labs["subject_id"])[0]
    labs["prior7d_min"] = kdigo.grouped(kdigo.prior_min, subj_code, labs["t"], labs["valuenum"],
                                        kdigo.BASE_WINDOW_S)

    # -- creatinine windows --
    inhosp = labs[labs["hadm_id"].notna()].copy()
    inhosp["hadm_id"] = inhosp["hadm_id"].astype("int64")
    m = inhosp.merge(adm[["hadm_id", "admit_s", "disch_s"]], on="hadm_id", how="inner")
    m = m[(m["t"] >= m["admit_s"]) & (m["t"] <= m["disch_s"])]
    hosp_labs = m[["hadm_id", "t", "valuenum", "prior7d_min", "lab_row"]]

    mi = hosp_labs.merge(icu[["hadm_id", "in_s", "out_s"]], on="hadm_id", how="inner")
    mi = mi[(mi["t"] >= mi["in_s"]) & (mi["t"] <= mi["out_s"])].drop_duplicates("lab_row")
    icu_labs = mi[["hadm_id", "t", "valuenum", "prior7d_min", "lab_row"]]

    win_hosp = kdigo.make_window(hosp_labs)
    win_icu = kdigo.make_window(icu_labs)
    idx = adm["hadm_id"].to_numpy()
    adm.index = pd.Index(idx)          # unnamed, so it never clashes with the hadm_id column
    adm["n_scr_hosp"] = pd.Series(win_hosp.counts, index=win_hosp.hadm).reindex(idx).fillna(0).astype(int)
    adm["n_scr_icu"] = pd.Series(win_icu.counts, index=win_icu.hadm).reindex(idx).fillna(0).astype(int)
    meta.n_no_scr_hosp = int((adm["n_scr_hosp"] == 0).sum())
    meta.n_no_scr_icu = int(((adm["n_scr_icu"] == 0) & adm["in_icu_cohort"]).sum())

    # -- baselines (admission-level scalars; B2 is per lab) --
    _log(cfg, "Computing B0 baselines with the existing compute_baselines() ...")
    b0_median = kdigo.b0_median_baselines(adm.reset_index(drop=True), labs)
    b0_median.index = adm.index
    first_min = pd.Series(np.minimum.reduceat(win_hosp.v, win_hosp.starts) if win_hosp.n_groups else [],
                          index=win_hosp.hadm).reindex(adm.index)
    b0 = b0_median.fillna(first_min)                     # existing fallback: in-admission minimum
    b1 = pd.Series(win_hosp.v[win_hosp.starts] if win_hosp.n_groups else [],
                   index=win_hosp.hadm).reindex(adm.index)
    b3 = pd.Series(kdigo.backcalc_scr(adm["age"].to_numpy(), adm["female"].to_numpy()), index=adm.index)
    b4 = kdigo.b4_measured(adm.reset_index(drop=True), labs).set_axis(adm.index).fillna(b3)
    base_scalar = {"B0": b0, "B1": b1, "B3": b3, "B4": b4}

    # -- creatinine stages for every window and baseline variant --
    _log(cfg, "Staging creatinine for each window and baseline variant ...")
    scr_hosp, scr_icu = {}, {}
    for name in BASELINES:
        for win, store in ((win_hosp, scr_hosp), (win_icu, scr_icu)):
            per_lab = win.prior7d if name == "B2" else win.per_lab(win.aligned(base_scalar[name]))
            store[name] = kdigo.stage_window(win, per_lab)
    for name in BASELINES:
        r = scr_hosp[name]
        adm[f"scr_HOSP_{name}"] = _col(r, "max_stage", adm.index, 0, int)
        adm[f"nobase_HOSP_{name}"] = ~_col(r, "has_baseline", adm.index, False, bool)
        adm[f"scr_ICU_{name}"] = _col(scr_icu[name], "max_stage", adm.index, 0, int)

    # -- urine output (ICU window only) --
    adm["uo_ICU"] = 0
    adm["uo_assessable"] = False
    if cfg.use_uo:
        _log(cfg, "Staging urine output ...")
        u = tab.urine.merge(icu[["stay_id", "in_s", "out_s"]], on="stay_id", how="inner")
        u["t"] = _secs(u["charttime"])
        u = u[(u["t"] >= u["in_s"]) & (u["t"] <= u["out_s"])]
        u["hour"] = (u["t"] - u["in_s"]) // 3600
        u["vol"] = np.where(u["itemid"] == criteria.IRRIGANT_IN, -u["value"], u["value"])
        st = criteria.uo_stage_by_stay(u[["stay_id", "hour", "vol"]], tab.weights)
        st = st.join(icu.set_index("stay_id")["hadm_id"])
        per_adm = st.groupby("hadm_id").agg(uo=("uo_stage", "max"), ok=("assessable", "max"))
        adm["uo_ICU"] = per_adm["uo"].reindex(adm.index, fill_value=0).astype(int)
        adm["uo_assessable"] = per_adm["ok"].reindex(adm.index, fill_value=False).astype(bool)
        w = tab.weights.reindex(icu["stay_id"])
        meta.n_stays = len(icu)
        meta.n_stays_weight = int(((w >= criteria.MIN_WEIGHT) & (w <= criteria.MAX_WEIGHT)).sum())
        meta.n_stays_urine = int(icu["stay_id"].isin(u["stay_id"]).sum())
        meta.n_uo_assessable = int(adm["uo_assessable"].sum())

    # -- dialysis start (stage 3), non-ESRD only --
    adm["rrt_ICU"] = False
    adm["rrt_HOSP"] = False
    if cfg.use_rrt:
        r = tab.rrt_proc.merge(icu[["stay_id", "hadm_id", "in_s", "out_s"]], on="stay_id", how="inner")
        r["t"] = _secs(r["starttime"])
        icu_set = set(r.loc[(r["t"] >= r["in_s"]) & (r["t"] <= r["out_s"]), "hadm_id"].astype("int64"))
        icd_set = criteria.rrt_icd_hadm(tab.proc_icd)
        any_icu = adm["hadm_id"].isin(icu_set)
        any_hosp = any_icu | adm["hadm_id"].isin(icd_set)
        adm["rrt_ICU"] = any_icu & ~adm["ESRD"]
        adm["rrt_HOSP"] = any_hosp & ~adm["ESRD"]
        meta.n_rrt_icu, meta.n_rrt_hosp = int(adm["rrt_ICU"].sum()), int(adm["rrt_HOSP"].sum())
        meta.n_rrt_esrd_excluded = int((any_hosp & adm["ESRD"]).sum())

    # -- combined KDIGO stage for the primary baseline --
    pb = cfg.primary_baseline
    adm["scr_primary_HOSP"] = adm[f"scr_HOSP_{pb}"]
    adm["scr_primary_ICU"] = adm[f"scr_ICU_{pb}"]
    adm["stage_HOSP"] = np.maximum(adm["scr_primary_HOSP"], 3 * adm["rrt_HOSP"].astype(int))
    adm["stage_ICU"] = np.maximum.reduce([adm["scr_primary_ICU"].to_numpy(), adm["uo_ICU"].to_numpy(),
                                          3 * adm["rrt_ICU"].to_numpy().astype(int)])
    adm["hosp_first_qual_t"] = _col(scr_hosp[pb], "first_qual_t", adm.index, -1, "int64")
    # the confirmed incident target always uses the patient's own baseline (B2), whatever the primary baseline is
    adm["stage_HOSP_B2"] = np.maximum(adm["scr_HOSP_B2"], 3 * adm["rrt_HOSP"].astype(int))
    adm["hosp_first_qual_t_B2"] = _col(scr_hosp["B2"], "first_qual_t", adm.index, -1, "int64")
    adm["icu_transient"] = _col(scr_icu[pb], "transient", adm.index, False, bool)

    # elevated on arrival: first SCr >= 1.5 x B4 and a later value is back below 1.5 x B4
    if win_hosp.n_groups:
        thr = win_hosp.per_lab(win_hosp.aligned(b4)) * kdigo.STAGE_RATIOS[0]
        first_t = win_hosp.t[win_hosp.starts][win_hosp.g]
        later_low = (win_hosp.t > first_t) & (win_hosp.v < thr)
        falls = np.logical_or.reduceat(later_low, win_hosp.starts)
        elevated = (win_hosp.v[win_hosp.starts] >= thr[win_hosp.starts]) & falls
        adm["elevated_on_arrival"] = pd.Series(elevated, index=win_hosp.hadm).reindex(adm.index, fill_value=False).astype(bool)
    else:
        adm["elevated_on_arrival"] = False

    windows = {"hosp": win_hosp, "icu": win_icu, "b0_median": b0_median, "b0": b0}
    return adm, meta, windows
