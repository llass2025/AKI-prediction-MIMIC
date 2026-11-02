"""
Flowchart numbers when AKI also uses KDIGO with the B3 baseline (opt-in). Two variants, same cohorts and boxes as
flowchart/compute_flowchart_counts.py (paper-run features_all.parquet, cmb_ckd / renal-impairment exclusions, CKD-survival exclusions):

  AND   AKI = incident_aki_label == 1 (ICD) and, at that admission, KDIGO stage >= 1 over the whole hospitalization with the
              B3 baseline (creatinine back-calculated for eGFR 75 from age and sex, plus dialysis). ICD-coded AKI without B3
              confirmation is a new excluded box. No AKI = ICD-negative, unchanged.
  OR    AKI = ICD-coded incident AKI, or an ICD-negative patient whose retained admission is B3-KDIGO-positive (index admission
              for the survival outcomes = that admission). No AKI = ICD-negative and B3-negative.

For AND, everything below AKI is the existing patient-level result restricted to the AKI patients. For OR the patients without
an AKI code have no existing survival record, so their outcomes are rebuilt from the admission-level file with the survival
script's logic (index = the B3-positive admission) and the lab-CKD script's detection. Both rebuilds are first run on the
ICD-coded AKI patients and must reproduce the verified numbers, otherwise the run stops. Only aggregate counts are written.
"""

import argparse
import importlib.util
import os
from pathlib import Path

import numpy as np
import pandas as pd

from . import data, mimic3
from .data import _CHUNK, _cached
from .config import Config
from .stats import Suppressor

FEATURES_DIR = os.environ.get("AKI_FEATURES_DIR", "features")     # folder holding features_all.parquet (set AKI_FEATURES_DIR for another run)

# (#, box, MIMIC-IV submitted, MIMIC-IV correct, MIMIC-III submitted, MIMIC-III correct, action) as given by the authors
REFERENCE = [
    (1, "All adults with admission(s)", 223452, 223452, 38552, 38552, "none"),
    (2, "Excluded (total)", 23630, 23630, 4371, 4371, "none"),
    (3, "Existing AKI at earliest admission", 2931, 2931, 402, 402, "none"),
    (4, "Pre-existing CKD", 11645, 11645, 2160, 2160, "reword"),
    (5, "Pre-existing ESRD/dialysis/kidney tx", 793, 793, 49, 49, "reword"),
    (6, "No sufficient admission data", 8261, 8261, 1760, 1760, "reword"),
    (7, "AKI", 26184, 26184, 6828, 6828, "none"),
    (8, "AKI, “Non-auto”", 16969, 15522, 5230, 3281, "fix"),
    (9, "No AKI", 173638, 173638, 27353, 27353, "none"),
    (10, "Excluded after AKI (total)", 1698, 1698, 227, 227, "none"),
    (11, "CKD/ESRD < 90 days", 798, 798, 25, 25, "none"),
    (12, "No follow-up data", 900, 900, 202, 202, "none"),
    (13, "Post-AKI analysis cohort", 24486, 24486, 6601, 6601, "none"),
    (14, "CKD (≥ 90 d)", 2472, 2742, 112, 112, "fix"),
    (15, "CKD, lab-based", 3063, 3063, 625, 625, "none"),
    (16, "PostCKD/ESRD (≥ 90 d)", 121, 121, 21, 21, "none"),
    (17, "Death", 3762, 3862, 3898, 3898, "fix"),
    (18, "Censored", 17761, 17761, 2570, 2570, "none"),
]
PRE_HOURS = 24.0     # sensitivity window: creatinine from this many hours before admittime (ED values) also counts
KEY = {7: "aki", 8: "nonauto", 9: "noaki", 10: "excl_total", 11: "excl_ckd", 12: "excl_nofu", 13: "post",
       14: "ckd", 15: "labckd", 16: "postckd", 17: "death", 18: "censor"}


# ---------------------------------------------------------------------------
# CKD-survival exclusions, rebuilt with the logic of ckd_survival/src/incident_ckd_target.py
# ---------------------------------------------------------------------------

def survival_pre_exclusion(adm: pd.DataFrame, index_hadm: dict = None) -> pd.DataFrame:
    """
    One row per patient before the <90 d / no-follow-up exclusions: event_type, time_days, reason (ckd / nofu / ''),
    plus age, female and index_admit. The index is the first is_aki admission, or, for patients in `index_hadm`
    ({(version, subject_id): hadm_id}), that admission (patients with neither are skipped).
    """
    index_hadm = index_hadm or {}
    adm = adm.copy()
    for c in ("admittime", "dischtime"):
        adm[c] = pd.to_datetime(adm[c], errors="coerce")
    rows = []
    for (ver, sid), g in adm.sort_values("admittime").groupby(["version", "subject_id"], sort=False):
        if (ver, sid) in index_hadm:
            pick = g[g["hadm_id"] == index_hadm[(ver, sid)]]
            assert len(pick) == 1, f"index admission {index_hadm[(ver, sid)]} not found for patient {sid}"
            aki = pick
        else:
            aki = g[g["is_aki"] == 1]
        if aki.empty:
            continue
        index = aki.iloc[0]["admittime"]
        events = []
        ckd = g[(g["event_ckd_flag"] == 1) & (g["admittime"] > index)]
        if not ckd.empty:
            events.append(("CKD", (ckd.iloc[0]["admittime"] - index).days))
        post = g[(g["event_postckd_flag"] == 1) & (g["admittime"] > index)]
        if not post.empty:
            events.append(("PostCKD", (post.iloc[0]["admittime"] - index).days))
        death = g[(g["event_death_flag"] == 1) & (g["dischtime"] >= index)]
        if not death.empty:
            events.append(("Death", (death.iloc[0]["dischtime"] - index).days))
        if not events:
            events.append(("Censor", (g["dischtime"].max() - index).days))
        ev, t = min(events, key=lambda x: x[1])
        is_ck = ev in ("CKD", "PostCKD")
        bad_t = pd.isna(t) or t <= 0
        reason = "ckd" if is_ck and (pd.isna(t) or t < 90 or t <= 0) else ("nofu" if (not is_ck) and bad_t else "")
        gender = aki.iloc[0].get("gender", "")
        rows.append(dict(version=ver, subject_id=sid, event_type=ev, time_days=t, reason=reason,
                         age=aki.iloc[0].get("age", np.nan), female=str(gender).upper().startswith("F"), index_admit=index))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Counts
# ---------------------------------------------------------------------------

def block(aki: pd.DataFrame, n_noaki: int, surv: pd.DataFrame, surv_lab: pd.DataFrame, pre: pd.DataFrame) -> dict:
    """Rows 7-18 for the AKI patients in `aki` (needs subject_id, is_autocar)."""
    sids = set(aki["subject_id"])
    s = surv[surv["subject_id"].isin(sids)]
    p = pre[pre["subject_id"].isin(sids)]
    out = dict(aki=len(aki), nonauto=int((aki["is_autocar"] == 0).sum()), noaki=n_noaki,
               excl_total=len(aki) - len(s["subject_id"].unique()),
               excl_ckd=int((p["reason"] == "ckd").sum()), excl_nofu=int((p["reason"] == "nofu").sum()),
               post=int(s["subject_id"].nunique()),
               ckd=int((s["event_type"] == "CKD").sum()), postckd=int((s["event_type"] == "PostCKD").sum()),
               death=int((s["event_type"] == "Death").sum()), censor=int((s["event_type"] == "Censor").sum()),
               labckd=int(((surv_lab["event_type"] == "LabCKD") & surv_lab["subject_id"].isin(sids)).sum()))
    assert out["excl_ckd"] + out["excl_nofu"] == out["excl_total"], "exclusion reasons do not add up to the excluded total"
    assert out["ckd"] + out["postckd"] + out["death"] + out["censor"] == out["post"], "outcomes do not add up to the cohort"
    return out


def _lab_module():
    p = Path(__file__).resolve().parents[2] / "ckd_definition_sensitivity" / "lab_ckd_definition.py"
    spec = importlib.util.spec_from_file_location("lab_ckd_definition", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def labckd_labs(root: str, version: str, cfg: Config) -> dict:
    """{subject_id: creatinine rows} with the lab-CKD script's own item selection (label regex) and cleaning."""
    lab = _lab_module()
    root = Path(root)
    if version == "mimic4":
        li, src = pd.read_csv(root / "hosp/d_labitems.csv.gz"), root / "hosp/labevents.csv.gz"
    else:
        li, src = pd.read_csv(mimic3._csv(root, "D_LABITEMS")), mimic3._csv(root, "LABEVENTS")
    ids = lab.select_creatinine_itemids(li)

    def build():
        parts = []
        for c in pd.read_csv(src, usecols=lambda x: x.lower() in ("subject_id", "hadm_id", "itemid", "charttime", "valuenum"),
                             chunksize=_CHUNK):
            c = c.rename(columns=str.lower)
            parts.append(c[c["itemid"].isin(ids)])
        return pd.concat(parts, ignore_index=True)
    raw = _cached(cfg, f"labckd_labs_{version}_{len(ids)}items", [src], build)
    return {sid: g for sid, g in lab.prep_creatinine_labs(raw, ids).groupby("subject_id")}


def lab_ckd_flags(rec: pd.DataFrame, labs: dict) -> pd.Series:
    """lab_ckd_flag (1/0) per subject for the patients that stay in the survival cohort, as ckd_definition_sensitivity does."""
    lab = _lab_module()
    empty = next(iter(labs.values())).iloc[0:0]
    keep = rec[rec["reason"] == ""]
    out = {int(r.subject_id): lab.detect_lab_ckd(int(r.subject_id), r.index_admit, float(r.age), bool(r.female),
                                                 labs.get(r.subject_id, empty), 90, 90)["lab_ckd_flag"]
           for r in keep.itertuples()}
    return pd.Series(out, dtype=int)


def compute(cfg: Config, a, log) -> dict:
    """Returns {version: {"icd": block, "b3": block (AND), "or": block, "extra": {...}}}."""
    feat = pd.read_parquet(a.features_all, columns=["version", "subject_id", "hadm_id", "incident_aki_label", "cmb_ckd",
                                                    "renal_impaired_at_adm"])
    mod = feat[(feat["cmb_ckd"] != 1) & (feat["renal_impaired_at_adm"] != 1)]
    inc = pd.read_parquet(a.incident_target, columns=["version", "hadm_id", "is_autocar"])
    surv_all = pd.read_csv(a.surv_icd)
    lab_all = pd.read_csv(a.surv_lab)
    adm_file = pd.read_csv(a.surv_adm)
    pre_all = survival_pre_exclusion(adm_file)

    log("MIMIC-IV: creatinine, B3 and dialysis flags ...")
    tab = data.load_tables(cfg)
    adm4, _, _ = data.build_admission_table(tab, cfg)
    f4 = ((adm4["scr_HOSP_B3"] >= 1) | adm4["rrt_HOSP"])
    log("MIMIC-IV: generic flag path vs the full admission table ...")
    gen = mimic3.hospital_flags(adm4[["hadm_id", "subject_id", "admittime", "dischtime", "age", "female"]],
                                tab.labs_raw, set(adm4.index[adm4["rrt_HOSP"]]), set(adm4.index[adm4["ESRD"]]))
    same = (gen["scr_B3"].to_numpy() == adm4.loc[gen.index, "scr_HOSP_B3"].to_numpy()) & \
           (gen["scr_B2"].to_numpy() == adm4.loc[gen.index, "scr_HOSP_B2"].to_numpy())
    assert same.all(), f"generic flag path differs from the admission table on {int((~same).sum())} MIMIC-IV admissions"
    log(f"  identical on {len(same)} MIMIC-IV admissions")

    log("MIMIC-IV: B3 flags with the 24 h before admission added ...")
    f4p = mimic3.hospital_flags(adm4[["hadm_id", "subject_id", "admittime", "dischtime", "age", "female"]], tab.labs_raw,
                                set(adm4.index[adm4["rrt_HOSP"]]), set(adm4.index[adm4["ESRD"]]), pre_hours=PRE_HOURS)
    log("MIMIC-III: reading raw v1.4 tables ...")
    f3_frame = mimic3.mimic3_flags(a.mimic3_dir, cfg)
    f3p = mimic3.mimic3_flags(a.mimic3_dir, cfg, pre_hours=PRE_HOURS)
    flags = {"mimic4": f4, "mimic3": f3_frame["kdigo_b3"]}
    flags_pre = {"mimic4": f4p["kdigo_b3"], "mimic3": f3p["kdigo_b3"]}
    flags_nad = {"mimic4": f4p["kdigo_b3n"], "mimic3": f3p["kdigo_b3n"]}
    roots = {"mimic4": a.mimic4_dir, "mimic3": a.mimic3_dir}

    res = {}
    for ver in ("mimic4", "mimic3"):
        m = mod[mod["version"] == ver]
        auto = inc[inc["version"] == ver][["hadm_id", "is_autocar"]].drop_duplicates("hadm_id")
        aki = m[m["incident_aki_label"] == 1].merge(auto, on="hadm_id", how="left")
        assert aki["is_autocar"].notna().all(), "AKI admission without an is_autocar value"
        ctrl = m[m["incident_aki_label"] == 0].merge(auto, on="hadm_id", how="left")
        n_noaki = len(ctrl)
        surv, lab, pre = (x[x["version"] == ver] for x in (surv_all, lab_all, pre_all))
        known = flags[ver].reindex(aki["hadm_id"])
        b3 = known.fillna(False).astype(bool).to_numpy()
        ctrl_b3 = flags[ver].reindex(ctrl["hadm_id"]).fillna(False).astype(bool).to_numpy()
        b3p = flags_pre[ver].reindex(aki["hadm_id"]).fillna(False).astype(bool).to_numpy()
        ctrl_b3p = flags_pre[ver].reindex(ctrl["hadm_id"]).fillna(False).astype(bool).to_numpy()
        assert (b3p | ~b3).all(), "adding the pre-admission hours removed a B3-positive admission"
        b3n = flags_nad[ver].reindex(aki["hadm_id"]).fillna(False).astype(bool).to_numpy()
        ctrl_b3n = flags_nad[ver].reindex(ctrl["hadm_id"]).fillna(False).astype(bool).to_numpy()
        assert (b3n | ~b3p).all(), "the nadir baseline removed a B3-positive admission"
        res[ver] = dict(
            icd=block(aki, n_noaki, surv, lab, pre),
            b3=block(aki[b3], n_noaki, surv, lab, pre),
            b3p=block(aki[b3p], n_noaki, surv, lab, pre),
            b3n=block(aki[b3n], n_noaki, surv, lab, pre),
            extra=dict(icd_not_b3=int((~b3).sum()), icd_not_b3p=int((~b3p).sum()), icd_not_b3n=int((~b3n).sum()), no_flag_row=int(known.isna().sum()),
                       controls_b3pos=int(ctrl_b3.sum()), controls_b3pos_p=int(ctrl_b3p.sum()),
                       controls_b3pos_n=int(ctrl_b3n.sum()), n_controls=n_noaki))

        # ---- OR variant: patients with no AKI code but a B3-positive retained admission join the AKI group ----
        log(f"{ver}: survival outcomes for B3-only patients ...")
        only = ctrl[ctrl_b3]
        assert ctrl["is_autocar"].notna().all(), "control admission without an is_autocar value"
        a_ver = adm_file[adm_file["version"] == ver]
        rec = survival_pre_exclusion(a_ver, {(ver, int(s)): int(h) for s, h in zip(only["subject_id"], only["hadm_id"])})
        icd_ids, only_ids = set(aki["subject_id"]), set(only["subject_id"])
        assert only_ids <= set(rec["subject_id"]), "a B3-only patient has no admission record"
        rec_icd = rec[rec["subject_id"].isin(icd_ids)]
        # the rebuild must equal the verified survival file for the ICD-coded patients
        v_surv = surv[surv["subject_id"].isin(icd_ids)]
        got = rec_icd[rec_icd["reason"] == ""]["event_type"].value_counts().to_dict()
        want = v_surv["event_type"].value_counts().to_dict()
        assert got == want, f"{ver}: rebuilt survival outcomes {got} != verified {want}"
        log(f"  {ver}: rebuilt survival outcomes equal the verified file for the ICD-coded patients")
        labs = labckd_labs(roots[ver], ver, cfg)
        flag = lab_ckd_flags(rec[rec["subject_id"].isin(icd_ids | only_ids)], labs)
        n_lab_icd = int(flag.reindex(list(icd_ids)).fillna(0).sum())
        want_lab = int(((lab["event_type"] == "LabCKD") & lab["subject_id"].isin(icd_ids)).sum())
        assert n_lab_icd == want_lab, f"{ver}: rebuilt lab-CKD events {n_lab_icd} != verified {want_lab}"
        log(f"  {ver}: rebuilt lab-CKD events equal the verified {want_lab}")

        union = pd.concat([aki[["subject_id", "is_autocar"]], only[["subject_id", "is_autocar"]]], ignore_index=True)
        rec_u = rec[rec["subject_id"].isin(icd_ids | only_ids)]
        surv_u = rec_u[rec_u["reason"] == ""][["subject_id", "event_type"]]
        lab_u = pd.DataFrame({"subject_id": flag.index[flag == 1], "event_type": "LabCKD"})
        res[ver]["or"] = block(union, n_noaki - len(only), surv_u, lab_u, rec_u[["subject_id", "reason"]])
        res[ver]["extra"].update(b3_only=len(only))
    return res


def _rows(res, sup, variant):
    """Table rows for the AND or OR variant."""
    fmt = lambda n: sup.count(n)  # noqa: E731
    main, comp = [], []
    for num, box, s4, c4, s3, c3, act in REFERENCE:
        k = KEY.get(num)
        v = {"mimic4": (c4, s4), "mimic3": (c3, s3)}
        icd_v = {ver: res[ver]["icd"][k] if k else v[ver][0] for ver in v}
        new_v = {ver: res[ver][variant][k] if k else v[ver][0] for ver in v}
        for ver in v:           # the ICD-only column must equal the verified numbers
            assert icd_v[ver] == v[ver][0], f"row {num} {box} {ver}: recomputed {icd_v[ver]} != verified {v[ver][0]}"
        changed = any(new_v[ver] != v[ver][0] for ver in v)
        new_act = "fix" if (changed or any(v[ver][0] != v[ver][1] for ver in v)) else act
        main.append([str(num), box, fmt(s4), fmt(new_v["mimic4"]), fmt(s3), fmt(new_v["mimic3"]), new_act])
        comp.append([str(num), box] + [fmt(x) for x in (s4, icd_v["mimic4"], new_v["mimic4"], s3, icd_v["mimic3"], new_v["mimic3"])])
        if num == 7 and variant in ("b3", "b3p", "b3n"):
            kx = {"b3": "icd_not_b3", "b3p": "icd_not_b3p", "b3n": "icd_not_b3n"}[variant]
            e4, e3 = res["mimic4"]["extra"][kx], res["mimic3"]["extra"][kx]
            main.append(["7a", "ICD-coded AKI, not B3-confirmed (excluded)", "n/a", fmt(e4), "n/a", fmt(e3), "add"])
            comp.append(["7a", "ICD-coded AKI, not B3-confirmed (excluded)", "n/a", "0", fmt(e4), "n/a", "0", fmt(e3)])
        if num == 7 and variant == "or":
            x4, x3 = res["mimic4"]["extra"], res["mimic3"]["extra"]
            for tag, label, k4, k3 in (
                    ("7a", "  of which ICD and B3", res["mimic4"]["b3"]["aki"], res["mimic3"]["b3"]["aki"]),
                    ("7b", "  of which ICD only (no B3)", x4["icd_not_b3"], x3["icd_not_b3"]),
                    ("7c", "  of which B3 only (no AKI code)", x4["b3_only"], x3["b3_only"])):
                main.append([tag, label, "n/a", fmt(k4), "n/a", fmt(k3), "add"])
                comp.append([tag, label, "n/a", "n/a", fmt(k4), "n/a", "n/a", fmt(k3)])
    return main, comp


def build_tables(res: dict, sup: Suppressor) -> dict:
    out = {}
    for variant, tag in (("b3", "AND"), ("b3p", f"AND, +{int(PRE_HOURS)} h pre-adm."),
                         ("b3n", f"AND, +{int(PRE_HOURS)} h, nadir"), ("or", "OR")):
        main, comp = _rows(res, sup, variant)
        out[variant] = (
            pd.DataFrame(main, columns=["#", "Box", "MIMIC-IV submitted", f"MIMIC-IV correct (ICD {tag} B3)", "MIMIC-III submitted",
                                        f"MIMIC-III correct (ICD {tag} B3)", "Action"]),
            pd.DataFrame(comp, columns=["#", "Box", "MIMIC-IV submitted", "MIMIC-IV ICD only", f"MIMIC-IV ICD {tag} B3",
                                        "MIMIC-III submitted", "MIMIC-III ICD only", f"MIMIC-III ICD {tag} B3"]))
    return out


def naki_only_table(res: dict, variant: str, sup: Suppressor) -> pd.DataFrame:
    """The verified table with only row 7 (n AKI) replaced by the `variant` count; every other row keeps its verified value."""
    rows = []
    for num, box, s4, c4, s3, c3, act in REFERENCE:
        n4, n3, new_act = c4, c3, act
        if num == 7:
            n4, n3, new_act = res["mimic4"][variant]["aki"], res["mimic3"][variant]["aki"], "fix"
        rows.append([str(num), box, sup.count(s4), sup.count(n4), sup.count(s3), sup.count(n3), new_act])
    return pd.DataFrame(rows, columns=["#", "Box", "MIMIC-IV submitted", "MIMIC-IV correct", "MIMIC-III submitted",
                                       "MIMIC-III correct", "Action"])


def md(df: pd.DataFrame) -> str:
    rows = ["| " + " | ".join(df.columns) + " |", "|" + "|".join("---" for _ in df.columns) + "|"]
    rows += ["| " + " | ".join(r) + " |" for r in df.astype(str).values.tolist()]
    return "\n".join(rows)


def run(a) -> None:
    cfg = Config(mimic_dir=a.mimic4_dir, cache_dir=a.cache_dir, min_cell=a.min_cell, use_uo=False, use_rrt=True,
                 quiet=a.quiet)
    log = (lambda m: None) if a.quiet else (lambda m: print(m, flush=True))
    sup = Suppressor(a.min_cell)
    res = compute(cfg, a, log)
    tabs = build_tables(res, sup)
    os.makedirs(a.outdir, exist_ok=True)
    out = Path(a.outdir)
    tabs["b3"][0].to_csv(out / "flowchart_b3_table.csv", index=False, lineterminator="\n")
    tabs["b3"][1].to_csv(out / "flowchart_b3_comparison.csv", index=False, lineterminator="\n")
    tabs["b3p"][0].to_csv(out / "flowchart_b3_pre24_table.csv", index=False, lineterminator="\n")
    tabs["b3p"][1].to_csv(out / "flowchart_b3_pre24_comparison.csv", index=False, lineterminator="\n")
    tabs["b3n"][0].to_csv(out / "flowchart_b3_nadir_table.csv", index=False, lineterminator="\n")
    tabs["b3n"][1].to_csv(out / "flowchart_b3_nadir_comparison.csv", index=False, lineterminator="\n")
    tabs["or"][0].to_csv(out / "flowchart_or_table.csv", index=False, lineterminator="\n")
    for variant, name in (("b3", "b3"), ("b3p", "b3_pre24"), ("b3n", "b3_nadir")):
        naki_only_table(res, variant, sup).to_csv(out / f"flowchart_naki_only_{name}.csv", index=False, lineterminator="\n")
    tabs["or"][1].to_csv(out / "flowchart_or_comparison.csv", index=False, lineterminator="\n")
    ex = {v: res[v]["extra"] for v in res}
    c = sup.count
    common = [
        "Rows 1-6 do not depend on the AKI definition and are copied from the verified table.",
        "'ICD only' is recomputed here and reproduces every verified number (rows 7-18); the run stops otherwise.",
        "B3 = KDIGO stage >= 1 over the whole hospitalization, creatinine vs the baseline for eGFR 75 (age, sex), or dialysis.",
        "MIMIC-III uses blood creatinine itemid 50912 only for B3; the pipeline's regex-based item selection would also take "
        "pleural, ascites, joint and body-fluid items. Lab-based CKD uses that script's own item selection.",
    ]
    notes_and = [
        "AKI = ICD-coded incident AKI and B3-KDIGO-positive at the incident admission; ICD-coded AKI without B3 confirmation is excluded (7a). "
        "No AKI is the ICD-negative group, unchanged "
        f"(B3-positive among them: MIMIC-IV {c(ex['mimic4']['controls_b3pos'])} of {c(ex['mimic4']['n_controls'])}, "
        f"MIMIC-III {c(ex['mimic3']['controls_b3pos'])} of {c(ex['mimic3']['n_controls'])}).",
        "Rows below AKI are the verified patient-level results restricted to the remaining AKI patients.",
    ]
    notes_pre = [
        f"Same as A, but a creatinine measured up to {int(PRE_HOURS)} h before the admission time (typically the ED draw, whatever its hadm_id) "
        "also counts, with the 48 h rise rule and the B3 baseline applied across that extended window. Admissions that were B3-positive in A "
        "stay positive. These patients most likely arrived with AKI, so A2 trades incident-onset purity for agreement with the ICD code. "
        f"B3-positive among No AKI: MIMIC-IV {c(ex['mimic4']['controls_b3pos_p'])} of {c(ex['mimic4']['n_controls'])}, "
        f"MIMIC-III {c(ex['mimic3']['controls_b3pos_p'])} of {c(ex['mimic3']['n_controls'])} (they stay in No AKI).",
    ]
    notes_nad = [
        f"Same window as A2, but the baseline is the lower of the eGFR-75 estimate and the patient's own lowest creatinine in that window "
        "(any order), so a creatinine that falls after arrival counts against that lowest value; this is the minimum-inpatient-creatinine "
        "surrogate described by Siew 2010. It closes the most of the remaining gap but also flags many more ICD-negative patients: "
        f"B3-positive among No AKI is {c(ex['mimic4']['controls_b3pos_n'])} of {c(ex['mimic4']['n_controls'])} in MIMIC-IV and "
        f"{c(ex['mimic3']['controls_b3pos_n'])} of {c(ex['mimic3']['n_controls'])} in MIMIC-III (they stay in No AKI).",
    ]
    notes_or = [
        "AKI = ICD-coded incident AKI, or an ICD-negative patient whose retained admission is B3-KDIGO-positive. No AKI = ICD-negative and B3-negative.",
        "For the B3-only patients (7c) the index admission is that retained admission. Their CKD / PostCKD / death / censoring and 90-day "
        "exclusions were rebuilt with the logic of ckd_survival/src/incident_ckd_target.py, and lab-based CKD with "
        "ckd_definition_sensitivity/lab_ckd_definition.py. Both rebuilds were first run on the ICD-coded AKI patients and reproduced the "
        "verified survival counts and the verified lab-CKD counts exactly.",
        "Patients are classed by the retained (representative) admission, not by their first B3-positive admission.",
    ]
    text = ("# Flowchart numbers with KDIGO-B3\n\n## A. AKI = ICD and B3\n\n" + md(tabs["b3"][0])
            + "\n\n" + "\n".join(f"- {n}" for n in notes_and)
            + f"\n\n## A2. AKI = ICD and B3, with the {int(PRE_HOURS)} h before admission added to the KDIGO window\n\n" + md(tabs["b3p"][0])
            + "\n\n" + "\n".join(f"- {n}" for n in notes_pre)
            + f"\n\n## A3. AKI = ICD and B3, {int(PRE_HOURS)} h pre-admission window and nadir baseline\n\n" + md(tabs["b3n"][0])
            + "\n\n" + "\n".join(f"- {n}" for n in notes_nad)
            + "\n\n## B. AKI = ICD or B3\n\n" + md(tabs["or"][0])
            + "\n\n" + "\n".join(f"- {n}" for n in notes_or)
            + "\n\n## Comparison with ICD only (A)\n\n" + md(tabs["b3"][1])
            + "\n\n## Comparison with ICD only (A2)\n\n" + md(tabs["b3p"][1])
            + "\n\n## Comparison with ICD only (A3)\n\n" + md(tabs["b3n"][1])
            + "\n\n## Comparison with ICD only (B)\n\n" + md(tabs["or"][1])
            + "\n\n## n AKI only (all other rows keep their verified values)\n\n"
            + md(naki_only_table(res, "b3n", sup))
            + "\n\nRow 7 above uses A3. A: " + f"{c(res['mimic4']['b3']['aki'])} / {c(res['mimic3']['b3']['aki'])}; "
            + f"A2: {c(res['mimic4']['b3p']['aki'])} / {c(res['mimic3']['b3p']['aki'])} (MIMIC-IV / MIMIC-III). "
            + "Rows 10 and 13 still describe all ICD-coded AKI patients, so row 7 no longer equals row 10 + row 13.\n"
            + "\n\n## Notes common to both\n\n" + "\n".join(f"- {n}" for n in common) + "\n")
    (out / "flowchart_b3.md").write_text(text)
    log(text)


def parse_args(argv=None):
    d = Config()
    ap = argparse.ArgumentParser(prog="python -m src.concordance.flowchart",
                                 description="Flowchart counts with AKI = ICD and B3-KDIGO (aggregate output only).")
    ap.add_argument("--mimic4-dir", default=d.mimic_dir)
    ap.add_argument("--mimic3-dir", default="../data/mimic-iii-clinical-database-1.4")
    ap.add_argument("--features-all", default=os.path.join(FEATURES_DIR, "features_all.parquet"),
                    help="Paper-run feature table that defines the modeling cohort (cmb_ckd / renal-impairment exclusions).")
    ap.add_argument("--surv-icd", default="ckd_survival_no_cmb_ckd/incident_ckd_survival.csv")
    ap.add_argument("--surv-lab", default="ckd_survival_no_cmb_ckd/incident_ckd_survival_labckd.csv")
    ap.add_argument("--surv-adm", default="ckd_survival_no_cmb_ckd/incident_ckd_admission.csv",
                    help="Admission-level file the survival target is built from (for the exclusion reasons).")
    ap.add_argument("--incident-target", default=d.incident_target, help="incident_aki_target.parquet (is_autocar).")
    ap.add_argument("--outdir", default="reports/flowchart_b3")
    ap.add_argument("--cache-dir", default=d.cache_dir)
    ap.add_argument("--min-cell", type=int, default=d.min_cell)
    ap.add_argument("--quiet", action="store_true")
    return ap.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
