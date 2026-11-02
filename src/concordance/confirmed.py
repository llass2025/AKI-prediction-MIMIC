"""
KDIGO-confirmed incident AKI target, and the overlap of the KDIGO and ICD targets (D6).

The pipeline label (incident_aki_label = 1) is ICD-based: first-ever AKI code, principal-diagnosis AKI
excluded. A code carries no timing, so some of these positives are AKI that was already there on arrival
(or never visible in creatinine). The confirmed target keeps a positive only when creatinine shows AKI
arising in hospital:

  CONFIRMED   KDIGO stage >= 1 over the whole hospitalization (creatinine vs the B2 baseline, or dialysis),
              with the first qualifying creatinine at least `min_onset_hours` after admission
  strict      positives that are not CONFIRMED are dropped, together with the patient
  light       only positives with a clear 'high on arrival, then falling' creatinine are dropped
              (and not CONFIRMED); everything else is kept

Negatives stay the ICD-negative admissions (optionally minus KDIGO-positive ones). The patient-level choice
is the pipeline's own (first positive, else last negative), so with no filter this reproduces target.parquet.
Row-level target files go next to target.parquet (gitignored); only suppressed aggregates reach reports/.
"""

import os

import numpy as np
import pandas as pd

from .stats import Suppressor, kappa

RULES = ("strict", "light")
ARRIVAL_PEAK_S = 24 * 3600          # a peak within 24 h of the first creatinine counts as 'on arrival'
FALL_ABS, FALL_RATIO = 0.3, 1.5     # KDIGO-sized fall from that peak (mg/dL, ratio)

_D6_ID = ["cohort", "stratum"]
D6_COLS = {
    "d6_target_overlap": _D6_ID + ["kdigo_target", "n", "n_icd_pos", "n_kdigo_pos", "both", "icd_only", "kdigo_only",
                                   "neither", "pct_icd_pos_also_kdigo", "pct_kdigo_pos_also_icd", "jaccard", "kappa"],
    "d6_three_way": _D6_ID + ["icd", "kdigo_b2_onset", "kdigo_b3", "n", "pct_of_cohort"],
    "d6_confirmation_funnel": ["section", "item", "n", "pct", "denominator"],
}


# ---------------------------------------------------------------------------
# per-admission flags
# ---------------------------------------------------------------------------

def arrival_fall(win) -> pd.Series:
    """
    True when the creatinine reached its running maximum within 24 h of the first measurement and a later value
    is at least 0.3 mg/dL or 1.5x below that maximum, i.e. AKI that was already there and is resolving.
    `win` is a kdigo.LabWindow (labs sorted by hadm, time).
    """
    res = np.zeros(win.n_groups, dtype=bool)
    for gi in range(win.n_groups):
        c = int(win.counts[gi])
        if c < 2:
            continue
        s = int(win.starts[gi])
        v, t = win.v[s:s + c], win.t[s:s + c]
        cm = np.maximum.accumulate(v)
        new = np.r_[True, v[1:] > cm[:-1]]
        pk = np.maximum.accumulate(np.where(new, np.arange(c), 0))     # index of the first value that reached each running max
        m, pki = cm[:-1], pk[:-1]
        later = v[1:]
        falls = ((m - later) >= FALL_ABS) | ((m / np.where(later > 0, later, np.nan)) >= FALL_RATIO)
        early = (t[pki] - t[0]) <= ARRIVAL_PEAK_S
        res[gi] = bool((falls & early).any())
    return pd.Series(res, index=win.hadm)


def confirmation_flags(adm: pd.DataFrame, win, min_onset_hours: float) -> pd.DataFrame:
    """Per-admission KDIGO onset facts (index = adm.index). Always the B2 baseline (the patient's own prior creatinine),
    whichever baseline is the primary one in D1-D5: B3 cannot tell in-hospital onset from AKI already present."""
    k_any = adm["stage_HOSP_B2"] >= 1
    t = adm["hosp_first_qual_t_B2"].to_numpy()
    onset_h = np.where(t >= 0, (t - adm["admit_s"].to_numpy()) / 3600.0, np.nan)    # -1 = no creatinine time (coded dialysis)
    f = pd.DataFrame(index=adm.index)
    f["kdigo_any"] = k_any.to_numpy()
    f["onset_hours"] = onset_h
    f["arrival_fall"] = arrival_fall(win).reindex(adm.index, fill_value=False).to_numpy()
    f["kdigo_b3"] = ((adm["scr_HOSP_B3"] >= 1) | adm["rrt_HOSP"]).to_numpy()
    f["kdigo_stage2"] = (adm["stage_HOSP_B2"] >= 2).to_numpy()
    f["n_scr"] = adm["n_scr_hosp"].to_numpy()
    for h in sorted({0.0, 24.0, 48.0, float(min_onset_hours)}):
        f[f"conf_{int(h)}h"] = confirmed(f, h).to_numpy()
    f["confirmed"] = confirmed(f, min_onset_hours).to_numpy()
    return f


def confirmed(f: pd.DataFrame, min_onset_hours: float) -> pd.Series:
    """KDIGO+ with the first qualifying creatinine at least min_onset_hours after admission (untimed dialysis only at 0 h)."""
    if min_onset_hours <= 0:
        return f["kdigo_any"].astype(bool)
    return f["kdigo_any"].astype(bool) & (f["onset_hours"] >= min_onset_hours)


# ---------------------------------------------------------------------------
# target construction
# ---------------------------------------------------------------------------

def label_status(inc: pd.DataFrame, f: pd.DataFrame, rule: str, drop_kdigo_neg: bool) -> pd.DataFrame:
    """Admission-level rows (labelled admissions only) with `kdigo_status` for the chosen rule ('none' = no filter)."""
    d = inc.merge(f[["confirmed", "arrival_fall", "onset_hours"]], left_on="hadm_id", right_index=True, how="left")
    d["confirmed"] = d["confirmed"].fillna(False).astype(bool)
    d["arrival_fall"] = d["arrival_fall"].fillna(False).astype(bool)
    pos = d["incident_aki_label"] == 1
    status = np.where(pos, "POS_KEPT", "NEG_KEPT")
    if rule == "strict":
        status = np.where(pos & ~d["confirmed"], "POS_DROPPED", status)
    elif rule == "light":
        status = np.where(pos & ~d["confirmed"] & d["arrival_fall"], "POS_DROPPED", status)
    elif rule != "none":
        raise ValueError(f"unknown rule {rule!r}")
    if drop_kdigo_neg:
        status = np.where(~pos & d["confirmed"], "NEG_DROPPED", status)
    d["kdigo_status"] = status
    return d


def dedup_confirmed(d: pd.DataFrame) -> pd.DataFrame:
    """The pipeline's patient rule on the kept rows; a patient with a dropped positive is excluded outright."""
    d = d.sort_values(["subject_id", "admittime", "hadm_id"], kind="mergesort")
    excluded = set(d.loc[d["kdigo_status"] == "POS_DROPPED", "subject_id"])
    pos = d[(d["kdigo_status"] == "POS_KEPT") & ~d["subject_id"].isin(excluded)].groupby("subject_id", sort=False).head(1)
    has_pos = set(d.loc[d["kdigo_status"] == "POS_KEPT", "subject_id"])
    neg = d[(d["kdigo_status"] == "NEG_KEPT") & ~d["subject_id"].isin(excluded | has_pos)].groupby("subject_id", sort=False).tail(1)
    return pd.concat([pos, neg], ignore_index=True)


def build_targets(inc: pd.DataFrame, f: pd.DataFrame, cfg) -> dict:
    """{'none': ..., 'strict': ..., 'light': ...} patient-level frames; 'none' is only for the reproduction check."""
    return {r: dedup_confirmed(label_status(inc, f, r, cfg.drop_kdigo_neg if r != "none" else False))
            for r in ("none",) + RULES}


def check_reproduces(reference: pd.DataFrame, none: pd.DataFrame) -> int:
    """The unfiltered rebuild must equal the existing patient-level target (patients and labels). Returns # differing hadm_ids."""
    a = reference.set_index("subject_id")["incident_aki_label"].astype(int)
    b = none.set_index("subject_id")["incident_aki_label"].astype(int)
    assert a.index.is_unique and b.index.is_unique, "patient appears twice in a patient-level target"
    assert set(a.index) == set(b.index), "unfiltered rebuild covers different patients than the existing target.parquet"
    assert (a.reindex(b.index) == b).all(), "unfiltered rebuild has different labels than the existing target.parquet"
    return len(set(reference["hadm_id"]) ^ set(none["hadm_id"]))


def paths(cfg) -> dict:
    base, ext = os.path.splitext(cfg.confirmed_out)
    return {"strict": cfg.confirmed_out, "light": f"{base}_light{ext}"}


def write_targets(built: dict, cfg) -> dict:
    p = paths(cfg)
    for r in RULES:
        t = built[r].copy()
        t["icd_incident_label"] = t["incident_aki_label"]
        t["kdigo_confirmed"] = t["confirmed"]
        cols = [c for c in t.columns if c not in ("confirmed",)]
        t[cols].to_parquet(p[r], index=False)
    return p


# ---------------------------------------------------------------------------
# D6 tables
# ---------------------------------------------------------------------------

def _strata(sub: pd.DataFrame):
    yield "all", sub
    yield "sex=F", sub[sub["female"]]
    yield "sex=M", sub[~sub["female"]]


def _row(sup, cohort, stratum, name, icd, k):
    icd, k = np.asarray(icd, bool), np.asarray(k, bool)
    both, icd_only = int((icd & k).sum()), int((icd & ~k).sum())
    k_only, neither = int((~icd & k).sum()), int((~icd & ~k).sum())
    cells = (both, icd_only, k_only, neither)
    n = sum(cells)
    return dict(cohort=cohort, stratum=stratum, kdigo_target=name, n=sup.count(n),
                n_icd_pos=sup.count(both + icd_only), n_kdigo_pos=sup.count(both + k_only),
                both=sup.count(both), icd_only=sup.count(icd_only), kdigo_only=sup.count(k_only),
                neither=sup.count(neither),
                pct_icd_pos_also_kdigo=sup.pct(both, both + icd_only, *cells),
                pct_kdigo_pos_also_icd=sup.pct(both, both + k_only, *cells),
                jaccard=sup.value(both / (both + icd_only + k_only) if both + icd_only + k_only else float("nan"), 4, *cells),
                kappa=sup.value(kappa(both, icd_only, k_only, neither), 4, *cells))


def overlap_tables(sub: pd.DataFrame, f: pd.DataFrame, cohort: str, cfg, sup: Suppressor, out) -> None:
    """sub: the modeling-cohort admissions (AKI_CORE = the model's own label). f: confirmation_flags."""
    f = f.reindex(sub.index)
    h = int(cfg.min_onset_hours)
    targets = {
        "B2_any_time": f["kdigo_any"],
        f"B2_onset_ge{h}h": f["confirmed"],
        "B3_any_time": f["kdigo_b3"],
        "B2_stage2plus": f["kdigo_stage2"],
    }
    for stratum, s in _strata(sub):
        fs = f.loc[s.index]
        for name, col in targets.items():
            out.add("d6_target_overlap", _row(sup, cohort, stratum, name, s["AKI_CORE"], col.loc[s.index]))
        icd = s["AKI_CORE"].to_numpy()
        k2, k3 = fs["confirmed"].to_numpy(), fs["kdigo_b3"].to_numpy()
        for i in (True, False):
            for a in (True, False):
                for b in (True, False):
                    n = int(((icd == i) & (k2 == a) & (k3 == b)).sum())
                    out.add("d6_three_way", dict(cohort=cohort, stratum=stratum, icd="+" if i else "-",
                                                 kdigo_b2_onset="+" if a else "-", kdigo_b3="+" if b else "-",
                                                 n=sup.count(n), pct_of_cohort=sup.pct(n, len(s))))


def funnel(sub: pd.DataFrame, f: pd.DataFrame, built: dict, cfg, sup: Suppressor, out) -> None:
    f = f.reindex(sub.index)
    icd = sub["AKI_CORE"].to_numpy()
    npos = int(icd.sum())
    add = lambda sec, item, n, den=None, dn="": out.add("d6_confirmation_funnel", dict(  # noqa: E731
        section=sec, item=item, n=sup.count(n), pct=sup.pct(n, den) if den else "", denominator=dn))
    add("cohort", "modeling cohort (MIMIC-IV, target.parquet)", len(sub))
    add("cohort", "ICD-positive (incident_aki_label = 1)", npos, len(sub), "modeling cohort")
    P = "ICD-positive"
    sel = lambda mask: int((icd & np.asarray(mask, bool)).sum())  # noqa: E731
    add("assessable", "  no creatinine in hospital", sel(f["n_scr"] == 0), npos, P)
    add("assessable", "  one creatinine only", sel(f["n_scr"] == 1), npos, P)
    add("kdigo", "  KDIGO+ at any time (creatinine vs B2, or dialysis)", sel(f["kdigo_any"]), npos, P)
    for h in sorted({0, 24, 48, int(cfg.min_onset_hours)}):
        add("kdigo", f"  KDIGO+ with first qualifying creatinine >= {h} h after admission", sel(f[f"conf_{h}h"]), npos, P)
    add("kdigo", "  KDIGO+ but first qualifying creatinine < %d h after admission" % int(cfg.min_onset_hours),
        sel(f["kdigo_any"] & ~f["confirmed"]), npos, P)
    add("arrival", "  high-on-arrival-then-falling creatinine pattern", sel(f["arrival_fall"]), npos, P)
    add("arrival", "    of which not KDIGO-confirmed", sel(f["arrival_fall"] & ~f["confirmed"]), npos, P)
    add("negatives", "ICD-negative admissions KDIGO-confirmed", int((~icd & f["confirmed"].to_numpy()).sum()),
        len(sub) - npos, "ICD-negative")
    for r in RULES:
        t = built[r]
        lab = t["incident_aki_label"].astype(int)
        add("final", f"rule={r}: patients in the new target", len(t), len(sub), "modeling cohort")
        add("final", f"rule={r}:   positives", int((lab == 1).sum()), npos, P)
        add("final", f"rule={r}:   negatives", int((lab == 0).sum()), len(sub) - npos, "ICD-negative")
    add("final", "rule=none rebuild: patients (equals the existing target)", len(built["none"]), len(sub), "modeling cohort")


def summary_lines(out, cfg) -> list:
    t = {k: out.frame(k) for k in D6_COLS}
    if t["d6_target_overlap"].empty:
        return []
    md = lambda rows, hdr: ["| " + " | ".join(hdr) + " |", "|" + "|".join("---" for _ in hdr) + "|"] + [  # noqa: E731
        "| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    L = ["## D6. KDIGO-confirmed incident target and overlap with the ICD target\n",
         "Confirmed = KDIGO stage >= 1 over the whole hospitalization (B2 baseline, creatinine + dialysis) "
         f"with the first qualifying creatinine >= {int(cfg.min_onset_hours)} h after admission. `strict` drops every ICD-positive that is not "
         "confirmed (and its patient); `light` drops only those with a clear high-on-arrival-then-falling creatinine. "
         "Row-level targets are written next to target.parquet; nothing below carries an ID.\n",
         "### Funnel (modeling cohort, MIMIC-IV)\n"]
    L += md([(r["item"], r["n"], r["pct"]) for _, r in t["d6_confirmation_funnel"].iterrows()], ["item", "n", "% of denominator"])
    L += ["", "### Overlap of the ICD target with each KDIGO target (modeling cohort)\n"]
    o = t["d6_target_overlap"]
    L += md([(r["kdigo_target"], r["stratum"], r["n_icd_pos"], r["n_kdigo_pos"], r["both"], r["icd_only"], r["kdigo_only"],
              r["pct_icd_pos_also_kdigo"], r["pct_kdigo_pos_also_icd"], r["jaccard"], r["kappa"]) for _, r in o.iterrows()],
            ["KDIGO target", "stratum", "ICD+", "KDIGO+", "both", "ICD only", "KDIGO only",
             "% ICD+ also KDIGO+", "% KDIGO+ also ICD+", "Jaccard", "kappa"])
    L += ["", "### ICD x confirmed-KDIGO x B3-KDIGO (all)\n"]
    w = t["d6_three_way"]
    w = w[w["stratum"] == "all"]
    L += md([(r["icd"], r["kdigo_b2_onset"], r["kdigo_b3"], r["n"], r["pct_of_cohort"]) for _, r in w.iterrows()],
            ["ICD", "KDIGO B2 confirmed", "KDIGO B3", "n", "% of cohort"])
    L.append("")
    return L


# ---------------------------------------------------------------------------
# entry point used by run.py
# ---------------------------------------------------------------------------

def run(adm: pd.DataFrame, win_hosp, modeling_sub: pd.DataFrame, modeling_ref: pd.DataFrame, cfg, sup: Suppressor, out, log) -> None:
    inc = pd.read_parquet(cfg.incident_target)
    inc = inc[(inc["version"] == "mimic4") & inc["incident_aki_label"].notna()].copy()
    inc["admittime"] = pd.to_datetime(inc["admittime"], errors="coerce")
    inc["incident_aki_label"] = inc["incident_aki_label"].astype(int)
    inc["hadm_id"] = inc["hadm_id"].astype("int64")
    log(f"Confirmed target: {sup.count(len(inc))} labelled MIMIC-IV admissions in {cfg.incident_target}")

    f = confirmation_flags(adm, win_hosp, cfg.min_onset_hours)
    built = build_targets(inc, f, cfg)
    diff = check_reproduces(modeling_ref, built["none"])
    log(f"Unfiltered rebuild equals the existing target.parquet ({sup.count(diff)} hadm_ids differ at equal patients and labels)")
    p = write_targets(built, cfg)
    log("Wrote " + ", ".join(p.values()))
    overlap_tables(modeling_sub, f, "modeling", cfg, sup, out)
    funnel(modeling_sub, f, built, cfg, sup, out)
    out.notes.append("Confirmed targets are MIMIC-IV only: MIMIC-III has no raw creatinine in this module, so it cannot be confirmed.")
