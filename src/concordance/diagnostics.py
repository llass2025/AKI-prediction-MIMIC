"""
D1-D5 concordance tables. Every table is built from in-memory admission-level
data and emitted as suppressed, aggregate-only rows (see stats.Suppressor).

2x2 convention (KDIGO is the reference, ICD is the "test"):
  tp = KDIGO+ & ICD+   fp = KDIGO- & ICD+   fn = KDIGO+ & ICD-   tn = KDIGO- & ICD-

`criteria` says which KDIGO criteria produced the label: SCR_ONLY, or the full set the
window supports (W_ICU: SCR+UO+RRT, W_HOSP: SCR+RRT; urine output is only charted in the ICU).
"""

from collections import defaultdict
from typing import Dict, List

import numpy as np
import pandas as pd

from .config import BASELINES, Config
from .confirmed import D6_COLS
from .stats import Suppressor, bootstrap_kappa_ci, kappa, wilson

THRESHOLDS = {"any": lambda s: s >= 1, "ge2": lambda s: s >= 2, "eq3": lambda s: s == 3}
STAGE_LABELS = (0, 1, 2, 3)
D2_GROUPS = ("SCR_ONLY", "UO_ONLY", "BOTH", "RRT_ONLY")

_ID = ["cohort", "window", "icd_set", "baseline", "criteria", "stratum"]
_2X2 = ["kdigo_threshold", "n", "tp", "fp", "fn", "tn",
        "sensitivity", "sens_ci_low", "sens_ci_high",
        "specificity", "spec_ci_low", "spec_ci_high",
        "ppv", "ppv_ci_low", "ppv_ci_high",
        "npv", "npv_ci_low", "npv_ci_high",
        "kappa", "kappa_ci_low", "kappa_ci_high"]
_D4 = ["direction", "level", "reason", "parent", "n", "pct_of_direction", "note"]

COLS = {
    "d1_agreement_by_stage": _ID + ["kdigo_stage", "n_adm", "n_icd_pos", "pct_icd_pos", "pct_ci_low", "pct_ci_high"],
    "d1_two_by_two": _ID + _2X2,
    "d2_criterion": _ID + ["group", "n", "pct_of_kdigo_pos", "pct_stage1", "pct_stage2", "pct_stage3",
                           "n_icd_pos", "pct_icd_pos", "status", "note"],
    "d3_baseline_incidence": _ID + ["category", "n_adm", "pct_cohort", "pct_ci_low", "pct_ci_high"],
    "d3_baseline_agreement": _ID + _2X2,
    "d3_variant_pairwise": _ID[:4] + ["baseline_b", "criteria", "stratum", "n", "n_same", "n_diff", "pct_same"],
    "d4_discordance_icd_pos": _ID + _D4,
    "d4_discordance_kdigo_pos": _ID + _D4,
    "d5_stratified": _ID + _2X2,
}
COLS.update(D6_COLS)


def crit_pairs(window: str, cfg: Config):
    """[(criteria label, admission-table stage column)], SCR_ONLY first, full criteria last."""
    scr = ("SCR_ONLY", f"scr_primary_{'ICU' if window == 'W_ICU' else 'HOSP'}")
    parts = ["SCR"] + (["UO"] if cfg.use_uo and window == "W_ICU" else []) + (["RRT"] if cfg.use_rrt else [])
    full = "+".join(parts)
    return [scr] if full == "SCR" else [scr, (full, f"stage_{'ICU' if window == 'W_ICU' else 'HOSP'}")]


def full_criteria(window: str, cfg: Config):
    return crit_pairs(window, cfg)[-1]


class Collector:
    def __init__(self):
        self.tables: Dict[str, List[dict]] = defaultdict(list)
        self.anomalies: List[str] = []
        self.notes: List[str] = []

    def add(self, table: str, row: dict) -> None:
        self.tables[table].append(row)

    def frame(self, table: str) -> pd.DataFrame:
        return pd.DataFrame(self.tables.get(table, []), columns=COLS[table])


def _cells(kd, icd):
    kd = np.asarray(kd, dtype=bool)
    icd = np.asarray(icd, dtype=bool)
    return (int((kd & icd).sum()), int((~kd & icd).sum()),
            int((kd & ~icd).sum()), int((~kd & ~icd).sum()))


def _prop(sup, k, d, cells):
    if d <= 0:
        return ("NA", "NA", "NA")
    lo, hi = wilson(k, d)
    return (sup.value(k / d, 4, *cells), sup.value(lo, 4, *cells), sup.value(hi, 4, *cells))


def two_by_two_row(ids, thr, cells, sup: Suppressor, cfg: Config, n_expected: int, key: str) -> dict:
    tp, fp, fn, tn = cells
    n = tp + fp + fn + tn
    assert n == n_expected, f"2x2 cells sum to {n}, expected cohort size {n_expected} ({key})"
    row = dict(ids, kdigo_threshold=thr, n=sup.count(n),
               tp=sup.count(tp), fp=sup.count(fp), fn=sup.count(fn), tn=sup.count(tn))
    for name, (k, d) in {"sens": (tp, tp + fn), "spec": (tn, tn + fp),
                         "ppv": (tp, tp + fp), "npv": (tn, tn + fn)}.items():
        v, lo, hi = _prop(sup, k, d, cells)
        long = {"sens": "sensitivity", "spec": "specificity"}.get(name, name)
        row[long], row[f"{name}_ci_low"], row[f"{name}_ci_high"] = v, lo, hi
    if sup.ok(*cells):
        klo, khi = bootstrap_kappa_ci(cells, cfg.n_boot, cfg.seed, key)
        row["kappa"] = sup.value(kappa(*cells), 4, *cells)
        row["kappa_ci_low"], row["kappa_ci_high"] = sup.value(klo, 4, *cells), sup.value(khi, 4, *cells)
    else:
        row["kappa"] = row["kappa_ci_low"] = row["kappa_ci_high"] = "suppressed"
    return row


def stage_rows(ids, stage, icd, sup: Suppressor, out: Collector, label: str) -> None:
    stage = np.asarray(stage)
    icd = np.asarray(icd, dtype=bool)
    total = 0
    pcts = []
    entries = [(s, stage == s) for s in STAGE_LABELS] + [("all", np.ones(len(stage), dtype=bool))]
    for s, m in entries:
        n, k = int(m.sum()), int((m & icd).sum())
        if s != "all":
            total += n
        cells = (n, k, n - k)
        lo, hi = wilson(k, n)
        out.add("d1_agreement_by_stage", dict(
            ids, kdigo_stage=str(s), n_adm=sup.count(n), n_icd_pos=sup.count(k),
            pct_icd_pos=sup.pct(k, n, *cells),
            pct_ci_low=sup.value(100 * lo, 2, *cells), pct_ci_high=sup.value(100 * hi, 2, *cells)))
        if s != "all" and sup.ok(*cells):
            pcts.append((s, k / n))
    assert total == len(stage), f"stage rows sum to {total}, expected {len(stage)} ({label})"
    drops = [(a, b) for (a, pa), (b, pb) in zip(pcts, pcts[1:]) if pb < pa]
    if drops:
        out.anomalies.append(f"{label}: % ICD+ is not monotonic in KDIGO stage "
                             f"(decreases at stage transitions {', '.join(f'{a}->{b}' for a, b in drops)})")


def _add_2x2(out, table, ids, sub, stage_col, icd_col, sup, cfg, thresholds=THRESHOLDS):
    stage = sub[stage_col].to_numpy()
    icd = sub[icd_col].to_numpy(dtype=bool)
    for thr, fn in thresholds.items():
        cells = _cells(fn(stage), icd)
        key = "|".join(str(ids[c]) for c in _ID) + f"|{thr}"
        out.add(table, two_by_two_row(ids, thr, cells, sup, cfg, len(sub), key))


def run_cohort(adm: pd.DataFrame, cohort: str, cfg: Config, sup: Suppressor, out: Collector) -> None:
    icu = adm[adm["in_icu_cohort"]]
    subs = {"W_HOSP": adm, "W_ICU": icu}
    pb = cfg.primary_baseline
    ids_for = lambda w, icd_set, crit, baseline=pb, stratum="all": dict(  # noqa: E731
        cohort=cohort, window=w, icd_set=icd_set, baseline=baseline, criteria=crit, stratum=stratum)

    # -- D1: SCR_ONLY next to the full criteria, so the effect of UO / RRT is visible --
    for w in cfg.windows:
        sub = subs[w]
        for s in cfg.icd_sets:
            for crit, col in crit_pairs(w, cfg):
                ids = ids_for(w, s, crit)
                stage_rows(ids, sub[col], sub[s], sup, out, f"{cohort}/{w}/{s}/{crit}")
                _add_2x2(out, "d1_two_by_two", ids, sub, col, s, sup, cfg)

    # -- D2 (W_ICU): which criterion fired --
    if "W_ICU" in cfg.windows:
        _d2(icu, cohort, cfg, sup, out)
    else:
        out.notes.append("D2 skipped: requires W_ICU.")

    # -- D3 (W_HOSP, creatinine only so that only the baseline varies) --
    if "W_HOSP" in cfg.windows:
        n = len(adm)
        for b in BASELINES:
            ids = ids_for("W_HOSP", "AKI_CORE", "SCR_ONLY", baseline=b)
            st = adm[f"scr_HOSP_{b}"].to_numpy()
            rows = [(f"stage_{s}", int((st == s).sum())) for s in STAGE_LABELS]
            rows += [("any_stage_1plus", int((st >= 1).sum())),
                     ("no_scr_in_window", int((adm["n_scr_hosp"] == 0).sum())),
                     ("no_usable_baseline", int(adm[f"nobase_HOSP_{b}"].sum()))]
            assert sum(c for _, c in rows[:4]) == n, f"D3 stage rows do not sum to cohort ({b})"
            for cat, k in rows:
                lo, hi = wilson(k, n)
                out.add("d3_baseline_incidence", dict(
                    ids, category=cat, n_adm=sup.count(k), pct_cohort=sup.pct(k, n, k, n - k),
                    pct_ci_low=sup.value(100 * lo, 2, k, n - k), pct_ci_high=sup.value(100 * hi, 2, k, n - k)))
            _add_2x2(out, "d3_baseline_agreement", ids, adm, f"scr_HOSP_{b}", "AKI_CORE", sup, cfg)
        for i, a in enumerate(BASELINES):
            for b in BASELINES[i + 1:]:
                same = int(((adm[f"scr_HOSP_{a}"] >= 1) == (adm[f"scr_HOSP_{b}"] >= 1)).sum())
                out.add("d3_variant_pairwise", dict(
                    cohort=cohort, window="W_HOSP", icd_set="AKI_CORE", baseline=a, baseline_b=b,
                    criteria="SCR_ONLY", stratum="all", n=sup.count(n), n_same=sup.count(same),
                    n_diff=sup.count(n - same), pct_same=sup.pct(same, n, same, n - same)))
    else:
        out.notes.append("D3 skipped: requires W_HOSP.")

    # -- D4 (needs both windows) --
    if {"W_ICU", "W_HOSP"} <= set(cfg.windows):
        _d4_icd_pos(icu, cohort, cfg, sup, out)
        _d4_kdigo_pos(icu, cohort, cfg, sup, out)
    else:
        out.notes.append("D4 skipped: requires both windows.")

    # -- D5 (W_ICU, AKI_CORE, any stage, full criteria) --
    if "W_ICU" in cfg.windows:
        crit, col = full_criteria("W_ICU", cfg)
        any_only = {"any": THRESHOLDS["any"]}
        strata = [("all", icu), ("excl_esrd", icu[icu["ckd_status"] != "esrd"])]
        for var in ("ckd_status", "anchor_year_group", "icd_version_group"):
            for val in sorted(icu[var].astype(str).unique()):
                strata.append((f"{var}={val}", icu[icu[var].astype(str) == val]))
        for name, sub in strata:
            if len(sub):
                _add_2x2(out, "d5_stratified", ids_for("W_ICU", "AKI_CORE", crit, stratum=name), sub,
                         col, "AKI_CORE", sup, cfg, any_only)
    else:
        out.notes.append("D5 skipped: requires W_ICU.")

    # -- D5 by sex, both windows (the modeling window is W_HOSP) --
    for w in cfg.windows:
        sub = subs[w]
        crit, col = full_criteria(w, cfg)
        for name, s in (("sex=F", sub[sub["female"]]), ("sex=M", sub[~sub["female"]])):
            if len(s):
                _add_2x2(out, "d5_stratified", ids_for(w, "AKI_CORE", crit, stratum=name), s,
                         col, "AKI_CORE", sup, cfg, {"any": THRESHOLDS["any"]})


# ---------------------------------------------------------------------------
# D2
# ---------------------------------------------------------------------------

def _d2(icu, cohort, cfg, sup, out):
    crit, col = full_criteria("W_ICU", cfg)
    ids = dict(cohort=cohort, window="W_ICU", icd_set="AKI_CORE", baseline=cfg.primary_baseline,
               criteria=crit, stratum="all")
    if not cfg.use_uo:
        out.add("d2_criterion", dict(
            ids, group="SCR_ONLY|UO_ONLY|BOTH", n="NA", pct_of_kdigo_pos="NA", pct_stage1="NA", pct_stage2="NA",
            pct_stage3="NA", n_icd_pos="NA", pct_icd_pos="NA", status="skipped",
            note="urine-output criterion switched off (--no-uo)"))
        return
    pos = icu[icu[col] >= 1]
    scr, uo = pos["scr_primary_ICU"].to_numpy() >= 1, pos["uo_ICU"].to_numpy() >= 1
    group = np.select([scr & uo, scr, uo], ["BOTH", "SCR_ONLY", "UO_ONLY"], default="RRT_ONLY")
    stage = pos[col].to_numpy()
    icd = pos["AKI_CORE"].to_numpy(dtype=bool)
    total = len(pos)
    counts = {g: int((group == g).sum()) for g in D2_GROUPS}
    assert sum(counts.values()) == total, "D2 groups do not sum to the KDIGO+ total"
    fn_plus_tp = int(_cells(icu[col] >= 1, icu["AKI_CORE"])[0] + _cells(icu[col] >= 1, icu["AKI_CORE"])[2])
    assert total == fn_plus_tp, "D2 total differs from the D1 any-stage KDIGO+ count"
    notes = {"RRT_ONLY": "KDIGO+ only through dialysis start (stage 3)",
             "SCR_ONLY": "creatinine criterion met, urine-output criterion not met",
             "UO_ONLY": "urine-output criterion met, creatinine criterion not met"}
    for g in D2_GROUPS + ("ALL",):
        m = np.ones(total, dtype=bool) if g == "ALL" else group == g
        n, k = int(m.sum()), int((m & icd).sum())
        row = dict(ids, group=g, n=sup.count(n),
                   pct_of_kdigo_pos=sup.pct(n, total, *((n,) if g == "ALL" else (n, total - n))),
                   n_icd_pos=sup.count(k), pct_icd_pos=sup.pct(k, n, k, n - k), status="ok", note=notes.get(g, ""))
        for s in (1, 2, 3):
            ns = int((m & (stage == s)).sum())
            row[f"pct_stage{s}"] = sup.pct(ns, n, ns, n - ns)
        out.add("d2_criterion", row)


# ---------------------------------------------------------------------------
# D4
# ---------------------------------------------------------------------------

def _assign(cand_len):
    reason = np.full(cand_len, "UNEXPLAINED", dtype=object)
    free = np.ones(cand_len, dtype=bool)

    def take(mask, label):
        hit = np.asarray(mask, dtype=bool) & free
        reason[hit] = label
        free[hit] = False
        return hit
    return reason, take


def _emit_d4(table, ids, direction, reason, order, sub_rows, sup, out, notes):
    parent = len(reason)
    counts = {r: int((reason == r).sum()) for r in order}
    assert sum(counts.values()) == parent, f"D4 {direction} reasons do not sum to parent total"
    for r in order:
        out.add(table, dict(ids, direction=direction, level="reason", reason=r, parent="",
                            n=sup.count(counts[r]), pct_of_direction=sup.pct(counts[r], parent),
                            note=notes.get(r, "")))
    for (p, r, c) in sub_rows:
        assert sum(x[2] for x in sub_rows if x[0] == p) == counts[p], f"D4 sub-reasons of {p} do not sum"
        out.add(table, dict(ids, direction=direction, level="subreason", reason=r, parent=p,
                            n=sup.count(c), pct_of_direction=sup.pct(c, parent), note=notes.get(r, "")))
    out.add(table, dict(ids, direction=direction, level="total", reason="ALL", parent="",
                        n=sup.count(parent), pct_of_direction=sup.pct(parent, parent, parent), note=""))
    return parent


def _d4_ids(cohort, cfg):
    crit, col = full_criteria("W_ICU", cfg)
    return dict(cohort=cohort, window="W_ICU", icd_set="AKI_CORE", baseline=cfg.primary_baseline,
                criteria=crit, stratum="all"), col


def _d4_icd_pos(icu, cohort, cfg, sup, out):
    ids, col = _d4_ids(cohort, cfg)
    cand = icu[icu["AKI_CORE"] & (icu[col] == 0)]
    reason, take = _assign(len(cand))
    take(cand["ESRD"], "ESRD")
    take(cand["n_scr_hosp"] < 2, "INSUFFICIENT_SCR")
    outside = take(cand["stage_HOSP"] >= 1, "OUTSIDE_ICU_WINDOW")
    take(cand["elevated_on_arrival"], "ELEVATED_ON_ARRIVAL")
    t = cand["hosp_first_qual_t"].to_numpy()
    pre = outside & (t >= 0) & (t < cand["first_intime_s"].to_numpy())     # t < 0: no creatinine time (e.g. coded dialysis)
    post = outside & ~pre
    order = ["ESRD", "INSUFFICIENT_SCR", "OUTSIDE_ICU_WINDOW", "ELEVATED_ON_ARRIVAL", "UNEXPLAINED"]
    subs = [("OUTSIDE_ICU_WINDOW", "PRE_ICU", int(pre.sum())),
            ("OUTSIDE_ICU_WINDOW", "POST_ICU", int(post.sum()))]
    notes = {"ELEVATED_ON_ARRIVAL": "first SCr >= 1.5 x B4 baseline and a later SCr back below 1.5 x B4",
             "OUTSIDE_ICU_WINDOW": "KDIGO+ over the whole hospitalization (SCR+RRT) but not in the ICU window",
             "PRE_ICU": "first stage-qualifying SCr before the first ICU intime",
             "POST_ICU": "first stage-qualifying SCr at/after the first ICU intime or outside ICU stays, or no SCr time (coded dialysis)"}
    parent = _emit_d4("d4_discordance_icd_pos", ids, "ICD+/KDIGO-", reason, order, subs, sup, out, notes)
    fp_any = _cells(icu[col] >= 1, icu["AKI_CORE"])[1]
    assert parent == fp_any, "D4 ICD+/KDIGO- total differs from the D1 any-stage 2x2 cell"


def _d4_kdigo_pos(icu, cohort, cfg, sup, out):
    ids, col = _d4_ids(cohort, cfg)
    cand = icu[(icu[col] >= 1) & ~icu["AKI_CORE"]]
    reason, take = _assign(len(cand))
    take(cand["ESRD"], "ESRD")
    take((cand["scr_primary_ICU"] == 0) & (cand["uo_ICU"] >= 1) & ~cand["rrt_ICU"], "UO_ONLY")
    take(cand["CKD"], "CKD")
    stage1_scr_only = (cand[col] == 1) & (cand["uo_ICU"] == 0)
    take(stage1_scr_only & cand["icu_transient"], "TRANSIENT_STAGE1")
    take(cand[col] == 1, "STAGE1_OTHER")
    reason[reason == "UNEXPLAINED"] = "STAGE2PLUS_UNCODED"
    order = ["ESRD", "UO_ONLY", "CKD", "TRANSIENT_STAGE1", "STAGE1_OTHER", "STAGE2PLUS_UNCODED"]
    notes = {"UO_ONLY": "only the urine-output criterion was met (creatinine criterion and dialysis not met)"
                        if cfg.use_uo else "not evaluated: urine-output criterion switched off (0 by construction)",
             "TRANSIENT_STAGE1": "max stage 1 by creatinine; a later SCr within 48h of the last qualifying SCr is below the stage-1 threshold",
             "STAGE2PLUS_UNCODED": "max stage 2 or 3, including dialysis-only"}
    parent = _emit_d4("d4_discordance_kdigo_pos", ids, "KDIGO+/ICD-", reason, order, [], sup, out, notes)
    fn_any = _cells(icu[col] >= 1, icu["AKI_CORE"])[2]
    assert parent == fn_any, "D4 KDIGO+/ICD- total differs from the D1 any-stage 2x2 cell"
