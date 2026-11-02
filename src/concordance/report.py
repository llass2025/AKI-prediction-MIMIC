"""summary.md writer. Reads only the already-suppressed output tables, so it cannot leak a small cell."""

import pandas as pd

from . import confirmed, icd
from .config import BASELINES, Config
from .diagnostics import Collector, full_criteria
from .stats import Suppressor


def _md(rows, headers):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _one(df, **flt):
    m = df
    for k, v in flt.items():
        m = m[m[k] == v]
    return m.iloc[0] if len(m) else None


def _two_by_two_line(r):
    if r is None:
        return "not computed"
    return (f"n={r['n']}, TP/FP/FN/TN={r['tp']}/{r['fp']}/{r['fn']}/{r['tn']}; "
            f"sensitivity {r['sensitivity']} "
            f"({r['sens_ci_low']}-{r['sens_ci_high']}), specificity {r['specificity']} "
            f"({r['spec_ci_low']}-{r['spec_ci_high']}), PPV {r['ppv']}, NPV {r['npv']}, "
            f"kappa {r['kappa']} ({r['kappa_ci_low']}-{r['kappa_ci_high']})")


def write_summary(path: str, cfg: Config, meta, out: Collector, sup: Suppressor, cohorts: dict) -> None:
    f = {t: out.frame(t) for t in ("d1_agreement_by_stage", "d1_two_by_two", "d2_criterion",
                                   "d3_baseline_incidence", "d3_baseline_agreement",
                                   "d3_variant_pairwise", "d4_discordance_icd_pos",
                                   "d4_discordance_kdigo_pos", "d5_stratified")}
    C = next(c for c in ("modeling", "pipeline", "all_adults") if c in cohorts)   # primary cohort
    L = []
    add = L.append

    add("# AKI label concordance: KDIGO vs ICD (MIMIC-IV)\n")
    add("Aggregate output only. Counts below MIN_CELL print as `<MIN_CELL`; rates and intervals "
        "built from such a cell print `suppressed`. Suppression is per cell as specified, so a "
        "suppressed cell can still be inferred from displayed margins; no secondary suppression is applied.\n")

    add("## Configuration\n")
    add(_md([(k, v) for k, v in cfg.as_rows()], ["setting", "value"]) + "\n")

    add("## Cohort\n")
    rows = [("adult MIMIC-IV admissions (W_HOSP cohort)", sup.count(meta.n_adults)),
            ("admissions with >= 1 ICU stay (W_ICU cohort)", sup.count(meta.n_icu_cohort)),
            ("  of which with > 1 ICU stay (aggregated to admission)", sup.count(meta.n_multi_stay)),
            ("admissions with no creatinine in W_HOSP", sup.count(meta.n_no_scr_hosp)),
            ("ICU-cohort admissions with no creatinine in W_ICU", sup.count(meta.n_no_scr_icu))]
    if cfg.use_uo:
        rows += [("ICU stays", sup.count(meta.n_stays)),
                 ("  with a plausible patient weight (20-400 kg)", sup.count(meta.n_stays_weight)),
                 ("  with any urine output charted", sup.count(meta.n_stays_urine)),
                 ("admissions assessable for the urine-output criterion", sup.count(meta.n_uo_assessable))]
    if cfg.use_rrt:
        rows += [("admissions with dialysis started in the ICU (non-ESRD)", sup.count(meta.n_rrt_icu)),
                 ("admissions with dialysis started anywhere in the hospitalization (non-ESRD)", sup.count(meta.n_rrt_hosp)),
                 ("dialysis admissions left out because they carry an ESRD code", sup.count(meta.n_rrt_esrd_excluded))]
    if "pipeline" in cohorts:
        rows.append(("pipeline cohort (target_admissions.parquet: adults, no primary AKI code)", sup.count(meta.n_pipeline)))
    if "modeling" in cohorts:
        rows.append(("modeling cohort (target.parquet, MIMIC-IV: the admissions the model is trained on)", sup.count(meta.n_modeling)))
        rows.append(("  of which incident_aki_label = 1", sup.count(meta.n_modeling_pos)))
    add(_md(rows, ["quantity", "n"]))
    add("\nAdmissions with no usable creatinine in a window count as KDIGO stage 0 so that every "
        "2x2 sums to the cohort size.\n")
    add(f"**Primary cohort for the results below: `{C}`.** Three cohorts are reported (`cohort` column in every CSV):\n")
    add(_md([("modeling", "the admissions in target.parquet: first-ever in-hospital AKI (label 1) vs. no AKI / no CKD history (label 0). "
                          "The ICD side of every table is the model's own `incident_aki_label`."),
             ("pipeline", "target_admissions.parquet: adults with no AKI code in the primary diagnosis position; ICD side = AKI_CORE."),
             ("all_adults", "every adult MIMIC-IV admission, including primary-AKI admissions; ICD side = AKI_CORE.")],
            ["cohort", "definition"]) + "\n")
    add("MIMIC-IV has no present-on-admission flag, so an AKI code outside the primary position can still be AKI that was "
        "present on arrival; the KDIGO comparison (D4 'elevated on arrival' and 'outside ICU window') is the only handle on that.\n")

    add("## Definitions used\n")
    add("**Existing pipeline label.** The primary outcome (`is_aki`) is ICD-coded; there is no stored "
        "admission-level KDIGO stage. The creatinine logic here extends `aki_timing_sensitivity/kdigo_onset_timing.py` "
        "(criteria: rise >= 0.3 mg/dL within 48h, or value >= 1.5 x baseline). Staging is added on top: "
        "stage 1 = those criteria, stage 2 = >= 2.0 x baseline, stage 3 = >= 3.0 x baseline or SCr >= 4.0 mg/dL "
        "with an acute rise.\n")
    add(f"**Criteria.** `criteria = SCR_ONLY` is creatinine alone. The full label adds, where the data allow: "
        f"{'urine output in the ICU window (stage 1: < 0.5 mL/kg/h over 6 h, stage 2: over 12 h, stage 3: < 0.3 mL/kg/h over 24 h or no urine for 12 h; a window counts only if >= 5/6 of its hours have urine charted; weight = stay median inputevents patientweight, 20-400 kg)' if cfg.use_uo else 'urine output is switched off'}; "
        f"{'dialysis start as stage 3 (procedureevents in an ICU stay; procedures_icd for the whole hospitalization only), excluding ESRD-coded admissions' if cfg.use_rrt else 'dialysis start is switched off'}. "
        f"The combined stage is the highest of the criteria. Urine output is only charted in the ICU, so W_HOSP uses creatinine + dialysis only.\n")
    add(f"**Primary baseline: `{cfg.primary_baseline}`** for D1, D2, D4 and D5 (D3 shows all five, creatinine-only). "
        f"B0 is the pipeline's own method and is what the equivalence check above uses.\n")
    add("**Windows.** W_HOSP = creatinine with the admission's `hadm_id` between admittime and dischtime. "
        "W_ICU = the same creatinine restricted to the union of the admission's ICU stays (intime-outtime).\n")
    add(_md([("B0", "pipeline method: median SCr in [admit-7d, admit) by subject; in-admission minimum if none. "
                    "Computed by the existing `compute_baselines()`"),
             ("B1", "first SCr of the admission"),
             ("B2", "rolling minimum of the subject's SCr over the 7 days strictly before each value"),
             ("B3", "back-calculated SCr for eGFR 75 mL/min/1.73m2 (CKD-EPI 2021, race-free), from age at admission and sex, for every admission"),
             ("B4", "most recent SCr within 365 days before admittime (any hadm_id, incl. outpatient), else B3")],
            ["baseline", "definition"]) + "\n")
    add("B3 as written in the spec ('use only when no measured pre-admission value exists; otherwise fall back to B4') "
        "is identical to B4 as written, so B3 is implemented as the pure back-calculation.\n")
    add("**Creatinine itemids.**\n")
    add(_md([(i, meta.item_labels[i], sup.count(meta.item_rows[i])) for i in cfg.creatinine_itemids],
            ["itemid", "d_labitems label (fluid, category)", "labevents rows"]))
    add(f"\nThe pipeline's own label regex (`\\bcreatinine\\b`, as in `pre_aki_lab_flags.py`) would select "
        f"{meta.existing_regex_itemids} itemids, including urine and body-fluid creatinine; only blood "
        f"creatinine items are used here.\n")
    add("**ICD code lists** (MIMIC stores codes without dots; match on icd_version + prefix; all seq_num).\n")
    add(_md(icd.describe_sets(), ["set", "ICD-9 prefixes", "ICD-10 prefixes"]) + "\n")

    # -- D1 --
    add("## D1. Agreement by KDIGO stage\n")
    for w in cfg.windows:
        st = f["d1_agreement_by_stage"]
        fc = full_criteria(w, cfg)[0]
        def stage_txt(s):
            r = _one(st, cohort=C, window=w, icd_set="AKI_CORE", criteria=fc, kdigo_stage=str(s))
            pct = r["pct_icd_pos"]
            return f"stage {s}: {pct if pct == 'suppressed' else pct + '%'} of {r['n_adm']}"
        line = ", ".join(stage_txt(s) for s in (0, 1, 2, 3))
        add(f"**{w}, AKI_CORE, criteria {fc}, baseline {cfg.primary_baseline}.** % ICD+ by max KDIGO stage: {line}.")
        add(f"Any-stage 2x2: {_two_by_two_line(_one(f['d1_two_by_two'], cohort=C, window=w, icd_set='AKI_CORE', criteria=fc, kdigo_threshold='any'))}.")
        for thr, lab in (("ge2", "stage >= 2"), ("eq3", "stage 3")):
            r = _one(f["d1_two_by_two"], cohort=C, window=w, icd_set="AKI_CORE", criteria=fc, kdigo_threshold=thr)
            add(f"{lab}: {_two_by_two_line(r)}.")
        if fc != "SCR_ONLY":
            r = _one(f["d1_two_by_two"], cohort=C, window=w, icd_set="AKI_CORE", criteria="SCR_ONLY", kdigo_threshold="any")
            add(f"Same cohort, creatinine only (SCR_ONLY), any stage: {_two_by_two_line(r)}.")
        if "AKI_BROAD" in cfg.icd_sets:
            r = _one(f["d1_two_by_two"], cohort=C, window=w, icd_set="AKI_BROAD", criteria=fc, kdigo_threshold="any")
            add(f"AKI_BROAD (secondary), any stage: {_two_by_two_line(r)}.")
        add("")
    for other in cohorts:
        if other == C:
            continue
        for w in cfg.windows:
            r = _one(f["d1_two_by_two"], cohort=other, window=w, icd_set="AKI_CORE", criteria=full_criteria(w, cfg)[0],
                     kdigo_threshold="any")
            add(f"For comparison, `{other}` cohort, {w}, AKI_CORE, any stage: {_two_by_two_line(r)}.")
        add("")

    # -- D2 --
    add("## D2. Which KDIGO criterion fired\n")
    if "W_ICU" in cfg.windows and cfg.use_uo:
        d2 = f["d2_criterion"]
        d2 = d2[d2["cohort"] == C]
        add("KDIGO+ admissions in the ICU window, by which criterion was met at any time (RRT_ONLY = dialysis start "
            "with neither creatinine nor urine output criterion):\n")
        add(_md([(r["group"], r["n"], r["pct_of_kdigo_pos"], r["pct_stage1"], r["pct_stage2"], r["pct_stage3"],
                  r["pct_icd_pos"]) for _, r in d2.iterrows()],
                ["group", "n", "% of KDIGO+", "% stage 1", "% stage 2", "% stage 3", "% ICD+"]) + "\n")
    else:
        add("Skipped: requires W_ICU and the urine-output criterion (see `d2_criterion.csv`).\n")

    # -- D3 --
    add("## D3. Baseline creatinine sensitivity (W_HOSP, creatinine only)\n")
    if "W_HOSP" in cfg.windows:
        inc, agr = f["d3_baseline_incidence"], f["d3_baseline_agreement"]
        rows = []
        for b in BASELINES:
            g = lambda cat: _one(inc, cohort=C, baseline=b, category=cat)  # noqa: E731
            a = _one(agr, cohort=C, baseline=b, kdigo_threshold="any")
            rows.append((b, g("any_stage_1plus")["pct_cohort"], g("stage_1")["pct_cohort"], g("stage_2")["pct_cohort"],
                         g("stage_3")["pct_cohort"], g("no_usable_baseline")["pct_cohort"],
                         a["sensitivity"], a["specificity"], a["kappa"]))
        add(_md(rows, ["baseline", "% any AKI", "% stage 1", "% stage 2", "% stage 3",
                       "% no usable baseline", "sens vs AKI_CORE", "spec", "kappa"]))
        add("\nPairwise agreement on the any-AKI label (% of admissions with the same label):\n")
        pw = f["d3_variant_pairwise"]
        pw = pw[pw["cohort"] == C]
        add(_md([(r["baseline"], r["baseline_b"], r["pct_same"]) for _, r in pw.iterrows()],
                ["variant", "vs variant", "% same"]) + "\n")
    else:
        add("Skipped: requires W_HOSP.\n")

    # -- D4 --
    add(f"## D4. Discordance decomposition (W_ICU, {full_criteria('W_ICU', cfg)[0]}, baseline {cfg.primary_baseline}, AKI_CORE)\n")
    if {"W_ICU", "W_HOSP"} <= set(cfg.windows):
        for tab, title in (("d4_discordance_icd_pos", "ICD+ / KDIGO-"), ("d4_discordance_kdigo_pos", "KDIGO+ / ICD-")):
            d = f[tab]
            d = d[d["cohort"] == C]
            add(f"**{title}** (first matching reason wins):\n")
            add(_md([(r["parent"] + " > " + r["reason"] if r["parent"] else r["reason"], r["n"], r["pct_of_direction"])
                     for _, r in d.iterrows()], ["reason", "n", "% of direction"]) + "\n")
    else:
        add("Skipped: requires both windows.\n")

    # -- D5 --
    add(f"## D5. Stratification (AKI_CORE, any stage, baseline {cfg.primary_baseline}; W_ICU: "
        f"{full_criteria('W_ICU', cfg)[0]}, W_HOSP: {full_criteria('W_HOSP', cfg)[0]}; sex strata in both windows)\n")
    if cfg.windows:
        d5 = f["d5_stratified"]
        d5 = d5[d5["cohort"] == C]
        add(_md([(r["window"], r["stratum"], r["n"], r["sensitivity"], r["specificity"], r["ppv"], r["kappa"])
                 for _, r in d5.iterrows()], ["window", "stratum", "n", "sensitivity", "specificity", "PPV", "kappa"]) + "\n")
    else:
        add("Skipped: no window selected.\n")

    L.extend(confirmed.summary_lines(out, cfg))

    # -- flags --
    add("## Flags and checks\n")
    b = meta.b0_check
    add(f"- B0 vs the existing code: {sup.count(b['n_checked'] - b['n_mismatch'])} of {sup.count(b['n_checked'])} "
        f"admissions with creatinine have an identical any-AKI label"
        f"{' (sampled)' if b.get('sampled') else ''}; {sup.count(b['n_mismatch'])} differ. Any difference would have to "
        f"be an admission with tied creatinine timestamps (the existing 48h rolling minimum depends on row order for "
        f"ties; ours does not), otherwise the run stops.")
    add("- Enforced at run time (the run aborts if any fails, so reaching this file means they held): 2x2 cells sum to the "
        "cohort size for every window / variant / stratum; stage-table rows sum to the cohort size; D3 stage rows sum "
        "to the cohort size; D4 reasons are mutually exclusive, sum to their direction totals and match the D1 any-stage "
        "FP / FN cells; the output scan finds no ID columns, timestamps or 8+ digit numbers in any CSV or in this file.")
    for a in out.anomalies:
        add(f"- ANOMALY: {a}")
    if not out.anomalies:
        add("- % ICD+ increases monotonically with KDIGO stage in every unsuppressed table.")
    for n in out.notes:
        add(f"- {n}")
    add("\n## Method notes and limitations\n")
    add("- Urine output is only available in the ICU, needs a patient weight, and a window counts only when most of its hours "
        "have urine charted (gaps are never read as oliguria); ICU stays without a plausible weight are not assessable for it. "
        "Dialysis is taken as a start in an ICU stay or a procedure code, never as maintenance dialysis (ESRD-coded admissions excluded).")
    add("- Admissions with no creatinine are KDIGO stage 0 unless urine output or dialysis says otherwise.")
    add("- D4 definitions I had to choose: 'SCr falls afterward' (ELEVATED_ON_ARRIVAL) = a later SCr below 1.5 x the B4 baseline; "
        "TRANSIENT_STAGE1 = a later SCr within 48h of the last qualifying value is below the stage-1 threshold "
        "(1.5 x baseline, or window minimum + 0.3 for rise-only cases); 'pre-ICU' = first qualifying SCr before the first ICU intime.")
    add(f"- Kappa CIs: {cfg.n_boot} bootstrap resamples of admissions (seed {cfg.seed}), implemented as a multinomial draw over "
        "the four 2x2 cell counts.")
    add("- Age is anchor_age + (admit year - anchor_year). The pipeline itself uses anchor_age; MIMIC-IV adults are >= 18 either way.")
    add("- MIMIC-III is not covered (no anchor_year_group; different ICU tables).")
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")
