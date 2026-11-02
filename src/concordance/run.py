"""Orchestration: load -> build admission table -> checks -> D1-D5 -> CSVs + summary.md."""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import checks, confirmed, data, diagnostics, report
from .config import BASELINES, ICD_SET_CHOICES, WINDOWS, Config
from .stats import Suppressor

OUTPUT_TABLES = ["d1_agreement_by_stage", "d1_two_by_two", "d2_criterion", "d3_baseline_incidence",
                 "d3_baseline_agreement", "d3_variant_pairwise", "d4_discordance_icd_pos",
                 "d4_discordance_kdigo_pos", "d5_stratified"]


def run(cfg: Config) -> None:
    cfg.validate()
    log = (lambda m: None) if cfg.quiet else (lambda m: print(m, flush=True))
    sup = Suppressor(cfg.min_cell)

    tab = data.load_tables(cfg)
    adm, meta, win = data.build_admission_table(tab, cfg)
    log(f"Adult admissions: {sup.count(meta.n_adults)}; with ICU stay: {sup.count(meta.n_icu_cohort)}")

    log("Checking that B0 reproduces the existing label ...")
    hosp = win["hosp"]
    stage_hosp_b0 = pd.DataFrame({"max_stage": adm["scr_HOSP_B0"].reindex(hosp.hadm).to_numpy()})
    meta.b0_check = checks.check_b0_against_existing(
        hosp, win["b0_median"].reindex(hosp.hadm).to_numpy(), stage_hosp_b0,
        cfg.b0_check_sample, sup, log)

    out = diagnostics.Collector()
    cohorts = {"all_adults": adm}
    modeling_ref = None
    if cfg.pipeline_target and Path(cfg.pipeline_target).exists():
        try:
            t = pd.read_parquet(cfg.pipeline_target, columns=["hadm_id", "version"])
        except Exception as e:      # e.g. parquet written by a newer pyarrow than this env has
            msg = f"pipeline cohort not reported: could not read {cfg.pipeline_target} ({type(e).__name__}: {e})"
            log(msg)
            out.notes.append(msg)
        else:
            ids = set(t.loc[t["version"] == "mimic4", "hadm_id"].astype("int64"))
            cohorts["pipeline"] = adm[adm["hadm_id"].isin(ids)]
            meta.n_pipeline = len(cohorts["pipeline"])
            log(f"Pipeline cohort (cached {cfg.pipeline_target}): {sup.count(meta.n_pipeline)} MIMIC-IV admissions")
    else:
        log("No cached pipeline target found")

    if cfg.modeling_target and Path(cfg.modeling_target).exists():
        try:
            m = pd.read_parquet(cfg.modeling_target, columns=["subject_id", "hadm_id", "version", "incident_aki_label"])
        except Exception as e:
            msg = f"modeling cohort not reported: could not read {cfg.modeling_target} ({type(e).__name__}: {e})"
            log(msg)
            out.notes.append(msg)
        else:
            m = m[(m["version"] == "mimic4") & m["incident_aki_label"].notna()]
            modeling_ref = m
            lab =pd.Series(m["incident_aki_label"].astype(int).to_numpy(), index=m["hadm_id"].astype("int64").to_numpy())
            sub = adm[adm["hadm_id"].isin(lab.index)].copy()
            # ICD side of every table for this cohort is the model's own target
            sub["AKI_CORE"] = lab.reindex(sub["hadm_id"]).to_numpy() == 1
            cohorts["modeling"] = sub
            meta.n_modeling = len(sub)
            meta.n_modeling_pos = int(sub["AKI_CORE"].sum())
            log(f"Modeling cohort ({cfg.modeling_target}): {sup.count(meta.n_modeling)} MIMIC-IV admissions, "
                f"{sup.count(meta.n_modeling_pos)} with incident_aki_label = 1")
    else:
        log("No modeling target found")

    for name, sub in cohorts.items():
        log(f"Diagnostics for cohort '{name}' ...")
        diagnostics.run_cohort(sub, name, cfg, sup, out)

    tables = list(OUTPUT_TABLES)
    if cfg.write_confirmed:
        if "modeling" in cohorts and Path(cfg.incident_target).exists():
            confirmed.run(adm, win["hosp"], cohorts["modeling"], modeling_ref, cfg, sup, out, log)
            tables += list(confirmed.D6_COLS)
        else:
            msg = (f"confirmed target not built: needs the modeling target ({cfg.modeling_target}) and the admission-level "
                   f"{cfg.incident_target}")
            log(msg)
            out.notes.append(msg)

    os.makedirs(cfg.outdir, exist_ok=True)
    for tbl in tables:
        out.frame(tbl).to_csv(Path(cfg.outdir) / f"{tbl}.csv", index=False, lineterminator="\n")
    report.write_summary(str(Path(cfg.outdir) / "summary.md"), cfg, meta, out, sup, cohorts)
    checks.verify_outputs(cfg.outdir)
    log(f"Wrote {len(tables)} tables + summary.md to {cfg.outdir}")


def parse_args(argv=None) -> Config:
    d = Config()
    ap = argparse.ArgumentParser(
        prog="python -m src.concordance",
        description="KDIGO-vs-ICD AKI label concordance diagnostics (aggregate output only).")
    ap.add_argument("--mimic-dir", default=d.mimic_dir, help="MIMIC-IV root containing hosp/ and icu/ (env MIMIC_IV_DIR).")
    ap.add_argument("--outdir", default=d.outdir)
    ap.add_argument("--cache-dir", default=d.cache_dir, help="Where the creatinine extract is cached (gitignored pickle).")
    ap.add_argument("--refresh-cache", action="store_true")
    ap.add_argument("--min-cell", type=int, default=d.min_cell)
    ap.add_argument("--n-boot", type=int, default=d.n_boot)
    ap.add_argument("--seed", type=int, default=d.seed)
    ap.add_argument("--windows", nargs="+", default=list(d.windows), choices=WINDOWS)
    ap.add_argument("--icd-sets", nargs="+", default=list(d.icd_sets), choices=ICD_SET_CHOICES,
                    help="ICD sets shown in D1 (AKI_CORE is always primary and always included).")
    ap.add_argument("--creatinine-itemids", nargs="+", type=int, default=list(d.creatinine_itemids))
    ap.add_argument("--pipeline-target", default=d.pipeline_target,
                    help="Cached target_admissions.parquet; if present, the pipeline cohort is reported too.")
    ap.add_argument("--modeling-target", default=d.modeling_target,
                    help="Final patient-level target.parquet; if present, the modeling cohort is reported "
                         "with incident_aki_label as the ICD side.")
    ap.add_argument("--b0-check-sample", type=int, default=d.b0_check_sample,
                    help="Compare B0 with the existing code on N evenly spaced admissions (0 = all).")
    ap.add_argument("--primary-baseline", default=d.primary_baseline, choices=BASELINES,
                    help="Baseline behind D1/D2/D4/D5 (default B3, SCr back-calculated for eGFR 75 from age and sex; "
                         "B2 is the patient's own rolling 7-day minimum). The confirmed target always uses B2.")
    ap.add_argument("--no-uo", action="store_true", help="Leave out the urine-output criterion (creatinine-only ICU label).")
    ap.add_argument("--no-rrt", action="store_true", help="Leave out the dialysis-start criterion.")
    ap.add_argument("--confirmed-target", action="store_true",
                    help="Also build the KDIGO-confirmed incident target (strict and light) next to target.parquet and "
                         "report the overlap with the ICD target (D6). Needs --modeling-target and --incident-target.")
    ap.add_argument("--incident-target", default=d.incident_target, help="Admission-level incident_aki_target.parquet.")
    ap.add_argument("--confirmed-out", default=d.confirmed_out,
                    help="Strict confirmed target path; the light variant is written next to it as <name>_light.parquet.")
    ap.add_argument("--min-onset-hours", type=float, default=d.min_onset_hours,
                    help="First qualifying creatinine must be at least this many hours after admission to count as in-hospital onset.")
    ap.add_argument("--drop-kdigo-negatives", action="store_true",
                    help="Also drop ICD-negative admissions that are KDIGO-confirmed from the confirmed targets.")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    return Config(mimic_dir=a.mimic_dir, outdir=a.outdir, cache_dir=a.cache_dir, refresh_cache=a.refresh_cache,
                  min_cell=a.min_cell, n_boot=a.n_boot, seed=a.seed, windows=tuple(a.windows),
                  icd_sets=tuple(a.icd_sets), creatinine_itemids=tuple(a.creatinine_itemids),
                  pipeline_target=a.pipeline_target, modeling_target=a.modeling_target,
                  b0_check_sample=a.b0_check_sample, primary_baseline=a.primary_baseline,
                  use_uo=not a.no_uo, use_rrt=not a.no_rrt, write_confirmed=a.confirmed_target,
                  incident_target=a.incident_target, confirmed_out=a.confirmed_out,
                  min_onset_hours=a.min_onset_hours, drop_kdigo_neg=a.drop_kdigo_negatives, quiet=a.quiet)


def main(argv=None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
