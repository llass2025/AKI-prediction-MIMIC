"""
Final KDIGO label (opt-in): B3 baseline with a 24 h pre-admission window, creatinine plus dialysis, per patient.

One row per patient of the flowchart cohort (paper-run features_all with the cmb_ckd / renal-impairment exclusions): the same
admission the ICD incident label is attached to. It sits beside, and never changes, target.parquet.

  kdigo_final            stage >= 1 over [admittime - 24 h, dischtime]: creatinine >= 1.5 x the eGFR-75 baseline (age, sex), a rise of
                         0.3 mg/dL within 48 h, higher stages by the usual ratios, or coded/ICU dialysis (never ESRD-coded)
  kdigo_stage            0-3 (dialysis counts as 3)
  onset_hours            first qualifying creatinine, hours after admittime (negative = the 24 h before admission, i.e. ED)
  onset_group            before admission / 0-24 h / 24-48 h / 48 h or later / dialysis only / none
  kdigo_incident_24h     kdigo_final with onset_hours >= 24 (the in-hospital-onset subset)
  concordance            both / ICD only / KDIGO only / neither, against incident_aki_label

Only the row-level parquet carries IDs (it is gitignored like every other parquet); the summary CSV is suppressed aggregates.
"""

import argparse
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from . import nadir
from .config import Config
from .stats import Suppressor

FEATURES_DIR = os.environ.get("AKI_FEATURES_DIR", "features")     # folder holding features_all.parquet (set AKI_FEATURES_DIR for another run)

GROUPS = ("before admission (ED value within 24 h)", "0 to <24 h", "24 to <48 h", "48 h or later",
          "dialysis only (no creatinine time)", "no KDIGO criterion met")
CONC = ("both", "ICD only", "KDIGO only", "neither")


def label_frame(version: str, frames: dict, cohort: pd.DataFrame) -> pd.DataFrame:
    """cohort: index = subject_id; columns hadm_id, icd (bool), female (bool). frames: {pre_hours: hospital_flags frame}."""
    f = frames[nadir.PRE_HOURS].reindex(cohort["hadm_id"].to_numpy())
    stage = np.maximum(f["scr_B3"].fillna(0).to_numpy().astype(int), 3 * f["rrt"].fillna(False).to_numpy().astype(int))
    pos = stage >= 1
    onset = f["onset_h_B3"].to_numpy(dtype=float)
    group = np.select(
        [~pos, pos & np.isnan(onset), onset < 0, onset < 24, onset < 48],
        [GROUPS[5], GROUPS[4], GROUPS[0], GROUPS[1], GROUPS[2]], default=GROUPS[3])
    icd = cohort["icd"].to_numpy()
    conc = np.select([icd & pos, icd & ~pos, ~icd & pos], [CONC[0], CONC[1], CONC[2]], default=CONC[3])
    return pd.DataFrame({
        "version": version, "subject_id": cohort.index.to_numpy(), "hadm_id": cohort["hadm_id"].to_numpy(),
        "female": cohort["female"].to_numpy(), "icd_incident_label": icd.astype(int),
        "kdigo_final": pos, "kdigo_stage": stage, "onset_hours": onset, "onset_group": group,
        "kdigo_incident_24h": pos & (onset >= 24), "n_creatinine_window": f["n_scr"].fillna(0).to_numpy().astype(int),
        "concordance": conc})


def summary(labels: pd.DataFrame, sup: Suppressor) -> pd.DataFrame:
    rows = []
    for ver, g in labels.groupby("version", sort=True):
        n = len(g)
        add = lambda sec, item, k, den=None: rows.append(dict(  # noqa: E731
            version=ver, section=sec, item=item, n=sup.count(k), pct=sup.pct(k, den, k, den - k) if den else ""))
        add("cohort", "patients", n)
        add("cohort", "ICD incident AKI", int(g["icd_incident_label"].sum()), n)
        add("cohort", "KDIGO final positive", int(g["kdigo_final"].sum()), n)
        add("cohort", "KDIGO final positive, onset >= 24 h after admission", int(g["kdigo_incident_24h"].sum()), n)
        for c in CONC:
            add("concordance", c, int((g["concordance"] == c).sum()), n)
        for s in range(4):
            add("stage", f"stage {s}", int((g["kdigo_stage"] == s).sum()), n)
        npos = int(g["kdigo_final"].sum())
        for gr in GROUPS[:5]:
            add("onset (KDIGO positive)", gr, int((g["onset_group"] == gr).sum()), npos)
    return pd.DataFrame(rows, columns=["version", "section", "item", "n", "pct"])


def run(a) -> None:
    cfg = Config(mimic_dir=a.mimic4_dir, cache_dir=a.cache_dir, min_cell=a.min_cell, use_uo=False, use_rrt=True, quiet=a.quiet)
    log = (lambda m: None) if a.quiet else (lambda m: print(m, flush=True))
    sup = Suppressor(a.min_cell)
    per_version = nadir.load(cfg, a, log)
    labels = pd.concat([label_frame(v, fr, co) for v, (fr, co) in per_version.items()], ignore_index=True)
    assert not labels.duplicated(["version", "subject_id"]).any(), "a patient appears twice in the label file"
    labels.to_parquet(a.label_out, index=False)
    os.makedirs(a.outdir, exist_ok=True)
    summ = summary(labels, sup)
    summ.to_csv(Path(a.outdir) / "final_label_summary.csv", index=False, lineterminator="\n")
    log(f"Wrote {a.label_out} ({len(labels)} patients)\n" + nadir.md(summ))


def parse_args(argv=None):
    d = Config()
    ap = argparse.ArgumentParser(prog="python -m src.concordance.final_label",
                                 description="Final KDIGO label: B3 baseline, 24 h pre-admission window, creatinine and dialysis.")
    ap.add_argument("--mimic4-dir", default=d.mimic_dir)
    ap.add_argument("--mimic3-dir", default="../data/mimic-iii-clinical-database-1.4")
    ap.add_argument("--features-all", default=os.path.join(FEATURES_DIR, "features_all.parquet"))
    ap.add_argument("--label-out", default="kdigo_final_label.parquet", help="Row-level output (gitignored); target.parquet is untouched.")
    ap.add_argument("--outdir", default="reports/final_kdigo")
    ap.add_argument("--cache-dir", default=d.cache_dir)
    ap.add_argument("--min-cell", type=int, default=d.min_cell)
    ap.add_argument("--quiet", action="store_true")
    return ap.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
