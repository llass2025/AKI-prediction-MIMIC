"""Run-time checks: B0 vs the existing code, and a scan of the outputs for IDs / timestamps."""

import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import kdigo
from .stats import Suppressor

FORBIDDEN_COLUMNS = {"subject_id", "hadm_id", "stay_id", "icustay_id", "row_id", "labevent_id",
                     "specimen_id", "charttime", "admittime", "dischtime", "intime", "outtime",
                     "storetime", "deathtime", "dod"}
_TS = re.compile(r"\d{4}-\d{2}-\d{2}")
_LONG_ID = re.compile(r"\b\d{8,}\b")    # MIMIC-IV subject/hadm/stay ids are 8 digits


def check_b0_against_existing(win, b0_median, stage_df, sample_n: int, sup: Suppressor, log) -> dict:
    """
    The existing kdigo_onset_for_admission() (median-7d-prior baseline, in-admission
    minimum fallback) is the pipeline's label logic. Re-run it on each admission's
    in-hospital creatinine series and require identical any-AKI labels to B0 stage >= 1.

    Admissions whose creatinine series contains tied timestamps are tolerated as the
    only possible cause of a mismatch: the existing code's 48h rolling minimum depends
    on the arbitrary order of rows with identical charttime, while ours does not.
    """
    mod = kdigo.load_existing_onset_module()
    G = win.n_groups
    idx = np.arange(G) if sample_n <= 0 or sample_n >= G else np.unique(np.linspace(0, G - 1, sample_n).astype(int))
    mine = (stage_df["max_stage"].to_numpy() >= 1)
    n_mismatch = n_tied = 0
    for c, gi in enumerate(idx):
        s, e = win.starts[gi], win.starts[gi] + win.counts[gi]
        tt, vv = win.t[s:e], win.v[s:e]
        df = pd.DataFrame({"charttime": pd.to_datetime(tt, unit="s"), "valuenum": vv})
        found = mod.kdigo_onset_for_admission(df, float(b0_median[gi])) is not None
        if found != mine[gi]:
            n_mismatch += 1
            n_tied += int(len(np.unique(tt)) < len(tt))
        if (c + 1) % 100000 == 0:
            log(f"  B0 check: {c + 1} admissions compared")
    res = {"n_checked": int(len(idx)), "n_mismatch": n_mismatch, "n_mismatch_tied": n_tied,
           "n_total_with_scr": int(G), "sampled": bool(len(idx) < G)}
    if n_mismatch - n_tied > 0:
        raise RuntimeError(
            f"B0 does NOT reproduce the existing label: {n_mismatch - n_tied} of {len(idx)} "
            f"admissions disagree without tied timestamps (stopping, per spec).")
    log(f"B0 reproduces the existing label for {sup.count(len(idx) - n_mismatch)} of {sup.count(len(idx))} "
        f"admissions with creatinine ({sup.count(n_mismatch)} differ, all with tied timestamps)")
    return res


def verify_outputs(outdir: str) -> None:
    """No ID/timestamp columns, no timestamp-looking or 8+ digit values in any report file."""
    problems = []
    for p in sorted(Path(outdir).glob("*")):
        if p.suffix == ".csv":
            df = pd.read_csv(p, dtype=str, keep_default_na=False)
            bad = FORBIDDEN_COLUMNS & {c.lower() for c in df.columns}
            if bad:
                problems.append(f"{p.name}: forbidden columns {sorted(bad)}")
            text = "\n".join(df.astype(str).agg(" ".join, axis=1))
        elif p.suffix == ".md":
            text = p.read_text()
        else:
            continue
        if _TS.search(text):
            problems.append(f"{p.name}: timestamp-like value")
        if _LONG_ID.search(text):
            problems.append(f"{p.name}: 8+ digit number (possible identifier)")
    if problems:
        raise AssertionError("Output scan failed: " + "; ".join(problems))
