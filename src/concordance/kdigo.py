"""
KDIGO creatinine staging with pluggable baselines.

B0 is the pipeline's existing method (aki_timing_sensitivity/kdigo_onset_timing.py):
median SCr in [admit-7d, admit) with the in-admission minimum as fallback, criteria
(A) rise >= 0.3 mg/dL within 48h, (B) value >= 1.5 x baseline. That code only says
"met / not met"; staging (1.5x / 2.0x / 3.0x or SCr >= 4.0) is added here, and the
baseline is a parameter so B1-B4 run through the same path.

Everything is vectorised over a "window" of labs sorted by (hadm_id, t, value).
Times are integer seconds. SCr only: neither UO nor RRT is used by the pipeline.
"""

import importlib.util
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ABS_RISE = 0.3
ABS_WINDOW_S = 48 * 3600
BASE_WINDOW_S = 7 * 24 * 3600
PRE_ADMIT_LOOKBACK_S = 365 * 24 * 3600
STAGE_RATIOS = (1.5, 2.0, 3.0)
STAGE3_ABS = 4.0
EGFR_ASSUMED = 75.0
_GROUP_STRIDE = 10 ** 10        # seconds; separates groups when windowing a flat array


# ---------------------------------------------------------------------------
# Reuse of the existing onset script
# ---------------------------------------------------------------------------

def load_existing_onset_module():
    path = Path(__file__).resolve().parents[2] / "aki_timing_sensitivity" / "kdigo_onset_timing.py"
    spec = importlib.util.spec_from_file_location("_kdigo_onset_timing", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Windowed minima (order-independent for tied timestamps)
# ---------------------------------------------------------------------------

def rolling_min_48h(t, v, window_s: int = ABS_WINDOW_S) -> np.ndarray:
    """min of v_j over t_i - window < t_j <= t_i, ties included; t ascending."""
    t = np.asarray(t).tolist()
    v = np.asarray(v, dtype=float).tolist()
    n = len(t)
    out = [0.0] * n
    dq = deque()
    j = 0
    for i in range(n):
        ti = t[i]
        while j < n and t[j] <= ti:
            vj = v[j]
            while dq and v[dq[-1]] >= vj:
                dq.pop()
            dq.append(j)
            j += 1
        lo = ti - window_s
        while t[dq[0]] <= lo:
            dq.popleft()
        out[i] = v[dq[0]]
    return np.asarray(out, dtype=float)


def prior_min(t, v, window_s: int = BASE_WINDOW_S) -> np.ndarray:
    """min of v_j over t_i - window <= t_j < t_i (strictly earlier); NaN if none."""
    t = np.asarray(t).tolist()
    v = np.asarray(v, dtype=float).tolist()
    n = len(t)
    out = [float("nan")] * n
    dq = deque()
    j = 0
    for i in range(n):
        ti = t[i]
        while j < n and t[j] < ti:
            vj = v[j]
            while dq and v[dq[-1]] >= vj:
                dq.pop()
            dq.append(j)
            j += 1
        lo = ti - window_s
        while dq and t[dq[0]] < lo:
            dq.popleft()
        if dq:
            out[i] = v[dq[0]]
    return np.asarray(out, dtype=float)


def grouped(fn, group_codes, t, v, window_s):
    """Apply a windowed-min fn across groups in one pass by offsetting time per group."""
    off = np.asarray(group_codes, dtype=np.int64) * _GROUP_STRIDE
    return fn(np.asarray(t, dtype=np.int64) + off, v, window_s)


# ---------------------------------------------------------------------------
# Lab window container
# ---------------------------------------------------------------------------

@dataclass
class LabWindow:
    hadm: np.ndarray        # unique hadm_ids, ascending (one per group)
    starts: np.ndarray      # first lab index of each group
    counts: np.ndarray
    g: np.ndarray           # group index of every lab
    t: np.ndarray           # int64 seconds
    v: np.ndarray           # mg/dL
    prior7d: np.ndarray     # subject-level min SCr in the preceding 7d (B2 baseline)
    min48: np.ndarray       # min SCr in (t-48h, t] within the window

    @property
    def n_groups(self) -> int:
        return len(self.hadm)

    def per_lab(self, per_group) -> np.ndarray:
        return np.repeat(np.asarray(per_group, dtype=float), self.counts)

    def aligned(self, series: pd.Series) -> np.ndarray:
        return series.reindex(self.hadm).to_numpy(dtype=float)


def make_window(labs: pd.DataFrame) -> LabWindow:
    """labs: columns hadm_id, t, valuenum, prior7d_min (any order)."""
    w = labs.sort_values(["hadm_id", "t", "valuenum"], kind="mergesort").reset_index(drop=True)
    hadm_vals = w["hadm_id"].to_numpy()
    if len(w) == 0:
        z = np.zeros(0, dtype=np.int64)
        return LabWindow(z, z, z, z, z, np.zeros(0), np.zeros(0), np.zeros(0))
    uniq, starts, counts = np.unique(hadm_vals, return_index=True, return_counts=True)
    g = np.repeat(np.arange(len(uniq)), counts)
    t = w["t"].to_numpy(dtype=np.int64)
    v = w["valuenum"].to_numpy(dtype=float)
    return LabWindow(
        hadm=uniq, starts=starts, counts=counts, g=g, t=t, v=v,
        prior7d=w["prior7d_min"].to_numpy(dtype=float),
        min48=grouped(rolling_min_48h, g, t, v, ABS_WINDOW_S),
    )


# ---------------------------------------------------------------------------
# Staging
# ---------------------------------------------------------------------------

def stage_labs(v, min48, baseline):
    """
    Per-lab stage (0-3) and the stage-1 threshold a later value must fall below
    to count as "back below". baseline is per-lab (NaN = unavailable).
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        abs_met = (v - min48) >= ABS_RISE
        r1 = v >= STAGE_RATIOS[0] * baseline
        r2 = v >= STAGE_RATIOS[1] * baseline
        r3 = v >= STAGE_RATIOS[2] * baseline
        acute = abs_met | r1
        stage = np.zeros(len(v), dtype=np.int8)
        stage[acute] = 1
        stage[r2] = 2
        stage[r3 | ((v >= STAGE3_ABS) & acute)] = 3
        thr_ratio = np.where(r1, STAGE_RATIOS[0] * baseline, np.inf)
        thr_abs = np.where(abs_met, min48 + ABS_RISE, np.inf)
        thr = np.minimum(thr_ratio, thr_abs)
    return stage, thr


def stage_window(win: LabWindow, baseline_per_lab) -> pd.DataFrame:
    """
    Per-group results indexed by hadm_id:
      max_stage, first_qual_t (-1 if none), transient (max stage 1 and a value
      < the stage-1 threshold within 48h after the last qualifying value),
      has_baseline (any lab in the group had a usable baseline).
    """
    if win.n_groups == 0:
        return pd.DataFrame(columns=["max_stage", "first_qual_t", "transient", "has_baseline"],
                            index=pd.Index([], name="hadm_id"))
    stage, thr = stage_labs(win.v, win.min48, baseline_per_lab)
    s = win.starts
    q = stage >= 1
    max_stage = np.maximum.reduceat(stage, s)
    big = np.iinfo(np.int64).max
    first_q = np.minimum.reduceat(np.where(q, win.t, big), s)
    last_q = np.maximum.reduceat(np.where(q, win.t, -1), s)
    thr_last = np.minimum.reduceat(np.where(q & (win.t == last_q[win.g]), thr, np.inf), s)
    lq = last_q[win.g]
    later = (lq >= 0) & (win.t > lq) & (win.t - lq <= ABS_WINDOW_S) & (win.v < thr_last[win.g])
    transient = np.logical_or.reduceat(later, s) & (max_stage == 1)
    has_base = np.logical_or.reduceat(~np.isnan(baseline_per_lab), s)
    return pd.DataFrame({
        "max_stage": max_stage.astype(np.int8),
        "first_qual_t": np.where(first_q == big, -1, first_q),
        "transient": transient,
        "has_baseline": has_base,
    }, index=pd.Index(win.hadm, name="hadm_id"))


# ---------------------------------------------------------------------------
# Baseline variants
# ---------------------------------------------------------------------------

def backcalc_scr(age, female, egfr: float = EGFR_ASSUMED) -> np.ndarray:
    """SCr (mg/dL) that gives `egfr` under CKD-EPI 2021 (race-free) for the given age and sex."""
    age = np.asarray(age, dtype=float)
    female = np.asarray(female, dtype=bool)
    kappa = np.where(female, 0.7, 0.9)
    alpha = np.where(female, -0.241, -0.302)
    e0 = 142.0 * 0.9938 ** age * np.where(female, 1.012, 1.0)   # eGFR at SCr == kappa
    return np.where(e0 >= egfr,
                    kappa * (e0 / egfr) ** (1.0 / 1.200),
                    kappa * (egfr / e0) ** (1.0 / alpha))


def ckd_epi_2021(scr, age, female) -> np.ndarray:
    scr = np.asarray(scr, dtype=float)
    female = np.asarray(female, dtype=bool)
    kappa = np.where(female, 0.7, 0.9)
    alpha = np.where(female, -0.241, -0.302)
    return (142.0 * np.minimum(scr / kappa, 1.0) ** alpha * np.maximum(scr / kappa, 1.0) ** -1.200
            * 0.9938 ** np.asarray(age, dtype=float) * np.where(female, 1.012, 1.0))


def b0_median_baselines(adm: pd.DataFrame, labs: pd.DataFrame, chunk_subjects: int = 20000) -> pd.Series:
    """
    The existing compute_baselines() (median SCr in [admit-7d, admit), by subject),
    run in subject chunks to bound the size of its subject x admission join.
    """
    mod = load_existing_onset_module()
    a = adm[["subject_id", "hadm_id", "admittime"]].copy()
    a["version"] = "mimic4"
    subjects = a["subject_id"].unique()
    labs_by = labs[["subject_id", "charttime", "valuenum"]]
    parts = []
    for i in range(0, len(subjects), chunk_subjects):
        chunk = subjects[i:i + chunk_subjects]
        res = mod.compute_baselines(a[a["subject_id"].isin(chunk)],
                                    labs_by[labs_by["subject_id"].isin(chunk)])
        parts.append(res.set_index("hadm_id")["baseline_creat"])
    return pd.concat(parts).reindex(adm["hadm_id"])


def b4_measured(adm_s: pd.DataFrame, labs: pd.DataFrame) -> pd.Series:
    """Most recent SCr strictly before admit_s and within 365d (any hadm_id, incl. null), by hadm_id."""
    a = adm_s[["hadm_id", "subject_id", "admit_s"]].sort_values("admit_s")
    l = labs[["subject_id", "t", "valuenum"]].sort_values(["t", "valuenum"], kind="mergesort")
    m = pd.merge_asof(a, l, left_on="admit_s", right_on="t", by="subject_id",
                      direction="backward", allow_exact_matches=False)
    ok = (m["admit_s"] - m["t"]) <= PRE_ADMIT_LOOKBACK_S
    return m["valuenum"].where(ok).set_axis(m["hadm_id"]).reindex(adm_s["hadm_id"])
