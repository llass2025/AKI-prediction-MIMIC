# Session Notes — AKI Project Context
*Last updated: 2026-06-03*

---

## What This Project Is

A full ML pipeline for predicting **incident AKI** and modeling **post-AKI CKD/death survival** using MIMIC-III and MIMIC-IV EHR data. Code is public at https://github.com/noboundarystat/AKI-prediction-MIMIC.

**Authors:** Rui Feng (UPenn Biostatistics) + Linge Lass (SIG)
**Submitted to:** Intelligence-Based Medicine (under revision)
**Funding:** DataWork!Prize Phase 1 Award

---

## Paper Summary

**Title:** "Sex Differences in Acute Kidney Injury Risk and Outcomes: Insights from MIMIC-III and MIMIC-IV"

**Central idea:** Treat sex as an *effect modifier* (not nuisance covariate) in AKI prediction and post-AKI outcome modeling.

**Key results:**
- XGBoost best AUC: 0.927 (MIMIC-IV), 0.829 (MIMIC-III external validation)
- Elastic Net most transportable (smallest AUC drop: -0.077)
- Creatinine, BUN, infection history, age, hypertension = top predictors across models
- Sex interactions: creatinine stronger in men; hemoglobin more protective in women; pH protective for CKD only in men; sodium bidirectional
- H-statistics (nonlinear): age, private insurance, hypertension, CAD, cerebrovascular disease, Black race all interact with sex
- Post-AKI CKD: Black race, widowed/divorced, diabetes, HF increase risk; cancer/infection/liver disease lower CKD hazard (competing mortality)
- Post-AKI death: infection (HR 2.59), liver disease (2.61), cancer (1.71) dominate; heart failure risk attenuated in men

---

## Pipeline Overview

```
src/compile_data.py            → data.pkl
src/compute_target_admissions  → target_admissions.parquet
src/incident_aki_target.py     → incident_aki_target.parquet
src/dedup_patient_level.py     → target.parquet
src/features/*.py              → features_*.parquet
src/build_features.py          → features/features_all.parquet
src/imputation.py              → features/features_all_imputed.parquet
src/train_*.py                 → artifacts/
ckd_survival/src/              → survival models (XGBoost CoxPH, DeepSurv)
```

**Imputation:** 3-tier (version+sex+age±2 → version+age±2 → version medians) + `*_missing` flags
**Splits:** GroupShuffleSplit on subject_id (60/20/20), train MIMIC-IV, test MIMIC-III

---

## Run Structure

| Folder | What it contains |
|---|---|
| Root + `features/` + `artifacts/` | Main run — **keepautocar** (includes auto accident admissions) |
| `noautocar/` + `features_noautocar/` + `artifacts_noautocar/` | Sensitivity run — **noautocar** (excludes auto accidents) |
| `ckd_survival/` | CKD/death survival pipeline (primary, ICD-based CKD) |
| `aki_timing_sensitivity/` | AKI timing sensitivity (KDIGO-anchored survival) |
| `ckd_definition_sensitivity/` | CKD definition sensitivity (lab-based eGFR CKD) |

**Important:** `run.sh` passes `--keep_autocar`; `run_noautocar.sh` does not.

---

## Cohort Numbers

### Paper (published Table 1) — keepautocar, original run
| | MIMIC-IV | MIMIC-III |
|---|---|---|
| All adult admissions | 546,028 | 50,766 |
| Eligible cohort | 199,822 | 34,181 |
| AKI | 26,184 | 6,828 |
| No AKI | 173,638 | 27,353 |
| Post-AKI CKD (≥90d) | 2,990 | 113 |
| Post-AKI death | 4,265 | 4,104 |

### Current keepautocar run — full verified pipeline flow

All numbers verified directly from pipeline parquet files. Source files noted per step.

| Step | MIMIC-IV | MIMIC-III |
|---|---|---|
| **1. All adult admissions (≥18y)** | 546,028 | 50,766 |
| Excluded: age <18 + AKI as primary dx (seq_num=1) | −7,419 | −690 |
| **2. Eligible admissions (target_admissions.parquet)** | 538,609 | 50,076 |
| Not incident AKI eligible (label NA): CKD-only, repeat AKI, prior renal impairment | −92,280 | −5,700 |
| **3. Labeled admissions (incident_aki_target.parquet)** | | |
| &nbsp;&nbsp;Incident AKI (label = 1) | 40,392 | 9,109 |
| &nbsp;&nbsp;No AKI (label = 0) | 405,937 | 35,267 |
| Dedup: keep first AKI+ or last AKI− per patient; drop rest | −232,299 | −7,914 |
| **4. Patient-level cohort (target.parquet)** | | |
| &nbsp;&nbsp;Incident AKI | 40,392 | 9,109 |
| &nbsp;&nbsp;No AKI | 173,638 | 27,353 |
| **5. Post-AKI survival cohort (incident_ckd_survival.csv)** | N = 40,392 | N = 9,109 |
| &nbsp;&nbsp;Incident CKD (≥90d post-discharge) | 9,477 | 274 |
| &nbsp;&nbsp;Progression to ESRD/dialysis | 926 | 60 |
| &nbsp;&nbsp;All-cause death (≥90d post-discharge) | 5,552 | 5,410 |
| &nbsp;&nbsp;Censored | 24,437 | 3,365 |

**Note:** AKI counts differ from paper (26,184 / 6,828) because original target.parquet was overwritten during re-runs. Paper numbers survive only in `artifacts/logreg_aki/logreg.predictions.csv`. The current numbers are fully reproducible from committed code.

---

## Autocar Sensitivity Analysis ✅ COMPLETE

**Finding:** Results extremely stable. MIMIC-IV AUC differences ≤0.003 across all models.

| Model | MIMIC-IV (keepauto) | MIMIC-IV (noauto) | MIMIC-III (keepauto) | MIMIC-III (noauto) |
|---|---|---|---|---|
| XGBoost | 0.917 | 0.916 | 0.807 | 0.784 |
| Logistic | 0.870 | 0.873 | 0.771 | 0.735 |
| Random Forest | 0.886 | 0.883 | 0.788 | 0.761 |
| DNN | 0.904 | 0.903 | 0.710 | 0.704 |
| Self-Attn | 0.907 | 0.908 | 0.733 | 0.724 |
| DCN | 0.902 | 0.902 | 0.694 | 0.698 |

**Output:** `sensitivity_autocar.docx`

---

## AKI Timing Sensitivity Analysis ✅ COMPLETE

**What:** Re-anchors post-AKI survival models to first KDIGO creatinine onset instead of admission time.
**Scope:** Survival models only — prediction model stays admission-anchored (feature leakage + task redefinition if shifted).

**KDIGO onset detection:**
- 54.7% of AKI admissions had KDIGO criteria confirmed in-admission labs
- Median onset: 38h after admission
- MIMIC-III coverage 67.8% vs MIMIC-IV 51.8% (ICU-only = more intensive monitoring)

**Survival time shift:** Minimal — median 1–3 days across event types (CKD: −1d, death: −2.7d)

**Model results:**

| Split | ICD C-idx (primary) | KDIGO C-idx | ICD AUROC | KDIGO AUROC |
|---|---|---|---|---|
| **CKD — Test** | 0.621 | 0.639 (+0.018) | 0.732 | 0.685 (−0.047) |
| **CKD — MIMIC-III** | 0.467 | 0.500 (+0.033) | 0.771 | 0.718 (−0.053) |
| **Death — Test** | 0.613 | 0.625 (+0.012) | 0.758 | 0.780 (+0.022) |
| **Death — MIMIC-III** | 0.535 | 0.528 (−0.006) | 0.725 | 0.726 (+0.000) |

**Conclusion:** Primary findings robust. KDIGO anchoring produces modest, consistent changes with no metric reversals.

**Outputs:** `aki_timing_sensitivity/` — docx, kdigo_onset_timing.py, kdigo_onset_times.parquet, incident_ckd_survival_kdigo.csv, features, model artifacts

---

## CKD Definition Sensitivity Analysis ✅ COMPLETE

**What:** Replaces ICD-coded CKD with lab-based CKD using 2021 race-free CKD-EPI equation (eGFR < 60 on two measurements ≥90 days apart, both ≥90 days post-AKI).

**ICD vs Lab-CKD concordance — key finding:**

| Version | ICD-CKD % | Lab-CKD % | Both | ICD-only | Lab-only |
|---|---|---|---|---|---|
| MIMIC-IV | 25.8% | 12.1% | 3,053 | 7,350 | 1,817 |
| MIMIC-III | 3.7% | 9.6% | 148 | 186 | 723 |

**Critical finding:** MIMIC-III *reversal* — ICD under-codes CKD (3.7%) while lab evidence finds 9.6%. Directly supports reviewer concern about healthcare-utilization bias: MIMIC-III AKI patients who developed CKD were often not followed at BIDMC and never received the ICD code.

**58–77% of patients lack sufficient post-AKI creatinine data** for lab-CKD assessment — own selection bias.

**Model performance — three-way comparison (test set):**

| Definition | C-index | AUROC | AUPRC |
|---|---|---|---|
| ICD-based (primary) | 0.621 | 0.732 | 0.768 |
| KDIGO-anchored | 0.639 | 0.685 | 0.707 |
| Lab-CKD (eGFR<60) | 0.555 | 0.571 | 0.367 |

Lab-CKD harder to predict — expected given sparse follow-up and different outcome population. MIMIC-III C-index improves slightly (0.467→0.534), consistent with removing ICD coding noise.

**Outputs:** `ckd_definition_sensitivity/` — docx, lab_ckd_definition.py, lab_ckd_events.parquet, lab_ckd_summary.csv, incident_ckd_survival_labckd.csv, features, model artifacts

---

## Uncertainty Intervals ✅ COMPLETE

**What:** Bootstrap 95% confidence intervals on AUROC and AUPRC for all 6 AKI prediction models, on both test set (MIMIC-IV) and MIMIC-III external validation.

**Method:** Percentile bootstrap, 1,000 resamples with replacement, seed=42. Reads from each model's `predictions.csv` (columns: `split`, `incident_aki_label`, `prob`). Skips degenerate resamples where only one class is present.

**Key results (test set AUROC 95% CI):**

| Model | Test AUROC (95% CI) | MIMIC-III AUROC (95% CI) |
|---|---|---|
| XGBoost | 0.917 (0.913–0.921) | 0.807 (0.801–0.813) |
| Self-Attn | 0.907 (0.903–0.910) | 0.733 (0.727–0.740) |
| DNN | 0.904 (0.901–0.908) | 0.710 (0.704–0.716) |
| DCN | 0.902 (0.898–0.905) | 0.694 (0.686–0.701) |
| Random Forest | 0.886 (0.881–0.891) | 0.788 (0.782–0.795) |
| Logistic | 0.870 (0.866–0.874) | 0.771 (0.765–0.777) |

CIs are narrow (±0.002–0.004 on test set), confirming stability. MIMIC-III CIs slightly wider due to external distribution shift.

**Outputs:** `uncertainty_intervals/` — bootstrap_ci.py, run_bootstrap_ci.sh, artifacts/bootstrap_ci_summary.csv, uncertainty_intervals.docx

---

## Age × Sex Interaction (RCS) ✅ COMPLETE

**What:** Re-fit Elastic Net logistic regression replacing linear age term with restricted cubic splines (RCS, 4 knots) plus explicit age × sex interaction terms, to formally model how the age-risk relationship differs between males and females.

**Method:**
- Harrell RCS parameterization, 4 knots at 5th/35th/65th/95th percentiles of training-set age (~25/52/67/87 yrs)
- 2 spline basis columns (age_rcs1, age_rcs2) + linear age term retained
- 2 interaction terms: male × age_rcs1, male × age_rcs2
- Same Elastic Net (l1_ratio=0.5, C=1.0, SAGA), same 60/20/20 splits, same imputation/scaling

**Key results:**

| Split | AUROC | AUPRC | N |
|---|---|---|---|
| Test (MIMIC-IV) | 0.877 | 0.576 | 39,965 |
| MIMIC-III (ext.) | 0.757 | 0.449 | 34,181 |
| Primary linear (test) | 0.870 | 0.539 | 59,947 |

**Key interaction coefficients (survive Elastic Net regularization):**
- male × age_rcs1 = **−0.527** (largest interaction; dampens male age-risk curve at intermediate ages)
- male × age_rcs2 = **+0.377** (increases male risk at higher ages)
- age linear = +0.688, age_rcs1 = −0.292 (nonlinear deceleration overall)

**Interpretation:** Opposing signs of the two interaction terms confirm the male age-risk curve has a meaningfully different shape than the female curve — directly addresses reviewer request and reinforces H-statistic findings in the main paper.

**Outputs:** `age_sex_rcs/` — rcs_logreg.py, run_age_sex_rcs.sh, artifacts/rcs_metrics.json, rcs_coefficients.csv, rcs_predictions.csv, age_sex_rcs.docx

---

## Calibration (Plots + Brier Score) ✅ COMPLETE

**What:** Calibration curves, Brier score, Expected Calibration Error (ECE), and Hosmer-Lemeshow test for all 6 AKI prediction models on test set and MIMIC-III external validation. Sex-stratified calibration plots for XGB, RF, and Self-Attn.

**Key results:**

| Model | Test Brier | Test BSS | Test ECE | MIMIC-III Brier | MIMIC-III ECE |
|---|---|---|---|---|---|
| XGBoost | 0.066 | 0.429 | 0.007 | 0.125 | 0.039 |
| DNN | 0.071 | 0.377 | 0.007 | 0.195 | 0.191 |
| Self-Attn | 0.070 | 0.383 | 0.012 | 0.141 | 0.031 |
| DCN | 0.072 | 0.365 | 0.013 | 0.194 | 0.191 |
| Random Forest | 0.075 | 0.347 | 0.013 | 0.125 | 0.042 |
| Logistic | 0.142 | −0.243 | 0.219 | 0.172 | 0.174 |

**Key findings:**
- XGBoost best calibrated on test (ECE=0.007, BSS=0.429)
- Logistic regression is poorly calibrated (ECE=0.219, negative BSS) — probabilities are not well-scaled, expected for Elastic Net without Platt scaling
- DNN/DCN degrade significantly on MIMIC-III (ECE jumps to ~0.19) — calibration doesn't transfer well externally
- XGBoost and RF maintain reasonable MIMIC-III calibration (ECE 0.039/0.042)
- All H-L p-values = 0.0 (trivial at these sample sizes — H-L is overpowered with N>30k)

**Outputs:** `calibration/` — calibration_utils.py, run_calibration.sh, artifacts/calibration_summary.csv, per-model calibration PNGs, sex-stratified PNGs, calibration.docx

---

## Reviewer Revisions Status

| Item | Status |
|---|---|
| **Flow diagram** | Two versions created: `flow_diagram_paper.docx` (published numbers) + `flow_diagram_current.docx` (current run) |
| **Transport/autocar sensitivity** | ✅ Complete — `sensitivity_autocar.docx` |
| **AKI timing (KDIGO onset)** | ✅ Complete — `aki_timing_sensitivity/` |
| **CKD definition (lab-based eGFR)** | ✅ Complete — `ckd_definition_sensitivity/` |
| **Uncertainty intervals** | ✅ Complete — `uncertainty_intervals/` (percentile bootstrap, 1000 resamples, AUROC + AUPRC with 95% CIs per model per split) |
| **Age × sex interaction (RCS)** | ✅ Complete — `age_sex_rcs/` (4-knot RCS, Harrell parameterization, male×rcs interaction terms, AUROC 0.877 test / 0.757 MIMIC-III) |
| **Calibration (plots + Brier score)** | ✅ Complete — `calibration/` (calibration curves + Brier score + ECE + H-L test per model per split, sex-stratified for XGB/RF/Self-Attn) |
| **TRIPOD+AI checklist** | ✅ Complete — `tripod_ai_checklist.docx` |

---

## Key Files

| File | Purpose |
|---|---|
| `AKI_RF_LL_IBM.pdf` | Submitted manuscript (22 pages) |
| `Response_Intelligence_Based_Medicine.docx` | Point-by-point reviewer response |
| `AKI_Cohort_Flow_Diagram.docx` | Original flow diagram template |
| `flow_diagram_paper.docx` | Flow diagram — published numbers |
| `flow_diagram_current.docx` | Flow diagram — current run numbers |
| `sensitivity_autocar.docx` | Autocar sensitivity write-up + table |
| `aki_timing_sensitivity/aki_timing_sensitivity.docx` | AKI timing sensitivity write-up + results |
| `ckd_definition_sensitivity/ckd_definition_sensitivity.docx` | CKD definition sensitivity write-up + results |
| `data.pkl` | Compiled MIMIC data bundle |
| `target.parquet` | Current keepautocar patient-level target |
| `artifacts/logreg_aki/logreg.predictions.csv` | Original paper cohort (paper numbers preserved here) |
| `ckd_survival/incident_ckd_survival.csv` | Primary post-AKI survival dataset (ICD-based) |
| `ckd_survival/artifacts/` | Primary CKD + death survival model artifacts |
| `aki_timing_sensitivity/artifacts/` | KDIGO-anchored survival model artifacts |
| `ckd_definition_sensitivity/artifacts/` | Lab-CKD survival model artifacts |

---

## Intermediate Datasets — Full Pipeline

All files listed below are produced by the current reproducible run. Data files (.parquet, .pkl, .csv with individual-level data) are not committed to git — only code and summary artifacts are committed.

### Prediction Pipeline (root directory)

| File | Produced by | What it contains |
|---|---|---|
| `data.pkl` | `compile_data.py` | Raw MIMIC-III + IV tables merged into one dict (admissions, patients, diagnoses, labs, etc.) |
| `target_admissions.parquet` | `compute_target_admissions.py` | Admission-level after age ≥18 + primary AKI exclusions. 538,609 (MIMIC-IV) + 50,076 (MIMIC-III) |
| `incident_aki_target.parquet` | `incident_aki_target.py` | Admission-level with incident_aki_label (1/0/NA), renal history flags, onset_aki_flag |
| `target.parquet` | `dedup_patient_level.py` | Patient-level, one row per patient. 214,030 (MIMIC-IV) + 36,462 (MIMIC-III) |
| `features/features_all.parquet` | `build_features.py` | All features merged at patient level (labs, vitals, demographics, comorbidities, etc.) |
| `features/features_all_imputed.parquet` | `imputation.py` | Features after 3-tier median imputation + missingness flags |
| `artifacts/*/` | `train_*.py` | Model artifacts, predictions CSV, metrics JSON, feature importance per model |
| `cohort_counts.csv` | `compute_target_admissions.py` | Step-by-step counts through the admission-level pipeline |
| `dedup_summary.csv` | `dedup_patient_level.py` | Patient counts after dedup (positive/negative/NA) |
| `incident_aki_summary.csv` | `incident_aki_target.py` | Label counts (1/0/NA) per run |
| `missingness_labs_preicu7d.csv` | `features/labs_preicu7d.py` | Lab missingness rates per feature |

### Survival Pipeline (`ckd_survival/`)

| File | Produced by | What it contains |
|---|---|---|
| `ckd_survival/target_admissions.parquet` | `ckd_survival/src/compute_target_admissions.py` | AKI patients eligible for post-AKI survival follow-up |
| `ckd_survival/incident_ckd_survival.csv` | survival pipeline | One row per AKI patient: event_type (CKD/PostCKD/Death/Censor) + time_days. 40,392 (MIMIC-IV) + 9,109 (MIMIC-III) |
| `ckd_survival/features_ckd.parquet` | `merge_ckd_features.py` | Features for survival models (joined to AKI cohort) |
| `ckd_survival/artifacts/` | survival training scripts | CKD + death model artifacts (XGBoost CoxPH, DeepSurv) |

### Sensitivity Analysis Datasets

| File | What it contains |
|---|---|
| `aki_timing_sensitivity/kdigo_onset_times.parquet` | KDIGO creatinine onset time per AKI admission (hours post-admission) |
| `aki_timing_sensitivity/incident_ckd_survival_kdigo.csv` | Survival dataset re-anchored to KDIGO onset instead of admission date |
| `aki_timing_sensitivity/artifacts/` | KDIGO-anchored survival model artifacts |
| `ckd_definition_sensitivity/lab_ckd_events.parquet` | Lab-based eGFR <60 CKD events (2 measurements ≥90d apart, race-free CKD-EPI) |
| `ckd_definition_sensitivity/incident_ckd_survival_labckd.csv` | Survival dataset with lab-CKD as outcome instead of ICD codes |
| `ckd_definition_sensitivity/artifacts/` | Lab-CKD survival model artifacts |
| `features_noautocar/` | Feature set for no-autocar sensitivity run |
| `artifacts_noautocar/` | Model artifacts for no-autocar sensitivity run |

### Summary / Analysis Artifacts (committed to git)

| File | What it contains |
|---|---|
| `uncertainty_intervals/artifacts/bootstrap_ci_summary.csv` | AUROC + AUPRC with 95% bootstrap CIs per model per split |
| `calibration/artifacts/calibration_summary.csv` | Brier score, ECE, H-L test per model per split |
| `age_sex_rcs/artifacts/rcs_metrics.json` | AUROC/AUPRC for RCS logistic model |
| `age_sex_rcs/artifacts/rcs_coefficients.csv` | All RCS model coefficients sorted by magnitude |
| `age_sex_rcs/artifacts/rcs_age_sex_curves.png` | Age-risk curves by sex with bootstrap CIs |
| `calibration/artifacts/cal_*.png` | Calibration curve plots per model per split |

---

## Bug Fixes Made During Session

| Script | Fix |
|---|---|
| `src/train_xgboost.py` | Added full evaluation, metrics JSON, predictions CSV, feature importance — was only saving H-stats |
| `ckd_survival/src/compute_target_admissions.py` | Fixed hardcoded `../../data/` path to use script-relative path |
| `ckd_survival/src/features/merge_ckd_features.py` | Added binary event indicators (is_ckd, is_death, etc.); preserve is_ckd from targets if already set |
| `ckd_survival/src/train_xgboost_time_to_ckd.py` | Added time_days_kdigo, time_days_original, hours_to_onset, kdigo_found to auto_exclude |
| `ckd_survival/src/train_xgboost_time_to_death.py` | Same auto_exclude fix |

---

## What "Sensitivity Analysis" Means in This Paper

Re-running the same analysis under different methodological assumptions to test robustness:
1. **Exclude transport admissions** — remove trauma/car crash patients (distinct AKI profile) ✅
2. **KDIGO onset timing** — use first creatinine criterion met as index time (vs. admission date) for survival models ✅
3. **Lab-based CKD** — use sustained eGFR < 60 (vs. ICD codes) as CKD outcome ✅
