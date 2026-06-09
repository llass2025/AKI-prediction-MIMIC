# Intermediate Data Files — AKI Prediction Pipeline

**Paper:** Sex Differences in Acute Kidney Injury Risk and Outcomes: Insights from MIMIC-III and MIMIC-IV  
**Journal:** Intelligence-Based Medicine (under revision)  
**Code:** https://github.com/llass2025/AKI-prediction-MIMIC

All files are patient- or admission-level outputs from the reproducible pipeline (keep-autocar run).  
`data.pkl` (37GB raw MIMIC data bundle) is excluded — obtain MIMIC-III/IV via PhysioNet (credentialed access).

---

## Pipeline Order

```
data.pkl (excluded)
    ↓ compute_target_admissions.py
target_admissions.parquet
    ↓ incident_aki_target.py
incident_aki_target.parquet
    ↓ dedup_patient_level.py
target.parquet
    ↓ build_features.py
features_all.parquet
    ↓ imputation.py
features_all_imputed.parquet   ← *** PRIMARY INPUT TO ALL MODEL TRAINING ***
    ↓ train_*.py
model artifacts (not included here)
```

---

## File Descriptions

### Prediction Pipeline

| File | Rows | Description |
|---|---|---|
| `target_admissions.parquet` | 588,685 admissions | Admission-level after age ≥18 and primary AKI diagnosis exclusions. Contains AKI/CKD flags, autocar flag, demographic info. MIMIC-IV: 538,609 \| MIMIC-III: 50,076 |
| `incident_aki_target.parquet` | 588,685 admissions | Adds `incident_aki_label` (1=incident AKI, 0=no AKI, NA=not eligible), `onset_aki_flag`, `renal_impaired_at_adm`, `aki_history_flag` per admission |
| `target.parquet` | 250,492 patients | **Patient-level**, one row per patient after dedup. Keeps first AKI+ or last AKI− admission per patient. MIMIC-IV: 214,030 \| MIMIC-III: 36,462 |
| `features_all.parquet` | 250,492 patients | All features joined to patient-level cohort: 35 lab values (7-day pre-ICU), 6 vital signs (48h pre-ICU), demographics, comorbidities, fluid balance, medication history |
| `features_all_imputed.parquet` | 250,492 patients | **⭐ PRIMARY MODEL INPUT** — `features_all.parquet` after 3-tier median imputation (version+sex+age±2yr → version+age±2yr → version medians) with binary `*_missing` indicator flags added for each imputed feature. This is the direct input to all 6 prediction models (Logistic, RF, XGBoost, DNN, Self-Attention, DCN) |

### Survival Pipeline

| File | Rows | Description |
|---|---|---|
| `incident_ckd_survival.csv` | 49,501 patients | One row per AKI patient. Columns: version, subject_id, age, gender, index_admit, event_type (CKD/PostCKD/Death/Censor), time_days. MIMIC-IV: 40,392 \| MIMIC-III: 9,109 |
| `features_ckd.parquet` | 49,501 patients | Features joined to survival cohort for XGBoost CoxPH and DeepSurv models |

### AKI Timing Sensitivity (KDIGO-Anchored)

| File | Rows | Description |
|---|---|---|
| `kdigo_onset_times.parquet` | ~21,500 AKI admissions | KDIGO creatinine onset time (hours post-admission) detected from in-admission labs. Coverage: 54.7% of AKI admissions |
| `incident_ckd_survival_kdigo.csv` | 49,501 patients | Same format as `incident_ckd_survival.csv` but survival time anchored to KDIGO onset instead of admission date |

### CKD Definition Sensitivity (Lab-Based eGFR)

| File | Rows | Description |
|---|---|---|
| `lab_ckd_events.parquet` | ~12,000 AKI patients | Lab-based CKD events: eGFR <60 on two measurements ≥90 days apart (both ≥90 days post-AKI), using 2021 race-free CKD-EPI equation |
| `incident_ckd_survival_labckd.csv` | 49,501 patients | Same format as `incident_ckd_survival.csv` but using lab-based CKD definition instead of ICD codes |

---

## Key Column Definitions

**`incident_aki_label`** (in `incident_aki_target.parquet` and `target.parquet`):
- `1` = incident AKI: first-ever AKI admission for this patient, no prior AKI or CKD history
- `0` = no AKI: admission with no AKI/CKD codes and no prior renal impairment
- `NA` = not eligible: CKD-only admission, repeat AKI (not first), or prior renal impairment

**`event_type`** (in survival CSVs):
- `CKD` = developed ICD-coded CKD ≥90 days post-discharge
- `PostCKD` = progressed to ESRD/dialysis ≥90 days post-discharge
- `Death` = all-cause death ≥90 days post-discharge
- `Censor` = no event observed within follow-up window

**`is_autocar`**: binary flag for transport/auto-accident admissions (ICD E810–E838 / V00–V89). These are *retained* in the keep-autocar run.

---

## Cohort Numbers (Current Run)

| Step | MIMIC-IV | MIMIC-III |
|---|---|---|
| All adult patients (≥18y) | 223,452 | 38,552 |
| After primary AKI exclusion | 222,538 | 38,253 |
| After CKD-only exclusion (no valid label) | 214,030 | 36,462 |
| Final: Incident AKI | 40,392 (18.9%) | 9,109 (25.0%) |
| Final: No AKI | 173,638 (81.1%) | 27,353 (75.0%) |
| Post-AKI: CKD ≥90d | 9,477 (23.5%) | 274 (3.0%) |
| Post-AKI: ESRD/dialysis | 926 (2.3%) | 60 (0.7%) |
| Post-AKI: Death ≥90d | 5,552 (13.7%) | 5,410 (59.4%) |
| Post-AKI: Censored | 24,437 (60.5%) | 3,365 (36.9%) |
