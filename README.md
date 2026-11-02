# Incident AKI / CKD Survival Modeling

This repo builds an **incident AKI/CKD prediction cohort** from MIMIC-III/IV, engineers features, and trains multiple machine learning and survival models.  

It supports:  

* Cohort construction with admission-level and patient-level de-duplication  
* Exclusion of primary diagnoses and lab-based removal of pre-admission AKI  
* Feature engineering across demographics, insurance, comorbidities, medications/procedures, **pre-ICU vitals (48h)**, **pre-ICU labs (7d)**, and **fluids (48h)**  
* Flexible imputation (3-tier by version, gender, age ±2; adds *_missing flags)  
* Training/evaluation of classical ML, deep models, and survival models (CoxPH, DeepSurv)  
* External validation: train on MIMIC-IV, test on MIMIC-III  

---

## 1) Data Compilation

**Script:** `src/compile_data.py`  

Loads raw MIMIC tables and outputs a single `data.pkl` bundle.  

**Tables loaded:**
- `patients`, `admissions`, `icustays`, `services`, `transfers`  
- Diagnoses (`diagnosis`), procedures, prescriptions  
- `labitems`, `labevents`  
- `d_items` (for CHARTEVENTS itemid mapping)  
- `chartevents` (vitals + ABG pH & lactate, extendable)  
- `inputevents`, `outputevents` (urine subset)  

---

## 2) Target Construction

**Scripts:**  
- `src/compute_target_admissions.py`  
- `src/incident_aki_target.py`  
- `src/dedup_patient_level.py`  

Rules:  
1. Adults only (≥18).  
2. Exclude **primary AKI** admissions (`seq_num==1 & is_aki==1`).  
   *Primary CKD admissions are retained for AKI→CKD trajectory tracking.*  
3. Flag admission-level AKI/CKD if codes appear in later admissions.  
4. Provide `ckd_only_flag` and `ckd_admission_flag`.  
5. Flag **pre-AKI** using KDIGO labs in prior 7d/48h (not excluded, only flagged).  
6. Exclude AutoCar admissions by default (`--keep_autocar` to override).  
7. Patient-level deduplication via `dedup_patient_level.py`.  

---

## 3) Feature Engineering

**Script:** `src/build_features.py`  

Feature modules under `src/features/`:  

- `demographics.py` – age, sex, race, marital status  
- `insurance.py` – insurance categories  
- `comorbidities.py` – Charlson/Elixhauser features  
- `meds_procedures_history.py` – medication/procedure history  
- `vitals_preicu_48h.py` – vitals aggregation  
- `labs_preicu_7d.py` – lab values aggregation  
- `fluids_preicu_48h.py` – fluid balance  
- `notes.py` – clinical text features (optional)  

Features for CKD survival modeling are built in a similar way, under
ckd_survivial directory.

---

## 4) Imputation

**Script:** `src/imputation.py`  

Fast 3-tier strategy (MIMIC-III/IV):  
1. version + gender + age±2  
2. version + age±2 (ignore gender)  
3. version-specific medians  

Adds `_missing` flags by default so models can capture missingness.  

---

## 5) Model Training

- **Classical ML:** `train_logreg.py`, `train_rf.py`, `train_xgboost.py`  
- **Deep models:** `train_dnn.py`, `train_selfattn.py`, `train_dcn.py`  
- **Survival:** `train_deepsurv_time_to_ckd.py` (DeepSurv Cox model)  

---

## 6) Evaluation

- **Classification models:** AUROC, AUPRC, calibration.  
- **Survival models:** Harrell’s C-index, time-dependent AUROC/AUPRC, mean AUC, iAUC/iAUPRC.  
- **External validation:** train on MIMIC-IV, test on MIMIC-III.  

---

## 7) Quickstart

```bash
# 1) Compile data
python -m src.compile_data --output_pickle data.pkl

# 2) Build targets
python -m src.compute_target_admissions --keep_autocar
python -m src.incident_aki_target
python -m src.dedup_patient_level

# 3) Engineer features
python -m src.features.demographics        --data-pkl data.pkl --target target.parquet --outdir .
python -m src.features.insurance           --data-pkl data.pkl --target target.parquet --outdir .
python -m src.features.comorbidities       --data-pkl data.pkl --target target.parquet --outdir .
python -m src.features.meds_procedures_history --data-pkl data.pkl --target target.parquet --outdir .
python -m src.features.vitals_preicu_48h   --data-pkl data.pkl --target target.parquet --outdir .
python -m src.features.labs_preicu_7d      --data-pkl data.pkl --target target.parquet --outdir .
python -m src.features.fluids_preicu_48h   --data-pkl data.pkl --target target.parquet --outdir .
python -m src.features.notes               --data-pkl data.pkl --target target.parquet --outdir .

# 4) Merge all features
python -m src.build_features --target target.parquet --features-dir . --outdir ./features

# 5) Imputation
python -m src.imputation

# 6) Train model (example: XGBoost)
python -m src.train_xgboost --input features_ckd.parquet --out-prefix artifacts/xgb_ckd/xgb_ckd
```
---

## 8) Notes & Caveats

* **Education** not in MIMIC → omitted  
* **Pre-AKI removal** uses KDIGO labs in prior 7d window (requires `labevents`)  
* **Regex safety**: meds regex uses non-capturing groups to avoid `str.contains` warnings  
* **Memory**: CHARTEVENTS and LABEVENTS are large → use chunked I/O for extensions  
* **Advanced ICU signals** (CO, PAP, VT, VE, GCS) not currently included; can be added by extending `itemids.py`, `compile_data.py`, and `vitals_preicu_48h.py`  
* **H statistics** were calculated using R; check R markdown `RF_Hstatistics.Rmd`
---

## 9) 📂 Source Tree (abridged)
```bash
src/
  compile_data.py
  compute_target_admissions.py
  incident_aki_target.py
  dedup_patient_level.py
  build_features.py
  imputation.py
  train_logreg.py / train_rf.py / train_xgboost.py
  train_dnn.py / train_selfattn.py / train_dcn.py
  train_deepsurv_time_to_ckd.py
  RF_Hstatistics.Rmd
  features/
    demographics.py, insurance.py, comorbidities.py,
    meds_procedures_history.py, vitals_preicu_48h.py,
    labs_preicu_7d.py, fluids_preicu_48h.py, notes.py
  utils/
    io.py, itemids.py, time_windows.py, meds_maps.py,
    dx_maps.py, agg.py
```

---

## 10) AKI label concordance diagnostics (opt-in)

**Package:** `src/concordance/` — compares KDIGO creatinine staging with ICD-coded AKI at the admission level in MIMIC-IV
(D1 agreement by stage, D2 criterion split [skipped: no urine output], D3 baseline-creatinine sensitivity B0–B4,
D4 discordance decomposition, D5 stratification). It does **not** run by default, does not change `is_aki`, and does not touch
any existing output or cache. Output is aggregate only, with small cells (< `--min-cell`, default 10) suppressed.

```bash
python -m src.concordance --mimic-dir ../data/mimic-iv-3.1        # writes reports/aki_label_concordance/
bash run.sh --skip-install --skip-compile --skip-targets --skip-features \
            --skip-imputation --skip-training --concordance        # same, via run.sh
python -m unittest tests.test_concordance                          # synthetic-data tests
```

Flags: `--min-cell`, `--n-boot`, `--seed`, `--windows`, `--icd-sets`, `--creatinine-itemids`, `--pipeline-target`, `--b0-check-sample`,
`--primary-baseline` (default B3: creatinine back-calculated for eGFR 75 from age and sex; B2 = the patient's own rolling 7-day minimum),
`--no-uo`, `--no-rrt`. D5 now also has sex strata in both windows. The confirmed target always uses B2. (Urine output and dialysis are now part of the ICU KDIGO label; D2 reports which criterion fired.)

**KDIGO-confirmed incident target (opt-in, `--confirmed-target`, D6).** The pipeline's positive is an ICD code with no timing, so some
"incident" positives are AKI already present on arrival. `--confirmed-target` writes `target_kdigo_confirmed.parquet` (strict: an
ICD-positive is kept only if creatinine shows KDIGO AKI whose first qualifying value is >= `--min-onset-hours`, default 24, after
admission; the patient is dropped otherwise) and `target_kdigo_confirmed_light.parquet` (drops only positives with a clear
high-on-arrival-then-falling creatinine), same columns as `target.parquet` plus `kdigo_status`, `onset_hours`, `kdigo_confirmed`.
`target.parquet` is not modified; MIMIC-IV only. The unfiltered rebuild is checked against `target.parquet` on every run, and D6 adds
the funnel, the overlap of ICD with each KDIGO target (overall and by sex), and a 3-way ICD x KDIGO-B2 x KDIGO-B3 table.
`bash run.sh <skip flags> --confirmed-target` runs it via run.sh.
The first run streams `labevents` once and caches creatinine rows under `concordance_cache/` (gitignored).
See `reports/aki_label_concordance/summary.md` for definitions, code lists, and flagged anomalies.

**Flowchart with AKI = ICD and B3 KDIGO (opt-in).** `python -m src.concordance.flowchart` recomputes the flowchart boxes for MIMIC-IV
and MIMIC-III when AKI requires both the ICD-coded incident label and a B3-KDIGO-positive admission (ICD-coded AKI without B3 confirmation is
shown as an excluded box; No AKI is unchanged). It first recomputes the ICD-only table and stops unless every verified number is reproduced.
It also writes the OR variant (AKI = ICD-coded or B3-positive; the extra patients' survival and lab-CKD outcomes are rebuilt from the
admission-level file and first checked against the verified ICD-only numbers). Output (aggregate only): `reports/flowchart_b3/`
`flowchart_b3_table.csv` / `flowchart_b3_comparison.csv` (AND), `flowchart_b3_pre24_*.csv` (AND with the 24 h before admission added to the KDIGO window), `flowchart_b3_nadir_*.csv` (same, with the lower of the B3 estimate and the patient's lowest creatinine as baseline), `flowchart_or_table.csv` / `flowchart_or_comparison.csv` (OR), `flowchart_b3.md` (all). The MIMIC-III
reader is `src/concordance/mimic3.py` (raw v1.4 files; blood creatinine itemid 50912).

**Nadir baseline, patient-level overlap (opt-in).** `python -m src.concordance.nadir` compares ICD-coded incident AKI with KDIGO using B2, B3 and the
nadir baseline (lower of the B3 estimate and the patient's lowest creatinine, with an optional 24 h pre-admission window) on the flowchart cohort, with
onset timing, stage, sex and discordance breakdowns. Output: `reports/nadir_baseline/` (CSVs, `nadir_baseline_report.md` / `.docx`).
`python -m src.concordance.nadir --primary "B3, +24 h before admission" --tag b3_pre24 --outdir reports/b3_pre24_baseline` produces the same note for the B3 baseline with the 24 h window (`reports/b3_pre24_baseline/`).

The concordance modules that read the feature tables take them from `features/` by default; set `AKI_FEATURES_DIR=/path/to/features` (or pass `--features-all` / `--features-dir`) to use another run, for example the paper-run tables.

**Final KDIGO label (opt-in).** `python -m src.concordance.final_label` writes `kdigo_final_label.parquet` (gitignored; one row per patient of the flowchart cohort: B3 baseline, creatinine from 24 h before admission through discharge plus dialysis; stage, onset hours, incident-onset flag, concordance with the ICD label) and `reports/final_kdigo/final_label_summary.csv`. `target.parquet` is untouched. The definition, headline numbers and the final flowchart table are in `reports/final_kdigo/final_kdigo_definition.md` / `.docx`.

**Label sensitivity (opt-in).** `python -m src.concordance.label_sensitivity` retrains an XGBoost model and recomputes the sex odds ratios under five AKI label variants (ICD, KDIGO final, KDIGO in-hospital onset, ICD-and-KDIGO agreement, ICD with arrival-onset cases removed) on the flowchart cohort; output and write-up in `reports/label_sensitivity/`.

**Case for the ICD outcome.** `reports/icd_target_justification/icd_target_justification.md` / `.docx` consolidates the evidence for using ICD-coded AKI as "first recorded AKI in patients without prior kidney disease" (concordance with KDIGO, onset timing, label sensitivity, limitations, draft methods/limitations text, likely reviewer questions).

**Rerunning the models with the KDIGO in-hospital-onset label.** `python -m src.concordance.model_inputs` writes `features_kdigo_in_hospital_onset/features_all.parquet` and `features_all_imputed.parquet` (the paper-run tables with `incident_aki_label` replaced: 1 = KDIGO with onset at least 24 h after admission, 0 = KDIGO-negative, early-onset KDIGO-positive patients left out; MIMIC-IV 12,695 / 170,523, MIMIC-III 5,984 / 20,877). `bash run_kdigo_onset24_models.sh` then trains the seven models on them exactly as step 5 of `run.sh` does, writing to `artifacts_kdigo_onset24/`. `--variant` selects another label (see `src/concordance/label_sensitivity.py`).
