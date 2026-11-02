"""Run configuration for the concordance diagnostics (mirrors the CLI flags)."""

import os
from dataclasses import dataclass
from typing import Optional, Tuple

WINDOWS = ("W_ICU", "W_HOSP")
ICD_SET_CHOICES = ("AKI_CORE", "AKI_BROAD")
BASELINES = ("B0", "B1", "B2", "B3", "B4")


@dataclass
class Config:
    mimic_dir: str = os.environ.get("MIMIC_IV_DIR", "../data/mimic-iv-3.1")
    outdir: str = "reports/aki_label_concordance"
    cache_dir: str = "concordance_cache"
    refresh_cache: bool = False
    min_cell: int = 10
    n_boot: int = 1000
    seed: int = 2024
    windows: Tuple[str, ...] = WINDOWS
    icd_sets: Tuple[str, ...] = ICD_SET_CHOICES      # AKI_CORE is always primary
    # 50912 (Creatinine, Blood, Chemistry) plus 52546 (same label/fluid/category).
    creatinine_itemids: Tuple[int, ...] = (50912, 52546)
    pipeline_target: Optional[str] = "target_admissions.parquet"
    modeling_target: Optional[str] = "target.parquet"     # final patient-level modeling set (incident_aki_label)
    b0_check_sample: int = 0                          # 0 = check every admission
    primary_baseline: str = "B3"                      # baseline behind D1/D2/D4/D5 (D3 shows all); confirmed target is always B2
    use_uo: bool = True                               # urine-output criterion (W_ICU only)
    use_rrt: bool = True                              # dialysis-start criterion (stage 3)
    # opt-in: KDIGO-confirmed incident target (confirmed.py); needs the modeling target and the admission-level one
    write_confirmed: bool = False
    incident_target: str = "incident_aki_target.parquet"
    confirmed_out: str = "target_kdigo_confirmed.parquet"    # 'light' rule goes to <name>_light.parquet
    min_onset_hours: float = 24.0
    drop_kdigo_neg: bool = False                      # also drop ICD-negative admissions that are KDIGO-confirmed
    quiet: bool = False

    def validate(self) -> None:
        if self.min_cell < 1:
            raise ValueError("--min-cell must be >= 1")
        if self.n_boot < 0:
            raise ValueError("--n-boot must be >= 0")
        bad = [w for w in self.windows if w not in WINDOWS]
        if bad or not self.windows:
            raise ValueError(f"--windows must be a non-empty subset of {WINDOWS}")
        bad = [s for s in self.icd_sets if s not in ICD_SET_CHOICES]
        if bad:
            raise ValueError(f"--icd-sets must be a subset of {ICD_SET_CHOICES}")
        if self.primary_baseline not in BASELINES:
            raise ValueError(f"--primary-baseline must be one of {BASELINES}")
        if self.min_onset_hours < 0:
            raise ValueError("--min-onset-hours must be >= 0")
        if "AKI_CORE" not in self.icd_sets:
            self.icd_sets = ("AKI_CORE",) + tuple(self.icd_sets)

    def as_rows(self):
        return [
            ("MIN_CELL", self.min_cell),
            ("bootstrap resamples", self.n_boot),
            ("seed", self.seed),
            ("windows", ", ".join(self.windows)),
            ("ICD sets in D1", ", ".join(self.icd_sets)),
            ("creatinine itemids", ", ".join(str(i) for i in self.creatinine_itemids)),
            ("B0 check sample (0 = all)", self.b0_check_sample),
            ("primary baseline", self.primary_baseline),
            ("urine-output criterion (W_ICU)", "on" if self.use_uo else "off"),
            ("dialysis-start criterion", "on" if self.use_rrt else "off"),
        ] + ([("confirmed target: min onset (h after admission)", self.min_onset_hours),
              ("confirmed target: also drop KDIGO+ negatives", "yes" if self.drop_kdigo_neg else "no")]
             if self.write_confirmed else [])
