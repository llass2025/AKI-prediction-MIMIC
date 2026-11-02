"""
ICD code sets for the concordance diagnostics.

Codes in MIMIC are stored without dots, so matching is on (icd_version, prefix).
ESRD codes are a subset of the CKD prefixes (585.6 / N18.6); ckd_status gives ESRD
precedence.
"""

import numpy as np
import pandas as pd

_CORE9 = ("5845", "5846", "5847", "5848", "5849")
_CORE10 = ("N170", "N171", "N172", "N178", "N179")

ICD_SETS = {
    "AKI_CORE":  {9: _CORE9, 10: _CORE10},
    "AKI_BROAD": {9: _CORE9 + ("586", "9975", "6693"),
                  10: _CORE10 + ("N19", "N990", "O9049")},
    "CKD":       {9: ("585",), 10: ("N18",)},
    "ESRD":      {9: ("5856", "V4511", "V56"), 10: ("N186", "Z992", "Z49")},
}


def describe_sets() -> list:
    """(set, ICD-9 prefixes, ICD-10 prefixes) rows for the summary."""
    return [(name, ", ".join(c[9]), ", ".join(c[10])) for name, c in ICD_SETS.items()]


def flag_admissions(dx: pd.DataFrame) -> pd.DataFrame:
    """
    One row per hadm_id with bool columns for each ICD set (any seq_num) and
    icd_version_group in {"9", "10", "mixed"}. Admissions without any diagnosis
    row are absent from the result (the caller treats them as ICD-negative).
    """
    d = dx[["hadm_id", "icd_code", "icd_version"]].copy()
    d["icd_code"] = d["icd_code"].astype(str).str.strip().str.upper()
    d["icd_version"] = d["icd_version"].astype(int)

    for name, by_version in ICD_SETS.items():
        hit = np.zeros(len(d), dtype=bool)
        for ver, prefixes in by_version.items():
            m = (d["icd_version"] == ver).to_numpy()
            hit |= m & d["icd_code"].str.startswith(prefixes).to_numpy()
        d[name] = hit

    flags = d.groupby("hadm_id")[list(ICD_SETS)].max()
    ver = d.groupby("hadm_id")["icd_version"].agg(["min", "max"])
    group = np.where(ver["min"] == ver["max"], ver["min"].astype(str), "mixed")
    flags["icd_version_group"] = group
    return flags


def ckd_status(ckd: pd.Series, esrd: pd.Series) -> pd.Series:
    return pd.Series(np.select([esrd.to_numpy(), ckd.to_numpy()],
                               ["esrd", "ckd_nonesrd"], default="none"),
                     index=ckd.index)
