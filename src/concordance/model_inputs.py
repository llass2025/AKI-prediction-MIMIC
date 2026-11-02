"""
Model-ready feature tables with a different AKI label (opt-in), so the existing training scripts can be rerun with
only a different --input.

Reads the paper-run feature tables (features_all.parquet and features_all_imputed.parquet), keeps the flowchart cohort
(the rows of kdigo_final_label.parquet), replaces `incident_aki_label` with the chosen variant (label_sensitivity.make_labels),
drops the patients that variant leaves out, and writes the tables under --out-dir. The columns are exactly the originals:
no extra label or onset column is added, so nothing can leak into the features. The original tables are not modified.

Variants: "KDIGO in-hospital onset" (default; positive = KDIGO with first qualifying creatinine >= 24 h after admission,
negative = KDIGO-negative, early-onset KDIGO-positive patients left out), "KDIGO final", "ICD and KDIGO agree",
"ICD, arrival-onset removed", "ICD incident" (the original label, for checking).
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .label_sensitivity import LABELS, make_labels

FEATURES_DIR = os.environ.get("AKI_FEATURES_DIR", "features")     # folder holding features_all.parquet (set AKI_FEATURES_DIR for another run)


def swap_label(feat: pd.DataFrame, labels: pd.DataFrame, variant: str) -> pd.DataFrame:
    """feat: features with version, hadm_id, incident_aki_label. labels: the kdigo_final_label frame. Returns the swapped table."""
    lab = labels.set_index(["version", "hadm_id"])
    made = make_labels(lab)[variant]
    key = pd.MultiIndex.from_frame(feat[["version", "hadm_id"]])
    new = made.reindex(key).to_numpy()                     # NaN for rows outside the cohort and for rows the variant leaves out
    out = feat.loc[~np.isnan(new)].copy()
    out["incident_aki_label"] = new[~np.isnan(new)]
    return out.reset_index(drop=True)


def slug(variant: str) -> str:
    return "".join(c.lower() if c.isalnum() else "_" for c in variant).strip("_").replace("__", "_")


def run(a) -> None:
    labels = pd.read_parquet(a.label_file)
    out_dir = Path(a.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    counts = []
    for name in ("features_all.parquet", "features_all_imputed.parquet"):
        src = Path(a.features_dir) / name
        if not src.exists():
            print(f"skipping {name}: not found in {a.features_dir}")
            continue
        feat = pd.read_parquet(src)
        out = swap_label(feat, labels, a.variant)
        assert list(out.columns) == list(feat.columns), "columns changed"
        out.to_parquet(out_dir / name, index=False)
        for ver, g in out.groupby("version"):
            counts.append(dict(table=name, version=ver, n=len(g), n_pos=int(g["incident_aki_label"].sum()),
                               n_neg=int((g["incident_aki_label"] == 0).sum())))
        print(f"{name}: {len(feat)} rows -> {len(out)} rows with label '{a.variant}' -> {out_dir / name}")
    pd.DataFrame(counts).to_csv(out_dir / "label_counts.csv", index=False)
    print(pd.DataFrame(counts).to_string(index=False))


def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog="python -m src.concordance.model_inputs",
                                 description="Write feature tables with a different AKI label for rerunning the models.")
    ap.add_argument("--features-dir", default=FEATURES_DIR)
    ap.add_argument("--label-file", default="kdigo_final_label.parquet")
    ap.add_argument("--variant", default="KDIGO in-hospital onset", choices=LABELS)
    ap.add_argument("--out-dir", default=None, help="Default: features_<variant slug>/")
    a = ap.parse_args(argv)
    if a.out_dir is None:
        a.out_dir = f"features_{slug(a.variant)}"
    return a


if __name__ == "__main__":
    run(parse_args())
