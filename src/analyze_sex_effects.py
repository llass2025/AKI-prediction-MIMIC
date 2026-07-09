"""
Decompose the net effect of sex in the elastic net model.

The model includes:
  - dem_sex_M  : main effect of male (vs reference)
  - dem_sex_F  : main effect of female (vs reference)
  - *_M        : continuous × male and categorical × male interaction terms

Net effect of being male   = dem_sex_M + sum(all _M interaction coefficients)
Net effect of being female = dem_sex_F  (no female interactions were built)
"""

import pandas as pd
import numpy as np

coef_path = "artifacts/elasticnet_aki/elasticnet.coefficients.csv"
df = pd.read_csv(coef_path)

# ── Split into groups ─────────────────────────────────────────────────────────
main_sex   = df[df["feature"].isin(["dem_sex_M", "dem_sex_F"])]
int_M      = df[df["feature"].str.endswith("_M") & ~df["feature"].isin(["dem_sex_M"])]
other      = df[~df["feature"].str.endswith("_M") & ~df["feature"].isin(["dem_sex_M", "dem_sex_F"])]

# ── Net male effect ───────────────────────────────────────────────────────────
dem_sex_M_coef = float(df.loc[df["feature"] == "dem_sex_M", "coefficient_elasticnet"])
int_M_sum      = float(int_M["coefficient_elasticnet"].sum())
net_male       = dem_sex_M_coef + int_M_sum

dem_sex_F_coef = float(df.loc[df["feature"] == "dem_sex_F", "coefficient_elasticnet"])

print("=" * 65)
print("NET SEX EFFECTS IN ELASTIC NET MODEL")
print("=" * 65)
print(f"\ndem_sex_M (main effect)          : {dem_sex_M_coef:+.4f}")
print(f"Sum of all _M interaction terms  : {int_M_sum:+.4f}  (n={len(int_M)})")
print(f"─────────────────────────────────────────")
print(f"Net effect of MALE               : {net_male:+.4f}")
print(f"\nNet effect of FEMALE (dem_sex_F) : {dem_sex_F_coef:+.4f}  (no interactions)")
print(f"\nMale vs Female difference        : {net_male - dem_sex_F_coef:+.4f}")

# ── Top interaction terms driving the male effect ─────────────────────────────
print("\n\nTOP 20 MALE INTERACTION TERMS (by absolute coefficient)")
print("=" * 65)
int_M_sorted = int_M.reindex(int_M["coefficient_elasticnet"].abs().sort_values(ascending=False).index)
print(int_M_sorted[["feature", "coefficient_elasticnet"]].head(20).to_string(index=False))

# ── Direction breakdown ───────────────────────────────────────────────────────
pos = int_M[int_M["coefficient_elasticnet"] > 0]
neg = int_M[int_M["coefficient_elasticnet"] < 0]
zero = int_M[int_M["coefficient_elasticnet"] == 0]

print(f"\n\nINTERACTION TERM DIRECTION SUMMARY")
print("=" * 65)
print(f"Positive (increase male risk) : {len(pos):3d}  sum={pos['coefficient_elasticnet'].sum():+.4f}")
print(f"Negative (decrease male risk) : {len(neg):3d}  sum={neg['coefficient_elasticnet'].sum():+.4f}")
print(f"Zero (zeroed out by LASSO)    : {len(zero):3d}")
print(f"Total _M terms                : {len(int_M):3d}")
