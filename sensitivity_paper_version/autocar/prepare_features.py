#!/usr/bin/env python3
"""
Filter aki-analysis features to exclude is_autocar=1 admissions.
Saves a filtered parquet that the train_*.py scripts consume.
"""
import argparse
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--target",   required=True)
    ap.add_argument("--out",      required=True)
    args = ap.parse_args()

    feat = pd.read_parquet(args.features)

    before = len(feat)
    excluded = (feat["is_autocar"] == 1).sum()
    feat = feat[feat["is_autocar"] != 1].drop(columns=["is_autocar"])

    print(f"Before: {before}  Excluded (autocar): {excluded}  After: {len(feat)}")
    feat.to_parquet(args.out, index=False)
    print(f"Saved -> {args.out}")

if __name__ == "__main__":
    main()
