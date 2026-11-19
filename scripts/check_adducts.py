"""Sanity-check the adduct delta table against train labels: neutral mass from the structure vs. from precursor_mz.

    python scripts/check_adducts.py [--n 200000]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from casmi.chem import ADDUCT_DELTA, exact_mass  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--train", default="data/train.parquet")
ap.add_argument("--n", type=int, default=200_000)
a = ap.parse_args()

df = pq.ParquetFile(a.train).read(columns=["adduct", "precursor_mz", "normalized_smiles", "precursor_error_ppm"]).to_pandas()
print("train adducts (top 30):")
print(df["adduct"].value_counts().head(30).to_string())
d = df.drop_duplicates("normalized_smiles").merge(df[["normalized_smiles"]].drop_duplicates(), on="normalized_smiles")
d = df.sample(min(a.n, len(df)), random_state=0)
cache = {s: exact_mass(s) for s in d["normalized_smiles"].unique()}
d["m"] = d["normalized_smiles"].map(cache)
d["delta"] = d["adduct"].map(ADDUCT_DELTA)
d = d.dropna(subset=["m", "delta"])
d["ppm"] = (d["precursor_mz"] - d["delta"] - d["m"]) / d["m"] * 1e6
print("\nppm error of (precursor - delta) vs structure mass, by test adduct (median / p5 / p95 / n):")
for ad, g in d.groupby("adduct"):
    print(f"{ad:>14}  {g['ppm'].median():8.2f} {g['ppm'].quantile(.05):9.2f} {g['ppm'].quantile(.95):9.2f}  {len(g):8d}")
