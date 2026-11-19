"""Print the schema and basic stats of test.parquet (a quick sanity check of the data layout)."""
import sys

import pandas as pd

path = sys.argv[1] if len(sys.argv) > 1 else "data/test.parquet"
t = pd.read_parquet(path)
print(t.dtypes.to_string())
print(len(t), "spectra,", t.molecule_id.nunique(), "molecules")
print(t.adduct.value_counts().to_string())
print(t.ionization_mode.value_counts().to_string())
print(t.groupby("molecule_id").size().describe().round(2).to_string())
r = t.iloc[0]
print(type(r.ms2_mzs), len(r.ms2_mzs), r.precursor_mz, r.collision_energy_ev)
