"""Validate a Kaggle kernel's submission.csv and print the key lines of its log.

    python scripts/check_kernel_output.py data/kernel_out
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from casmi.chem import key14  # noqa: E402

out = Path(sys.argv[1] if len(sys.argv) > 1 else "data/kernel_out")
log = next(out.glob("*.log"))
try:
    entries = json.loads(log.read_text(encoding="utf-8"))
    text = "".join(e.get("data", "") for e in entries)
except Exception:
    text = log.read_text(encoding="utf-8", errors="replace")
keep = ("python", "rdkit", "pool", "library", "vectorised", "test:", "predicted", "elapsed", "Error", "Traceback", "WARNING")
for line in text.splitlines():
    line = line.split("\r")[-1]
    if any(k in line for k in keep) and "it/s]" not in line and "s/it]" not in line:
        print("LOG |", line[:200])

sub = pd.read_csv(out / "submission.csv")
test = pd.read_parquet("data/test.parquet")
n = sub["smiles"].str.split(";").str.len()
print(f"\nrows {len(sub)}, unique ids {sub.molecule_id.nunique()}, test molecules {test.molecule_id.nunique()}")
print(f"ids match test: {set(sub.molecule_id) == set(test.molecule_id)}; nulls: {int(sub.isna().sum().sum())}")
print(f"guesses per molecule: min {n.min()}, median {int(n.median())}, max {n.max()}; fallback-only rows: {(sub.smiles == 'C').sum()}")
sample = sub["smiles"].str.split(";").explode().sample(300, random_state=0)
print(f"parsable SMILES in a 300 sample: {sum(key14(s) is not None for s in sample)}/300")
