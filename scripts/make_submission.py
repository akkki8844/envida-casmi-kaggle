"""Rank candidates for every molecule in test.parquet and write submission.csv.

NOTE: the local test.parquet is a placeholder drawn from train.parquet, so its score is meaningless
(every spectrum self-matches). Use this as a format and runtime smoke test only.

    python scripts/make_submission.py [--test data/test.parquet] [--out submission.csv]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from casmi.baseline import BaselineRanker, Params, Query  # noqa: E402
from casmi.chem import ADDUCT_DELTA  # noqa: E402
from casmi.library import build_library, build_pool, load_train_meta  # noqa: E402
from casmi.spectra import spectrum_vectors, stack_csr  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data")
ap.add_argument("--cache", default="data/cache")
ap.add_argument("--test", default="data/test.parquet")
ap.add_argument("--out", default="submission.csv")
ap.add_argument("--limit", type=int, default=0, help="only the first N molecules (smoke test)")
a = ap.parse_args()

train = Path(a.data) / "train.parquet"
cache = Path(a.cache)
meta = load_train_meta(train)
pool = build_pool(meta, cache)
lib = build_library(train, meta, pool, cache)
ranker = BaselineRanker(lib, pool, Params())

test = pd.read_parquet(a.test)
rows = []
mols = list(test.groupby("molecule_id", sort=False))
if a.limit:
    mols = mols[: a.limit]
for mol_id, g in tqdm(mols, desc="molecules"):
    vecs = [spectrum_vectors(r.ms2_mzs, r.ms2_normalized_intensities, float(r.precursor_mz)) for r in g.itertuples()]
    masses = [r.precursor_mz - ADDUCT_DELTA[r.adduct] for r in g.itertuples() if r.adduct in ADDUCT_DELTA]
    if not masses:
        masses = [g["precursor_mz"].median() - 1.00728]
    q = Query(pk=stack_csr([v[0] for v in vecs]), nl=stack_csr([v[1] for v in vecs]),
              neutral_mass=float(np.median(masses)), mode=int((g["ionization_mode"] == "negative").mean() > 0.5))
    guesses, _ = ranker.rank(q)
    rows.append((mol_id, ";".join(guesses) if guesses else "C"))
sub = pd.DataFrame(rows, columns=["molecule_id", "smiles"])
sub.to_csv(a.out, index=False)
print(f"wrote {a.out}: {len(sub)} molecules")
