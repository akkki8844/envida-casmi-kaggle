"""Turn test.parquet into a submission with any ranker that has .rank(Query) -> (smiles, diag)."""
from __future__ import annotations

import traceback

import numpy as np
import pandas as pd

from .baseline import Query
from .chem import ADDUCT_DELTA, M_PROTON
from .spectra import spectrum_vectors, stack_csr

FALLBACK = "C"  # a valid SMILES so a failed molecule never invalidates the file


def molecule_query(g: pd.DataFrame) -> Query:
    vecs = [spectrum_vectors(np.asarray(r.ms2_mzs), np.asarray(r.ms2_normalized_intensities), float(r.precursor_mz))
            for r in g.itertuples()]
    masses = [r.precursor_mz - ADDUCT_DELTA[r.adduct] for r in g.itertuples() if r.adduct in ADDUCT_DELTA]
    if not masses:  # unknown adduct: assume protonated / deprotonated by mode
        neg = (g["ionization_mode"] == "negative").to_numpy()
        masses = list(g["precursor_mz"].to_numpy() + np.where(neg, M_PROTON, -M_PROTON))
    mode = int((g["ionization_mode"] == "negative").mean() > 0.5)
    peaks = [(np.asarray(r.ms2_mzs), np.asarray(r.ms2_normalized_intensities), float(r.precursor_mz), r.adduct)
             for r in g.itertuples()]
    return Query(pk=stack_csr([v[0] for v in vecs]), nl=stack_csr([v[1] for v in vecs]),
                 neutral_mass=float(np.median(masses)), mode=mode, peaks=peaks)


def predict(test: pd.DataFrame, ranker, k: int = 25, progress=None) -> pd.DataFrame:
    rows, failures = [], 0
    groups = list(test.groupby("molecule_id", sort=False))
    it = progress(groups) if progress else groups
    for mol_id, g in it:
        try:
            guesses, _ = ranker.rank(molecule_query(g), k=k)
        except Exception:
            failures += 1
            if failures <= 3:
                traceback.print_exc()
            guesses = []
        guesses = [s for s in guesses if s and ";" not in s][:k]
        rows.append((mol_id, ";".join(guesses) if guesses else FALLBACK))
    sub = pd.DataFrame(rows, columns=["molecule_id", "smiles"])
    assert sub["molecule_id"].is_unique and sub["smiles"].notna().all()
    print(f"predicted {len(sub)} molecules ({failures} failures)")
    return sub
