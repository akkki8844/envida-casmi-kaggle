"""Build and cache the spectral library and the candidate structure pool from train.parquet."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy.sparse as sp
from tqdm import tqdm

from .chem import ADDUCT_DELTA, exact_mass, key14, morgan_bits
from .spectra import spectrum_vectors, stack_csr

META_COLS = [
    "spectrum_id", "adduct", "ionization_mode", "precursor_mz", "collision_energy_ev",
    "normalized_smiles", "inchikey14", "ingest_lib", "precursor_error_ppm", "base_peak_intensity",
]


@dataclass
class Pool:
    """Unique candidate structures. struct_id is the row index."""
    df: pd.DataFrame          # inchikey14, smiles, mass (sorted by mass)
    fps: np.ndarray           # (n, 2048) uint8, aligned with df
    key_to_id: dict[str, int]

    @property
    def mass(self) -> np.ndarray:
        return self.df["mass"].to_numpy()


@dataclass
class Library:
    meta: pd.DataFrame        # per spectrum: struct_id, ingest_lib, mode, adduct, precursor_mz, neutral_mass
    pk: sp.csr_matrix         # peak vectors, rows aligned with meta
    nl: sp.csr_matrix         # neutral-loss vectors


def load_train_meta(path: Path) -> pd.DataFrame:
    pf = pq.ParquetFile(path)
    cols = [c for c in META_COLS if c in pf.schema_arrow.names]
    df = pf.read(columns=cols).to_pandas()
    df["row"] = np.arange(len(df))
    return df


def build_pool(meta: pd.DataFrame, cache: Path) -> Pool:
    f_df, f_fp = cache / "pool.parquet", cache / "pool_fps.npy"
    if f_df.exists() and f_fp.exists():
        df = pd.read_parquet(f_df)
        return Pool(df, np.load(f_fp), {k: i for i, k in enumerate(df["inchikey14"])})
    u = meta.drop_duplicates("inchikey14")[["inchikey14", "normalized_smiles"]]
    rows = []
    for k, smi in tqdm(zip(u["inchikey14"], u["normalized_smiles"]), total=len(u), desc="pool masses"):
        m = exact_mass(smi)
        if m is not None:
            rows.append((k, smi, m))
    df = pd.DataFrame(rows, columns=["inchikey14", "smiles", "mass"]).sort_values("mass").reset_index(drop=True)
    fps = np.zeros((len(df), 2048), dtype=np.uint8)
    for i, smi in enumerate(tqdm(df["smiles"], desc="fingerprints")):
        fps[i] = morgan_bits(smi)
    cache.mkdir(parents=True, exist_ok=True)
    df.to_parquet(f_df)
    np.save(f_fp, fps)
    return Pool(df, fps, {k: i for i, k in enumerate(df["inchikey14"])})


def select_library_rows(meta: pd.DataFrame, per_lib: int, max_ppm: float, seed: int = 0) -> pd.DataFrame:
    """Quality-filter and thin: keep <= per_lib spectra per (structure, ingest_lib)."""
    m = meta[meta["adduct"].isin(ADDUCT_DELTA)]
    if "precursor_error_ppm" in m:
        m = m[m["precursor_error_ppm"].abs() <= max_ppm]
    m = m.sample(frac=1.0, random_state=seed)
    rank = m.groupby(["inchikey14", "ingest_lib"]).cumcount()
    return m[rank < per_lib].sort_values("row")


def build_library(
    train_path: Path, meta: pd.DataFrame, pool: Pool, cache: Path,
    per_lib: int = 4, max_ppm: float = 20.0, batch: int = 50_000,
) -> Library:
    f_meta, f_pk, f_nl = cache / "lib_meta.parquet", cache / "lib_pk.npz", cache / "lib_nl.npz"
    if f_meta.exists() and f_pk.exists() and f_nl.exists():
        return Library(pd.read_parquet(f_meta), sp.load_npz(f_pk), sp.load_npz(f_nl))
    sel = select_library_rows(meta, per_lib, max_ppm)
    sel = sel[sel["inchikey14"].isin(pool.key_to_id)]
    want = np.zeros(len(meta), dtype=bool)
    want[sel["row"].to_numpy()] = True
    pf = pq.ParquetFile(train_path)
    pk_v, nl_v, kept_rows = [], [], []
    offset = 0
    t0 = time.time()
    for b in tqdm(pf.iter_batches(batch_size=batch, columns=["ms2_mzs", "ms2_normalized_intensities", "precursor_mz"]),
                  total=pf.num_row_groups, desc="library spectra"):
        n = b.num_rows
        w = want[offset:offset + n]
        if w.any():
            mzs = b.column("ms2_mzs").to_pylist()
            ints = b.column("ms2_normalized_intensities").to_pylist()
            prec = b.column("precursor_mz").to_numpy(zero_copy_only=False)
            for j in np.flatnonzero(w):
                p, l = spectrum_vectors(mzs[j], ints[j], float(prec[j]))
                if len(p[0]) == 0:
                    continue
                pk_v.append(p)
                nl_v.append(l)
                kept_rows.append(offset + j)
        offset += n
    print(f"vectorised {len(kept_rows)} spectra in {time.time() - t0:.0f}s")
    lib_meta = meta.iloc[kept_rows].copy()
    lib_meta["struct_id"] = lib_meta["inchikey14"].map(pool.key_to_id).astype(np.int64)
    lib_meta["mode"] = (lib_meta["ionization_mode"] == "negative").astype(np.int8)
    delta = lib_meta["adduct"].map(ADDUCT_DELTA).to_numpy()
    lib_meta["neutral_mass"] = lib_meta["precursor_mz"].to_numpy() - delta
    keep_cols = ["row", "struct_id", "ingest_lib", "mode", "adduct", "precursor_mz", "neutral_mass", "collision_energy_ev"]
    lib_meta = lib_meta[keep_cols].reset_index(drop=True)
    pk, nl = stack_csr(pk_v), stack_csr(nl_v)
    # sort by (mode, neutral_mass) so a mass window is a contiguous slice per mode
    order = np.lexsort((lib_meta["neutral_mass"].to_numpy(), lib_meta["mode"].to_numpy()))
    lib_meta = lib_meta.iloc[order].reset_index(drop=True)
    pk, nl = pk[order], nl[order]
    cache.mkdir(parents=True, exist_ok=True)
    lib_meta.assign(collision_energy_ev=lib_meta["collision_energy_ev"].astype(str)).to_parquet(f_meta)
    sp.save_npz(f_pk, pk, compressed=False)
    sp.save_npz(f_nl, nl, compressed=False)
    return Library(lib_meta, pk, nl)
