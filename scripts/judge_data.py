"""Job B of docs/RANK1_JUDGE.md: the private Kaggle dataset the judge's training job reads.

    python scripts/judge_data.py --extra data/coconut/coconut_pool.parquet [--val-frac 0.1]

Writes build/judge_data/:
  spectra.parquet  train.parquet rows of the fold-0 'hold' structures (never seen by fpnet_fold0.pt), test-set
                   adducts only, with a structure-disjoint judge split column ('train' / 'val')
  pool.parquet     our candidate pool (train + COCONUT) in pid order (sorted by mass): inchikey14, smiles, mass, in_train
  pool_fp.npy      packed 2048-bit Morgan fingerprints (pid order) for the hard-negative Tanimoto
  split_info.json  counts
"""
import argparse
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from casmi.chem import ADDUCT_DELTA  # noqa: E402
from casmi.library import build_pool, load_train_meta  # noqa: E402

COLS = ["spectrum_id", "inchikey14", "normalized_smiles", "adduct", "ionization_mode", "precursor_mz",
        "collision_energy_ev", "ingest_lib", "ms2_mzs", "ms2_normalized_intensities"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--extra", default=None)
    ap.add_argument("--split", default="data/ext_code/foldsafe/split.parquet")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--per-group", type=int, default=4)
    ap.add_argument("--out", default="build/judge_data")
    a = ap.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    train = Path(a.data) / "train.parquet"
    meta = load_train_meta(train)
    pool = build_pool(meta, Path(a.cache), extra=a.extra)

    hold = set(pd.read_parquet(a.split).query("fold == 'hold'").inchikey14.astype(str))
    rows = np.flatnonzero(meta.inchikey14.isin(hold).to_numpy() & meta.adduct.isin(list(ADDUCT_DELTA)).to_numpy())
    pf = pq.ParquetFile(train)
    have = set(pf.schema_arrow.names)
    t = pf.read(columns=[c for c in COLS if c in have]).take(rows).to_pandas()
    t["row"] = rows
    # at most --per-group spectra per (structure, adduct, library): keeps diversity, stops big libraries dominating
    t = t.sample(frac=1.0, random_state=0).groupby(["inchikey14", "adduct", "ingest_lib"], sort=False).head(a.per_group)
    t = t.sort_values("row").reset_index(drop=True)
    # structure-disjoint judge split, stable across runs
    t["judge"] = np.where([zlib.crc32(k.encode()) % 1000 < a.val_frac * 1000 for k in t.inchikey14], "val", "train")
    t["pid"] = t.inchikey14.map(pool.key_to_id)
    missing = int(t.pid.isna().sum())
    t = t[t.pid.notna()].copy()
    t["pid"] = t.pid.astype(np.int64)
    t.to_parquet(out / "spectra.parquet", index=False)

    df = pool.df[["inchikey14", "smiles", "mass"]].copy()
    df["in_train"] = pool.df["in_train"].to_numpy() if "in_train" in pool.df else True
    assert np.all(np.diff(df.mass.to_numpy()) >= 0), "pool must be sorted by mass (pid order)"
    df.to_parquet(out / "pool.parquet", index=False)
    np.save(out / "pool_fp.npy", np.packbits(pool.fps.astype(bool), axis=1))

    info = dict(spectra=len(t), structures=int(t.inchikey14.nunique()),
                judge_train_structures=int(t[t.judge == "train"].inchikey14.nunique()),
                judge_val_structures=int(t[t.judge == "val"].inchikey14.nunique()),
                adducts=t.adduct.value_counts().to_dict(), libs=t.ingest_lib.value_counts().to_dict(),
                missing_from_pool=missing, pool=len(df))
    (out / "split_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
