"""Build the cached candidate pool and spectral library from data/train.parquet.

    python scripts/prepare.py [--per-lib 4] [--max-ppm 20]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from casmi.library import build_library, build_pool, load_train_meta  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data")
ap.add_argument("--cache", default="data/cache")
ap.add_argument("--per-lib", type=int, default=4, help="max spectra kept per (structure, library)")
ap.add_argument("--max-ppm", type=float, default=20.0)
a = ap.parse_args()

train = Path(a.data) / "train.parquet"
cache = Path(a.cache)
meta = load_train_meta(train)
print(f"train metadata: {len(meta):,} spectra, {meta['inchikey14'].nunique():,} structures")
pool = build_pool(meta, cache)
print(f"pool: {len(pool.df):,} structures")
lib = build_library(train, meta, pool, cache, per_lib=a.per_lib, max_ppm=a.max_ppm)
print(f"library: {len(lib.meta):,} spectra; libs: {lib.meta['ingest_lib'].value_counts().to_dict()}")
