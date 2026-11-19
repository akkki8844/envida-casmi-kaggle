"""Rank candidates for every molecule in test.parquet and write submission.csv.

NOTE: the local test.parquet is a placeholder drawn from train.parquet, so its score is meaningless
(every spectrum self-matches). Use this as a format and runtime smoke test only.

    python scripts/make_submission.py [--test data/test.parquet] [--out submission.csv] [--limit N]
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from casmi.baseline import BaselineRanker, Params  # noqa: E402
from casmi.library import build_library, build_pool, load_train_meta  # noqa: E402
from casmi.predict import predict  # noqa: E402

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
if a.limit:
    keep = test["molecule_id"].drop_duplicates().head(a.limit)
    test = test[test["molecule_id"].isin(keep)]
t0 = time.time()
sub = predict(test, ranker, progress=lambda x: tqdm(x, desc="molecules"))
print(f"{(time.time() - t0) / max(len(sub), 1):.2f}s per molecule")
sub.to_csv(a.out, index=False)
print(f"wrote {a.out}")
