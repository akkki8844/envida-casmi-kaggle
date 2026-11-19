"""Run the class-simulating validation harness on the baseline ranker.

    python scripts/evaluate.py [--n-per-class 60] [--seed 0] [--out data/eval_baseline.csv]
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from casmi.baseline import BaselineRanker, Params  # noqa: E402
from casmi.library import build_library, build_pool, load_train_meta  # noqa: E402
from casmi.validation import evaluate, make_items, summarise  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data")
ap.add_argument("--cache", default="data/cache")
ap.add_argument("--n-per-class", type=int, default=60)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--power", type=float, nargs="+", default=[3.0], help="one or more values to sweep")
ap.add_argument("--ppm", type=float, default=10.0)
ap.add_argument("--out", default="data/eval_baseline.csv")
a = ap.parse_args()

train = Path(a.data) / "train.parquet"
cache = Path(a.cache)
meta = load_train_meta(train)
pool = build_pool(meta, cache)
lib = build_library(train, meta, pool, cache)
del meta
items = make_items(lib, pool, a.n_per_class, seed=a.seed)
print(f"{len(items)} query molecules", flush=True)
frames = []
for p in a.power:
    ranker = BaselineRanker(lib, pool, Params(power=p, ppm=a.ppm))
    df = evaluate(ranker, lib, pool, items, desc=f"power={p}")
    df["power"] = p
    frames.append(df)
    print(f"\n=== power {p}, ppm {a.ppm}\n{summarise(df)}", flush=True)
pd.concat(frames).to_csv(a.out, index=False)
